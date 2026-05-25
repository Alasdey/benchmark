#!/usr/bin/env python3
"""
EventStoryLine v0.9 — ECI (Event Causality Identification) dataprep
====================================================================
Produces a dataset aligned with Nofing/CausalTimeBank-standard:

  • Dataset version   : v0.9 (the standard benchmarked by all SOTA papers)
  • Mentions          : ACTION_* only (event triggers)
  • Relations         : PLOT_LINK with relType PRECONDITION or FALLING_ACTION
                        → both mapped to directional CauseEffect / EffectCause
  • Negative pairs    : all intra-sentence ordered pairs not in CauseEffect/EffectCause
                        → stored as NoRel
  • Dev topics        : 37 and 41 (excluded from fold CV, marked split="dev")
                        Source: Caselli & Vossen (2017); confirmed in HOTECI impl.
  • Fold assignment   : doc_idx % 5  (train docs only; dev docs carry split="dev")
  • No coref propagation (ESL has directed cycles that break transitivity)

Output schema per row (identical to CausalTimeBank-standard, plus topic_id / split):
  id          str
  doc_idx     int                stable sequential index (across all docs, sorted)
  topic_id    int                numeric topic folder (e.g. 1, 3, … 41)
  split       str                "train" | "dev"
  tokens      list[str]          flat token list (all sentences)
  mentions    list[str]          mention IDs present, position-ordered
  spans       list[list[int]]    token index lists per mention (0-based, doc-level)
  relations   dict               {"CauseEffect": [[src, tgt], ...],
                                  "EffectCause": [[tgt, src], ...],
                                  "NoRel":       [[i, j], ...]}
  sentences   list[[int, int]]   sentence token boundary pairs [start, end)
  pair_list   list[[int, int]]   all directed intra-sentence pairs (mention indices)

Fold usage (train split only):
  train_fold = ds.filter(lambda x: x["split"] == "train" and x["doc_idx"] % 5 != k)
  test_fold  = ds.filter(lambda x: x["split"] == "train" and x["doc_idx"] % 5 == k)
  dev        = ds.filter(lambda x: x["split"] == "dev")

References
  Caselli & Vossen (ESC, ACL 2017)   https://aclanthology.org/W17-2711
  Man et al. (DiffusECI, AAAI 2024)
  Liao et al. (CLECI, Information 2026)
"""
from __future__ import annotations

import os
import glob
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple

from datasets import Dataset, Features, Sequence, Value


# ──────────────────────── constants ──────────────────────────────────────────

# Topic folder numbers used as development set (not included in 5-fold CV)
DEV_TOPICS = {37, 41}

# Only these tag prefixes are kept as event mentions (action triggers)
ACTION_TAGS = {
    "ACTION_OCCURRENCE", "ACTION_REPORTING", "ACTION_PERCEPTION",
    "ACTION_ASPECTUAL", "ACTION_STATE", "ACTION_CAUSATIVE", "ACTION_GENERIC",
    "NEG_ACTION_OCCURRENCE", "NEG_ACTION_REPORTING", "NEG_ACTION_PERCEPTION",
    "NEG_ACTION_ASPECTUAL", "NEG_ACTION_STATE", "NEG_ACTION_CAUSATIVE",
    "NEG_ACTION_GENERIC", "ACTION_GOLD", "ACTION_SILVER",
}

# PLOT_LINK relTypes that encode causality (both map to CauseEffect direction)
CAUSAL_RELTYPES = {"PRECONDITION", "FALLING_ACTION"}


# ──────────────────────── XML parser ─────────────────────────────────────────

def _parse_xml(path: str, doc_idx: int, topic_id: int) -> Dict | None:
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        print(f"  Skipping malformed XML {path}: {exc}")
        return None

    root = tree.getroot()
    doc_id = root.get("doc_name") or os.path.splitext(os.path.basename(path))[0]

    # ── 1. Tokens + sentence boundaries ──────────────────────────────────────
    # Each <token> carries a `sentence` attribute (integer sentence id).
    tokens: List[str] = []
    tid_to_docidx: Dict[int, int] = {}   # t_id (1-based) → 0-based token index
    sent_id_of_token: Dict[int, int] = {}  # token doc-index → sentence id

    raw_tokens = sorted(root.findall(".//token"), key=lambda t: int(t.get("t_id", 0)))
    for tok in raw_tokens:
        t_id = int(tok.get("t_id", 0))
        sent_id = int(tok.get("sentence", 0))
        doc_pos = len(tokens)
        tokens.append(tok.text or "")
        tid_to_docidx[t_id] = doc_pos
        sent_id_of_token[doc_pos] = sent_id

    # Build sentences as [start, end) token slices from the sent_id mapping
    sent_buckets: Dict[int, List[int]] = defaultdict(list)
    for doc_pos, sent_id in sent_id_of_token.items():
        sent_buckets[sent_id].append(doc_pos)

    sentences: List[List[int]] = []
    for sent_id in sorted(sent_buckets):
        positions = sorted(sent_buckets[sent_id])
        sentences.append([positions[0], positions[-1] + 1])  # [start, end)

    # ── 2. Mentions (ACTION_* only) ───────────────────────────────────────────
    mentions: List[str] = []
    spans: List[List[int]] = []
    mid_to_idx: Dict[str, int] = {}   # raw m_id → position in mentions list

    # Collect and sort by first token position for stable ordering
    raw_mentions = []
    for m in root.findall(".//Markables/*"):
        if m.tag not in ACTION_TAGS:
            continue
        m_id = m.get("m_id")
        if not m_id:
            continue
        anchors = sorted(
            [tid_to_docidx[int(a.get("t_id"))]
             for a in m.findall("token_anchor")
             if int(a.get("t_id", 0)) in tid_to_docidx]
        )
        if not anchors:
            continue
        raw_mentions.append((anchors[0], m_id, anchors))

    raw_mentions.sort(key=lambda x: x[0])
    for _, m_id, anchors in raw_mentions:
        idx = len(mentions)
        mid_to_idx[m_id] = idx
        mentions.append(m_id)
        spans.append(anchors)

    if len(mentions) < 2:
        return None   # nothing to pair

    # ── 3. Sentence membership per mention ───────────────────────────────────
    midx_to_sent: Dict[int, int] = {}
    for m_idx, span in enumerate(spans):
        if span:
            midx_to_sent[m_idx] = sent_id_of_token.get(span[0], -1)

    # ── 4. Causal relations (PLOT_LINK only, intra-sentence) ─────────────────
    cause_effect: List[List[int]] = []
    effect_cause: List[List[int]] = []

    for rel in root.findall(".//Relations/PLOT_LINK"):
        rel_type = rel.get("relType", "")
        if rel_type not in CAUSAL_RELTYPES:
            continue
        src_el = rel.find("source")
        tgt_el = rel.find("target")
        if src_el is None or tgt_el is None:
            continue
        src_mid = src_el.get("m_id")
        tgt_mid = tgt_el.get("m_id")
        if src_mid not in mid_to_idx or tgt_mid not in mid_to_idx:
            continue
        src_idx = mid_to_idx[src_mid]
        tgt_idx = mid_to_idx[tgt_mid]
        # Intra-sentence filter
        if midx_to_sent.get(src_idx, -1) != midx_to_sent.get(tgt_idx, -2):
            continue
        cause_effect.append([src_idx, tgt_idx])
        effect_cause.append([tgt_idx, src_idx])

    # ── 5. pair_list: all directed intra-sentence ordered pairs ──────────────
    sent_to_midxs: Dict[int, List[int]] = defaultdict(list)
    for m_idx, sent_id in midx_to_sent.items():
        sent_to_midxs[sent_id].append(m_idx)

    pair_list: List[List[int]] = []
    for midxs in sent_to_midxs.values():
        if len(midxs) < 2:
            continue
        midxs_sorted = sorted(midxs, key=lambda i: spans[i][0])
        for i in midxs_sorted:
            for j in midxs_sorted:
                if i != j:
                    pair_list.append([i, j])

    labeled = {(p[0], p[1]) for p in cause_effect + effect_cause}
    no_rel = [[i, j] for i, j in pair_list if (i, j) not in labeled]

    return {
        "id": doc_id,
        "doc_idx": doc_idx,
        "topic_id": topic_id,
        "split": "dev" if topic_id in DEV_TOPICS else "train",
        "tokens": tokens,
        "mentions": mentions,
        "spans": spans,
        "relations": {
            "CauseEffect": cause_effect,
            "EffectCause": effect_cause,
            "NoRel": no_rel,
        },
        "sentences": sentences,
        "pair_list": pair_list,
    }


# ──────────────────────── dataset builder ────────────────────────────────────

def build_dataset(root_dir: str) -> Dataset:
    # Gather all XML files, sorted by (topic_id, filename) for stable doc_idx
    xml_files = glob.glob(os.path.join(root_dir, "**", "*.xml"), recursive=True)
    if not xml_files:
        raise FileNotFoundError(f"No .xml files found under {root_dir}")

    def _sort_key(p: str) -> Tuple[int, str]:
        topic_str = os.path.basename(os.path.dirname(p))
        try:
            topic_id = int(topic_str)
        except ValueError:
            topic_id = 0
        return (topic_id, os.path.basename(p))

    xml_files = sorted(xml_files, key=_sort_key)

    rows = []
    for doc_idx, path in enumerate(xml_files):
        topic_str = os.path.basename(os.path.dirname(path))
        try:
            topic_id = int(topic_str)
        except ValueError:
            topic_id = 0
        row = _parse_xml(path, doc_idx, topic_id)
        if row is not None:
            rows.append(row)

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_dev   = sum(1 for r in rows if r["split"] == "dev")
    n_ce    = sum(len(r["relations"]["CauseEffect"]) for r in rows)
    n_pairs = sum(len(r["pair_list"]) for r in rows)
    n_norel = sum(len(r["relations"]["NoRel"]) for r in rows)
    print(f"Parsed {len(rows)} documents  ({n_train} train / {n_dev} dev)")
    print(f"Total intra-sentence pairs  : {n_pairs}")
    print(f"CauseEffect (unique)        : {n_ce}")
    print(f"Directional labeled pairs   : {n_ce * 2}  (CauseEffect + EffectCause)")
    print(f"NoRel pairs                 : {n_norel}")
    print(f"Imbalance ratio             : 1 : {n_norel // max(n_ce * 2, 1)}")

    features = Features({
        "id":       Value("string"),
        "doc_idx":  Value("int64"),
        "topic_id": Value("int64"),
        "split":    Value("string"),
        "tokens":   Sequence(Value("string")),
        "mentions": Sequence(Value("string")),
        "spans":    Sequence(Sequence(Value("int64"))),
        "relations": {
            "CauseEffect": Sequence(Sequence(Value("int64"))),
            "EffectCause": Sequence(Sequence(Value("int64"))),
            "NoRel":       Sequence(Sequence(Value("int64"))),
        },
        "sentences": Sequence(Sequence(Value("int64"))),
        "pair_list": Sequence(Sequence(Value("int64"))),
    })
    return Dataset.from_list(rows, features=features)


# ──────────────────────── HF push ────────────────────────────────────────────

def push_to_hub(dataset: Dataset, repo_id: str, token: str | None, private: bool) -> None:
    dataset.push_to_hub(
        repo_id,
        token=token or os.getenv("HF_TOKEN"),
        private=private,
    )
    print(f"Dataset pushed: https://huggingface.co/datasets/{repo_id}")


# ─────────────────────────── CLI ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Parse EventStoryLine v0.9 XML files into ECI format "
            "(intra-sentence PLOT_LINKs only) and push to HuggingFace Hub."
        )
    )
    p.add_argument(
        "--root_dir",
        default="data/EventStoryLine/annotated_data/v0.9",
        help="Path to the v0.9 root folder containing per-topic subfolders.",
    )
    p.add_argument(
        "--repo_id",
        default="Nofing/EventStoryLine-0.9-standard",
        help="Target HF dataset repo.",
    )
    p.add_argument("--private", action="store_true")
    p.add_argument("--token", default=None, help="HF token (defaults to $HF_TOKEN env-var).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ds = build_dataset(root_dir=args.root_dir)
    push_to_hub(ds, repo_id=args.repo_id, token=args.token, private=args.private)
