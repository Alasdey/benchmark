#!/usr/bin/env python3
"""
EventStoryLine v0.9 — ECI dataprep (master)
============================================
Schema aligned with Nofing/CausalTimeBank-standard.

Decisions
---------
version       v0.9  — the only version benchmarked by published ECI papers.
mentions      ACTION_* tags only (event triggers).
relations     PLOT_LINK relType ∈ {PRECONDITION, FALLING_ACTION}
              → causes [src, tgt] / caused_by [tgt, src].
              All annotated pairs kept regardless of sentence distance.
dev topics    {37, 41} — Caselli & Vossen (2017), confirmed in HOTECI source.
              split="dev", excluded from 5-fold CV.
fold          doc_idx % 5  (train docs only).
coref         No propagation — ESL v0.9 has directed coreference cycles.

Output schema (one row per document)
-------------------------------------
  id          str
  doc_idx     int          stable 0-based index, sorted by (topic_id, filename)
  topic_id    int          numeric topic folder (1, 3, … 41)
  split       str          "train" | "dev"
  tokens      list[str]    flat token list (doc-level)
  mentions    list[str]    m_id strings, sorted by first token position
  spans       list[list[int]]  token indices per mention (0-based, doc-level)
  relations   dict         {"causes":    [[src_idx, tgt_idx], …],
                             "caused_by": [[tgt_idx, src_idx], …]}
  sentences   list[list[int]]  [[start, end), …] token boundary slices

Fold usage
----------
  train = ds.filter(lambda x: x["split"] == "train" and x["doc_idx"] % 5 != k)
  test  = ds.filter(lambda x: x["split"] == "train" and x["doc_idx"] % 5 == k)
  dev   = ds.filter(lambda x: x["split"] == "dev")

References
----------
  Caselli & Vossen, ACL 2017  https://aclanthology.org/W17-2711
  Man et al. (DiffusECI), AAAI 2024
  Liao et al. (CLECI), Information 2026
"""
from __future__ import annotations

import os
import glob
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, Features, Sequence, Value

# ── constants ─────────────────────────────────────────────────────────────────

DEV_TOPICS: set[int] = {37, 41}

ACTION_TAGS: set[str] = {
    "ACTION_OCCURRENCE", "ACTION_REPORTING", "ACTION_PERCEPTION",
    "ACTION_ASPECTUAL",  "ACTION_STATE",     "ACTION_CAUSATIVE",
    "ACTION_GENERIC",    "ACTION_GOLD",       "ACTION_SILVER",
    "NEG_ACTION_OCCURRENCE", "NEG_ACTION_REPORTING", "NEG_ACTION_PERCEPTION",
    "NEG_ACTION_ASPECTUAL",  "NEG_ACTION_STATE",     "NEG_ACTION_CAUSATIVE",
    "NEG_ACTION_GENERIC",
}

CAUSAL_RELTYPES: set[str] = {"PRECONDITION", "FALLING_ACTION"}

# ── XML parser ────────────────────────────────────────────────────────────────

def _parse_xml(path: str, doc_idx: int, topic_id: int) -> Optional[Dict]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        print(f"  Skipping malformed XML {path}: {exc}")
        return None

    doc_id: str = root.get("doc_name") or os.path.splitext(os.path.basename(path))[0]

    # 1. Tokens — each <token> carries a `sentence` integer attribute
    tokens: List[str] = []
    tid_to_pos: Dict[int, int] = {}
    pos_to_sent: Dict[int, int] = {}

    for tok in sorted(root.findall(".//token"), key=lambda t: int(t.get("t_id", 0))):
        t_id  = int(tok.get("t_id", 0))
        sent  = int(tok.get("sentence", 0))
        pos   = len(tokens)
        tokens.append(tok.text or "")
        tid_to_pos[t_id] = pos
        pos_to_sent[pos]  = sent

    # 2. Sentence boundaries [start, end)
    sent_buckets: Dict[int, List[int]] = defaultdict(list)
    for pos, sid in pos_to_sent.items():
        sent_buckets[sid].append(pos)

    sentences: List[List[int]] = [
        [sorted(v)[0], sorted(v)[-1] + 1]
        for _, v in sorted(sent_buckets.items())
    ]

    # 3. ACTION mentions, sorted by first token position
    raw: List[Tuple[int, str, List[int]]] = []
    for m in root.findall(".//Markables/*"):
        if m.tag not in ACTION_TAGS:
            continue
        m_id = m.get("m_id")
        if not m_id:
            continue
        anchors = sorted(
            tid_to_pos[int(a.get("t_id"))]
            for a in m.findall("token_anchor")
            if int(a.get("t_id", 0)) in tid_to_pos
        )
        if anchors:
            raw.append((anchors[0], m_id, anchors))

    raw.sort(key=lambda x: x[0])
    mentions: List[str]        = [r[1] for r in raw]
    spans:    List[List[int]]  = [r[2] for r in raw]
    mid_to_idx: Dict[str, int] = {m: i for i, m in enumerate(mentions)}

    if len(mentions) < 2:
        return None

    # 4. Causal relations — all PLOT_LINKs, no sentence filter
    causes:     List[List[int]] = []
    caused_by:  List[List[int]] = []

    for rel in root.findall(".//Relations/PLOT_LINK"):
        if rel.get("relType", "") not in CAUSAL_RELTYPES:
            continue
        src_el = rel.find("source")
        tgt_el = rel.find("target")
        if src_el is None or tgt_el is None:
            continue
        src_idx = mid_to_idx.get(src_el.get("m_id", ""))
        tgt_idx = mid_to_idx.get(tgt_el.get("m_id", ""))
        if src_idx is None or tgt_idx is None:
            continue
        causes.append([src_idx, tgt_idx])
        caused_by.append([tgt_idx, src_idx])

    return {
        "id":       doc_id,
        "doc_idx":  doc_idx,
        "topic_id": topic_id,
        "split":    "dev" if topic_id in DEV_TOPICS else "train",
        "tokens":   tokens,
        "mentions": mentions,
        "spans":    spans,
        "relations": {
            "causes":    causes,
            "caused_by": caused_by,
        },
        "sentences": sentences,
    }

# ── dataset builder ───────────────────────────────────────────────────────────

def build_dataset(root_dir: str) -> Dataset:
    xml_files = glob.glob(os.path.join(root_dir, "**", "*.xml"), recursive=True)
    if not xml_files:
        raise FileNotFoundError(f"No .xml files found under {root_dir}")

    def _key(p: str) -> Tuple[int, str]:
        t = os.path.basename(os.path.dirname(p))
        return (int(t) if t.isdigit() else 0, os.path.basename(p))

    xml_files = sorted(xml_files, key=_key)

    rows: List[Dict] = []
    for doc_idx, path in enumerate(xml_files):
        t = os.path.basename(os.path.dirname(path))
        row = _parse_xml(path, doc_idx, int(t) if t.isdigit() else 0)
        if row is not None:
            rows.append(row)

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_dev   = sum(1 for r in rows if r["split"] == "dev")
    n_ce    = sum(len(r["relations"]["causes"]) for r in rows)

    print(f"Documents : {len(rows)}  ({n_train} train / {n_dev} dev)")
    print(f"causes    : {n_ce}  →  {n_ce * 2} directional")

    features = Features({
        "id":       Value("string"),
        "doc_idx":  Value("int64"),
        "topic_id": Value("int64"),
        "split":    Value("string"),
        "tokens":   Sequence(Value("string")),
        "mentions": Sequence(Value("string")),
        "spans":    Sequence(Sequence(Value("int64"))),
        "relations": {
            "causes":    Sequence(Sequence(Value("int64"))),
            "caused_by": Sequence(Sequence(Value("int64"))),
        },
        "sentences": Sequence(Sequence(Value("int64"))),
    })
    return Dataset.from_list(rows, features=features)

# ── HF push ───────────────────────────────────────────────────────────────────

def push_to_hub(dataset: Dataset, repo_id: str, token: Optional[str], private: bool) -> None:
    dataset.push_to_hub(
        repo_id,
        token=token or os.getenv("HF_TOKEN"),
        private=private,
    )
    print(f"Pushed → https://huggingface.co/datasets/{repo_id}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EventStoryLine v0.9 → ECI HF dataset."
    )
    p.add_argument(
        "--root_dir",
        default="data/EventStoryLine/annotated_data/v0.9",
        help="Root folder containing per-topic subfolders of v0.9 XML files.",
    )
    p.add_argument(
        "--repo_id",
        default="Nofing/EventStoryLine-master",
        help="HuggingFace dataset repo to push to.",
    )
    p.add_argument("--private", action="store_true")
    p.add_argument("--token", default=None, help="HF token (falls back to $HF_TOKEN).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ds = build_dataset(root_dir=args.root_dir)
    push_to_hub(ds, repo_id=args.repo_id, token=args.token, private=args.private)
