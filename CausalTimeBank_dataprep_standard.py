#!/usr/bin/env python3
"""
CausalTimeBank — ECI (Event Causality Identification) dataprep (directional)
=============================================================================
Variant of CausalTimeBank_dataprep_aligned.py restricted to the ECI task,
with directional CLINK labeling:

  • relations contains "CauseEffect" and "EffectCause" — no TLINKs.
  • Each annotated CLINK a→b produces two entries:
      "CauseEffect": [a_idx, b_idx]   (a causes b)
      "EffectCause": [b_idx, a_idx]   (b is the effect of a)
    This aligns with pair_list, which covers both orderings of every pair.
  • Only intra-sentence CLINKs are kept (286 / 318 = 90% of annotations).
    Cross-sentence CLINKs are dropped: they fall outside pair_list scope
    and no published ECI paper on this corpus evaluates them.
  • pair_list covers all annotated positive CLINKs (both directions) by construction.

Output schema per row (identical fields to _aligned, narrower relations):
  id         str
  doc_idx    int                stable sequential document index
  tokens     list[str]          flat token list (all sentences)
  mentions   list[str]          eiids present in TEXT, position-ordered
  spans      list[list[int]]    token index lists per mention (0-based, doc-level)
  relations  dict               {"CauseEffect": [[src_idx, tgt_idx], ...],
                                 "EffectCause": [[tgt_idx, src_idx], ...],
                                 "NoRel":       [[i, j], ...]}  ← all unlabeled pairs
  sentences  list[[int, int]]   sentence token boundary pairs [start, end)
  pair_list  list[[int, int]]   directed intra-sentence pairs (mention indices)

Fold assignment (done after loading, not baked in):
  ds = ds.map(lambda x: {"fold": x["doc_idx"] % 5})
  train = ds.filter(lambda x: x["fold"] != k)
  test  = ds.filter(lambda x: x["fold"] == k)

References
  Mirza & Tonelli (CATENA, COLING 2016)  https://aclanthology.org/C16-1007
  Ning et al. (ACL 2018)                 https://aclanthology.org/P18-1212
  Zuo et al. (COLING 2020)               https://aclanthology.org/2020.coling-main.135
"""
from __future__ import annotations

import os
import glob
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple

from datasets import Dataset, Features, Sequence, Value


# ─────────────────── sentence-aware TEXT element parsing ─────────────────────

def _extract_tokens_and_sentences(
    text_el: ET.Element,
) -> Tuple[List[str], Dict[str, List[int]], List[List[int]]]:
    """Walk <TEXT>, split on \\n for sentence boundaries.

    Returns:
        tokens       : flat list of all tokens (document-level)
        eid_to_span  : eid → list of token indices (document-level, 0-based)
        sentences    : [[start, end), ...] token boundary slices per sentence
    """
    tokens: List[str] = []
    eid_to_span: Dict[str, List[int]] = {}
    sentence_starts: List[int] = [0]

    def _add(text: str | None, eid: str | None = None) -> None:
        if not text:
            return
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i > 0 and tokens:
                sentence_starts.append(len(tokens))
            words = line.split()
            if not words:
                continue
            if eid is not None:
                start = len(tokens)
                tokens.extend(words)
                eid_to_span[eid] = list(range(start, len(tokens)))
            else:
                tokens.extend(words)

    _add(text_el.text)
    for child in text_el:
        child_eid = child.get("eid") if child.tag == "EVENT" else None
        _add(child.text, eid=child_eid)
        _add(child.tail)

    sentence_ends = sentence_starts[1:] + [len(tokens)]
    sentences = [[s, e] for s, e in zip(sentence_starts, sentence_ends) if s < e]
    return tokens, eid_to_span, sentences


# ──────────────────────────────── .tml parser ────────────────────────────────

def _parse_tml(path: str, doc_idx: int) -> Dict | None:
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        print(f"  Skipping malformed XML {path}: {exc}")
        return None
    root = tree.getroot()

    docid_el = root.find("DOCID")
    doc_id = (
        docid_el.text.strip()
        if docid_el is not None
        else os.path.splitext(os.path.basename(path))[0]
    )

    text_el = root.find("TEXT")
    if text_el is None:
        print(f"  No <TEXT> in {path}, skipping")
        return None

    tokens, eid_to_span, sentences = _extract_tokens_and_sentences(text_el)

    # eiid → eid (MAKEINSTANCE links instance IDs to surface events)
    eiid_to_eid: Dict[str, str] = {}
    for mi in root.findall("MAKEINSTANCE"):
        eiid = mi.get("eiid")
        eid = mi.get("eventID")
        if eiid and eid:
            eiid_to_eid[eiid] = eid

    # Position-order eiids by their first token index in the document
    eiid_span_pairs = [
        (eiid, eid_to_span[eid])
        for eiid, eid in eiid_to_eid.items()
        if eid in eid_to_span
    ]
    eiid_span_pairs.sort(key=lambda x: x[1][0])

    mentions = [eiid for eiid, _ in eiid_span_pairs]
    spans = [span for _, span in eiid_span_pairs]
    eiid_to_idx = {eiid: i for i, eiid in enumerate(mentions)}

    # Map each mention index to its sentence for intra-sentence filtering
    midx_to_sent: Dict[int, int] = {}
    for sent_idx, (sent_start, sent_end) in enumerate(sentences):
        for m_idx, span in enumerate(spans):
            if span and sent_start <= span[0] < sent_end:
                midx_to_sent[m_idx] = sent_idx

    # Intra-sentence CLINKs only (ECI task scope), stored directionally
    cause_effect: List[List[int]] = []
    effect_cause: List[List[int]] = []
    for clink in root.findall("CLINK"):
        src = clink.get("eventInstanceID")
        tgt = clink.get("relatedToEventInstance")
        if not (src and tgt and src in eiid_to_idx and tgt in eiid_to_idx):
            continue
        src_idx, tgt_idx = eiid_to_idx[src], eiid_to_idx[tgt]
        if midx_to_sent.get(src_idx) == midx_to_sent.get(tgt_idx):
            cause_effect.append([src_idx, tgt_idx])
            effect_cause.append([tgt_idx, src_idx])

    relations: Dict[str, List[List[int]]] = {
        "CauseEffect": cause_effect,
        "EffectCause": effect_cause,
    }

    # pair_list: all directed intra-sentence event-pair candidates
    sent_to_midxs: Dict[int, List[int]] = defaultdict(list)
    for m_idx, sent_idx in midx_to_sent.items():
        sent_to_midxs[sent_idx].append(m_idx)

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
    relations["NoRel"] = no_rel

    return {
        "id": doc_id,
        "doc_idx": doc_idx,
        "tokens": tokens,
        "mentions": mentions,
        "spans": spans,
        "relations": relations,
        "sentences": sentences,
        "pair_list": pair_list,
    }


# ──────────────────────────── dataset builder ────────────────────────────────

def build_dataset(root_dir: str) -> Dataset:
    files = sorted(glob.glob(os.path.join(root_dir, "**", "*.tml"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .tml files found under {root_dir}")

    rows = [_parse_tml(f, idx) for idx, f in enumerate(files)]
    rows = [r for r in rows if r is not None]
    print(f"Parsed {len(rows)} / {len(files)} documents.")

    n_clinks = sum(len(r["relations"]["CauseEffect"]) for r in rows)
    n_pairs = sum(len(r["pair_list"]) for r in rows)
    n_norel = sum(len(r["relations"]["NoRel"]) for r in rows)
    print(f"Total intra-sentence pairs : {n_pairs}")
    print(f"Intra-sentence CLINKs      : {n_clinks}  (cross-sentence dropped)")
    print(f"Labeled directional pairs  : {n_clinks * 2}  (CauseEffect + EffectCause)")
    print(f"NoRel pairs                : {n_norel}")
    print(f"Imbalance ratio            : 1 : {n_norel // max(n_clinks * 2, 1)}")

    features = Features({
        "id": Value("string"),
        "doc_idx": Value("int64"),
        "tokens": Sequence(Value("string")),
        "mentions": Sequence(Value("string")),
        "spans": Sequence(Sequence(Value("int64"))),
        "relations": {
            "CauseEffect": Sequence(Sequence(Value("int64"))),
            "EffectCause": Sequence(Sequence(Value("int64"))),
            "NoRel": Sequence(Sequence(Value("int64"))),
        },
        "sentences": Sequence(Sequence(Value("int64"))),
        "pair_list": Sequence(Sequence(Value("int64"))),
    })
    return Dataset.from_list(rows, features=features)


# ──────────────────────────────── HF push ────────────────────────────────────

def push_to_hub(dataset: Dataset, repo_id: str, token: str | None, private: bool) -> None:
    dataset.push_to_hub(
        repo_id,
        token=token or os.getenv("HF_TOKEN"),
        private=private,
    )
    print(f"Dataset pushed: https://huggingface.co/datasets/{repo_id}")


# ─────────────────────────────────── CLI ─────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Parse Causal-TimeBank .tml files into ECI format "
            "(intra-sentence CLINKs only) and push to HuggingFace Hub."
        )
    )
    p.add_argument(
        "--root_dir",
        default="data/CausalTimeBank/TimeML",
        help="Directory containing .tml files (after unzipping Causal-TimeBank-TimeML.zip).",
    )
    p.add_argument("--repo_id", default="Nofing/CausalTimeBank-standard", help="Target HF dataset repo, e.g. user/my-dataset.")
    p.add_argument("--private", action="store_true")
    p.add_argument("--token", default=None, help="HF token (defaults to $HF_TOKEN env-var).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ds = build_dataset(root_dir=args.root_dir)
    push_to_hub(ds, repo_id=args.repo_id, token=args.token, private=args.private)
