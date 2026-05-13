#!/usr/bin/env python3
"""
Parse Causal-TimeBank TimeML .tml files and push to HuggingFace Hub.

Each document yields:
    id        : document ID (from <DOCID>)
    tokens    : list[str]  (whitespace-tokenised TEXT content)
    mentions  : list[eiid] (one per MAKEINSTANCE whose event appears in TEXT)
    spans     : list[list[int]]  (token indices, 0-based)
    relations : dict[relType, list[[src_idx, tgt_idx]]]
                includes TLINK relTypes (BEFORE, AFTER, …) and CLINK
"""
from __future__ import annotations

import os
import glob
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

from datasets import Dataset


# ─────────────────────────── parsing helpers ──────────────────────────────────

def _extract_tokens_and_spans(
    text_el: ET.Element,
) -> Tuple[List[str], Dict[str, List[int]]]:
    """Walk <TEXT>, tokenise on whitespace, return tokens + eid→token-indices.

    EVENT child text is tracked by eid; TIMEX3 and C-SIGNAL text is included in
    the token stream but not mapped to any mention.
    """
    tokens: List[str] = []
    eid_to_span: Dict[str, List[int]] = {}

    def _add(text: str | None, eid: str | None = None) -> None:
        if not text:
            return
        words = text.split()
        if not words:
            return
        start = len(tokens)
        tokens.extend(words)
        if eid is not None:
            eid_to_span[eid] = list(range(start, len(tokens)))

    _add(text_el.text)
    for child in text_el:
        _add(child.text, eid=child.get("eid") if child.tag == "EVENT" else None)
        _add(child.tail)

    return tokens, eid_to_span


def _parse_tml(path: str) -> Dict | None:
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        print(f"⚠️  Skipping malformed XML {path}: {exc}")
        return None
    root = tree.getroot()

    # doc id
    docid_el = root.find("DOCID")
    doc_id = (
        docid_el.text.strip()
        if docid_el is not None
        else os.path.splitext(os.path.basename(path))[0]
    )

    # 1. tokens + event-span map
    text_el = root.find("TEXT")
    if text_el is None:
        print(f"⚠️  No <TEXT> element in {path}, skipping")
        return None
    tokens, eid_to_span = _extract_tokens_and_spans(text_el)

    # 2. eiid → eid via MAKEINSTANCE
    eiid_to_eid: Dict[str, str] = {}
    for mi in root.findall("MAKEINSTANCE"):
        eiid = mi.get("eiid")
        eid = mi.get("eventID")
        if eiid and eid:
            eiid_to_eid[eiid] = eid

    # 3. mentions + spans  (only eiids whose EVENT surface appears in TEXT)
    mentions: List[str] = []
    spans: List[List[int]] = []
    for eiid, eid in eiid_to_eid.items():
        span = eid_to_span.get(eid)
        if span is not None:
            mentions.append(eiid)
            spans.append(span)

    eiid_to_idx = {m: i for i, m in enumerate(mentions)}

    # 4. relations — TLINKs (event-event only) + CLINKs
    relations: Dict[str, List[List[int]]] = {}

    for tlink in root.findall("TLINK"):
        rel = tlink.get("relType", "UNKNOWN")
        src = tlink.get("eventInstanceID")
        tgt = tlink.get("relatedToEventInstance")   # None for event-time links
        if src and tgt and src in eiid_to_idx and tgt in eiid_to_idx:
            relations.setdefault(rel, []).append(
                [eiid_to_idx[src], eiid_to_idx[tgt]]
            )

    for clink in root.findall("CLINK"):
        src = clink.get("eventInstanceID")
        tgt = clink.get("relatedToEventInstance")
        if src and tgt and src in eiid_to_idx and tgt in eiid_to_idx:
            relations.setdefault("CLINK", []).append(
                [eiid_to_idx[src], eiid_to_idx[tgt]]
            )

    return {
        "id": doc_id,
        "tokens": tokens,
        "mentions": mentions,
        "spans": spans,
        "relations": relations,
    }


# ─────────────────────────── dataset builder ──────────────────────────────────

def build_dataset(root_dir: str) -> Dataset:
    files = sorted(glob.glob(os.path.join(root_dir, "**", "*.tml"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .tml files found under {root_dir}")

    rows = [_parse_tml(f) for f in files]
    rows = [r for r in rows if r is not None]
    print(f"Parsed {len(rows)} / {len(files)} documents.")
    return Dataset.from_list(rows)


# ─────────────────────────── HF push ──────────────────────────────────────────

def push_to_hub(
    dataset,
    repo_id: str,
    token: str | None,
    private: bool,
) -> None:
    dataset.push_to_hub(
        repo_id,
        token=token or os.getenv("HF_TOKEN"),
        private=private,
    )
    print(f"✅  Dataset pushed: https://huggingface.co/datasets/{repo_id}")


# ─────────────────────────── CLI ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Parse Causal-TimeBank .tml files and push a HuggingFace dataset."
    )
    p.add_argument(
        "--root_dir",
        default="data/CausalTimeBank/TimeML",
        help="Directory containing .tml files (after unzipping Causal-TimeBank-TimeML.zip).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--repo_id", required=True,
                   help="Target HuggingFace dataset repo, e.g. user/my-dataset.")
    p.add_argument("--private", action="store_true")
    p.add_argument("--token", default=None,
                   help="HF token (defaults to $HF_TOKEN env-var).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ds = build_dataset(args.root_dir)

    if args.seed is not None:
        ds = ds.shuffle(seed=args.seed)
    if args.test_size:
        ds = ds.train_test_split(test_size=args.test_size, seed=args.seed or 0)

    push_to_hub(ds, repo_id=args.repo_id, token=args.token, private=args.private)
