"""
Pre-processing TimeML-style SML XML files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
• Crawls 1.5/**.xml
• Extracts tokens, mentions, spans and relations
• Builds a HuggingFace Dataset and uploads it
"""

from __future__ import annotations
import os, glob, json
import xml.etree.ElementTree as ET
from typing import List, Dict
from datasets import Dataset
import argparse

# ---------- low-level helpers -------------------------------------------------
def _sorted_by_int_attr(elems, attr):
    """Return elements sorted by the integer value of `attr`."""
    return sorted(elems, key=lambda el: int(el.get(attr, "0")))

def _parse_single_xml(path: str, exclude: str = []) -> Dict:
    """Parse one SML file and return a single row ready for HF Datasets."""
    tree = ET.parse(path)
    root = tree.getroot()

    doc_id = root.get("doc_name") or os.path.basename(path)
    if os.path.basename(path) in exclude or doc_id in exclude:
        # Skip anything whose basename *or* doc_name is black-listed
        print(f"🚫  Skipping excluded file: {path}")
        return None
    
    # 1. tokens ----------------------------------------------------------------
    tokens_el = _sorted_by_int_attr(root.findall(".//token"), "t_id")
    tokens: List[str] = [tok.text or "" for tok in tokens_el]

    # 2. mentions & spans ------------------------------------------------------
    mentions, spans = [], []
    markables = root.find("Markables") or ET.Element("dummy")         # safety
    for m in markables:
        m_id = m.get("m_id")
        if not m_id:           # skip malformed markables
            continue
        anchors = _sorted_by_int_attr(m.findall("token_anchor"), "t_id")
        t_ids = [int(a.get("t_id")) for a in anchors]
        if len(t_ids)>0:
            mentions.append(m_id)
            # spans.append([min(t_ids)-1, max(t_ids)])
            spans.append([t_id-1 for t_id in t_ids])

    # 3. relations -------------------------------------------------------------
    relations: Dict[str, List[List[str]]] = {}
    relations_el = root.find("Relations") # or ET.Element("dummy")
    for rel in relations_el:
        rel_type = rel.get("relType", "UNKNOWN")
        if rel_type == "": # Debug
            print("This is why there is a blank rel type", path) 
        # if rel_type == "UNKNOWN": # Debug
        #     print("This is why there is an UNKNOWN rel type", path) 
        src_id  = rel.findtext("source/@m_id") if rel.find("source") is None else rel.find("source").get("m_id")
        tgt_id  = rel.findtext("target/@m_id") if rel.find("target") is None else rel.find("target").get("m_id")
        if src_id and tgt_id:
            if src_id in mentions and tgt_id in mentions:
                src_idx = mentions.index(src_id)
                tgt_idx = mentions.index(tgt_id)
                relations.setdefault(rel_type, []).append([src_idx, tgt_idx])
        # if src_id not in mentions or tgt_id not in mentions: # Debug
        #     print("Missing mentions", path, rel, src_id, tgt_id) 

    # 4. meta ------------------------------------------------------------------
    doc_id = root.get("doc_name") or os.path.basename(path)

    return {
        "id":        doc_id,
        "tokens":    tokens,
        "mentions":  mentions,
        "spans":     spans,
        "relations": relations,
    }

# ---------- dataset builder ---------------------------------------------------
def build_dataset(root_dir: str = "1.5", exclude: str = []) -> Dataset:
    """Walk `root_dir/**.xml`, parse files, and return a HuggingFace Dataset."""
    xml_files = glob.glob(os.path.join(root_dir, "**", "*.xml"), recursive=True)
    if not xml_files:
        raise FileNotFoundError(f"No XML files found under {root_dir}/**.xml")

    rows = [_parse_single_xml(p, exclude) for p in xml_files]
    rows.remove(None)
    
    return Dataset.from_list(rows)

# ---------- push to HuggingFace Hub ------------------------------------------
def push_to_hub(
    dataset: Dataset,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> None:
    """Push the dataset to the HF Hub; `token` can come from the env var HF_TOKEN."""
    dataset.push_to_hub(
        repo_id,
        token=token or os.getenv("HF_TOKEN"),
        private=private,
    )
    print(f"✅ Dataset pushed: https://huggingface.co/datasets/{repo_id}")

# ─────────────────────────── CLI ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Parse SML XML files and push a HuggingFace dataset."
    )
    p.add_argument("--root_dir", default="data/EventStoryLine/annotated_data/v1.5",
                   help="Root directory containing XML files.")
    p.add_argument("--exclude", default="1_10ecbplus.xml",
                   help="Comma-separated list of filenames (or doc_name) to skip.")
    p.add_argument("--seed", type=int, default=42,
                   help="Shuffle seed (omit to skip shuffling).")
    p.add_argument("--test_size", type=float, default=0.1,
                   help="Train/test split fraction, e.g. 0.1 (omit for no split).")
    p.add_argument("--repo_id", required=True,
                   help="Target HuggingFace dataset repo, e.g. user/my-dataset.")
    p.add_argument("--private", action="store_true",
                   help="If set, push the dataset privately.")
    p.add_argument("--token", default=None,
                   help="HF token (defaults to $HF_TOKEN env-var).")
    return p.parse_args()

# ─────────────────────────── main ─────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    exclude_set = {e.strip() for e in args.exclude.split(",") if e.strip()}

    # 1) Build
    ds = build_dataset(args.root_dir, exclude_set)

    # 2) Optional shuffle / split
    if args.seed is not None:
        ds = ds.shuffle(seed=args.seed)
    if args.test_size:
        ds = ds.train_test_split(test_size=args.test_size, seed=args.seed or 0)

    # 3) Push
    push_to_hub(ds, repo_id=args.repo_id,
                private=args.private, token=args.token)