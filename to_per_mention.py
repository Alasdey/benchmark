#!/usr/bin/env python
"""
Transform a document-level causal event dataset into a mention-level dataset.

Each row in the output corresponds to a single event mention, enriched with:
- All original document fields (for context)
- The mention identifier + its token(s)
- gold: outgoing relations (this mention → other)
- sym_gold: incoming relations (other → this mention)
"""

import argparse
import json
from datasets import load_dataset, DatasetDict, Dataset


def get_mention_token(mention_idx: int, spans: list, tokens: list) -> str:
    """Resolve a mention index to its token string."""
    token_indices = spans[mention_idx]
    return " ".join(tokens[idx] for idx in token_indices)


def build_relation_dicts(mention_idx: int, relations: dict, spans: list, tokens: list):
    """
    Build gold (outgoing) and sym_gold (incoming) relation dicts for a given mention.

    Relations are stored as pairs [source, target].
    - gold: pairs where source == mention_idx  →  we collect the targets
    - sym_gold: pairs where target == mention_idx  →  we collect the sources
    """
    gold = {}
    sym_gold = {}

    for rel_type, pairs in relations.items():
        outgoing = []
        incoming = []
        for pair in pairs:
            src, tgt = pair[0], pair[1]
            if src == mention_idx:
                tgt_token = get_mention_token(tgt, spans, tokens)
                outgoing.append(f"e{tgt} {tgt_token}")
            if tgt == mention_idx:
                src_token = get_mention_token(src, spans, tokens)
                incoming.append(f"e{src} {src_token}")
        gold[rel_type] = outgoing
        sym_gold[rel_type] = incoming

    return gold, sym_gold


def process_split(dataset_split) -> Dataset:
    """Expand a single split to mention-level rows."""
    rows = []
    for sample in dataset_split:
        sample_id = sample["id"]
        tokens = sample["tokens"]
        mentions = sample["mentions"]
        spans = sample["spans"]
        relations = sample["relations"]
        text = sample["text"]
        annots = sample["annots"]

        for mention_idx, mention_id in enumerate(mentions):
            mention_token = get_mention_token(mention_idx, spans, tokens)
            mention_str = f"{mention_id} {mention_token}"

            gold, sym_gold = build_relation_dicts(
                mention_idx, relations, spans, tokens
            )

            rows.append(
                {
                    # Original sample fields
                    "id": sample_id,
                    "tokens": tokens,
                    "mentions": mentions,
                    "spans": spans,
                    "relations": json.dumps(relations),
                    "text": text,
                    "annots": annots,
                    # Mention-level fields
                    "mention": mention_str,
                    "mention_idx": mention_idx,
                    "gold": json.dumps(gold),
                    "sym_gold": json.dumps(sym_gold),
                }
            )

    return Dataset.from_list(rows)


def process_dataset(dataset_name: str) -> DatasetDict:
    """Load the source dataset and expand all splits to mention-level rows."""
    ds = load_dataset(dataset_name)

    new_splits = {}
    for split_name in ds:
        print(f"  Processing split '{split_name}' ({len(ds[split_name])} samples)...")
        new_splits[split_name] = process_split(ds[split_name])
        print(f"    → {len(new_splits[split_name])} mention-level rows")

    return DatasetDict(new_splits)


def main():
    parser = argparse.ArgumentParser(
        description="Transform document-level causal event dataset to mention-level."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Nofing/MAVEN-ERE-Causal-Events",
        help="HuggingFace dataset name to load.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_dataset",
        help="HuggingFace repo name to push the new dataset to.",
    )
    args = parser.parse_args()

    print(f"Loading dataset '{args.dataset}'...")
    new_ds = process_dataset(args.dataset)

    # Preview
    first_split = list(new_ds.keys())[0]
    sample = new_ds[first_split][0]
    print(f"\n--- Sample row (from '{first_split}') ---")
    print(f"  id:        {sample['id']}")
    print(f"  mention:   {sample['mention']}")
    print(f"  gold:      {sample['gold']}")
    print(f"  sym_gold:  {sample['sym_gold']}")

    print(f"\nPushing to HuggingFace as '{args.output}'...")
    new_ds.push_to_hub(args.output)
    print("Done.")


if __name__ == "__main__":
    main()