#!/usr/bin/env python3
"""
Evaluate chain_of_thought predictions against answers in a HuggingFace dataset,
adding a 'reward' column with F1 scores from accuracy_reward.
"""

import argparse
from datasets import load_dataset, Dataset
from accuracy_reward import accuracy_reward  # assumes accuracy_reward.py is in the path
from huggingface_hub import login
import os

login(token=os.environ["HF_TOKEN"])

def compute_reward(example: dict) -> dict:
    """
    Applies accuracy_reward(pred, gold) where:
      - pred = chain_of_thought  (model prediction)
      - gold = answer            (ground truth)

    Returns a dict with the 'reward' key added.
    """
    pred = example["chain_of_thought"] or ""
    gold = example["answer"] or ""

    reward = accuracy_reward(pred, gold)

    # accuracy_reward returns -1 when both pred and gold are empty → treat as 0.0
    example["reward"] = float(reward) if reward != -1 else 0.0
    return example


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load dataset                                                      #
    # ------------------------------------------------------------------ #
    print(f"Loading dataset '{args.dataset}' (split='{args.split}') …")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"  Loaded {len(ds):,} rows with columns: {ds.column_names}")

    # ------------------------------------------------------------------ #
    # 2. Compute rewards                                                   #
    # ------------------------------------------------------------------ #
    print(f"Computing rewards using {args.num_proc} process(es) …")
    ds = ds.map(
        compute_reward,
        num_proc=args.num_proc,
        desc="Scoring chain_of_thought vs answer",
    )

    # ------------------------------------------------------------------ #
    # 3. Quick summary statistics                                          #
    # ------------------------------------------------------------------ #
    rewards = ds["reward"]
    n = len(rewards)
    mean_r  = sum(rewards) / n
    min_r   = min(rewards)
    max_r   = max(rewards)

    print("\n── Reward statistics ──────────────────────────────────────")
    print(f"  Rows evaluated : {n:,}")
    print(f"  Mean reward    : {mean_r:.4f}")
    print(f"  Min  reward    : {min_r:.4f}")
    print(f"  Max  reward    : {max_r:.4f}")
    print("────────────────────────────────────────────────────────────\n")

    # ------------------------------------------------------------------ #
    # 4. Push to Hub                                                       #
    # ------------------------------------------------------------------ #
    output_repo = args.output or args.dataset.rstrip("/") + "-rewarded"
    print(f"Pushing dataset to Hub repo '{output_repo}' …")
    ds.push_to_hub(output_repo, split=args.split)
    print(f"Done. Load it with: load_dataset('{output_repo}', split='{args.split}')")



def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate chain_of_thought vs answer and add reward column."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Nofing/maven-ere-llm-sft-agg",
        help="HuggingFace dataset name or local path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to evaluate (default: train).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Hub repo ID to push the rewarded dataset to. "
            "Defaults to '{source_dataset}-rewarded'."
        ),
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the resulting dataset to the HuggingFace Hub.",
    )
    parser.add_argument(
        "--hub_repo",
        type=str,
        default=None,
        help="Hub repo ID to push to (e.g. 'username/my-dataset'). Required if --push_to_hub is set.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of processes for dataset.map (default: 4).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()