import os
import json
import argparse
import itertools
from collections import defaultdict
from datasets import Dataset, DatasetDict

# Citation: MAVEN-ERE Dataset https://github.com/THU-KEG/MAVEN-ERE
BIDIRECTIONAL_TYPES = {"SIMULTANEOUS", "BEGINS-ON", "coreference"}

def get_global_spans(tokens_list, sentence_list, sent_id, local_offset):
    """Calculates document-level start/end indices from sentence-level offsets."""
    # Calculate how many tokens appeared in sentences prior to sent_id
    prior_token_count = sum(len(sent) for sent in tokens_list[:sent_id])
    
    start = prior_token_count + local_offset[0]
    end = prior_token_count + local_offset[1]
    
    # Return list of indices covered by this span (matching original logic)
    return list(range(start, end))

def process_document(example):
    """
    Transforms a raw MAVEN-ERE json object into the training format.
    """
    # 1. Flatten tokens
    flat_tokens = [token for sent in example["tokens"] for token in sent]

    # 2. Collect all mentions (Events + TIMEX)
    # map: original_string_id -> integer_index_in_output_lists
    id_to_idx = {} 
    mentions_ids = []  # List of string IDs (e.g., "EVENT_1", "TIME_2")
    mentions_spans = [] # List of spans (indices)
    
    current_idx = 0

    # Process Events
    # In MAVEN-ERE, 'events' contains coreference chains. 
    # We iterate the mentions inside the chains.
    if "events" in example:
        for chain in example["events"]:
            for mention in chain["mention"]:
                uid = mention["id"]
                spans = get_global_spans(example["tokens"], example["sentences"], mention["sent_id"], mention["offset"])
                
                id_to_idx[uid] = current_idx
                mentions_ids.append(uid)
                mentions_spans.append(spans)
                current_idx += 1
    elif "event_mentions" in example: # Test set format
         for mention in example["event_mentions"]:
                uid = mention["id"]
                spans = get_global_spans(example["tokens"], example["sentences"], mention["sent_id"], mention["offset"])
                
                id_to_idx[uid] = current_idx
                mentions_ids.append(uid)
                mentions_spans.append(spans)
                current_idx += 1

    # Process TIMEX
    for timex in example.get("TIMEX", []):
        uid = timex["id"]
        spans = get_global_spans(example["tokens"], example["sentences"], timex["sent_id"], timex["offset"])
        
        id_to_idx[uid] = current_idx
        mentions_ids.append(uid)
        mentions_spans.append(spans)
        current_idx += 1

    # 3. Process Relations
    formatted_relations = defaultdict(list)

    # Helper to add relation if both ends exist
    def add_rel(rtype, src_id, tgt_id):
        if src_id in id_to_idx and tgt_id in id_to_idx:
            u, v = id_to_idx[src_id], id_to_idx[tgt_id]
            formatted_relations[rtype].append([u, v])
            if rtype in BIDIRECTIONAL_TYPES:
                formatted_relations[rtype].append([v, u])

    # A. Temporal, Causal, Subevent (Explicit lists in JSON)
    for category in ["temporal_relations", "causal_relations"]:
        if category in example:
            for rel_type, pairs in example[category].items():
                for (src, tgt) in pairs:
                    add_rel(rel_type, src, tgt)

    if "subevent_relations" in example:
        for (src, tgt) in example["subevent_relations"]:
            add_rel("subevent", src, tgt)

    # B. Coreference (Implicit in 'events' lists)
    # Generate pairwise bidirectional links for all mentions in the same event chain
    if "events" in example:
        for chain in example["events"]:
            mention_ids = [m["id"] for m in chain["mention"]]
            # Generate all permutations of length 2
            for m1, m2 in itertools.permutations(mention_ids, 2):
                add_rel("coreference", m1, m2)

    return {
        "id": example["id"],
        "tokens": flat_tokens,
        "mentions": mentions_ids,
        "spans": mentions_spans,
        "relations": dict(formatted_relations)
    }

def main():
    parser = argparse.ArgumentParser(description="DataPrep for MAVEN-ERE (Simplified)")
    parser.add_argument("--repo_id", default="Nofing/MAVEN-ERE-toto", help="HF Repo ID")
    parser.add_argument("--path", default="data/MAVEN-ERE/MAVEN_ERE/", help="Local path to JSONL files")
    args = parser.parse_args()

    # Load raw JSONL files directly into a DatasetDict
    # We map 'train' locally to 'train', and 'valid' locally to 'test' temporarily
    data_files = {
        "train": os.path.join(args.path, "train.jsonl"),
        "test_original": os.path.join(args.path, "valid.jsonl") 
    }
    
    # Load dataset
    raw_ds = load_dataset_from_jsonl(data_files)
    
    # Process dataset
    print(f"Processing data...")
    processed_ds = raw_ds.map(process_document, remove_columns=raw_ds["train"].column_names)

    # Split: 
    # 1. Take 10% of 'train' to make 'dev'
    # 2. Rename original 'test_original' (which was valid.jsonl) to 'test'
    split_train = processed_ds["train"].train_test_split(test_size=0.1, seed=42)
    
    final_dd = DatasetDict({
        "train": split_train["train"],
        "dev": split_train["test"],
        "test": processed_ds["test_original"]
    })

    print(f"Resulting split sizes: {final_dd}")
    print(f"Sample Entry: {final_dd['train'][0]}")

    # Push to Hub
    if args.repo_id:
        final_dd.push_to_hub(args.repo_id, token=os.getenv("HF_TOKEN"))
        print(f"✅ Dataset pushed to https://huggingface.co/datasets/{args.repo_id}")

def load_dataset_from_jsonl(data_files):
    """Helper to load local JSONL files into a Dataset object without failing on missing files."""
    dsets = {}
    for split, path in data_files.items():
        if os.path.exists(path):
            dsets[split] = Dataset.from_json(path)
        else:
            print(f"Warning: {path} not found.")
    return DatasetDict(dsets)

if __name__ == "__main__":
    main()