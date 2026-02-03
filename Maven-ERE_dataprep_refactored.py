import os
import json
import argparse
import itertools
from collections import defaultdict
from datasets import Dataset, DatasetDict

# Configuration Constants
BIDIRECTIONAL_TYPES = {"SIMULTANEOUS", "BEGINS-ON", "coreference"}

RELATION_GROUPS = {
    "temporal": ["BEFORE", "OVERLAP", "CONTAINS", "SIMULTANEOUS", "ENDS-ON", "BEGINS-ON"],
    "causal": ["CAUSE", "PRECONDITION"],
    "subevent": ["subevent"],
    "coreference": ["coreference"]
}

def get_doc_offset_map(sentences):
    """
    Creates a mapping to convert sentence-relative offsets to document-relative offsets.
    Returns: [0, len(sent0), len(sent0)+len(sent1), ...]
    """
    offsets = [0]
    for start, sent in enumerate(sentences):
        offsets.append(offsets[-1] + len(sent))
    return offsets

def process_item(example, keep_mentions, keep_relations):
    """
    Transforms a raw MAVEN-ERE entry into a simplified span-based format.
    Renames mentions to e0..eN and t0..tN.
    Filters nodes and relations based on user arguments.
    """
    doc_id = example["id"]
    original_sentences = example["tokens"]
    
    # 1. Flatten Tokens
    flat_tokens = [t for sent in original_sentences for t in sent]
    sent_offsets = get_doc_offset_map(original_sentences)

    # 2. Collect all raw mentions depending on args
    # Stored as: { "start": int, "end": int, "indices": [], "concept_id": str, "type": "event"|"timex" }
    raw_mentions = []

    # A. Extract Events
    if "events" in example and ("events" in keep_mentions or "both" in keep_mentions):
        for chain in example["events"]:
            concept_id = chain["id"] 
            for m in chain["mention"]:
                s_id = m["sent_id"]
                start = sent_offsets[s_id] + m["offset"][0]
                end = sent_offsets[s_id] + m["offset"][1]
                raw_mentions.append({
                    "start": start,
                    "end": end,
                    "indices": list(range(start, end)),
                    "concept_id": concept_id,
                    "type": "event"
                })
    
    # B. Extract TIMEX
    if "TIMEX" in example and ("timex" in keep_mentions or "both" in keep_mentions):
        for t in example["TIMEX"]:
            concept_id = t["id"]
            s_id = t["sent_id"]
            start = sent_offsets[s_id] + t["offset"][0]
            end = sent_offsets[s_id] + t["offset"][1]
            raw_mentions.append({
                "start": start,
                "end": end,
                "indices": list(range(start, end)),
                "concept_id": concept_id,
                "type": "timex"
            })

    # 3. Sort Mentions by position and Rename
    # This ensures e0/t0 appears before e1/t1 in the text.
    raw_mentions.sort(key=lambda x: (x["start"], x["end"]))

    formatted_mentions = [] 
    formatted_spans = []    
    
    # Lookups for relation construction
    # concept_to_indices: Maps abstract ID (E123) to list of integer indices in the final list [0, 2, ...]
    concept_to_indices = defaultdict(list)
    
    event_counter = 0
    timex_counter = 0

    for idx, m in enumerate(raw_mentions):
        # Generate ID (e.g., e0 or t0)
        if m["type"] == "event":
            new_id = f"e{event_counter}"
            event_counter += 1
        else:
            new_id = f"t{timex_counter}"
            timex_counter += 1
            
        formatted_mentions.append(new_id)
        formatted_spans.append(m["indices"])
        
        # We store the index `idx` which refers to the position in `formatted_mentions`
        concept_to_indices[m["concept_id"]].append(idx)

    # 4. Process Relations
    valid_relations = defaultdict(list)

    def add_relation(rtype, idx_a, idx_b):
        # Only add if the relation type matches the user filter
        if keep_relations == ["all"] or rtype in keep_relations:
            # Check if self-loop (rare but possible in bad data)
            if idx_a == idx_b: return 

            valid_relations[rtype].append([idx_a, idx_b])
            if rtype in BIDIRECTIONAL_TYPES:
                valid_relations[rtype].append([idx_b, idx_a])

    # A. Interaction Relations (Temporal, Causal, Subevent)
    for rel_cat in ["temporal_relations", "causal_relations", "subevent_relations"]:
        if rel_cat not in example: continue
        
        source_data = example[rel_cat]
        if isinstance(source_data, list):
            source_data = {"subevent": source_data}

        for rel_type, pairs in source_data.items():
            for (id_a, id_b) in pairs:
                # Retrieve indices. If an ID is missing (because we filtered out "timex" for example),
                # get() returns [], limiting the loop to 0 iterations -> relation is dropped.
                mentions_a = concept_to_indices.get(id_a, [])
                mentions_b = concept_to_indices.get(id_b, [])
                
                for ma, mb in itertools.product(mentions_a, mentions_b):
                    add_relation(rel_type, ma, mb)

    # B. Coreference Relations
    # Only applicable if we are keeping events (TIMEX usually don't have coref chains in this dataset)
    if "coreference" in keep_relations or "all" in keep_relations:
        for concept_id, indices in concept_to_indices.items():
            if len(indices) > 1:
                # Add coref link for every pair in the chain
                for m1, m2 in itertools.combinations(indices, 2):
                    add_relation("coreference", m1, m2)

    return {
        "id": doc_id,
        "tokens": flat_tokens,
        "mentions": formatted_mentions,
        "spans": formatted_spans,
        "relations": dict(valid_relations)
    }

def main():
    parser = argparse.ArgumentParser(description="Clean DataPrep for MAVEN-ERE")
    parser.add_argument("--repo_id", default="Nofing/MAVEN-ERE-causal", help="Target HuggingFace Repo")
    parser.add_argument("--path", default="data/MAVEN-ERE/MAVEN_ERE/", help="Local path to JSONL")
    
    # Filter Arguments
    parser.add_argument("--keep_mentions", choices=["events", "timex", "both"], default="both", 
                        help="Which nodes to include in the dataset.")
    parser.add_argument("--keep_relations", nargs="+", default=["all"], 
                        help="List specific relation labels to keep (e.g. 'BEFORE CAUSE') or 'all'.")

    args = parser.parse_args()

    # Expand relation filtering logic
    # If user provided groups like "temporal", expand them.
    selected_rels = []
    if "all" in args.keep_relations:
        selected_rels = ["all"]
    else:
        for r in args.keep_relations:
            if r in RELATION_GROUPS:
                selected_rels.extend(RELATION_GROUPS[r])
            else:
                selected_rels.append(r)
    
    print(f"Node Configuration: Keeping {args.keep_mentions}")
    print(f"Relation Configuration: Keeping {selected_rels}")

    # Load Data
    def load_jsonl(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    files = {
        "train": os.path.join(args.path, "train.jsonl"),
        "test_raw": os.path.join(args.path, "valid.jsonl") 
    }
    
    raw_ds = DatasetDict({
        "train": Dataset.from_list(load_jsonl(files["train"])),
        "test_raw": Dataset.from_list(load_jsonl(files["test_raw"]))
    })

    # Apply Processing
    # We pass args via fn_kwargs to the map function
    ds = raw_ds.map(
        process_item, 
        fn_kwargs={"keep_mentions": args.keep_mentions, "keep_relations": selected_rels},
        remove_columns=raw_ds["train"].column_names
    )

    # 1. Create Dev split (10% of Train)
    split_train = ds["train"].train_test_split(test_size=0.1, seed=42)
    
    # 2. Final Structure
    final_dataset = DatasetDict({
        "train": split_train["train"],
        "dev": split_train["test"],
        "test": ds["test_raw"]
    })

    final_dataset = final_dataset.shuffle(seed=42)

    # Debug Preview
    sample = final_dataset['train'][0]
    print("\n--- SAMPLE OUTPUT ---")
    print(f"ID: {sample['id']}")
    print(f"Mentions: {sample['mentions'][:10]} ...")
    print(f"Relations Keys: {list(sample['relations'].keys())}")
    
    # Push
    if args.repo_id:
        print(f"Pushing to {args.repo_id}...")
        final_dataset.push_to_hub(args.repo_id, token=os.getenv("HF_TOKEN"))
        print("Done.")

if __name__ == "__main__":
    main()