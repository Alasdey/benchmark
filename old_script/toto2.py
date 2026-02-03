import os
import json
import argparse
import itertools
from collections import defaultdict
from datasets import Dataset, DatasetDict

# Symmetrical relations need to be added in both directions (A,B) and (B,A)
BIDIRECTIONAL_TYPES = {"SIMULTANEOUS", "BEGINS-ON", "coreference"}

def get_doc_offset_map(sentences):
    """
    Creates a mapping to convert sentence-relative offsets to document-relative offsets.
    Returns: [0, len(sent0), len(sent0)+len(sent1), ...]
    """
    offsets = [0]
    for start, sent in enumerate(sentences):
        offsets.append(offsets[-1] + len(sent))
    return offsets

def process_item(example):
    """
    Transforms a raw MAVEN-ERE entry into a simplified span-based format.
    Renames all mentions to e0, e1, e2... based on document order.
    """
    doc_id = example["id"]
    original_sentences = example["tokens"]
    
    # 1. Flatten Tokens
    flat_tokens = [t for sent in original_sentences for t in sent]
    sent_offsets = get_doc_offset_map(original_sentences)

    # 2. Collect all mentions (Events + TIMEX)
    # Stored as: (global_start, global_end, concept_id)
    # 'concept_id' is the ID of the Event Chain or the TIMEX ID.
    raw_mentions = []

    # A. Extract Events (which are grouped by Coref Chains in MAVEN)
    if "events" in example:
        for chain in example["events"]:
            concept_id = chain["id"] # The Event ID (e.g., "E123")
            for m in chain["mention"]:
                s_id = m["sent_id"]
                # Convert features to document-level indices
                start = sent_offsets[s_id] + m["offset"][0]
                end = sent_offsets[s_id] + m["offset"][1]
                raw_mentions.append({
                    "start": start,
                    "end": end,
                    "indices": list(range(start, end)),
                    "concept_id": concept_id
                })
    
    # B. Extract TIMEX (No chains, usually singletons)
    if "TIMEX" in example:
        for t in example["TIMEX"]:
            concept_id = t["id"]
            s_id = t["sent_id"]
            start = sent_offsets[s_id] + t["offset"][0]
            end = sent_offsets[s_id] + t["offset"][1]
            raw_mentions.append({
                "start": start,
                "end": end,
                "indices": list(range(start, end)),
                "concept_id": concept_id
            })

    # 3. Sort Mentions by position and Rename to e0, e1, e2...
    # Sorting ensures e0 appears before e1 in the text.
    raw_mentions.sort(key=lambda x: (x["start"], x["end"]))

    formatted_mentions = [] # ["e0", "e1", ...]
    formatted_spans = []    # [[0,1], [5,6,7], ...]
    
    # Mapping: Concept_ID (Source JSON ID) -> List of new Integer Indices (e.g., [0, 5])
    # This handles the 1-to-many logic where an Event ID maps to multiple specific mention spans
    concept_to_indices = defaultdict(list)

    for idx, m in enumerate(raw_mentions):
        new_id = f"e{idx}"
        formatted_mentions.append(new_id)
        formatted_spans.append(m["indices"])
        
        # Link the abstract concept ID to this specific mention index
        concept_to_indices[m["concept_id"]].append(idx)

    # 4. Process Relations
    # We map relations from Concept_ID -> Concept_ID to Mention_Index -> Mention_Index
    relations = defaultdict(list)

    def add_relation(rtype, idx_a, idx_b):
        relations[rtype].append([idx_a, idx_b])
        if rtype in BIDIRECTIONAL_TYPES:
            relations[rtype].append([idx_b, idx_a])

    # A. Interaction Relations (Temporal, Causal, Subevent)
    for rel_cat in ["temporal_relations", "causal_relations", "subevent_relations"]:
        if rel_cat not in example: continue
        
        # Subevent relations format is typically a list, others are dicts
        source_data = example[rel_cat]
        if isinstance(source_data, list):
            # Normalize list format to dict format for generic processing
            # assuming list elements are [parent, child]
            source_data = {"subevent": source_data}

        for rel_type, pairs in source_data.items():
            for (id_a, id_b) in pairs:
                # Get all mentions belonging to Concept A and Concept B
                mentions_a = concept_to_indices.get(id_a, [])
                mentions_b = concept_to_indices.get(id_b, [])
                
                # Create edges between ALL mentions of Concept A and ALL mentions of Concept B
                for ma, mb in itertools.product(mentions_a, mentions_b):
                    add_relation(rel_type, ma, mb)

    # B. Coreference Relations
    # Implicit in the 'events' list: all mentions sharing a concept_id correlate
    for concept_id, indices in concept_to_indices.items():
        if len(indices) > 1:
            # Create pairs for every combination of mentions in the same chain
            for m1, m2 in itertools.combinations(indices, 2):
                add_relation("coreference", m1, m2)

    return {
        "id": doc_id,
        "tokens": flat_tokens,
        "mentions": formatted_mentions,
        "spans": formatted_spans,
        "relations": dict(relations)
    }

def main():
    parser = argparse.ArgumentParser(description="Clean DataPrep for MAVEN-ERE")
    parser.add_argument("--repo_id", default="Nofing/MAVEN-ERE-toto2", help="Target HuggingFace Repo")
    parser.add_argument("--path", default="data/MAVEN-ERE/MAVEN_ERE/", help="Local path to JSONL")
    args = parser.parse_args()

    # Define file paths
    # Note: MAVEN usually provides train.jsonl and valid.jsonl.
    # We will split train -> train/dev later, and use valid.jsonl as test.
    files = {
        "train": os.path.join(args.path, "train.jsonl"),
        "test_raw": os.path.join(args.path, "valid.jsonl") 
    }

    print(f"Loading data from {args.path}...")
    
    # helper to load jsonl safely
    def load_jsonl(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    train_data = load_jsonl(files["train"])
    test_data = load_jsonl(files["test_raw"])

    # Create HF Dataset
    raw_ds = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test_raw": Dataset.from_list(test_data)
    })

    # Apply transformation
    print("Processing documents...")
    ds = raw_ds.map(process_item, remove_columns=raw_ds["train"].column_names)

    # Create Splits
    # 1. Use 10% of training data as Dev
    split_train = ds["train"].train_test_split(test_size=0.1, seed=42)
    
    # 2. Re-assemble final dict
    final_dataset = DatasetDict({
        "train": split_train["train"],
        "dev": split_train["test"],
        "test": ds["test_raw"]
    })

    print("Shuffling...")
    final_dataset = final_dataset.shuffle(seed=42)

    print(f"Sample Document Mentions: {final_dataset['train'][0]['mentions'][:5]}")
    print(f"Sample Document Spans: {final_dataset['train'][0]['spans'][:5]}")

    # Push
    print(f"Pushing to {args.repo_id}...")
    final_dataset.push_to_hub(args.repo_id, token=os.getenv("HF_TOKEN"))
    print("Done.")

if __name__ == "__main__":
    main()