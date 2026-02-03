
"""
Simplified Pre-processing for EventStoryLine/ECB+ XML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Extracts tokens, mentions, spans, and relations into a HF Dataset.
"""
import os, glob, argparse
import xml.etree.ElementTree as ET
from datasets import Dataset
from collections import defaultdict

# Comprehensive mapping of EventStoryLine/ECB+ tags to prefixes
TYPE_MAP = {
    # Actions
    "ACTION_OCCURRENCE": "a",
    "ACTION_REPORTING": "a",
    "ACTION_PERCEPTION": "a",
    "ACTION_ASPECTUAL": "a",
    "ACTION_STATE": "a",
    "ACTION_CAUSATIVE": "a",
    "ACTION_GENERIC": "a",
    "NEG_ACTION_OCCURRENCE": "a",
    "NEG_ACTION_REPORTING": "a",
    "NEG_ACTION_PERCEPTION": "a",
    "NEG_ACTION_ASPECTUAL": "a",
    "NEG_ACTION_STATE": "a",
    "NEG_ACTION_CAUSATIVE": "a",
    "NEG_ACTION_GENERIC": "a",
    "ACTION_GOLD": "a",
    "ACTION_SILVER": "a",
    # Locations
    "LOC_GEO": "l",
    "LOC_FAC": "l",
    "LOC_OTHER": "l",
    # Time
    "TIME_DATE": "t",
    "TIME_DURATION": "t",
    "TIME_OF_THE_DAY": "t",
    "TIME_REPETITION": "t",
    "TIME_Zone": "t",
    # Humans / Actors
    "HUMAN_PART_PER": "h",
    "HUMAN_PART_ORG": "h",
    "HUMAN_PART_GPE": "h",
    "HUMAN_PART_FAC": "h",
    "HUMAN_PART_VEH": "h",
    "HUMAN_PART_MET": "h",
    # Non-Humans
    "NON_HUMAN_PART": "n",
    "NON_HUMAN_PART_GENERIC": "n",
}

def get_union_find_roots(num_items, pairs):
    """Simple Union-Find implementation to get cluster representatives."""
    parent = list(range(num_items))
    
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    
    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i, j in pairs:
        union(i, j)
    
    return [find(i) for i in range(num_items)]

def parse_xml(path: str, exclude_set: set, propagate_coref: bool, 
              allowed_mentions: set = None, allowed_relations: set = None) -> dict:
    """Parses a single SML/XML file with optional usage of filters."""
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        print(f"⚠️  Parse Error: {path}")
        return None

    root = tree.getroot()
    doc_id = root.get("doc_name") or os.path.basename(path)

    if doc_id in exclude_set or os.path.basename(path) in exclude_set:
        print(f"🚫 Skipping: {doc_id}")
        return None

    # --- 1. Tokens ---
    by_tid = lambda el: int(el.get("t_id", 0))
    tokens = [t.text or "" for t in sorted(root.findall(".//token"), key=by_tid)]

    # --- 2. Mentions & Spans ---
    mentions_list = []
    spans_list = []
    raw_to_new_id = {} 
    
    for m in root.findall(".//Markables/*"):
        raw_id = m.get("m_id")
        tag = m.tag
        
        # Get anchors (text span)
        anchors = [int(a.get("t_id")) - 1 for a in sorted(m.findall("token_anchor"), key=by_tid)]
        
        # Validation
        if raw_id and anchors:
            prefix = TYPE_MAP.get(tag, "x") 

            # Filter Mention Type
            if allowed_mentions and prefix not in allowed_mentions:
                continue

            new_id = f"{prefix}{raw_id}"
            
            raw_to_new_id[raw_id] = new_id
            mentions_list.append(new_id)
            spans_list.append(anchors)

    id_to_idx = {mid: i for i, mid in enumerate(mentions_list)}

    # --- 3. Relations & Coref extraction ---
    # We segregate relations into 'coref_pairs' (for logic) and 'raw_relations' (for output)
    raw_relations = []       
    coref_pairs = []       
    
    for rel in root.findall(".//Relations/*"):
        src, tgt = rel.find("source"), rel.find("target")
        
        if src is not None and tgt is not None:
            raw_s, raw_t = src.get("m_id"), tgt.get("m_id")
            s_id, t_id = raw_to_new_id.get(raw_s), raw_to_new_id.get(raw_t)

            # Only proceed if both mentions survived the mention filter
            if s_id in id_to_idx and t_id in id_to_idx:
                s_idx, t_idx = id_to_idx[s_id], id_to_idx[t_id]
                rel_type = rel.get("relType", "UNKNOWN")

                if rel_type == "COREFERENCE":
                    coref_pairs.append((s_idx, t_idx))
                    # Only add COREFERENCE to output list if explicitly allowed or no filter set
                    if not allowed_relations or "COREFERENCE" in allowed_relations:
                        # We store it in raw_relations only if propagation is OFF
                        # If propagation is ON, coref is implicit in the logic, usually not desired as output
                        # unless specifically requested. We handle final aggregation below.
                        if not propagate_coref:
                            raw_relations.append((rel_type, s_idx, t_idx))
                else:
                    # Filter Relation Type (non-coref)
                    if not allowed_relations or rel_type in allowed_relations:
                        raw_relations.append((rel_type, s_idx, t_idx))

    # --- 4. Logic & Propagation ---
    final_relations = defaultdict(list)

    if propagate_coref:
        # Calculate clusters using internal coref_pairs
        roots = get_union_find_roots(len(mentions_list), coref_pairs)
        
        cluster_map = defaultdict(list)
        for idx, rt in enumerate(roots):
            cluster_map[rt].append(idx)
        
        processed_pairs = set()

        for r_type, s_idx, t_idx in raw_relations:
            s_root, t_root = roots[s_idx], roots[t_idx]
            
            if (r_type, s_root, t_root) in processed_pairs: continue
            
            # Propagate: All mentions in Source Cluster -> All mentions in Target Cluster
            for true_s in cluster_map[s_root]:
                for true_t in cluster_map[t_root]:
                    if true_s != true_t: 
                        final_relations[r_type].append([true_s, true_t])
            processed_pairs.add((r_type, s_root, t_root))
            
        # If user specifically asked for COREFERENCE in output with propagation on
        if allowed_relations and "COREFERENCE" in allowed_relations:
             for u, v in coref_pairs:
                 final_relations["COREFERENCE"].append([u, v])

    else:
        # No propagation: just dump what we collected
        for r_type, s_idx, t_idx in raw_relations:
            final_relations[r_type].append([s_idx, t_idx])
            

    return {
        "id": doc_id,
        "tokens": tokens,
        "mentions": mentions_list,
        "spans": spans_list,
        "relations": dict(final_relations),
    }

def main():
    parser = argparse.ArgumentParser(description="Parse EventStoryLine XML to HF Dataset.")
    parser.add_argument("--root", default="data/EventStoryLine/annotated_data/v1.5",
                        help="Path to root folder containing XML files.")
    parser.add_argument("--exclude", default="1_10ecbplus.xml",
                        help="Files to exclude (comma separated).")
    
    parser.add_argument("--filter_mentions", default="a",
                        help="Comma-separated list of prefixes to keep (e.g., 'a,h'). Default: keep all.")
    parser.add_argument("--filter_relations", default="PRECONDITION,FALLING_ACTION",
                        help="Comma-separated list of relation types to keep (e.g., 'TLINK,PRECONDITION'). Default: keep all.")
    parser.add_argument("--propagate_coref", default=True,
                        help="If set to True, relations are propagated through coreference clusters.")
    
    parser.add_argument("--repo_id", default="Nofing/EventStoryLine-1.5-Causal",
                        help="Target HF repository ID.")
    parser.add_argument("--private", action="store_true", help="Make repo private.")
    args = parser.parse_args()

    # Pre-process arguments
    exclude_set = {x.strip() for x in args.exclude.split(",") if x.strip()}
    
    allowed_mentions = None
    if args.filter_mentions:
        allowed_mentions = {x.strip() for x in args.filter_mentions.split(",") if x.strip()}
        print(f"ℹ️  Filtering mentions for types: {allowed_mentions}")

    allowed_relations = None
    if args.filter_relations:
        allowed_relations = {x.strip() for x in args.filter_relations.split(",") if x.strip()}
        print(f"ℹ️  Filtering relations for types: {allowed_relations}")

    # 1. Gather files
    files = glob.glob(os.path.join(args.root, "**", "*.xml"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No XML files found in {args.root}")
    
    # 2. Parse all
    print(f"Processing {len(files)} files...")
    rows = [parse_xml(f, exclude_set, args.propagate_coref, allowed_mentions, allowed_relations) for f in files]
    valid_rows = [r for r in rows if r is not None]

    # 3. Build & Split
    print("Building dataset...")
    ds = Dataset.from_list(valid_rows).shuffle(seed=42)
    
    if len(ds) > 10:
        ds = ds.train_test_split(test_size=0.1, seed=42)

    # 4. Push
    print(f"Pushing to {args.repo_id}...")
    ds.push_to_hub(args.repo_id, private=args.private)
    print(f"✅ Dataset ({len(valid_rows)} docs) successfully pushed!")

if __name__ == "__main__":
    main()
