import re
from typing import List, Dict, Any
from collections import defaultdict
import json
import pprint
from datasets import Dataset, DatasetDict
import os

def _tokenize_with_offsets(text: str):
    """
    Split *text* on whitespace, returning both tokens and their
    character offsets (start, end – end exclusive).
    """
    tokens, offsets = [], []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group(0))
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def prepare_document(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one raw document (as in your example) into a dictionary with:
      • id          – str
      • tokens      – List[str]        (original sentences)
      • words       – List[str]        (flat list of all tokens)
      • mentions    – List[int]        (original mention *ids*)
      • spans       – List[List[int]]  (token indices per mention, parallel to *mentions*)
      • relations   – Dict[str, List[List[int]]]
                      (pairs of *mention indices*, **not** mention ids)

    The token indices in *spans* and the pairs in *relations* are all based on
    the flattened *words* list.
    """
    sentences: List[str] = raw["text"]

    # ---- 1. tokenise every sentence, collect global token offsets ----
    words: List[str] = []
    sent_token_meta: List[List[tuple]] = []       # [(tok, start, end, global_idx), ...] per sent

    for sent in sentences:
        toks, offs = _tokenize_with_offsets(sent)
        base_idx = len(words)                     # first global token idx for this sentence
        words.extend(toks)
        sent_token_meta.append(
            [(tok, s, e, base_idx + i) for i, (tok, (s, e)) in enumerate(zip(toks, offs))]
        )

    # ---- 2. build mentions → span (token indices) ----
    mentions: List[int] = []
    spans: List[List[int]] = []
    id_to_pos: Dict[int, int] = {}               # mention id  → position in *mentions*

    for pos, m in enumerate(raw["events"]):
        m_id = m["id"]
        mentions.append(m_id)
        id_to_pos[m_id] = pos

        sent_id = m["sent_id"]
        char_start, char_end = m["offset"]

        # tokens whose char span overlaps the mention’s char span
        tok_indices = [
            glob_idx
            for _tok, t_start, t_end, glob_idx in sent_token_meta[sent_id]
            if not (t_end <= char_start or t_start >= char_end)
        ]
        spans.append(tok_indices)

    # ---- 3. rewrite relations to use mention *positions* ----
    relations: Dict[str, List[List[int]]] = {}
    for rel_type, pairs in raw["relations"].items():
        mapped: List[List[int]] = [
            [id_to_pos[a], id_to_pos[b]]
            for a, b in pairs
            if a in id_to_pos and b in id_to_pos
        ]
        relations[rel_type] = mapped

    # ---- 4. package result ----
    return {
        "id": raw["id"],
        "tokens": words,
        "mentions": mentions,
        "spans": spans,
        "relations": relations,
    }

# ──────────────────────────────────────────────────────────────────────────────
def push_to_hub(dd: DatasetDict, repo_id: str, private: bool, token: str | None):
    dd.push_to_hub(repo_id,
                   token=token or os.getenv("HF_TOKEN"),
                   private=private)
    print(f"✅  Dataset pushed to https://huggingface.co/datasets/{repo_id}")

def main():
    
    root_dir = 'data/MAVEN-ERE/processed/hievents/'
    file_type = '.json'
    
    split_rows = defaultdict(list)
    for split in ("train", "test", "dev"):
        path = root_dir + split + file_type
        with open(path, encoding="utf-8") as fh:
            docs = [json.loads(ln) for ln in fh]
        
        for raw_doc in docs:
            prep_doc = prepare_document(raw_doc)
    
            split_rows[split].append(prep_doc)
        
        if not split_rows:
            raise RuntimeError(f"No documents found under {root_dir}")

    ds_hievents = DatasetDict({sp: Dataset.from_list(rows) for sp, rows in split_rows.items()})

    for sp in ds_hievents:
        ds_hievents[sp] = ds_hievents[sp].shuffle(seed=42)

    push_to_hub(ds_hievents,
                repo_id="Nofing/Hievents-span",
                private=False,
                token=None)


if __name__ == "__main__":
    main()