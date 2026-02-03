import json, pathlib, pprint
from collections import defaultdict
from datasets import Dataset, DatasetDict
import os
import re
from typing import List, Dict, Any, Tuple


# -----------------------------------------------------------
#  Utilities
# -----------------------------------------------------------

def _sent_starts(tokenised_sents: List[List[str]]) -> List[int]:
    """Return the starting *global* token index for every sentence."""
    starts, c = [], 0
    for sent in tokenised_sents:
        starts.append(c)
        c += len(sent)
    return starts


def _tokens_from_sent_strings(sent_strings: List[str]) -> List[List[str]]:
    """
    Fallback tokenizer (simple whitespace) when the document
    has *sentences* strings but no explicit *tokens* field.
    """
    return [re.findall(r"\S+", s) for s in sent_strings]


# -----------------------------------------------------------
#  Core converter
# -----------------------------------------------------------

def prepare_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one raw record into the canonical structure:

        {
          "id":          str,              # document id
          "tokens":      List[str],        # original sentence strings
          "words":       List[str],        # flat list of all tokens
          "mentions":    List[str],        # mention *ids* (events + TIMEX)
          "spans":       List[List[int]],  # token indices per mention (match *mentions*)
          "relations":   Dict[str, List[List[int]]]  # pairs of mention *indices*
        }

    Notes
    -----
    * Works whether offsets are **token indices** (your new sample) **or
      character offsets** (first sample).  The rule is:
        – If the document already carries a *tokens* field, offsets are
          interpreted as token indices.
        – Otherwise they are taken as character positions and an
          on-the-fly whitespace tokeniser is used.
    * TIMEX items are treated exactly like event mentions, so they get
      their own entry in *mentions* / *spans* and can participate in
      relations.
    """

    # -------------------------------------------------------
    # 1.  Tokens & sentences
    # -------------------------------------------------------
    tokenised_sents: List[List[str]]
    sentence_strings: List[str]

    if "tokens" in doc and doc["tokens"]:
        tokenised_sents = doc["tokens"]
        sentence_strings = doc.get("sentences", [" ".join(t) for t in tokenised_sents])
        offsets_are_token = True
    else:
        # fall back to whitespace-split
        if "sentences" not in doc:
            raise ValueError("Document needs either 'tokens' or 'sentences'.")
        sentence_strings = doc["sentences"]
        tokenised_sents = _tokens_from_sent_strings(sentence_strings)
        offsets_are_token = False  # will treat offsets as characters

    starts = _sent_starts(tokenised_sents)           # first global idx per sentence
    words: List[str] = [tok for sent in tokenised_sents for tok in sent]

    # -------------------------------------------------------
    # 2.  Gather mentions (events + TIMEX)
    # -------------------------------------------------------
    mentions:   List[str]        = []
    spans:      List[List[int]]  = []
    id2pos:     Dict[str, int]   = {}

    def _add_mention(m: Dict[str, Any], offset_key: str = "offset") -> None:
        """Helper: add one mention to mentions/spans/id2pos."""
        m_id     = m["id"]
        sent_id  = m["sent_id"]
        start, end = m[offset_key]        # end exclusive

        if offsets_are_token:             # token indices → direct
            span = list(range(starts[sent_id] + start,
                               starts[sent_id] + end))
        else:                             # character offsets → map to tokens
            char_start, char_end = start, end
            # examine each token in that sentence
            sent_tokens = tokenised_sents[sent_id]
            sent_text   = sentence_strings[sent_id]
            local_span  = []
            pos = 0
            for i, token in enumerate(sent_tokens):
                pos = sent_text.find(token, pos)     # next occurrence
                if pos == -1:
                    break
                if not (pos + len(token) <= char_start or pos >= char_end):
                    local_span.append(i)
                pos += len(token)
            span = [starts[sent_id] + i for i in local_span]

        idx = len(mentions)
        mentions.append(m_id)
        spans.append(span)
        id2pos[m_id] = idx

    # event mentions
    for ev in doc.get("event_mentions", []):
        _add_mention(ev)

    # TIMEX mentions
    for tx in doc.get("TIMEX", []):
        _add_mention(tx)

    # -------------------------------------------------------
    # 3.  Relations (if present) – rewrite ids→indices
    # -------------------------------------------------------
    relations: Dict[str, List[List[int]]] = {}
    for rel_type, pairs in doc.get("relations", {}).items():
        mapped = []
        for a, b in pairs:
            if a in id2pos and b in id2pos:           # safeguard
                mapped.append([id2pos[a], id2pos[b]])
        relations[rel_type] = mapped

    # -------------------------------------------------------
    # 4.  Package
    # -------------------------------------------------------
    return dict(
        id        = doc.get("id", ""),
        tokens    = sentence_strings,
        words     = words,
        mentions  = mentions,
        spans     = spans,
        relations = relations,
    )



# ──────────────────────────────────────────────────────────────────────────────
def push_to_hub(dd: DatasetDict, repo_id: str, private: bool, token: str | None):
    dd.push_to_hub(repo_id,
                   token=token or os.getenv("HF_TOKEN"),
                   private=private)
    print(f"✅  Dataset pushed to https://huggingface.co/datasets/{repo_id}")


def main():
    root_dir = "data/MAVEN-ERE/MAVEN_ERE/"
    file_type = '.jsonl'
    
    split_rows = defaultdict(list)
    for split in ("test", ):
        path = root_dir + split + file_type
        with open(path, encoding="utf-8") as fh:
            docs = [json.loads(ln) for ln in fh]

        for raw_doc in docs:
            prep_doc = prepare_document(raw_doc)
    
            split_rows[split].append(prep_doc)
        
        if not split_rows:
            raise RuntimeError(f"No documents found under {root_dir}")

    ds_maven_ere = DatasetDict({sp: Dataset.from_list(rows) for sp, rows in split_rows.items()})

    for sp in ds_maven_ere:
        ds_maven_ere[sp] = ds_maven_ere[sp].shuffle(seed=42)

    print(ds_maven_ere['test'][0])

    push_to_hub(ds_maven_ere,
                repo_id="Nofing/Maven-ERE-span",
                private=False,
                token=None)


# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()