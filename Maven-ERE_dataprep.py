import os
import json
import copy
import itertools
import random
import math
from collections import defaultdict
from tqdm import tqdm
import argparse
from datasets import Dataset, DatasetDict

#from constant import BIDIRECTIONAL_REL
#from template import TASK_DESC_TEMPORAL
BIDIRECTIONAL_REL = ["SIMULTANEOUS", "BEGINS-ON"]

class Document:
    def __init__(self, data):
        self.id = data["id"]
        self.words = data["tokens"]
        self.mentions = []
        self.events = []
        self.eid2mentions = {}

        if "events" in data:
            for e in data["events"]:
                self.events += e["mention"]
            for e in data["events"]:
                self.eid2mentions[e["id"]] = e["mention"]
        else:
            self.events = copy.deepcopy(data['event_mentions'])

        self.timexes = data["TIMEX"]
        for t in data["TIMEX"]:
            self.eid2mentions[t["id"]] = [t]
        self.events_all = self.events + self.timexes

        self.sort_events()
        self.map_events()
        self.map_timexes()

        if "events" in data: # ?
            self.temporal_relations = data["temporal_relations"]
            self.causal_relations = data["causal_relations"]
            self.subevent_relations = {"subevent": data["subevent_relations"]}
            self.coref_relations = None

            self.temporal_labels = self.get_relation_labels(self.temporal_relations)
            self.causal_labels = self.get_relation_labels(self.causal_relations)
            self.subevent_labels = self.get_relation_labels(self.subevent_relations)
            self.coref_labels = self.get_coref_labels(data)
        else:
            self.temporal_relations = {}
            self.causal_relations = {}
            self.subevent_relations = {}
            self.coref_relations = {}

        self.get_id2mention()

        self.text = self.annot_text()

        self.ere_types = [
            'BEFORE',
            'OVERLAP',
            'CONTAINS',
            'SIMULTANEOUS',
            'ENDS-ON',
            'BEGINS-ON',
            'CAUSE',
            'PRECONDITION',
            'subevent',
            'coreference'
        ]
        self.get_all_num2frozenset()
        self.get_all_labels()
        # self.get_word_set_annotation()
        self.get_word_list()
    
    def sort_events(self):
        self.events_all = sorted(self.events_all, key=lambda x: (x["sent_id"], x["offset"][0]))

    def get_id2mention(self):
        self.events_all_id2mention = {}
        self.all_id2mention = {}
        for index, event in enumerate(self.events_all):
            if event["id"].startswith("TIME"):
                mention = event["mention"]
            else:
                mention = event["trigger_word"]
            self.events_all_id2mention[event["id"]] = mention
            self.all_id2mention[event["id"]] = event

    def map_events(self):
        self.events_sorted = sorted(self.events, key=lambda x: (x["sent_id"], x["offset"][0]))
        self.event_num2id = {f"e{index}": e["id"] for index, e in enumerate(self.events_sorted)}
        self.event_id2num = {e["id"]: f"e{index}" for index, e in enumerate(self.events_sorted)}

    def map_timexes(self):
        self.timexes_sorted = sorted(self.timexes, key=lambda x: (x["sent_id"], x["offset"][0]))
        self.timex_num2id = {f"t{index}": e["id"] for index, e in enumerate(self.timexes_sorted)}
        self.timex_id2num = {e["id"]: f"t{index}" for index, e in enumerate(self.timexes_sorted)}

    def get_relation_labels(self, relations):
        new_relations = copy.deepcopy(relations)
        for rel in relations:
            pair_set = set()
            for pair in relations[rel]:
                for e1 in self.eid2mentions[pair[0]]:
                    for e2 in self.eid2mentions[pair[1]]:
                        if e1["id"].startswith("TIME"):
                            e1_map_id = self.timex_id2num[e1["id"]]
                        else:
                            e1_map_id = self.event_id2num[e1["id"]]

                        if e2["id"].startswith("TIME"):
                            e2_map_id = self.timex_id2num[e2["id"]]
                        else:
                            e2_map_id = self.event_id2num[e2["id"]]

                        pair_set.add((e1_map_id, e2_map_id))
                        if rel in BIDIRECTIONAL_REL:
                            pair_set.add((e2_map_id, e1_map_id))
            new_relations[rel] = list(pair_set)

        return new_relations

    def get_coref_labels(self, data):
        pair_list = []
        for event in data["events"]:
            for mention1, mention2 in itertools.permutations(event["mention"], 2):
                # coref is bidirectional
                m1_map_id = self.event_id2num[mention1["id"]]
                m2_map_id = self.event_id2num[mention2["id"]]
                pair_list.append((m1_map_id, m2_map_id))
        relations = {"coreference": pair_list}

        return relations

    def annot_text(self):
        new_sent_list = []
        for sent_id, sent in enumerate(self.words):
            new_sent = []
            offset = 0
            for event in self.events_all:
                if sent_id == event["sent_id"]:
                    event_map = self.timex_id2num[event["id"]] if event["id"].startswith("TIME") else \
                        self.event_id2num[event["id"]]
                    sp1 = event["offset"][0]
                    sp2 = event["offset"][1]
                    # Events are marked with special symbols
                    new_sent.extend(sent[offset: sp1])
                    new_sent.extend(["<" + event_map])
                    new_sent.extend([" ".join(sent[sp1: sp2]) + ">"])
                    offset = sp2
            new_sent.extend(sent[offset:])
            new_sent_list.append(" ".join(new_sent))
        text = " ".join(new_sent_list)
        return text
    
    def get_sent_offset(self):
        self.sent_offset = [0]
        for sent in self.words:
            self.sent_offset.append(self.sent_offset[-1] + len(sent))
    
    def get_all_num2frozenset(self):
        self.get_sent_offset()
        self.all_num2id = self.event_num2id | self.timex_num2id
        self.all_num2frozenset = {}
        for num, idy in self.all_num2id.items():
            sent_id = self.all_id2mention[idy]['sent_id']
            sent_off = self.sent_offset[sent_id]
            offset = self.all_id2mention[idy]['offset']
            self.all_num2frozenset[num] = frozenset([sent_off + word_off for word_off in range(offset[0], offset[1])])

    def get_all_labels(self):
        labels = [
            self.temporal_labels, 
            self.causal_labels, 
            self.subevent_labels, 
            self.coref_labels
        ]
        self.all_labels = {}
        for label in labels:
            self.all_labels.update(label)
    
    def get_word_set_annotation(self):
        #self.ere_types = list(self.all_labels.keys())
        self.wsa = {}
        pair2fst = {}
        for i in self.all_num2id.keys():
            for j in self.all_num2id.keys():
                a = self.all_num2frozenset[i]
                b = self.all_num2frozenset[j]
                key = (a, b)
                pair2fst[(i, j)] = key
                self.wsa[key] = [0]*len(self.ere_types)
        for label, pairs in self.all_labels.items():
            for pair in pairs:
                key = pair2fst[pair]
                self.wsa[key][self.ere_types.index(label)] = 1


    def get_word_list(self):
        self.word_list = []
        for sent in self.words:
            self.word_list += sent


class maven_ere_em():
    def __init__(self, path = "../../data/MAVEN_ERE_split/"):
        self.modes = ['test', 'train', 'valid', 'debug']
        self.data_path = path
        
        self.ere_types = [
            'BEFORE',
            'OVERLAP',
            'CONTAINS',
            'SIMULTANEOUS',
            'ENDS-ON',
            'BEGINS-ON',
            'CAUSE',
            'PRECONDITION',
            'subevent',
            'coreference'
        ]

    def set_dataset(self, mode="train"):
        """
        """
        if mode not in self.modes:
            raise ValueError(f"Wrong name for the dataset segment. Expected one of: {self.split}")
        self.split = mode
        with open(os.path.join(self.data_path, f"{self.split}.jsonl")) as f:
            lines = f.readlines()
        self.data_dict = {'word_list': [], 'wsa':[]}
        for line in tqdm(lines):
            data = json.loads(line.strip())
            doc = Document(data)
            self.data_dict['word_list'].append(doc.word_list)
            self.data_dict['wsa'].append(doc.wsa)
            self.ere_types = doc.ere_types

    def word_set_annotation(self, no_identical=True, frame_scope='frames'):
        """
        """
        return self.data_dict['wsa']

    def word_list(self):
        """
        """
        return self.data_dict['word_list']


def prepare_document(raw):

    doc = Document(raw)
    
    mentions = []
    spans = []
    for k, v in doc.all_num2frozenset.items():
        mentions.append(k)
        v_l = list(v)
        v_l.sort()
        spans.append(v_l)
    relations = {}
    for rel_type, rel_list in doc.all_labels.items():
        relations[rel_type] = []
        for rel in rel_list:
            a = mentions.index(rel[0])
            b = mentions.index(rel[1])
            relations[rel_type].append([a, b])

    return {
        "id": raw["id"],
        "tokens": doc.word_list,
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
    return

def main():
    parser = argparse.ArgumentParser(description="DataPrep for MAVEN-ERE")
    parser.add_argument("--dataset", default="Nofing/MAVEN-ERE-span", help="Dataset name")
    parser.add_argument("--path", default="data/MAVEN-ERE/MAVEN_ERE/", help="Dataset path")
    args = parser.parse_args()

    split_rows = defaultdict(list)
    for split in ['train', 'valid']:
        print(split)
        with open(os.path.join(args.path, f"{split}.jsonl")) as f:
            lines = f.readlines()
        for line in tqdm(lines):
            raw_doc = json.loads(line.strip())
            prep_doc = prepare_document(raw_doc)
            split_rows[split].append(prep_doc)
        if not split_rows:
            raise RuntimeError(f"No documents found under {root_dir}")

    print(split_rows['train'][0])

    ds = Dataset.from_list(split_rows['train'])
    ds = ds.train_test_split(test_size=0.1)
    ds = DatasetDict({
        'train': ds['train'],
        'dev': ds['test'],
        'test': Dataset.from_list(split_rows['valid'])
    })

    for sp in ds:
        ds[sp] = ds[sp].shuffle(seed=42)

    push_to_hub(ds,
                repo_id="Nofing/MAVEN-ERE-span",
                private=False,
                token=None)
    
    return

if __name__ == "__main__":
    main()
