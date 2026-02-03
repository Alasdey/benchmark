from bs4 import BeautifulSoup
import re
import pandas as pd

def parse_matres(tml_folder, annotations):
    
    # Parse the TML file
    with open(tml_folder, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    # Extract document ID
    doc_id = soup.find('docid').text.strip()

    # Extract text while tracking word positions
    text_element = soup.find('text')
    words = []
    events = []
    eid_to_eiid = {}
    word_index = 0

    for element in text_element.descendants:
        if isinstance(element, str):  # Plain text part
            for word in element.split():
                words.append(word)
                word_index += 1
        elif element.name == 'event':  # Event-tagged word
            eid = element['eid']
            event_text = element.text.strip()
            eiid = None
            
            # Find the MAKEINSTANCE matching this event eid
            for makeinstance in soup.find_all('makeinstance'):
                if makeinstance['eventid'] == eid:
                    eiid = makeinstance['eiid']
                    break
            
            # Store mapping between eid and eiid
            eid_to_eiid[eid] = eiid
            
            # Store event with its specific word index
            event_indexes = list(range(word_index, word_index + len(event_text.split())))
            events.append({'eid': eid, 'eiid': eiid, 'indexes': event_indexes})
    
    
    relations = []
    relations_duplication = {"AFTER": "BEFORE", "BEFORE": "AFTER", "EQUAL": "EQUAL", "VAGUE": "VAGUE"}
    #print(len([r for _, r in annotations[annotations['doc_id'] == doc_id].iterrows()]))
    for _, row in annotations[annotations['doc_id'] == doc_id].iterrows():
        # Use the eiids directly from the annotations
        event1_eiid = 'ei' + str(row['index1'])
        event2_eiid = 'ei' + str(row['index2'])

        if event1_eiid and event2_eiid:
            relations.append({
                'event1_eiid': event1_eiid,
                'event2_eiid': event2_eiid,
                'relation': row['relation']
            })
            #relations.append({
            #    'event1_eiid': event2_eiid,
            #    'event2_eiid': event1_eiid,
            #    'relation': relations_duplication[row['relation']]
            #})
        else:
            print(doc_id, row['index1'], row['index2'], event1_eiid, event2_eiid)
    #print(len(relations))
    return words, events, relations

def data_wsa(events, relations, ere_types):
    """
    """
    lost_rel = 0
    evt = {}
    for e in events:
        evt[e['eiid']] = set(e['indexes'])
    rel = {}
    for r in relations:
        label_array = [0] * len(ere_types)
        label_array[ere_types.index(r['relation'])] = 1
        try:
            rel[(frozenset(evt[r['event1_eiid']]), frozenset(evt[r['event2_eiid']]))] = label_array
        except:
            lost_rel += 1
            print("UwU")
    return rel

def all_file(tml_folder, annotation_file, ere_types):
    """
    """
    
    # Load annotations to extract relations
    annotations = pd.read_csv(annotation_file, sep='\t', header=None)
    annotations.columns = ['doc_id', 'event1', 'event2', 'index1', 'index2', 'relation']

    docs = list(set(annotations['doc_id']))

    rel = []
    wor = []
    did = []
    for file_name in docs:
        #print(file_name)
        words, events, relations = parse_matres(tml_folder+file_name+'.tml', annotations)
        rel.append(data_wsa(events, relations, ere_types))
        wor.append(words)
        did.append(file_name)

    return {d:(d, w, r) for d, w, r in zip(did, wor, rel)}

class matres():
    """
    """
    def __init__(self, data_fold = "./../../data/"):
        # Example usage
        tml_folder = [
            data_fold+'TempEval3/Evaluation/te3-platinum/',
            data_fold+'TempEval3/Training/TBAQ-cleaned/AQUAINT/',
            data_fold+'TempEval3/Training/TBAQ-cleaned/TimeBank/'
        ]
        annotation_file = [
            data_fold+'MATRES/platinum.txt',
            data_fold+'MATRES/aquaint.txt',
            data_fold+'MATRES/timebank.txt'
        ]
        split = {
            'train': data_fold+'MATRES/train.txt',
            'dev': data_fold+'MATRES/dev.txt',
            'test': data_fold+'MATRES/test.txt'
        }
    
        self.ere_types = ['AFTER', 'BEFORE', 'VAGUE', 'EQUAL']

        data = {}
        for tml, annot in zip(tml_folder, annotation_file):
            data.update(all_file(tml, annot, self.ere_types))
        
        self.matres_train = [data[l.strip()] for l in open(split['train'])] #(str: doc_name, list: text, dict: word_set_annotation)
        self.matres_dev = [data[l.strip()] for l in open(split['dev'])]
        self.matres_test = [data[l.strip()] for l in open(split['test'])]


    def set_dataset(self, mode="train"):
        """
        """
        if mode=="train":
            self.db=self.matres_train
        elif mode=="valid":
            self.db=self.matres_dev
        elif mode=="eval":
            self.db=self.matres_test
        elif mode=="debug":
            self.db=self.matres_train[:5]
        else:
            raise ValueError('Wrong name for the dataset segment. Expected train, valid or eval, got:' + mode)

    def word_list(self):
        return [data[1] for data in self.db]

    def word_set_annotation(self, no_identical=True, frame_scope='events'):
        return [data[2] for data in self.db]