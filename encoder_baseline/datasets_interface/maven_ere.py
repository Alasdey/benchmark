import random
import copy
from utils import tupling


def _to_interface(in_data):
    """
    Adds keys and values to the dictionary that is the data in order to facilitate data_prep opperations
    """
    
    data = copy.deepcopy(in_data)
    
    # Copy all events and time expressions in a uniform 'frame' format
    data['frame'] = copy.deepcopy(data['events']) + [{'id': i['id'], 'frame_type': 'TIMEX','type': i['type'], 
                                        'mention': [{'trigger_word': i['mention'], 'sent_id': i['sent_id'], 'offset': i['offset']}]} 
                                       for i in data['TIMEX']]
    
    # Sort entities and mentions by order of (first) apparition in the text
    data['frame'].sort(key=lambda x: min([i['sent_id']*9000+i['offset'][1] for i in x['mention']]))
    for frame in data['frame']:
        frame['mention'].sort(key=lambda i: i['sent_id']*9000+i['offset'][1])
    
    # Add an frame name by first mention and number of apparition in the text
    for i, frame in enumerate(data['frame']):
        frame['name'] = frame['mention'][0]['trigger_word'] + '_' + str(i)
    
    # Add the frame_type for the events
    for i in data['frame']:
        if not 'frame_type' in i:
            i['frame_type'] = 'EVENT'
    
    # List of all relations in uniform format
    event_id = [i['id'] for i in data['frame']]
    b = lambda rel: [event_id.index(rel[0]), event_id.index(rel[1])]
    a = lambda data: [[b(rel), key] for key in data.keys() for rel in data[key]]
    data['relation'] = a(data['temporal_relations'])+a(data['causal_relations'])+[[b(rel), 'SUBEVENT'] for rel in data['subevent_relations']]
    
    # List of relation in frame (the starting one)
    for frame in data['frame']:
        frame['relation'] = []
    for rel in data['relation']:
        data['frame'][rel[0][0]]['relation'] += [rel]
    
    # Adds a complete text
    data['flat_tokens'] = [word for sen in data['tokens'] for word in sen]
    data['text'] = ' '.join(data['flat_tokens'])
    
    # Give relation in frame index by relation class
    data['relation_type'] = {}
    for rel_type in ['BEFORE','OVERLAP','CONTAINS','SIMULTANEOUS','ENDS-ON','BEGINS-ON','CAUSE','PRECONDITION','SUBEVENT']:
        data['relation_type'][rel_type] = []
    for rel in data['relation']:
        data['relation_type'][rel[1]] += [rel[0]]

    # Creates a unique list of mentions and adds the frame index for each
    data['mention'] = []
    for frame in data['frame']:
        for mention in frame['mention']:
            mention['frame'] = data['frame'].index(frame)
            data['mention'] += [mention]
    
    return data


def _mention_to_text_word_span(data, ment):
    """
    """
    sent_offset = sum([len(data['tokens'][i]) for i in range(ment['sent_id'])])
    res = ment['offset'].copy()
    res[0] += sent_offset
    res[1] += sent_offset
    return res


def _frame_to_text_word_clust(data, frame):
    """
    """
    res = []
    for ment in frame['mention']:
        res += [_mention_to_text_word_span(data, ment)]
    return res

def span_clust_pair_rel(self, label_list, no_identical=True, frame_scope="events"):
    """
    """
    clust_rel = []
    for frames in tupling(self.event_clust(frame_scope)):
        clust_rel.append(dict())
        for clusta in frames:
            for clustb in frames:
                if no_identical and clusta==clustb:
                    continue 
                clust_rel[-1][(clusta, clustb)] = [0]*len(label_list)
    for rel, docs in self.rel_dict(frame_ref='span_clust').items():
        #print(rel)
        for doc_idx, pairs in enumerate(docs):
            for pair in pairs:
                array = clust_rel[doc_idx][tupling(pair)]
                array[label_list.index(rel)] = 1
    return clust_rel


def span_to_word_set(span):
    """
    """
    res = set(list(range(span[0], span[1])))
    return frozenset(res)


def span_clust_to_word_set(span_clust):
    """
    """
    res = set()
    for span in span_clust:
        res.update(span_to_word_set(span))
    return frozenset(res)


def word_clust_pair_rel(clust_rel):
    """
    """
    res = []
    for doc in clust_rel:
        res.append(dict())
        for span_clust_pair, gold_label in doc.items():
            word_clust_pair = tuple([span_clust_to_word_set(span_clust) for span_clust in span_clust_pair])
            res[-1][word_clust_pair] = gold_label
    return res

class maven_ere():
    def __init__(self, path='../../data/MAVEN_ERE/', seed=42):
        """
        """
        
        # Read the dataset from the files
        path_train = path + 'train.jsonl'
        path_eval = path + 'valid.jsonl'
        
        db_file_train = open(path_train, 'r')
        db_file_eval = open(path_eval, 'r')
        
        db_train_temp = [_to_interface(eval(i)) for i in db_file_train]
        self.db_eval = [_to_interface(eval(i)) for i in db_file_eval]
        
        # Cut train into train and valid at 1/10 to 9/10 ratios
        random.seed(a=seed)
        random.shuffle(db_train_temp)
        
        cutoff = len(db_train_temp)//10
        self.db_valid = db_train_temp[:cutoff]
        self.db_train = db_train_temp[cutoff:]
        
        self.db=self.db_train
        
        self.ere_types = ["BEFORE", "OVERLAP", "CONTAINS", "SIMULTANEOUS", "BEGINS-ON", "ENDS-ON", "CAUSE", "PRECONDITION", "SUBEVENT"]
        
    
    def set_dataset(self, mode="train"):
        """
        """
        if mode=="train":
            self.db=self.db_train
        elif mode=="valid":
            self.db=self.db_valid
        elif mode=="eval":
            self.db=self.db_eval
        elif mode=="debug":
            self.db=self.db_valid[:5]
        else:
            raise ValueError('Wrong name for the dataset segment. Expected train, valid or eval, got:' + mode)
    
    
    def ment_list(self):
        """
        """
        res = []
        for data in self.db:
            res += [[]]
            for ment in data['mention']:
                res[-1] += [_mention_to_text_word_span(data, ment)]
        return res
    
    
    def event_clust(self, key='events'):
        """
        """
        assert key in ['events', 'frame']
        res = []
        for data in self.db:
            res += [[]]
            for frame in data[key]:
                res[-1] += [_frame_to_text_word_clust(data, frame)]
        return res
    
    
    def rel_list(self, relation_types=None):
        """
        """
        if relation_types==None:
            relation_types=self.ere_types
        relations = []
        for data in self.db:
            relations += [[]]
            for rel in data['relation']:
                relations[-1] += [(rel[0], relation_types.index(rel[1]))]
        return relations
    
    
    def rel_clust_list(self, relation_types=None):
        """
        """
        if relation_types==None:
            relation_types=self.ere_types
        relations = []
        for data in self.db:
            relations.append([])
            for rel_type in relation_types:
                for rel in data['relation']:
                    relations[-1].append((_frame_to_text_word_clust(data, data['frame'][rel[0]]), _frame_to_text_word_clust(data, data['frame'][rel[1]])))
        return relations
        
    
    def relat_list(self, rel_type, frame_ref='idx'):
        """
        """
        res = []
        for data in self.db:
            res.append([])
            for rel in data['relation_type'][rel_type]:
                if frame_ref=='span_clust':
                    res[-1].append((_frame_to_text_word_clust(data, data['frame'][rel[0]]),
                                 _frame_to_text_word_clust(data, data['frame'][rel[1]])))
                else:
                    res[-1].append((rel[0], rel[1]))
        return res
    
    
    def rel_dict(self, frame_ref='idx'):
        """
        """
        rel_dict = {}
        for rel_type in self.ere_types:
            rel_dict[rel_type] = self.relat_list(rel_type, frame_ref)
        return rel_dict

    def span_clust_pair_rel(self, label_list, no_identical=True, frame_scope="events"):
        """
        """
        clust_rel = []
        for frames in tupling(self.event_clust(frame_scope)):
            clust_rel.append(dict())
            for clusta in frames:
                for clustb in frames:
                    if no_identical and clusta==clustb:
                        continue 
                    clust_rel[-1][(clusta, clustb)] = [0]*len(label_list)
        for rel, docs in self.rel_dict(frame_ref='span_clust').items():
            #print(rel)
            for doc_idx, pairs in enumerate(docs):
                for pair in pairs:
                    array = clust_rel[doc_idx][tupling(pair)]
                    array[label_list.index(rel)] = 1
        return clust_rel


    def word_set_annotation(self, no_identical=True, frame_scope='events'):
        """
        """
        assert frame_scope in ['events', 'frame']
        span_clust = span_clust_pair_rel(self, self.ere_types, no_identical=no_identical, frame_scope=frame_scope)
        wsa = word_clust_pair_rel(span_clust)
        return wsa
    
        
    def word_list(self):
        """
        """
        return [data['flat_tokens'] for data in self.db]