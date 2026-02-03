
from tqdm import tqdm
from datasets import load_dataset

class data_preped():
    def __init__(self, hf_path = "Nofing/Hievents-span"):
        self.hf_path = hf_path
        self.ds = load_dataset(self.hf_path)
        self.modes = list(self.ds.keys())
        self.ere_types = list(self.ds[self.modes[0]][0]['relations'].keys())

    def sample2wsa(self, sample, NoRel='NoRel'):
        frosetspa = []
        wsa = {}
        for span in sample['spans']:
            frosetspa.append(frozenset(span))
        # # This is removed to avoid adding 'norel' relations to the dataset. This made things worst
        # # Though Maven-ere needs this, and Hievents and EventStoryLine should be better prep
        # for fs1 in frosetspa:
        #     for fs2 in frosetspa:
        #         if fs1==fs2:
        #             continue
        #         wsa[(fs1, fs2)] = [0]*len(self.ere_types)
        for rel_type in sample['relations']:
            for pair in sample['relations'][rel_type]:
                fs1 = frosetspa[pair[0]]
                fs2 = frosetspa[pair[1]]
                if (fs1, fs2) not in wsa.keys():
                    wsa[(fs1, fs2)] = [0]*len(self.ere_types)
                wsa[(fs1, fs2)][self.ere_types.index(rel_type)] = 1
        return wsa
    
    def set_dataset(self, mode="train"):
        """
        """
        if mode not in self.modes:
            raise ValueError(f"Wrong name for the dataset segment. Expected one of: {self.split}")
        self.split = mode
        self.ds_split = self.ds[self.split]
        self.data_dict = {'word_list': [], 'wsa':[]}
        for sample in tqdm(self.ds_split):
            self.data_dict['word_list'].append(sample['tokens'])
            self.data_dict['wsa'].append(self.sample2wsa(sample))

    def word_set_annotation(self, no_identical=True, frame_scope='frames'):
        """
        """
        return self.data_dict['wsa']

    def word_list(self):
        """
        """
        return self.data_dict['word_list']
