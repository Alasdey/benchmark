import os
import pickle
import math
import time
import random
import statistics
import gc
import sys
import datetime
import json
import shutil
import argparse
import cProfile
import pstats
from pstats import SortKey
import itertools

import numpy as np
from sklearn.metrics import f1_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerModel, LongformerTokenizerFast, AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel, AutoModelForMaskedLM, AutoModelForCausalLM


from utils import tupling, type_depth
from datasets_interface.old_format import data_preped

# Configuration and Hyperparameters
class Config:
    ### Random Seed
    RANDOM_SEED = 42

    # Hyperparameters
    MAX_SEQ_LENGTH = 4096   # Maximum sequence length for Longformer 
    TRUNCATION = True       # Truncation of the sequence at MAX_SEQ_LENGTH
    PADDING = 'longest'     # Padding strategy for the tokenizer
    WORD_SPLIT = False      # Wether the text is split into a list of words # Not ready for True
    BATCH_SIZE = 8          # Batch size
    NUM_EPOCHS = 20         # Number of training epochs
    LEARNING_RATE = 2e-5    # Learning rate for optimizer
    FRAME_SCOPE = 'frames'  # The argument given to dataset.event_clust()
    FRAME_SCOPE_LIST = ['events', 'frames']
    BATCH_SHUFFLE = True    # Shuffling to form batches
    ### Data Prep
    # Dataset used
    DATASET = 'Nofing/MAVEN-ERE-Causal-Events'
    DATASET_LIST = [
        'Nofing/Maven-ERE-span', 
        'Nofing/MECI-v0.1-public-span', 
        'Nofing/Hievents-span', 
        'Nofing/EventStoryLine-1.5-span',
        'Nofing/MAVEN-ERE-Causal-Events',
    ]
                            # Possible datasets
    NO_IDENTICAL = True     # The relations cannot be from a cluster to itself
    UNDER_SAMPLE = False    # Undersampling the train dataset
    NOTHING_LABEL = False   # If the labels contain a NOTHING label
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model configuration
    ENCODER_MODEL = "longformer"
    ENCODER_MODEL_LIST = ["longformer", "modernbert", "roberta"]
    if ENCODER_MODEL == "longformer":
        MODEL_NAME = "allenai/longformer-base-4096"
    elif ENCODER_MODEL == "modernbert":
        MODEL_NAME = "answerdotai/ModernBERT-large" # TBDone
    elif ENCODER_MODEL == "qwen-0.6":
        MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    
    ENCODER_TRAINED_LAYERS = [-1, -2, -3, -4, -5, -6]
    GLOBAL_MENTION = True # Integrate the mentions' tokens in the global_attention_mask (only for longformer)
    # Classification Head configuration
    CLASS_HEAD = 'big'
    VALID_CLASS_HEAD = ['big', 'small', 'split']

    # Aggregation Strategy
    AGGREG = 'mean'
    VALID_AGGREG = ['mean', 'max', 'lse']

    # Dataset segment
    DATASET_TRAIN = 'dev'
    DATASET_TEST = 'dev'
    DATASET_SET_LIST = ['train', 'dev', 'test']
    DATASET_LOAD = False
    DATASET_SAVE = False
    TRAIN_PRED_SAVE = False
    TEST_PRED_SAVE = True

    # Wether to discriminate between fundamental and deducible relations
    DISCRIM = False
    DISCRIM_FUND_GEN_ALPHA = 5
    DISCRIM_GEN_ALPHA = 1
    DISCRIM_FUND_ALPHA = 2
    DISCRIM_DEDU_ALPHA = 1
    DISCRIM_SCALE = lambda s, x=0: 1

    # Optimization
    ALL_CONSTRAINTS = True

    # Incoherence loss (sequential)
    INCO_LOSS = False
    INCO_LOSS_ALPHA = lambda s, x=0: 1
    # Incoherence loss (GPU)
    INCO_LOSS_MAT = False
    INCO_LOSS_MAT_ALPHA = lambda s, x=0: 2e-4
    

    # Solver 
    DELTA_SIGMOID = 0.5745
    # Train time
    # Whether to use the optimizer at training time
    TRAIN_OPTIMIZE = False
    LABEL_GRADIENT = False # Check Veracity of prior preds
    TRAIN_OPTIMIZE_ALPHA = lambda s, x=0: 2
    TRAIN_NON_OPTIMIZE_ALPHA = lambda s, x=0: 1

    # Wether to go through the matrix interface
    TRAIN_MATRIX_INTERFACE = TRAIN_OPTIMIZE or INCO_LOSS_MAT

    # Test time
    # Whether to use the optimizer at testing time
    TEST_OPTIMIZE = False 
    TEST_OPTIMIZE_ALPHA = lambda s, x=0: 1
    TEST_NON_OPTIMIZE_ALPHA = lambda s, x=0: 0

    # Wether to go through the matrix interface
    TEST_MATRIX_INTERFACE = TEST_OPTIMIZE or INCO_LOSS_MAT

    # Wether to get the constraints
    CONSTRAINTS = TEST_OPTIMIZE or TRAIN_OPTIMIZE or INCO_LOSS or INCO_LOSS_MAT or DISCRIM

    
    ### Loss
    # Mask (Quick fix horror)
    # REL_TYPE_NAMES = [
    #     'BEFORE',
    #     'OVERLAP',
    #     'CONTAINS',
    #     'SIMULTANEOUS',
    #     'ENDS-ON',
    #     'BEGINS-ON',
    #     'CAUSE',
    #     'PRECONDITION',
    #     'subevent',
    #     'coreference'
    # ]
    REL_TYPE_MASK = []
    REL_TYPE_IDX = []
    EXCLUDED_REL = ['NoRel']
    # Loss for the prediciton
    LOSS = 'asl'
    LOSS_OPT = 'asl'
    VALID_LOSSES = ['bce', 'foc', 'asl']
    # Reduction
    REDUCTION = 'mean'
    VALID_REDUCTIONS = ['mean', 'sum', 'none']
    # Focal Loss parameters
    ALPHA = 0.25
    GAMMA = 2.0
    # ASL parameters
    GAMMA_NEG=4 
    GAMMA_POS=1
    CLIP=0.05
    EPS=1e-8
    DTGFL=True

    ### Sklearn Report
    THRESHOLD = 0.5
    ZERO_DIVISION = 0.0 #np.nan
    REPORT_DIGITS = 4
    
    ### Dataset descritpions to be set 
    LABEL_LIST = None
    NUM_LABELS = None


    ### Loading the model
    START_EPOCH = 0
    LOAD_MODEL = False
    MODEL_PATH = ''
    
    ### Saving model
    BEST_MODEL = True
    top_micro_f1 = 0.0 # To be changed
    top_threshold = THRESHOLD
    top_epoch = -1
    THRESH_FLOOR = 3
    THRESH_STEPS = 100
    
    # Meta program
    PROFILE = True          # Use of cprofile
    LINE_PROFILE = 100      # Number of line printed of the profiler report
    ### Logging
    TIME_START = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    LOG_DIR = f"./encoder_baseline/logs/MatrixIEO/{TIME_START}/"
    TRAIN_LOG_FILE = LOG_DIR+"training_log.txt"
    CONFIG_LOG_FILE = LOG_DIR+"config.json"
    PREDS_LOG = LOG_DIR+"eval_preds"
    GOLDS_LOG = LOG_DIR+"eval_golds"
    IDX_LOG = LOG_DIR+"eval_idx"
    IND_LOG = LOG_DIR+"eval_inds"
    TRAIN_PREDS_LOG = LOG_DIR+"train_preds"
    TRAIN_GOLDS_LOG = LOG_DIR+"train_golds"
    TRAIN_IDX_LOG = LOG_DIR+"train_idx"
    TRAIN_IND_LOG = LOG_DIR+"train_inds"
    LOG_OVERALL = LOG_DIR+"overall.txt"
    LOG_PROFILE = LOG_DIR+"profile.prof"
    ### Model save
    MODEL_SAVE = LOG_DIR+"model.pkl"

    ### Dataset Save
    DS_SUF = "_"
    DS_SUF += "nn_" if NOTHING_LABEL else ""
    DS_SUF += "us_" if UNDER_SAMPLE else ""
    DS_PATH_START = "encoder_baseline/datasets/matrixie_"+DATASET+DS_SUF

    ### Error checks
    if FRAME_SCOPE not in FRAME_SCOPE_LIST: 
        raise ValueError(f"Invalid dataset '{FRAME_SCOPE}'. Valid options are: {', '.join(FRAME_SCOPE_LIST)}")
    if DATASET not in DATASET_LIST: 
        raise ValueError(f"Invalid dataset '{DATASET}'. Valid options are: {', '.join(DATASET_LIST)}")
    if LOSS not in VALID_LOSSES: 
        raise ValueError(f"Invalid loss function '{LOSS}'. Valid options are: {', '.join(VALID_LOSSES)}")
    if LOSS_OPT not in VALID_LOSSES: 
        raise ValueError(f"Invalid loss function '{LOSS_OPT}'. Valid options are: {', '.join(VALID_LOSSES)}")
    if REDUCTION not in VALID_REDUCTIONS: 
        raise ValueError(f"Invalid reduction function '{REDUCTION}'. Valid options are: {', '.join(VALID_REDUCTIONS)}")
    if DATASET_TRAIN not in DATASET_SET_LIST:
        raise ValueError(f"Invalid train dataset set '{DATASET_TRAIN}'. Valid options are: {', '.join(DATASET_SET_LIST)}")
    if DATASET_TEST not in DATASET_SET_LIST:
        raise ValueError(f"Invalid test dataset set '{DATASET_TEST}'. Valid options are: {', '.join(DATASET_SET_LIST)}")
    if AGGREG not in VALID_AGGREG:
        raise ValueError(f"Invalid aggregation function '{AGGREG}'. Valid options are: {', '.join(VALID_AGGREG)}")
    if CLASS_HEAD not in VALID_CLASS_HEAD:
        raise ValueError(f"Invalid classification head model '{CLASS_HEAD}'. Valid options are: {', '.join(VALID_CLASS_HEAD)}")
    if ENCODER_MODEL not in ENCODER_MODEL_LIST:
        raise ValueError(f"Invalid encoder model '{ENCODER_MODEL}'. Valid options are: {', '.join(ENCODER_MODEL_LIST)}")


#############################################################################################

class ClassHeadSplit(nn.Module):
    """
    """
    def __init__(self, hidden_size, config):
        super(ClassHeadSplit, self).__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(2 * hidden_size, 3 * hidden_size), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(3 * hidden_size, 2 * hidden_size), 
        )
        self.classifiers = [
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 1)
            ) for _ in range(config.NUM_LABELS)
        ]

    def forward(self, pair_embedding):
        """
        """
        hidden_rep = self.ffnn(pair_embedding) # Shape: (total_pairs, hidden_size * 2)
        outputs = [classifier(hidden_rep).squeeze(-1) for classifier in self.classifiers] # List[(total_pairs, 1)]
        logits = torch.cat(outputs, dim=0)  # Shape: (total_pairs, num_labels)
        return logits

class LongformerPairClassifier(nn.Module):
    """
    """
    def __init__(self, config):
        super(LongformerPairClassifier, self).__init__()

        if config.ENCODER_MODEL == "longformer":
            self.encoder = LongformerModel.from_pretrained(config.MODEL_NAME)
            self.layers = self.encoder.base_model.encoder.layer
            self.global_mask = True
        elif config.ENCODER_MODEL == "modernbert":
            self.encoder = AutoModel.from_pretrained(config.MODEL_NAME)
            self.layers = self.encoder.base_model.layers
            self.global_mask = False
        elif config.ENCODER_MODEL == "qwen-0.6":
            self.encoder = AutoModel.from_pretrained(config.MODEL_NAME)
            self.layers = self.encoder.base_model.encoder.layer
            self.global_mask = False
        
        
        # Freeze all layers
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Unfreeze the config.ENCODER_TRAINED_LAYERS attention blocks
        for i in config.ENCODER_TRAINED_LAYERS:
            for param in self.layers[i].parameters():
                param.requires_grad = True
        
        
        # Define the classification head
        hidden_size = self.encoder.config.hidden_size

        if config.CLASS_HEAD=='small':
            self.ffnn = nn.Linear(hidden_size * 2, config.NUM_LABELS)

        elif config.CLASS_HEAD=='big':
            self.ffnn = nn.Sequential(
                nn.Linear(2 * hidden_size, 3 * hidden_size), 
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(3 * hidden_size, 2 * hidden_size), 
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2 * hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, config.NUM_LABELS)
            )
        elif config.CLASS_HEAD=='split':
            self.ffnn = ClassHeadSplit(hidden_size, config)

    def forward(self, input_ids, attention_mask=None, global_attention_mask=None, pair_indices=None, doc_indices=None, pair_labels=None):
        """
        """
        if self.global_mask:
            # Pass through Longformer
            outputs = self.encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                global_attention_mask=global_attention_mask
            )
        else:
            # Pass through encoder
            outputs = self.encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )

        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        pair_embeddings = []
        total_indices = []
        all_labels = []

        for batch_idx in range(input_ids.size(0)):
            doc_hidden_state = last_hidden_state[batch_idx]  # (seq_len, hidden_size)
            doc_pair_indices = pair_indices[batch_idx]       # List of (i, j) tuples
            doc_pair_labels = pair_labels[batch_idx]         # List of labels for each pair

            for idx, (i, j) in enumerate(doc_pair_indices):
                emb_i = doc_hidden_state[i]  # (hidden_size)
                emb_j = doc_hidden_state[j]  # (hidden_size)
                pair_emb = torch.cat([emb_i, emb_j], dim=-1)  # (hidden_size * 2)
                pair_embeddings.append(pair_emb)
                total_indices.append((doc_indices[batch_idx], batch_idx, i, j))
                all_labels.append(doc_pair_labels[idx])

        # Stack pair embeddings and total indices
        pair_embeddings = torch.stack(pair_embeddings)  # (total_pairs, hidden_size * 2)
        all_labels = torch.tensor(all_labels).to(pair_embeddings.device)  # (total_pairs, num_labels)

        logits = self.ffnn(pair_embeddings) # (total_pairs, num_labels)

        return logits, total_indices, all_labels


class PairDataset(Dataset):
    """
    """
    def __init__(self, documents, config):
        """
        documents: Dictionary with keys 'tokens', 'pair_indices', 'pair_labels'
        """
        self.documents = documents
        self.max_length = config.MAX_SEQ_LENGTH

    def __len__(self):
        return len(self.documents['tokens']['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.documents['tokens'].items()}
        item['pair_indices'] = self.documents['pair_indices'][idx] # List of (i, j) indices
        item['pair_labels'] = self.documents['pair_labels'][idx]   # List of labels for each pair
        item['doc_indices'] = idx
        return item


def collate_fn(batch):
    """
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    global_attention_mask = None
    if 'global_attention_mask' in batch[0]:
        global_attention_mask = torch.stack([item['global_attention_mask'] for item in batch])
    pair_indices = [item['pair_indices'] for item in batch]
    pair_labels = [item['pair_labels'] for item in batch]
    doc_indices = [item['doc_indices'] for item in batch]

    batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'global_attention_mask': global_attention_mask,
        'pair_indices': pair_indices,
        'pair_labels': pair_labels,
        'doc_indices': doc_indices
    }
    return batch

#############################################################################################

def define_const_maven(config):
    """
    """
    LABEL_LIST = config.LABEL_LIST

    nl = len(LABEL_LIST)

    if config.ALL_CONSTRAINTS:
        neg_forw = [('BEFORE', 'CONTAINS'), ('BEFORE', 'SIMULTANEOUS'), ('BEFORE', 'OVERLAP'), ('BEFORE', 'ENDS-ON'), ('BEFORE', 'BEGINS-ON'), ('CONTAINS', 'BEFORE'), ('CONTAINS', 'SIMULTANEOUS'), ('CONTAINS', 'OVERLAP'), ('CONTAINS', 'ENDS-ON'), ('CONTAINS', 'BEGINS-ON'), ('SUBEVENT', 'CAUSE'), ('SUBEVENT', 'SIMULTANEOUS'), ('SUBEVENT', 'ENDS-ON'), ('CAUSE', 'SUBEVENT'), ('CAUSE', 'PRECONDITION'), ('CAUSE', 'ENDS-ON'), ('SIMULTANEOUS', 'BEFORE'), ('SIMULTANEOUS', 'CONTAINS'), ('SIMULTANEOUS', 'SUBEVENT'), ('SIMULTANEOUS', 'PRECONDITION'), ('SIMULTANEOUS', 'OVERLAP'), ('SIMULTANEOUS', 'ENDS-ON'), ('SIMULTANEOUS', 'BEGINS-ON'), ('PRECONDITION', 'CAUSE'), ('PRECONDITION', 'SIMULTANEOUS'), ('OVERLAP', 'BEFORE'), ('OVERLAP', 'CONTAINS'), ('OVERLAP', 'SIMULTANEOUS'), ('OVERLAP', 'ENDS-ON'), ('OVERLAP', 'BEGINS-ON'), ('ENDS-ON', 'BEFORE'), ('ENDS-ON', 'CONTAINS'), ('ENDS-ON', 'SUBEVENT'), ('ENDS-ON', 'CAUSE'), ('ENDS-ON', 'SIMULTANEOUS'), ('ENDS-ON', 'OVERLAP'), ('ENDS-ON', 'BEGINS-ON'), ('BEGINS-ON', 'BEFORE'), ('BEGINS-ON', 'CONTAINS'), ('BEGINS-ON', 'SIMULTANEOUS'), ('BEGINS-ON', 'OVERLAP'), ('BEGINS-ON', 'ENDS-ON')]
        neg_back = [('BEFORE', 'BEFORE'), ('BEFORE', 'CONTAINS'), ('BEFORE', 'SIMULTANEOUS'), ('BEFORE', 'OVERLAP'), ('BEFORE', 'ENDS-ON'), ('BEFORE', 'BEGINS-ON'), ('CONTAINS', 'BEFORE'), ('CONTAINS', 'CONTAINS'), ('CONTAINS', 'SUBEVENT'), ('CONTAINS', 'CAUSE'), ('CONTAINS', 'SIMULTANEOUS'), ('CONTAINS', 'PRECONDITION'), ('CONTAINS', 'OVERLAP'), ('CONTAINS', 'ENDS-ON'), ('CONTAINS', 'BEGINS-ON'), ('SUBEVENT', 'CONTAINS'), ('SUBEVENT', 'SUBEVENT'), ('SUBEVENT', 'CAUSE'), ('SUBEVENT', 'PRECONDITION'), ('SUBEVENT', 'ENDS-ON'), ('CAUSE', 'CONTAINS'), ('CAUSE', 'SUBEVENT'), ('CAUSE', 'CAUSE'), ('CAUSE', 'SIMULTANEOUS'), ('CAUSE', 'PRECONDITION'), ('CAUSE', 'ENDS-ON'), ('CAUSE', 'BEGINS-ON'), ('SIMULTANEOUS', 'BEFORE'), ('SIMULTANEOUS', 'CONTAINS'), ('SIMULTANEOUS', 'CAUSE'), ('SIMULTANEOUS', 'SIMULTANEOUS'), ('SIMULTANEOUS', 'PRECONDITION'), ('SIMULTANEOUS', 'OVERLAP'), ('SIMULTANEOUS', 'ENDS-ON'), ('SIMULTANEOUS', 'BEGINS-ON'), ('PRECONDITION', 'CONTAINS'), ('PRECONDITION', 'SUBEVENT'), ('PRECONDITION', 'CAUSE'), ('PRECONDITION', 'SIMULTANEOUS'), ('PRECONDITION', 'PRECONDITION'), ('PRECONDITION', 'OVERLAP'), ('PRECONDITION', 'BEGINS-ON'), ('OVERLAP', 'BEFORE'), ('OVERLAP', 'CONTAINS'), ('OVERLAP', 'SIMULTANEOUS'), ('OVERLAP', 'PRECONDITION'), ('OVERLAP', 'OVERLAP'), ('OVERLAP', 'ENDS-ON'), ('OVERLAP', 'BEGINS-ON'), ('ENDS-ON', 'BEFORE'), ('ENDS-ON', 'CONTAINS'), ('ENDS-ON', 'SUBEVENT'), ('ENDS-ON', 'CAUSE'), ('ENDS-ON', 'SIMULTANEOUS'), ('ENDS-ON', 'OVERLAP'), ('ENDS-ON', 'ENDS-ON'), ('ENDS-ON', 'BEGINS-ON'), ('BEGINS-ON', 'BEFORE'), ('BEGINS-ON', 'CONTAINS'), ('BEGINS-ON', 'CAUSE'), ('BEGINS-ON', 'SIMULTANEOUS'), ('BEGINS-ON', 'PRECONDITION'), ('BEGINS-ON', 'OVERLAP'), ('BEGINS-ON', 'ENDS-ON')]
        tp_rules_lit = [['BEFORE', 'BEFORE', 'BEFORE'], ['BEFORE', 'CONTAINS', 'BEFORE'], ['BEFORE', 'PRECONDITION', 'BEFORE'], ['BEFORE', 'SIMULTANEOUS', 'BEFORE'], ['CONTAINS', 'SIMULTANEOUS', 'CONTAINS'], ['CONTAINS', 'CONTAINS', 'CONTAINS'], ['CAUSE', 'SUBEVENT', 'CAUSE'], ['SIMULTANEOUS', 'BEFORE', 'BEFORE'], ['SIMULTANEOUS', 'CONTAINS', 'CONTAINS'], ['SIMULTANEOUS', 'OVERLAP', 'OVERLAP'], ['SIMULTANEOUS', 'SIMULTANEOUS', 'SIMULTANEOUS'], ['SIMULTANEOUS', 'BEGINS-ON', 'BEGINS-ON'], ['PRECONDITION', 'CAUSE', 'PRECONDITION'], ['PRECONDITION', 'SUBEVENT', 'PRECONDITION'], ['OVERLAP', 'SIMULTANEOUS', 'OVERLAP'], ['OVERLAP', 'ENDS-ON', 'BEFORE'], ['BEGINS-ON', 'SIMULTANEOUS', 'BEGINS-ON']]
    else:
        neg_forw = []
        neg_back = [('BEFORE', 'BEFORE')]
        tp_rules_lit = [['BEFORE', 'BEFORE', 'BEFORE']]

    bnf_rules =  [[LABEL_LIST.index(i) for i in rule] for rule in neg_forw]
    bnb_rules =  [[LABEL_LIST.index(i) for i in rule] for rule in neg_back]

    tp_rules = [[LABEL_LIST.index(i) for i in rule] for rule in tp_rules_lit]

    b_rel = [[[[] for j in range(2)] for i in range(2)] for l in range(nl)]
    t_rel = [[[] for i in range(3)] for l in range(nl)]
    for nf in bnf_rules:
        b_rel[nf[0]][0][1].append(nf[1])
    for nb in bnb_rules:
        b_rel[nb[0]][1][1].append(nb[1])
        #b_rel[nb[1]][1][1].append(nb[0])
    for i in range(3):
        for tp in tp_rules:
            temp = tp.copy()
            l = temp.pop(i)
            p, q = temp
            t_rel[l][i].append((p, q))

    config.t_rel = t_rel
    config.b_rel = b_rel

    return

def define_const_maven_em(config):
    """
    """
    LABEL_LIST = config.LABEL_LIST

    nl = len(LABEL_LIST)

    if config.ALL_CONSTRAINTS:
        pos_back = [('coreference', 'coreference')]#, ('SIMULTANEOUS', 'SIMULTANEOUS'), ('BEGINS-ON', 'BEGINS-ON')]
        neg_forw = [('BEFORE', 'CONTAINS'), ('BEFORE', 'SIMULTANEOUS'), ('BEFORE', 'OVERLAP'), ('BEFORE', 'ENDS-ON'), ('BEFORE', 'BEGINS-ON'), ('CONTAINS', 'BEFORE'), ('CONTAINS', 'SIMULTANEOUS'), ('CONTAINS', 'OVERLAP'), ('CONTAINS', 'ENDS-ON'), ('CONTAINS', 'BEGINS-ON'), ('subevent', 'CAUSE'), ('subevent', 'SIMULTANEOUS'), ('subevent', 'ENDS-ON'), ('CAUSE', 'subevent'), ('CAUSE', 'PRECONDITION'), ('CAUSE', 'ENDS-ON'), ('SIMULTANEOUS', 'BEFORE'), ('SIMULTANEOUS', 'CONTAINS'), ('SIMULTANEOUS', 'subevent'), ('SIMULTANEOUS', 'PRECONDITION'), ('SIMULTANEOUS', 'OVERLAP'), ('SIMULTANEOUS', 'ENDS-ON'), ('SIMULTANEOUS', 'BEGINS-ON'), ('PRECONDITION', 'CAUSE'), ('PRECONDITION', 'SIMULTANEOUS'), ('OVERLAP', 'BEFORE'), ('OVERLAP', 'CONTAINS'), ('OVERLAP', 'SIMULTANEOUS'), ('OVERLAP', 'ENDS-ON'), ('OVERLAP', 'BEGINS-ON'), ('ENDS-ON', 'BEFORE'), ('ENDS-ON', 'CONTAINS'), ('ENDS-ON', 'subevent'), ('ENDS-ON', 'CAUSE'), ('ENDS-ON', 'SIMULTANEOUS'), ('ENDS-ON', 'OVERLAP'), ('ENDS-ON', 'BEGINS-ON'), ('BEGINS-ON', 'BEFORE'), ('BEGINS-ON', 'CONTAINS'), ('BEGINS-ON', 'SIMULTANEOUS'), ('BEGINS-ON', 'OVERLAP'), ('BEGINS-ON', 'ENDS-ON')]
        neg_back = [('BEFORE', 'BEFORE'), ('BEFORE', 'CONTAINS'), ('BEFORE', 'SIMULTANEOUS'), ('BEFORE', 'OVERLAP'), ('BEFORE', 'ENDS-ON'), ('BEFORE', 'BEGINS-ON'), ('CONTAINS', 'BEFORE'), ('CONTAINS', 'CONTAINS'), ('CONTAINS', 'subevent'), ('CONTAINS', 'CAUSE'), ('CONTAINS', 'SIMULTANEOUS'), ('CONTAINS', 'PRECONDITION'), ('CONTAINS', 'OVERLAP'), ('CONTAINS', 'ENDS-ON'), ('CONTAINS', 'BEGINS-ON'), ('subevent', 'CONTAINS'), ('subevent', 'subevent'), ('subevent', 'CAUSE'), ('subevent', 'PRECONDITION'), ('subevent', 'ENDS-ON'), ('CAUSE', 'CONTAINS'), ('CAUSE', 'subevent'), ('CAUSE', 'CAUSE'), ('CAUSE', 'SIMULTANEOUS'), ('CAUSE', 'PRECONDITION'), ('CAUSE', 'ENDS-ON'), ('CAUSE', 'BEGINS-ON'), ('SIMULTANEOUS', 'BEFORE'), ('SIMULTANEOUS', 'CONTAINS'), ('SIMULTANEOUS', 'CAUSE'), ('SIMULTANEOUS', 'SIMULTANEOUS'), ('SIMULTANEOUS', 'PRECONDITION'), ('SIMULTANEOUS', 'OVERLAP'), ('SIMULTANEOUS', 'ENDS-ON'), ('SIMULTANEOUS', 'BEGINS-ON'), ('PRECONDITION', 'CONTAINS'), ('PRECONDITION', 'subevent'), ('PRECONDITION', 'CAUSE'), ('PRECONDITION', 'SIMULTANEOUS'), ('PRECONDITION', 'PRECONDITION'), ('PRECONDITION', 'OVERLAP'), ('PRECONDITION', 'BEGINS-ON'), ('OVERLAP', 'BEFORE'), ('OVERLAP', 'CONTAINS'), ('OVERLAP', 'SIMULTANEOUS'), ('OVERLAP', 'PRECONDITION'), ('OVERLAP', 'OVERLAP'), ('OVERLAP', 'ENDS-ON'), ('OVERLAP', 'BEGINS-ON'), ('ENDS-ON', 'BEFORE'), ('ENDS-ON', 'CONTAINS'), ('ENDS-ON', 'subevent'), ('ENDS-ON', 'CAUSE'), ('ENDS-ON', 'SIMULTANEOUS'), ('ENDS-ON', 'OVERLAP'), ('ENDS-ON', 'ENDS-ON'), ('ENDS-ON', 'BEGINS-ON'), ('BEGINS-ON', 'BEFORE'), ('BEGINS-ON', 'CONTAINS'), ('BEGINS-ON', 'CAUSE'), ('BEGINS-ON', 'SIMULTANEOUS'), ('BEGINS-ON', 'PRECONDITION'), ('BEGINS-ON', 'OVERLAP'), ('BEGINS-ON', 'ENDS-ON')]
        tp_rules_lit = [['BEFORE', 'BEFORE', 'BEFORE'], ['BEFORE', 'CONTAINS', 'BEFORE'], ['BEFORE', 'PRECONDITION', 'BEFORE'], ['BEFORE', 'SIMULTANEOUS', 'BEFORE'], ['CONTAINS', 'SIMULTANEOUS', 'CONTAINS'], ['CONTAINS', 'CONTAINS', 'CONTAINS'], ['CAUSE', 'subevent', 'CAUSE'], ['SIMULTANEOUS', 'BEFORE', 'BEFORE'], ['SIMULTANEOUS', 'CONTAINS', 'CONTAINS'], ['SIMULTANEOUS', 'OVERLAP', 'OVERLAP'], ['SIMULTANEOUS', 'SIMULTANEOUS', 'SIMULTANEOUS'], ['SIMULTANEOUS', 'BEGINS-ON', 'BEGINS-ON'], ['PRECONDITION', 'CAUSE', 'PRECONDITION'], ['PRECONDITION', 'subevent', 'PRECONDITION'], ['OVERLAP', 'SIMULTANEOUS', 'OVERLAP'], ['OVERLAP', 'ENDS-ON', 'BEFORE'], ['BEGINS-ON', 'SIMULTANEOUS', 'BEGINS-ON']]
        tp_rules_lit += [['coreference', 'coreference', 'coreference']]
        tp_rules_lit += [['coreference', 'BEFORE', 'BEFORE'], ['coreference', 'OVERLAP', 'OVERLAP'], ['coreference', 'CONTAINS', 'CONTAINS'], ['coreference', 'SIMULTANEOUS', 'SIMULTANEOUS'], ['coreference', 'ENDS-ON', 'ENDS-ON'], ['coreference', 'BEGINS-ON', 'BEGINS-ON'], ['coreference', 'CAUSE', 'CAUSE'], ['coreference', 'PRECONDITION', 'PRECONDITION'], ['coreference', 'subevent', 'subevent']]
        tp_rules_lit += [['BEFORE', 'coreference', 'BEFORE'], ['OVERLAP', 'coreference', 'OVERLAP'], ['CONTAINS', 'coreference', 'CONTAINS'], ['SIMULTANEOUS', 'coreference', 'SIMULTANEOUS'], ['ENDS-ON', 'coreference', 'ENDS-ON'], ['BEGINS-ON', 'coreference', 'BEGINS-ON'], ['CAUSE', 'coreference', 'CAUSE'], ['PRECONDITION', 'coreference', 'PRECONDITION'], ['subevent', 'coreference', 'subevent']]
    else:
        pos_forw = []
        neg_forw = []
        neg_back = [('BEFORE', 'BEFORE')]
        tp_rules_lit = [['BEFORE', 'BEFORE', 'BEFORE']]

    bpb_rules = [[LABEL_LIST.index(i) for i in rule] for rule in pos_back]
    bnf_rules = [[LABEL_LIST.index(i) for i in rule] for rule in neg_forw]
    bnb_rules = [[LABEL_LIST.index(i) for i in rule] for rule in neg_back]

    tp_rules = [[LABEL_LIST.index(i) for i in rule] for rule in tp_rules_lit]

    b_rel = [[[[] for j in range(2)] for i in range(2)] for l in range(nl)]
    t_rel = [[[] for i in range(3)] for l in range(nl)]

    for pf in bpb_rules:
        b_rel[pf[0]][1][0].append(pf[1])
    for nf in bnf_rules:
        b_rel[nf[0]][0][1].append(nf[1])
    for nb in bnb_rules:
        b_rel[nb[0]][1][1].append(nb[1])
        #b_rel[nb[1]][1][1].append(nb[0])
    for i in range(3):
        for tp in tp_rules:
            temp = tp.copy()
            l = temp.pop(i)
            p, q = temp
            t_rel[l][i].append((p, q))

    config.t_rel = t_rel
    config.b_rel = b_rel

    return

# def define_const_maven_em(config):
#     """
#     SEGMENTED VERSION
#     """
#     LABEL_LIST = config.LABEL_LIST

#     nl = len(LABEL_LIST)

#     if config.ALL_CONSTRAINTS:
#         pos_back = [('coreference', 'coreference'), ('SIMULTANEOUS', 'SIMULTANEOUS'), ('BEGINS-ON', 'BEGINS-ON')]
#         neg_forw = [('BEFORE', 'CONTAINS'), ('BEFORE', 'SIMULTANEOUS'), ('BEFORE', 'OVERLAP'), ('BEFORE', 'ENDS-ON'), ('BEFORE', 'BEGINS-ON'), ('CONTAINS', 'BEFORE'), ('CONTAINS', 'SIMULTANEOUS'), ('CONTAINS', 'OVERLAP'), ('CONTAINS', 'ENDS-ON'), ('CONTAINS', 'BEGINS-ON'), 
#                     #('subevent', 'CAUSE'), ('subevent', 'SIMULTANEOUS'), ('subevent', 'ENDS-ON'), ('CAUSE', 'subevent'), 
#                     ('CAUSE', 'PRECONDITION'), 
#                     #('CAUSE', 'ENDS-ON'), 
#                     ('SIMULTANEOUS', 'BEFORE'), ('SIMULTANEOUS', 'CONTAINS'), 
#                     #('SIMULTANEOUS', 'subevent'), ('SIMULTANEOUS', 'PRECONDITION'), 
#                     ('SIMULTANEOUS', 'OVERLAP'), ('SIMULTANEOUS', 'ENDS-ON'), ('SIMULTANEOUS', 'BEGINS-ON'), ('PRECONDITION', 'CAUSE'), 
#                     #('PRECONDITION', 'SIMULTANEOUS'), 
#                     ('OVERLAP', 'BEFORE'), ('OVERLAP', 'CONTAINS'), ('OVERLAP', 'SIMULTANEOUS'), ('OVERLAP', 'ENDS-ON'), ('OVERLAP', 'BEGINS-ON'), ('ENDS-ON', 'BEFORE'), ('ENDS-ON', 'CONTAINS'), 
#                     #('ENDS-ON', 'subevent'), ('ENDS-ON', 'CAUSE'), 
#                     ('ENDS-ON', 'SIMULTANEOUS'), ('ENDS-ON', 'OVERLAP'), ('ENDS-ON', 'BEGINS-ON'), ('BEGINS-ON', 'BEFORE'), ('BEGINS-ON', 'CONTAINS'), ('BEGINS-ON', 'SIMULTANEOUS'), ('BEGINS-ON', 'OVERLAP'), ('BEGINS-ON', 'ENDS-ON')]
#         neg_back = [('BEFORE', 'BEFORE'), ('BEFORE', 'CONTAINS'), ('BEFORE', 'SIMULTANEOUS'), ('BEFORE', 'OVERLAP'), ('BEFORE', 'ENDS-ON'), ('BEFORE', 'BEGINS-ON'), ('CONTAINS', 'BEFORE'), ('CONTAINS', 'CONTAINS'), 
#                     #('CONTAINS', 'subevent'), ('CONTAINS', 'CAUSE'), 
#                     ('CONTAINS', 'SIMULTANEOUS'), 
#                     #('CONTAINS', 'PRECONDITION'), 
#                     ('CONTAINS', 'OVERLAP'), ('CONTAINS', 'ENDS-ON'), ('CONTAINS', 'BEGINS-ON'), 
#                     #('subevent', 'CONTAINS'), 
#                     ('subevent', 'subevent'), 
#                     #('subevent', 'CAUSE'), ('subevent', 'PRECONDITION'), ('subevent', 'ENDS-ON'), ('CAUSE', 'CONTAINS'), ('CAUSE', 'subevent'), 
#                     ('CAUSE', 'CAUSE'), 
#                     #('CAUSE', 'SIMULTANEOUS'), 
#                     ('CAUSE', 'PRECONDITION'), 
#                     #('CAUSE', 'ENDS-ON'), ('CAUSE', 'BEGINS-ON'), 
#                     ('SIMULTANEOUS', 'BEFORE'), ('SIMULTANEOUS', 'CONTAINS'), 
#                     #('SIMULTANEOUS', 'CAUSE'), 
#                     ('SIMULTANEOUS', 'SIMULTANEOUS'), 
#                     #('SIMULTANEOUS', 'PRECONDITION'), 
#                     ('SIMULTANEOUS', 'OVERLAP'), ('SIMULTANEOUS', 'ENDS-ON'), ('SIMULTANEOUS', 'BEGINS-ON'), 
#                     #('PRECONDITION', 'CONTAINS'), ('PRECONDITION', 'subevent'), 
#                     ('PRECONDITION', 'CAUSE'), 
#                     #('PRECONDITION', 'SIMULTANEOUS'), 
#                     ('PRECONDITION', 'PRECONDITION'), 
#                     #('PRECONDITION', 'OVERLAP'), ('PRECONDITION', 'BEGINS-ON'), 
#                     ('OVERLAP', 'BEFORE'), ('OVERLAP', 'CONTAINS'), ('OVERLAP', 'SIMULTANEOUS'), 
#                     #('OVERLAP', 'PRECONDITION'), 
#                     ('OVERLAP', 'OVERLAP'), ('OVERLAP', 'ENDS-ON'), ('OVERLAP', 'BEGINS-ON'), ('ENDS-ON', 'BEFORE'), ('ENDS-ON', 'CONTAINS'), 
#                     #('ENDS-ON', 'subevent'), ('ENDS-ON', 'CAUSE'), 
#                     ('ENDS-ON', 'SIMULTANEOUS'), ('ENDS-ON', 'OVERLAP'), ('ENDS-ON', 'ENDS-ON'), ('ENDS-ON', 'BEGINS-ON'), ('BEGINS-ON', 'BEFORE'), ('BEGINS-ON', 'CONTAINS'), 
#                     #('BEGINS-ON', 'CAUSE'), 
#                     ('BEGINS-ON', 'SIMULTANEOUS'), 
#                     #('BEGINS-ON', 'PRECONDITION'), 
#                     ('BEGINS-ON', 'OVERLAP'), ('BEGINS-ON', 'ENDS-ON')]
#         tp_rules_lit = [['BEFORE', 'BEFORE', 'BEFORE'], ['BEFORE', 'CONTAINS', 'BEFORE'], 
#                         #['BEFORE', 'PRECONDITION', 'BEFORE'], 
#                         ['BEFORE', 'SIMULTANEOUS', 'BEFORE'], ['CONTAINS', 'SIMULTANEOUS', 'CONTAINS'], ['CONTAINS', 'CONTAINS', 'CONTAINS'], 
#                         #['CAUSE', 'subevent', 'CAUSE'], 
#                         ['SIMULTANEOUS', 'BEFORE', 'BEFORE'], ['SIMULTANEOUS', 'CONTAINS', 'CONTAINS'], ['SIMULTANEOUS', 'OVERLAP', 'OVERLAP'], ['SIMULTANEOUS', 'SIMULTANEOUS', 'SIMULTANEOUS'], ['SIMULTANEOUS', 'BEGINS-ON', 'BEGINS-ON'], ['PRECONDITION', 'CAUSE', 'PRECONDITION'], 
#                         #['PRECONDITION', 'subevent', 'PRECONDITION'], 
#                         ['OVERLAP', 'SIMULTANEOUS', 'OVERLAP'], ['OVERLAP', 'ENDS-ON', 'BEFORE'], ['BEGINS-ON', 'SIMULTANEOUS', 'BEGINS-ON']]
#         tp_rules_lit += [['coreference', 'coreference', 'coreference']]
#         tp_rules_lit += [#['coreference', 'BEFORE', 'BEFORE'], ['coreference', 'OVERLAP', 'OVERLAP'], ['coreference', 'CONTAINS', 'CONTAINS'], ['coreference', 'SIMULTANEOUS', 'SIMULTANEOUS'], ['coreference', 'ENDS-ON', 'ENDS-ON'], ['coreference', 'BEGINS-ON', 'BEGINS-ON'], ['coreference', 'CAUSE', 'CAUSE'], ['coreference', 'PRECONDITION', 'PRECONDITION'], ['coreference', 'subevent', 'subevent']
#         ]
#         tp_rules_lit += [#['BEFORE', 'coreference', 'BEFORE'], ['OVERLAP', 'coreference', 'OVERLAP'], ['CONTAINS', 'coreference', 'CONTAINS'], ['SIMULTANEOUS', 'coreference', 'SIMULTANEOUS'], ['ENDS-ON', 'coreference', 'ENDS-ON'], ['BEGINS-ON', 'coreference', 'BEGINS-ON'], ['CAUSE', 'coreference', 'CAUSE'], ['PRECONDITION', 'coreference', 'PRECONDITION'], ['subevent', 'coreference', 'subevent']
#         ]
#     else:
#         pos_forw = []
#         neg_forw = []
#         neg_back = [('BEFORE', 'BEFORE')]
#         tp_rules_lit = [['BEFORE', 'BEFORE', 'BEFORE']]

#     bpb_rules = [[LABEL_LIST.index(i) for i in rule] for rule in pos_back]
#     bnf_rules = [[LABEL_LIST.index(i) for i in rule] for rule in neg_forw]
#     bnb_rules = [[LABEL_LIST.index(i) for i in rule] for rule in neg_back]

#     tp_rules = [[LABEL_LIST.index(i) for i in rule] for rule in tp_rules_lit]

#     b_rel = [[[[] for j in range(2)] for i in range(2)] for l in range(nl)]
#     t_rel = [[[] for i in range(3)] for l in range(nl)]

#     for pf in bpb_rules:
#         b_rel[pf[0]][1][0].append(pf[1])
#     for nf in bnf_rules:
#         b_rel[nf[0]][0][1].append(nf[1])
#     for nb in bnb_rules:
#         b_rel[nb[0]][1][1].append(nb[1])
#         #b_rel[nb[1]][1][1].append(nb[0])
#     for i in range(3):
#         for tp in tp_rules:
#             temp = tp.copy()
#             l = temp.pop(i)
#             p, q = temp
#             t_rel[l][i].append((p, q))

#     config.t_rel = t_rel
#     config.b_rel = b_rel

#     return

#############################################################################################

def _word_to_token(word_lists, token_offsets):
    """
    Maps each word to its corresponding token indices based on character offsets.

    Args:
        word_lists (list of list of str): Batch of word lists.
        token_offsets (list of list of tuple): Batch of token offsets.

    Returns:
        list of list of set: Each set contains token indices corresponding to a word.
    """
    res = []
    for words, token_offset in zip(word_lists, token_offsets):

        word_offset = []
        char_offset = 0
        for word in words:
            word_offset.append([char_offset, char_offset + len(word)])
            char_offset += len(word)+1

        res.append([])
        t_id = 0
        for w_off in word_offset:

            res[-1].append(set())
            while t_id<len(token_offset) and token_offset[t_id][0]<w_off[1]:

                if w_off[0]<token_offset[t_id][1]:
                    res[-1][-1].add(t_id)
                t_id += 1
    return res


def set_word_to_tok(word_set, word_to_token):
    """
    """
    res = set()
    for word_idx in word_set:
        res.update(word_to_token[word_idx])
    return frozenset(res)


def tok_clust_pair_rel(annots, word_to_token):
    """
    """
    res = list()
    for doc_idx, doc_annot in enumerate(annots):
        res.append(dict())
        for clust_pair, gold in doc_annot.items():
            key = (
                set_word_to_tok(clust_pair[0], word_to_token[doc_idx]), 
                set_word_to_tok(clust_pair[1], word_to_token[doc_idx])
            )
            res[-1][key] = gold
    return res


def tok_pair_annot(tok_set_annot):
    """
    """
    pair_indices = list()
    pair_labels = list()
    for _, doc_annot in enumerate(tok_set_annot):
        pair_indices.append(list())
        pair_labels.append(list())
        for tok_set_pair, gold in doc_annot.items():
            for idx in tok_set_pair[0]:
                for jdx in tok_set_pair[1]:
                    pair_indices[-1].append((idx, jdx))
                    pair_labels[-1].append(gold)
    return pair_indices, pair_labels


def compute_class_counts(pair_labels, config):
    """
    """
    counts = np.zeros(config.NUM_LABELS)
    n_total = 0
    for doc_labels in pair_labels:
        for label_list in doc_labels:
            counts += np.array(label_list)
            n_total += 1
    return counts, n_total


def under_sample(tok_set_annot, config):
    """
    """
    # Accumulate the labels accros the dataset
    labels = [gold.copy() for doc_tsa in tok_set_annot for gold in doc_tsa.values()]
    # Number of samples in the dataset
    count = len(labels)
    # Count the number of positive samples per labels
    label_count = [sum([x[i] for x in labels]) for i in range(len(labels[0]))]
    # Count the number of all negative samples
    noth_count = sum([0 if 1 in gold else 1 for gold in labels])
    # Extract the median support for the labels
    med_count = statistics.median(label_count)
    # Get a proportion of kept samples for each labels
    lab_thres = [med_count/lc for lc in label_count]
    for doc_idx, doc_tsa in enumerate(tok_set_annot):
        pop_keys = []
        for set_pair, list_label in doc_tsa.items():
            # Get the maximum proportion for the positive labels of the sample
            lamb = max(map(lambda x: x[0]*x[1], zip(list_label.copy(), lab_thres)))
            # If there are none consider the all negative proportion instead
            if lamb == 0 and random.random()>med_count/noth_count:
                pop_keys.append(set_pair)
            elif random.random()>lamb:
                pop_keys.append(set_pair)
        for key in pop_keys:
            doc_tsa.pop(key)
    return tok_set_annot

def create_global_attention_mask(mentions_list, attention_mask):
    """
    Create a global attention mask for documents.

    Parameters:
        mentions_list (list of list of int): A list of mentions for each document. Each document has a list of mention indices.
        attention_mask (torch.Tensor): Attention mask for the documents (batch_size x seq_len).

    Returns:
        torch.Tensor: Global attention mask for the documents (batch_size x seq_len).
    """
    # Initialize the global attention mask with the same shape as attention_mask
    global_attention_mask = torch.zeros_like(attention_mask, dtype=torch.long)

    for doc_idx, doc_mentions in enumerate(mentions_list):
        # Set indices corresponding to mentions to 1 in the global attention mask
        for index in doc_mentions:
            global_attention_mask[doc_idx, 0] = 1
            global_attention_mask[doc_idx, -1] = 1 # Changed may not work
            if 0 <= index < global_attention_mask.size(1):
                global_attention_mask[doc_idx, index] = 1

    return global_attention_mask


def data_prep(dataset, config, undsamp=False):
    """
    """
    word_set_annot = dataset.word_set_annotation(no_identical=config.NO_IDENTICAL, frame_scope=config.FRAME_SCOPE)
    word_lists = dataset.word_list()

    config.LABEL_LIST = dataset.ere_types

    if config.NOTHING_LABEL:
        config.LABEL_LIST = ['NOTHING']+config.LABEL_LIST
    config.NUM_LABELS = len(config.LABEL_LIST)        # Total number of possible labels

    if config.NOTHING_LABEL:
        for doc in word_set_annot:
            for labels in doc.values():
                if sum(labels)==0:
                    labels = [1] + labels
                else:
                    labels = [0] + labels

    if config.WORD_SPLIT:
        words = word_lists
    else:
        words = list(map(' '.join, word_lists))
    
    if config.ENCODER_MODEL == "longformer":
        tokenizer = LongformerTokenizerFast.from_pretrained(config.MODEL_NAME, add_prefix_space=True)
    elif config.ENCODER_MODEL == "modernbert":
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, add_prefix_space=True)
    elif config.ENCODER_MODEL == "qwen-0.6":
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, add_prefix_space=True)#, padding_side='right') # May be right or left
    
    tokens = tokenizer(words, is_split_into_words=config.WORD_SPLIT, return_offsets_mapping=True, padding=config.PADDING, truncation=config.TRUNCATION, max_length=config.MAX_SEQ_LENGTH, return_tensors='pt')
    w_to_t = _word_to_token(dataset.word_list(), tokens['offset_mapping'])

    tok_set_annot = tok_clust_pair_rel(word_set_annot, w_to_t)

    # There to put relations subset
    # This should be the end of the funciton as tok_set_annot and the tokens are enough
    # From this i could build a matrix dataset with masking and other torch operations as a better way to do things
    # Except this would require an adoc collate function and modifying the model handling which is difficult and 
    # may be more complex
    # Also no longer aligned with the constraints

    if undsamp:
        under_sample(tok_set_annot, config)

    pair_indices, pair_labels = tok_pair_annot(tok_set_annot)

    if config.GLOBAL_MENTION:
        mentions = [list(set([j for i in pi for j in i])) for pi in pair_indices]
        tokens['global_attention_mask'] = create_global_attention_mask(mentions, tokens['attention_mask'])

    documents = {'tokens': tokens , 'pair_indices': pair_indices, 'pair_labels': pair_labels}

    counts, n_total = compute_class_counts(pair_labels, config)

    return documents, tok_set_annot, counts, n_total

#############################################################################################

def segment(golds, config):
    """
    """
    nl = config.NUM_LABELS
    bbp = [(i, j) for i in range(len(config.b_rel)) for j in config.b_rel[i][1][0]]
    bfn = [(i, j) for i in range(len(config.b_rel)) for j in config.b_rel[i][0][1]]
    bbn = [(i, j) for i in range(len(config.b_rel)) for j in config.b_rel[i][1][1]]
    tft = [(i, j, k) for i in range(len(config.t_rel)) for (j, k) in config.t_rel[i][0]]

    stuff = list(set([obj for samp in golds for obj in samp]))
    stuff.sort(key=min)
    mask = {pair: [0]*nl for pair in golds}
    part_mask = {pair: [0]*nl for pair in golds}
    gen_mask = {pair: [0]*nl for pair in golds}
    for obja in stuff:
        for objb in stuff:
            if not (obja, objb) in golds:
                continue
            for i, j in bfn:
                if golds[(obja, objb)][i] == 1:
                    mask[(obja, objb)][j] = 1
                    part_mask[(obja, objb)][j] = 1
                    gen_mask[(obja, objb)][i] += 1
            for i, j in bbn:
                if golds[(obja, objb)][i] == 1:
                    mask[(objb, obja)][j] = 1
                    part_mask[(objb, obja)][j] = 1
                    gen_mask[(obja, objb)][i] += 1
            # The following for loop could be removed if all coreference are fundamental
            # Similarily an exception on the ternary relations for coreference could be made
            for i, j in bbp:
                if golds[(obja, objb)][i] == 1:
                    mask[(objb, obja)][j] = 1
                    part_mask[(objb, obja)][j] = 1
                    gen_mask[(obja, objb)][i] += 1
            for objc in stuff:
                if not (objb, objc) in golds or not (obja, objc) in golds:
                    continue
                for i, j, k in tft:
                    if golds[(obja, objb)][i] == 1 and golds[(objb, objc)][j] == 1:
                        mask[(obja, objc)][k] = 1
                        part_mask[(obja, objc)][k] = 1
                        gen_mask[(obja, objb)][i] += 1
                        gen_mask[(objb, objc)][j] += 1
                    if golds[(obja, objb)][i] == 1 and golds[(obja, objc)][k] == 0:
                        mask[(objb, objc)][j] = 1
                        gen_mask[(obja, objb)][i] += 1
                        gen_mask[(obja, objc)][k] += 1
                    if golds[(objb, objc)][j] == 1 and golds[(obja, objc)][k] == 0:
                        mask[(obja, objb)][i] = 1
                        gen_mask[(objb, objc)][j] += 1
                        gen_mask[(obja, objc)][k] += 1
    return mask, part_mask, gen_mask

def dedu_mask_to_loss_weight(train_loss_weights, config):
    """
    """
    for ftlw, ptlw, gtlw in train_loss_weights:
        for flw_mask, glw_mask in zip(ftlw.values(), gtlw.values(), strict=True):
            for i, _ in enumerate(flw_mask):
                if flw_mask[i]==1:
                    if glw_mask[i]>0:
                      flw_mask[i] = config.DISCRIM_GEN_ALPHA
                    else:
                      flw_mask[i] = config.DISCRIM_DEDU_ALPHA
                elif glw_mask[i]>0:
                    flw_mask[i] = config.DISCRIM_FUND_GEN_ALPHA
                else:
                    flw_mask[i] = config.DISCRIM_FUND_ALPHA
    return [tlw[0] for tlw in train_loss_weights]

#############################################################################################

def check_cons(a, b):
    """
    """
    if a>0 and b==1 or a<0 and b==0:
        res = True
    else:
        res = False
    return res

def check_veracity(new, lab, pred):
    """
    Check if a wrong prediction is used to deduct a correct one
    """
    n = pred.shape[0]
    res = pred
    if n>1:
        cons = np.array([check_cons(
            new[pred[i, 0], pred[i, 1], pred[i, 2]]*pred[i, 3], 
            lab[pred[i, 0], pred[i, 1], pred[i, 2]]) 
            for i in range(n)], dtype=bool)
        if np.any(cons):
            res = pred[cons, :]
    return res

def change(new, lab, i, j, l, pred, pile, move, label_gradient=False):
    """
    """
    if not np.abs(new[i, j, l]) > 0.:
        if label_gradient:
            pred = check_veracity(new, lab, pred)
        a = np.argmin(np.abs(new[pred[:, 0], pred[:, 1], pred[:, 2]]*pred[:, 3]))
        new[i, j, l] = new[pred[a, 0], pred[a, 1], pred[a, 2]]*pred[a, 3]
        pile.append((i, j, l))  # Add the new relation to the pile
        move[i, j, l] = move[pred[a, 0], pred[a, 1], pred[a, 2]]
        move[i, j, l, 3] *= pred[a, 3]
    return

def binary_change(new, lab, i, j, l, pred, pile, b_rule, move, label_gradient=False):
    """
    Perform a binary assignment to the solution `new` and ensure consistency.
    """
    change(new, lab, i, j, l, pred, pile, move, label_gradient)
    binary(new, lab, i, j, l, pile, b_rule, move, label_gradient)
    return

def binary(new, lab, i, j, l, pile, b_rule, move, label_gradient=False):
    """
    Perform a binary assignment to the solution `new` and ensure consistency.
    """
    if new[i, j, l] > 0:
        for p in b_rule[l][0][0]:  # Forward True
            change(new, lab, i, j, p,  np.array(([i, j, l, 1],)), pile, move, label_gradient)
        for p in b_rule[l][0][1]:  # Forward False
            change(new, lab, i, j, p,  np.array(([i, j, l, -1],)), pile, move, label_gradient)
        for p in b_rule[l][1][0]:  # Backward True
            change(new, lab, j, i, p,  np.array(([i, j, l, 1],)), pile, move, label_gradient)
        for p in b_rule[l][1][1]:  # Backward False
            change(new, lab, j, i, p,  np.array(([i, j, l, -1],)), pile, move, label_gradient)
    return

def ternary(new, lab, i, j, l, pile, b_rule, t_rule, move, label_gradient=False):
    """
    Perform a Ternary assignment to the solution `new` and ensure consistency.
    """
    for k in range(len(new)):
        if k == j or k == i:
            continue

        if new[i, j, l] > 0:
            for p, q in t_rule[l][0]:  # l is first in sequence : i, j, k (l, p, q)
                if new[j, k, p] > 0:
                    binary_change(new, lab, i, k, q, np.array(([i, j, l, 1], [j, k, p, 1])), pile, b_rule, move, label_gradient)
                if new[i, k, q] < 0:
                    binary_change(new, lab, j, k, p, np.array(([i, j, l, -1], [i, k, q, 1])), pile, b_rule, move, label_gradient)
            for p, q in t_rule[l][1]:  # l is second in sequence : k, i, j (p, l, q)
                if new[k, i, p] > 0:
                    binary_change(new, lab, k, j, q, np.array(([i, j, l, 1], [k, i, p, 1])), pile, b_rule, move, label_gradient)
                if new[k, j, q] < 0:
                    binary_change(new, lab, k, i, p, np.array(([i, j, l, -1], [k, j, q, 1])), pile, b_rule, move, label_gradient)
        if new[i, j, l] < 0:
            for p, q in t_rule[l][2]:  # l is last in sequence : i, k, j (p, q, l)
                if new[i, k, p] > 0:
                    binary_change(new, lab, k, j, q, np.array(([i, j, l, 1], [i, k, p, -1])), pile, b_rule, move, label_gradient)
                if new[k, j, q] > 0:
                    binary_change(new, lab, i, k, l, np.array(([i, j, l, 1], [k, j, q, -1])), pile, b_rule, move, label_gradient)
    return

def check(new, lab, pile, b_rule, t_rule, move, label_gradient=False):
    """
    Check and propagate constraints for the assignment.
    """
    i, j, l = pile[-1]
    binary(new, lab, i, j, l, pile, b_rule, move, label_gradient)
    while len(pile) > 0:
        i, j, l = pile.pop(-1)
        if i == j:
            continue
        ternary(new, lab, i, j, l, pile, b_rule, t_rule, move, label_gradient)
    return True

def greedy_search(inp, lab, b_rule, t_rule, label_gradient=False):
    """
    Perform a greedy search to find the optimal assignment based on the probability matrix.
    """
    n, _, l = inp.shape
    res = np.zeros_like(inp)
    move = np.array([[[[i, j, k, 1] for k in range(l)] for j in range(n)] for i in range(n)])

    # Get the sorted indices in descending order of absolute value
    sorted_indices = np.argsort(-np.abs(inp).ravel())
    sorted_multi_indices = np.unravel_index(sorted_indices, inp.shape)
    sorted_multi_indices = list(zip(*sorted_multi_indices))  # Convert to list of tuples

    # Iterate over the sorted indices
    for i, j, r in sorted_multi_indices:
        # Checking to avoid self relations
        if i == j:
            continue
        if abs(res[i, j, r]) > 0.:
            continue
        res[i, j, r] = inp[i, j, r]
        pile = [(i, j, r)]
        check(res, lab, pile, b_rule, t_rule, move, label_gradient)
    return res, move

def opt_move(inp, n, l):
    # Initialize array B of size (n*n*l, 7)
    move = np.zeros((n * n * l, 7), dtype=int)

    # Fill array B according to the given rule
    index = 0
    for p in range(n):
        for q in range(n):
            for r in range(l):
                coord = inp[p, q, r]
                move[index] = np.concatenate(([p, q, r], coord))
                index += 1
    return move

def copy_with_sign_change(A, coord):
    coord = torch.tensor(coord, dtype=torch.long)
    dest_i = coord[:, 0]
    dest_j = coord[:, 1]
    dest_r = coord[:, 2]
    src_i = coord[:, 3]
    src_j = coord[:, 4]
    src_r = coord[:, 5]
    sign_change = coord[:, 6]

    # Copy values with possible sign change
    A[dest_i, dest_j, dest_r] = A[src_i, src_j, src_r] * sign_change
    return A

def optimize_with_gradient(pred, gold, b_rel, t_rel, label_gradient=False):
    n, _, l = pred.shape

    start_time = time.time()
    A = np.random.randn(n, n, l)
    np_pred = pred.clone().detach().numpy()
    np.copyto(A, np_pred, casting='safe')
    B = np.random.randn(n, n, l)
    np_gold = pred.clone().detach().numpy()
    np.copyto(B, np_gold, casting='safe')
    end_time = time.time()
    #print(type(A))
    #print(f"To Numpy Copy CPU time: {end_time - start_time} seconds")

    start_time = time.time()
    opt_array, coord = greedy_search(A, B, b_rel, t_rel, label_gradient)
    end_time = time.time()
    #print(f"Total Greedy Search CPU time: {end_time - start_time} seconds")

    move = opt_move(coord, n, l)

    start_time = time.time()
    sol = copy_with_sign_change(pred.clone(), move)
    end_time = time.time()
    #print(f"Gradient Copying Execution time: {end_time - start_time:.6f} seconds")
    print(" UwU ", end="")
    
    return sol

#####################################################################################

def split_batch_in_docs(preds, golds, inds, doc_idx_list, nd):
    """
    Splitting the batch into documents
    """

    # List of preds and golds split by document
    preds_split = [[] for i in range(nd)]
    golds_split = [[] for i in range(nd)]
    inds_split = [[] for i in range(nd)]

    # Iterate over every pair prediction, gold and indices
    for idx, (doc_idx, pair) in enumerate(inds):
        # Get the index of the document in the batch
        batch_idx = doc_idx_list.index(doc_idx)
        # Copy the preds, golds, and inds to the document
        preds_split[batch_idx].append(preds[idx])
        golds_split[batch_idx].append(golds[idx])
        inds_split[batch_idx].append(inds[idx])

    return preds_split, golds_split, inds_split

def doc_to_mat(pred, gold, ind, nl):
    """
    Taking one doc at a time to output matrix style tensors for pred and gold
    """

    evt_list = list(set([e for pair in ind for e in pair[1]]))
    #print(evt_list)
    # Uncertain order from the set() is sorted
    evt_list.sort(key=min)
    evt_dic = {evt: idx for idx, evt in enumerate(evt_list)}

    ne = math.ceil(math.sqrt(len(pred)))
    mat_pred = torch.zeros(ne, ne, nl, requires_grad=pred[0].requires_grad, dtype=pred[0].dtype, device=pred[0].device).clone()
    mat_gold = torch.zeros(ne, ne, nl, requires_grad=gold[0].requires_grad, dtype=gold[0].dtype, device=gold[0].device).clone()
    mat_mask = torch.zeros(ne, ne, nl, requires_grad=False, dtype=gold[0].dtype, device=gold[0].device)

    check_doc_idx = -1
    for idx, (doc_idx, pair) in enumerate(ind):
        if check_doc_idx == -1:
            check_doc_idx = doc_idx
        elif check_doc_idx != doc_idx:
            raise ValueError(f"Differing doc_idx values in ind: {(check_doc_idx, doc_idx)}")
        x, y = evt_dic[pair[0]], evt_dic[pair[1]]
        mat_pred[x, y] = pred[idx]
        mat_gold[x, y] = gold[idx]
        mat_mask[x, y] = torch.ones_like(gold[idx], requires_grad=False)

    return mat_pred, mat_gold, mat_mask, evt_dic, check_doc_idx

def pred_to_mat(preds, golds, inds, n_max=139):
    """
    Getting the predictions (and gold) 
    and making them into well shaped tensors one per document
    """

    # List of document idx
    doc_idx_list = list(set([i[0] for i in inds]))
    # Sort what has undefined order
    doc_idx_list.sort()
    nd = len(doc_idx_list)
    # Number labels and number of documents in the batch
    nl = preds.size()[-1]

    # Split the batch into lists of documents
    preds_split, golds_split, inds_split = split_batch_in_docs(preds, golds, inds, doc_idx_list, nd)

    # Result list of tensors for preds and golds
    res_preds, res_golds, res_masks, res_evts = list(), list(), list(), list()

    # Iterates over the documents
    for batch_idx in range(nd):

        pred = preds_split[batch_idx]
        gold = golds_split[batch_idx]
        ind = inds_split[batch_idx]

        mat_pred, mat_gold, mat_mask, evt_dic, check_doc_idx = doc_to_mat(pred, gold, ind, nl)

        if check_doc_idx != doc_idx_list[batch_idx]:
            raise ValueError(f"Differing doc_idx between ind and doc_idx_list in pred_to_mat: {(doc_idx_list[batch_idx], check_doc_idx)}")

        res_preds.append(mat_pred)
        res_golds.append(mat_gold)
        res_masks.append(mat_mask)
        res_evts.append(evt_dic)

    return res_preds, res_golds, res_masks, res_evts, doc_idx_list

def mat_to_pred(mat_preds, mat_golds, mat_masks, dic_evts, doc_idx_list, inds):
    """
    """

    np = len(inds)
    nl = mat_preds[0].shape[-1]

    res_preds = torch.zeros(np, nl, requires_grad=mat_preds[0].requires_grad, dtype=mat_preds[0].dtype, device=mat_preds[0].device).clone()
    res_golds = torch.zeros(np, nl, requires_grad=mat_golds[0].requires_grad, dtype=mat_golds[0].dtype, device=mat_golds[0].device).clone()
    res_masks = torch.zeros(np, nl, requires_grad=False, dtype=mat_golds[0].dtype, device=mat_golds[0].device).clone()

    for idx, (doc_idx, pair) in enumerate(inds):
        batch_idx = doc_idx_list.index(doc_idx)
        evt0_idx = dic_evts[batch_idx][pair[0]]
        evt1_idx = dic_evts[batch_idx][pair[1]]
        res_preds[idx] = mat_preds[batch_idx][evt0_idx, evt1_idx]
        res_golds[idx] = mat_golds[batch_idx][evt0_idx, evt1_idx]
        res_masks[idx] = mat_masks[batch_idx][evt0_idx, evt1_idx]

    return res_preds, res_golds, res_masks

def mat_interface(preds, golds, inds, config, train=False, label_gradient=False):
    """
    """

    print("Matrix Interface", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
    mpreds, mgolds, mmasks, devts, doc_idx = pred_to_mat(preds-config.DELTA_SIGMOID, golds, inds)

    if (train and config.TRAIN_OPTIMIZE) or ((not train) and config.TEST_OPTIMIZE):
        print("optimize", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
        b_rel, t_rel = config.b_rel, config.t_rel
        chan_preds = []
        for i, _ in enumerate(mpreds):
            chan_preds.append(optimize_with_gradient(mpreds[i], mgolds[i], b_rel, t_rel, label_gradient))
        
        print("Back to prediction", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
        res_preds, res_golds, res_masks = mat_to_pred(chan_preds, mgolds, mmasks, devts, doc_idx, inds)
    else:
        res_preds, res_golds, res_masks = mat_to_pred(mpreds, mgolds, mmasks, devts, doc_idx, inds)
        #print(res_preds)

    if train and config.INCO_LOSS_MAT:
        print("incoherence loss", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
        res_loss = mat_incoh_loss(mpreds, config)
    else:
        res_loss = torch.tensor(0.0)

    return res_preds+config.DELTA_SIGMOID, res_golds, res_masks, res_loss

#############################################################################################

def aggregate_rel(tok_set_annot, indices, logits, tlabels, config):
    """
    """
    doc_pair = dict()
    for idx, pair in enumerate(indices):
        if not pair[0] in doc_pair.keys():
            doc_pair[pair[0]] = dict()
        doc_pair[pair[0]][(pair[2], pair[3])] = idx
    preds = list()
    golds = list()
    inds = list()
    incon = 0
    for doc_idx, pred_pairs in doc_pair.items():
        for set_pair, list_label in tok_set_annot[doc_idx].items():
            label = torch.tensor(list_label)
            temp = list()
            for idx in set_pair[0]:
                for jdx in set_pair[1]:
                    temp.append(logits[pred_pairs[(idx, jdx)]])
                    if False in (tlabels[pred_pairs[(idx, jdx)]]==label):
                        #print(doc_idx, idx, jdx, tlabels[pred_pairs[(idx, jdx)]], label)
                        #print(set_pair, list_label)
                        incon += 1
            #print(temp)
            if config.AGGREG=='mean':
                preds.append(torch.mean(torch.stack(temp), dim=0))
            elif config.AGGREG=='max':
                preds.append(torch.max(torch.stack(temp), dim=0)[0])
            elif config.AGGREG=='lse':
                preds.append(torch.logsumexp(torch.stack(temp), dim=0))
            golds.append(label)
            inds.append((doc_idx, set_pair))
    return torch.stack(preds), torch.stack(golds), inds

#############################################################################################

def train(config, model, dataloader, tok_set_annot, optimizer, loss_fn, loss_fn_opt, loss_weights, epoch):
    """
    """
    train_preds = []
    train_golds = []
    train_idx = []
    train_inds = []
    epoch_loss = 0
    model.train()
    batch_idx=0
    for batch in dataloader:
        batch_idx+=1
        print(batch_idx, "batch prep", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        global_attention_mask = batch['global_attention_mask']
        if global_attention_mask is not None:
            global_attention_mask = global_attention_mask.to(config.DEVICE)
        pair_indices = batch['pair_indices']
        pair_labels = batch['pair_labels']
        doc_indices = batch['doc_indices']

        optimizer.zero_grad()

        print(batch_idx, "model pred", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
        logits, indices, tlabels = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            pair_indices=pair_indices,
            doc_indices=doc_indices,
            pair_labels=pair_labels
        )

        #temp_labels = [pair_labels[idx[1]][pair_indices[idx[1]].index((idx[2], idx[3]))] for idx in indices]
        #flat_pair_labels = [pair_label for doc_labels in pair_labels for pair_label in doc_labels]
        #print("Failed asserts due to double pairs : ", sum([1 if a!=b else 0 for a, b in zip(temp_labels,flat_pair_labels)]), "/", len(temp_labels))
        #labels = torch.tensor(flat_pair_labels).to(logits.device)
        #print(labels==tlabels)

        print(batch_idx, "aggregation", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
        agg_logits, golds, inds = aggregate_rel(tok_set_annot, indices, logits.to("cpu"), tlabels.to("cpu"), config)

        print(batch_idx, "loss", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
        #loss = loss_fn(logits.to(config.DEVICE), labels.float().to(config.DEVICE))
        loss = loss_fn(agg_logits.to(config.DEVICE), golds.float().to(config.DEVICE))

        if config.DISCRIM:
            print(batch_idx, "discrim", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
            # Select the correct loss weights
            lw = []
            for idx, (doc_idx, (evta, evtb)) in enumerate(inds):
                #print(loss_weights[doc_idx])
                #print(idx, doc_idx, evta, evtb)
                lw.append(loss_weights[doc_idx][(evta, evtb)])
            # Apply the loss weigths
            loss = loss * torch.tensor(lw).to(config.DEVICE) * config.DISCRIM_SCALE(epoch) + loss * (1 - config.DISCRIM_SCALE(epoch))
        
        if config.TRAIN_MATRIX_INTERFACE:
            opt_logits, opt_golds, opt_masks, res_loss = mat_interface(agg_logits, golds, inds, config, train=True, label_gradient=config.LABEL_GRADIENT)
            #loggit(config, str(torch.sum(agg_logits==opt_logits).detach().cpu()) + str(torch.sum(torch.abs(agg_logits-opt_logits)).detach().cpu()))

            if config.TRAIN_OPTIMIZE:
                print(batch_idx, "optimized logits loss", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
                loss_opt = loss_fn_opt(opt_logits.to(config.DEVICE), golds.float().to(config.DEVICE))
                loss = config.TRAIN_NON_OPTIMIZE_ALPHA(epoch) * loss + config.TRAIN_OPTIMIZE_ALPHA(epoch) * loss_opt

        loss = loss * torch.tensor(config.REL_TYPE_MASK).to(config.DEVICE)
        
        if config.REDUCTION != 'none':
            print(batch_idx, "reduction", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
            # Reduce the loss
            if config.REDUCTION == 'sum':
                loss = torch.sum(loss)
            elif config.REDUCTION == 'mean':
                loss = torch.mean(loss)
            else:
                raise ValueError(f"The Reduction mode: {config.REDUCTION} is not taken into account")

        if config.TRAIN_MATRIX_INTERFACE and config.INCO_LOSS_MAT:
            loss = loss + res_loss * config.INCO_LOSS_MAT_ALPHA(epoch)

        if config.INCO_LOSS:
            print(batch_idx, "incoherence loss", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
            loss = loss + config.INCO_LOSS_ALPHA(epoch) * incoherence_loss(agg_logits, golds, inds, config)

        loss_item = loss.item()
        epoch_loss += loss.item()

        print(batch_idx, "backprop", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
        loss.backward()
        optimizer.step()

        print(batch_idx, "saving preds", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
        preds = torch.sigmoid(opt_logits).detach() if config.TRAIN_OPTIMIZE else torch.sigmoid(agg_logits).detach()
        train_preds.append(preds.detach().to("cpu"))
        train_golds.append(golds.detach().to("cpu"))
        train_idx.append({'pair_indices': pair_indices, 'doc_indices': doc_indices, 'indices': indices})
        train_inds.append(inds)

        print("\n" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + ': '+"Batch number:", batch_idx, "| Loss:", loss_item)

        with open(config.TRAIN_LOG_FILE, "a") as f:
            f.write(f"Epoch {epoch} | batch {batch_idx} | Loss: {loss.item()}\n")

    loggit(config, f"The epoch loss at train is {epoch_loss}")

    print(batch_idx, "detach to cpu", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")

    loss.detach()
    logits.detach().cpu()
    tlabels.detach().cpu()
    golds.detach().cpu()
    input_ids.detach().cpu()
    attention_mask.detach().cpu()
    if global_attention_mask!=None:
        global_attention_mask.detach().cpu()
    optimizer.zero_grad()

    print(batch_idx, "train epoch done", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")

    tpreds = torch.cat(train_preds, dim=0)
    tgolds = torch.cat(train_golds, dim=0)
    npreds = tpreds[:, 1:].detach().cpu().numpy() if config.NOTHING_LABEL else tpreds.detach().to("cpu").numpy()
    ngolds = tgolds[:, 1:].detach().cpu().numpy() if config.NOTHING_LABEL else tgolds.detach().to("cpu").numpy()
    bpreds = (npreds > config.THRESHOLD).astype(int)

    target_names = config.LABEL_LIST[1:] if config.NOTHING_LABEL else config.LABEL_LIST
    # Generate the classification report
    train_report = classification_report(ngolds, bpreds, target_names=target_names, zero_division=config.ZERO_DIVISION, digits=config.REPORT_DIGITS)

    return train_report, train_preds, train_golds, train_idx, train_inds

def thresh_lin(floor, steps):
    """
    """
    bot = math.exp(-floor)
    alp = abs(1-bot)/steps
    for i in range(steps):
        yield bot + alp*i

def thresh_find(config, np_preds, np_golds):
    """
    """
    top = 0.0
    top_th = 0.0
    f1_average = 'micro'
    if len(config.REL_TYPE_IDX)==1:
        f1_average = 'binary'
    for thresh in itertools.chain(thresh_lin(config.THRESH_FLOOR, config.THRESH_STEPS), [0.6225]):
        binary_preds = (np_preds[:, config.REL_TYPE_IDX] > thresh).astype(int)
        score = f1_score(np_golds[:, config.REL_TYPE_IDX], binary_preds, average=f1_average)
        print(thresh, score)
        if score >= top:
            top_th = thresh
            top = score
    return top, thresh

def test(config, model, dataloader, tok_set_annot, loss_fn, loss_weights, epoch):
    """
    """
    #dataset.set_dataset('eval')
    # Initialize lists to collect all predictions and true labels
    all_preds = []
    all_golds = [] 
    eval_preds = []
    eval_golds = []
    eval_idx = []
    eval_inds = []
    epoch_loss = 0
    
    # Inference example
    model.eval()
    with torch.no_grad():
    
        batch_idx=0
        for batch in dataloader:
            batch_idx+=1
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            global_attention_mask = batch['global_attention_mask']
            if global_attention_mask is not None:
                global_attention_mask = global_attention_mask.to(config.DEVICE)
            pair_indices = batch['pair_indices']
            doc_indices = batch['doc_indices']
            #print(doc_indices)
            logits, indices, tlabels = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                pair_indices=pair_indices,
                doc_indices=doc_indices,
                pair_labels=[[[0 for i in range(config.NUM_LABELS)] for j in pairs] for pairs in pair_indices]  # Dummy labels
            )
    
            print(batch_idx, "aggregation", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
            agg_logits, golds, inds = aggregate_rel(tok_set_annot, indices, logits.to("cpu"), tlabels.to("cpu"), config)
    
            # print(batch_idx, "loss", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
            # #loss = loss_fn(logits.to(config.DEVICE), labels.float().to(config.DEVICE))
            # loss = loss_fn(agg_logits.to(config.DEVICE), golds.float().to(config.DEVICE))
    
            # if config.DISCRIM:
            #     print(batch_idx, "discrim", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
            #     # Select the correct loss weights
            #     lw = []
            #     for idx, (doc_idx, (evta, evtb)) in enumerate(inds):
            #         #print(loss_weights[doc_idx])
            #         #print(idx, doc_idx, evta, evtb)
            #         lw.append(loss_weights[doc_idx][(evta, evtb)])
            #     # Apply the loss weigths
            #     loss = loss * torch.tensor(lw).to(config.DEVICE)
            
            # if config.TEST_MATRIX_INTERFACE:
            #     print(batch_idx, "matrix interface", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
            #     opt_logits, opt_golds, opt_masks, res_loss = mat_interface(agg_logits, golds, inds, config, train=True)
            #     #loggit(config, str(torch.sum(agg_logits==opt_logits).detach().cpu()) + str(torch.sum(torch.abs(agg_logits-opt_logits)).detach().cpu()))
    
            #     if config.TEST_OPTIMIZE:
            #         print(batch_idx, "optimized logits loss", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
            #         loss_opt = loss_fn(opt_logits.to(config.DEVICE), golds.float().to(config.DEVICE))
            #         loss = config.TEST_NON_OPTIMIZE_ALPHA(epoch) * loss + config.TEST_OPTIMIZE_ALPHA(epoch) * loss_opt
            
            # if config.REDUCTION != 'none':
            #     print(batch_idx, "reduction", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
            #     # Reduce the loss
            #     if config.REDUCTION == 'sum':
            #         loss = torch.sum(loss)
            #     elif config.REDUCTION == 'mean':
            #         loss = torch.mean(loss)
            #     else:
            #         raise ValueError(f"The Reduction mode: {config.REDUCTION} is not taken into account")
    
            # if config.TEST_MATRIX_INTERFACE and config.INCO_LOSS_MAT:
            #     loss = loss + res_loss * config.INCO_LOSS_MAT_ALPHA(epoch)
    
            # if config.INCO_LOSS:
            #     print(batch_idx, "incoherence loss", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
            #     loss = loss + config.INCO_LOSS_ALPHA(epoch) * incoherence_loss(agg_logits, golds, inds, config)
    
            # loss_item = loss.item()
            # epoch_loss += loss.item()
    
            print(batch_idx, "detach to cpu", datetime.datetime.now().strftime('%H_%M_%S'), end=" § ")
            
            logits.detach().cpu()
    
            preds = torch.sigmoid(agg_logits).detach()
            preds_np = preds[:, 1:].detach().cpu().numpy() if config.NOTHING_LABEL else preds.detach().to("cpu").numpy()
            golds_np = golds[:, 1:].detach().cpu().numpy() if config.NOTHING_LABEL else golds.detach().to("cpu").numpy()
    
            # Collect the predictions and true labels
            all_preds.append(preds_np)
            all_golds.append(golds_np)
    
            eval_preds.append(preds.detach().to("cpu"))
            eval_golds.append(golds.detach().to("cpu"))
            eval_idx.append({'pair_indices': pair_indices, 'doc_indices': doc_indices, 'indices': indices})
            eval_inds.append(inds)

            # loss.detach().cpu()
            logits.detach().cpu()
            tlabels.detach().cpu()
            preds.detach().cpu()
            golds.detach().cpu()
            input_ids.detach().cpu()
            attention_mask.detach().cpu()
            # del loss, logits, tlabels, preds, golds, input_ids, attention_mask
            if global_attention_mask!=None:
                global_attention_mask.detach().cpu()
                # del global_attention_mask
            
            print(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+':', "Test Batch number:", batch_idx)#, "| Loss:", loss_item)

    
    # loggit(config, f"The epoch loss at test is {epoch_loss}")

    # Concatenate the collected predictions and true labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_golds = np.concatenate(all_golds, axis=0)
    
    # Convert probabilities to binary predictions
    threshold = config.THRESHOLD  # You may need to adjust this threshold
    binary_preds = (all_preds > threshold).astype(int)

    target_names = config.LABEL_LIST[1:] if config.NOTHING_LABEL else config.LABEL_LIST
    # Generate the classification report
    report = classification_report(all_golds, binary_preds, target_names=target_names, zero_division=config.ZERO_DIVISION, digits=config.REPORT_DIGITS)
    
    if config.BEST_MODEL:
        top, thresh = thresh_find(config, all_preds, all_golds)
        if top>config.top_micro_f1:
            config.top_micro_f1 = top
            config.top_threshold = thresh
            config.top_epoch = epoch
            with open(config.MODEL_SAVE, 'wb') as file:
                pickle.dump(model, file)
            loggit(config, f"Model Save Done with {top}")

    return report, eval_preds, eval_golds, eval_idx, eval_inds

#############################################################################################

def set_global_seed(config):
    """
    Set random seeds for reproducibility across multiple libraries.
    
    Args:
        seed (int): Seed value to use. Default is 42.
    """
    # Python's built-in random module
    random.seed(config.RANDOM_SEED)
    
    # Numpy
    np.random.seed(config.RANDOM_SEED)
    
    # PyTorch
    torch.manual_seed(config.RANDOM_SEED)
    torch.cuda.manual_seed(config.RANDOM_SEED)
    torch.cuda.manual_seed_all(config.RANDOM_SEED)  # for multi-GPU
    
    # Set a fixed environment seed
    os.environ['PYTHONHASHSEED'] = str(config.RANDOM_SEED)

    return

def loggit(config, *strings):
    """
    Log a message with timestamp to a file and print it.
    
    :param config: Configuration object with LOG_OVERALL attribute
    :param strings: Variable number of strings to be logged
    """
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + ': '
    message = timestamp + ' '.join(map(str, strings)) + '\n'
    
    with open(config.LOG_OVERALL, "a") as f:
        f.write(message)
    
    print(message)
    return

def log_config(config):
    """
    Logs configuration attributes to a JSON file.

    Converts configuration object attributes to a dictionary, 
    handling special types like datetime and torch.device.
    Writes the configuration to the specified log file.

    :param config: Configuration object to be logged
    :return: None
    """
    # Create a dictionary to hold the config attributes
    config_dict = {}
    for attribute in dir(config):
        # Filter out special and private attributes
        if not attribute.startswith('__') and not callable(getattr(config, attribute)):
            value = getattr(config, attribute)
            # Handle non-serializable attributes
            if isinstance(value, (datetime.datetime, datetime.date)):
                value = value.isoformat()
            elif isinstance(value, torch.device):
                value = str(value)
            # Add to the dictionary
            config_dict[attribute] = value
    # Write to the file
    with open(config.CONFIG_LOG_FILE, 'w') as f:
        json.dump(config_dict, f, indent=4)
    return

def save_current_file(config):
    """
    """
    # Get the path of the current file
    current_file_path = os.path.abspath(__file__)
    # Construct the backup file path
    backup_file_path = os.path.join(config.LOG_DIR, os.path.basename(current_file_path))
    # Make sure the backup directory exists
    os.makedirs(config.LOG_DIR, exist_ok=True)
    # Copy the current file to the backup location
    shutil.copy2(current_file_path, backup_file_path)
    loggit(config, f"File saved to: {config.LOG_DIR}")
    return

def data_save(todel):
    """
    """
    for obj in todel:
        try:
            obj.detach().cpu()
        except:
            pass
        del obj
    return

def check_gpu_tensors():
    tensors_on_gpu = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                tensors_on_gpu.append(obj)
        except:
            pass
    # After your epoch
    res = ""
    res += f"Number of tensors remaining on GPU: {len(tensors_on_gpu)}\n"
    #if tensors_on_gpu:
    #    res += "Tensors on GPU:"
    #    for tensor in tensors_on_gpu:
    #        res += f"{tensor.device, tensor.size()}\n"
    return res

#############################################################################################

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return F_loss

class AsymmetricLoss(nn.Module):
    """
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        loss = -loss

        return loss

def incoherence_loss(preds, golds, inds, config, thresh=0):
    """
    """
    res_loss = torch.tensor(0, requires_grad=preds[0].requires_grad, dtype=preds[0].dtype, device=preds[0].device)
    n_loss = 0

    nl = config.NUM_LABELS
    bfn = [(i, j) for i in range(len(config.b_rel)) for j in config.b_rel[i][0][1]]
    bbn = [(i, j) for i in range(len(config.b_rel)) for j in config.b_rel[i][1][1]]
    tft = [(i, j, k) for i in range(len(config.t_rel)) for (j, k) in config.t_rel[i][0]]

    doc_idx_list = list(set([i[0] for i in inds]))
    doc_idx_list.sort()
    #print(doc_idx_list)
    nd = len(doc_idx_list)

    preds_split, golds_split, inds_split = split_batch_in_docs(preds, golds, inds, doc_idx_list, nd)

    for batch_idx, _ in enumerate(golds_split):

        pred = preds_split[batch_idx]
        gold = golds_split[batch_idx]
        ind = inds_split[batch_idx]
        evts = list(set([j for i in inds for j in i[1]]))
        evts.sort(key=min)

        for idxa, (doc_idx, (evta, evtb)) in enumerate(ind):

            for i, j in bfn:
                if pred[idxa][i] > thresh and pred[idxa][j] > thresh:
                    res_loss = res_loss + torch.abs(pred[idxa][i]-thresh + pred[idxa][j]-thresh)
                    n_loss += 2

            if (doc_idx, (evta, evtb)) in ind:
                idxb = ind.index((doc_idx, (evta, evtb)))
                for i, j in bbn:
                    if pred[idxa][i] > thresh and pred[idxb][j] > thresh:
                        res_loss = res_loss + torch.abs(pred[idxa][i]-thresh + pred[idxb][j]-thresh)
                        n_loss += 2

            for idxb, (doc_idx, (evtc, evtd)) in enumerate(ind):
                if evtb!=evtc or (doc_idx, (evta, evtd)) not in enumerate(ind):
                    continue
                idxc = ind.index((doc_idx, (evta, evtd)))
                for i, j, k in tft:
                    if pred[idxa][i] > thresh and pred[idxb][j] > thresh and pred[idxc][k] < thresh:
                        res_loss = res_loss + torch.abs(pred[idxa][i]-thresh + pred[idxb][j]-thresh - (pred[idxc][k]-thresh))
                        n_loss += 3

    return res_loss

def mat_incoh_loss(preds, config):
    """
    """
    res_loss = torch.tensor(0, requires_grad=preds[0].requires_grad, dtype=preds[0].dtype, device=preds[0].device)
    n_loss = 0

    nl = config.NUM_LABELS
    bfn = [(i, j) for i in range(len(config.b_rel)) for j in config.b_rel[i][0][1]]
    bbn = [(i, j) for i in range(len(config.b_rel)) for j in config.b_rel[i][1][1]]
    tft = [(i, j, k) for i in range(len(config.t_rel)) for (j, k) in config.t_rel[i][0]]


    for batch_idx, _ in enumerate(preds):

        pred = preds[batch_idx]

        for i, j in bfn:
            res_loss = res_loss + torch.sum(F.relu(pred[:, :, i]) * F.relu(pred[:, :, j]))
        
        for i, j in bbn:
            res_loss = res_loss + torch.sum(F.relu(pred[:, :, i]) * F.relu(torch.t(pred[:, :, j])))
        
        for i, j, k in tft:
            res_loss = res_loss + torch.sum(F.relu(pred[:, :, i]).unsqueeze(2) * F.relu(pred[:, :, j]).unsqueeze(0) * F.relu(pred[:, :, k]).unsqueeze(1))

    return res_loss


#############################################################################################

def __main__():
    """
    """
    # Initializing Config and Logging
    config = Config()
    # Logging directory creation
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_config(config)
    loggit(config, "Config Done")
    save_current_file(config)
    loggit(config, "Code save Done")
    # Fix the random seed
    set_global_seed(config)

    loggit(config, f"Device is: {config.DEVICE}")

    if config.PROFILE:
        # Create a profiler
        profiler = cProfile.Profile()
        # Start profiling
        profiler.enable()
        loggit(config, "Profile Set Up Done")

    dataset = data_preped(hf_path=config.DATASET)
    loggit(config, "Dataset Interface Load Done")
    config.LABEL_LIST = dataset.ere_types
    print("\n", "aya", "config.LABEL_LIST:", config.LABEL_LIST, "\n")
    config.NUM_LABELS = len(config.LABEL_LIST)
    print("\n", "aya", "config.REL_TYPE_MASK, config.REL_TYPE_IDX:", config.REL_TYPE_MASK, config.REL_TYPE_IDX, "\n")
    if len(config.REL_TYPE_MASK)+len(config.REL_TYPE_IDX)==0:
        print("\n", "aya", "If neither rel type mask nor rel type idx list filled", "\n")
        config.REL_TYPE_MASK = [1.0]*config.NUM_LABELS
        config.REL_TYPE_IDX = list(range(config.NUM_LABELS))
        if 'NoRel' in config.LABEL_LIST:
            norel_idx = config.LABEL_LIST.index('NoRel')
            config.REL_TYPE_MASK[norel_idx] = 0.0
            config.REL_TYPE_IDX.pop(norel_idx)
            print("\n", "aya", "again: norel_idx, config.REL_TYPE_MASK, config.REL_TYPE_IDX", norel_idx, config.REL_TYPE_MASK, config.REL_TYPE_IDX)
    
    if not config.DATASET_LOAD:
        dataset.set_dataset(config.DATASET_TRAIN)
        train_documents, train_tok_set_annot, _, _ = data_prep(dataset, config, undsamp=config.UNDER_SAMPLE)
        train_ds = PairDataset(train_documents, config)
        loggit(config, "Dataset Train Prep Done")
        if config.CONSTRAINTS:
            raise NotImplementedError()
            if config.DATASET == 'maven':
                define_const_maven(config)
            elif config.DATASET == 'maven_em':
                define_const_maven_em(config)
            else:
                raise ValueError(f"Constraints not implemented for this dataset: {config.DATASET}.")
            loggit(config, "Constraints Set")
        if config.DISCRIM:
            raise NotImplementedError()
            train_loss_weights = [segment(tsa, config) for tsa in train_tok_set_annot]
            train_loss_weights = dedu_mask_to_loss_weight(train_loss_weights, config)
            loggit(config, "Dataset Train Fundamental Relations Mask Done")
        else:
            nl = config.NUM_LABELS
            train_loss_weights = [{pair: [1]*nl for pair in tsa} for tsa in train_tok_set_annot]
        #
        dataset.set_dataset(config.DATASET_TEST)
        test_documents, test_tok_set_annot, _, _ = data_prep(dataset, config, undsamp=False)
        test_ds = PairDataset(test_documents, config)
        loggit(config, "Dataset Test Prep Done")
        if config.DISCRIM:
            raise NotImplementedError()
            test_loss_weights = [segment(tsa, config) for tsa in test_tok_set_annot]
            test_loss_weights = dedu_mask_to_loss_weight(test_loss_weights, config)
            loggit(config, "Dataset Test Fundamental Relations Mask Done")
        else:
            nl = config.NUM_LABELS
            test_loss_weights = [{pair: [1]*nl for pair in tsa} for tsa in test_tok_set_annot]
    else:
        with open(config.DS_PATH_START+config.DATASET_TRAIN+"_train_dataset.pkl", 'rb') as file:
            train_ds = pickle.load(file)
        with open(config.DS_PATH_START+config.DATASET_TRAIN+"_train_tok_annot.pkl", 'rb') as file:
            train_tok_set_annot = pickle.load(file)
        if config.DISCRIM:
            raise NotImplementedError()
            with open(config.DS_PATH_START+config.DATASET_TRAIN+"_train_loss_weights.pkl", 'rb') as file:
                train_loss_weights = pickle.load(file)
            with open(config.DS_PATH_START+config.DATASET_TEST+"_test_loss_weights.pkl", 'rb') as file:
                test_loss_weights = pickle.load(file)
        with open(config.DS_PATH_START+config.DATASET_TEST+"_test_dataset.pkl", 'rb') as file:
            test_ds = pickle.load(file)
        with open(config.DS_PATH_START+config.DATASET_TEST+"_test_tok_annot.pkl", 'rb') as file:
            test_tok_set_annot = pickle.load(file)
        loggit(config, "Dataset Load Done")
        if config.CONSTRAINTS:
            raise NotImplementedError()
            if config.DATASET == 'maven':
                define_const_maven(config)
            elif config.DATASET == 'maven_em':
                define_const_maven_em(config)
            else:
                raise ValueError(f"Constraints not implemented for this dataset: {config.DATASET}.")
            loggit(config, "Constraints Set")
        if not config.DISCRIM:
            nl = config.NUM_LABELS
            train_loss_weights = [{pair: [1]*nl for pair in tsa} for tsa in train_tok_set_annot]
            test_loss_weights = [{pair: [1]*nl for pair in tsa} for tsa in test_tok_set_annot]

    if config.DATASET_SAVE:
        with open(config.DS_PATH_START+config.DATASET_TRAIN+"_train_dataset.pkl", 'wb') as file:
            pickle.dump(train_ds, file)
        with open(config.DS_PATH_START+config.DATASET_TRAIN+"_train_tok_annot.pkl", 'wb') as file:
            pickle.dump(train_tok_set_annot, file)
        if config.DISCRIM:
            raise NotImplementedError()
            with open(config.DS_PATH_START+config.DATASET_TRAIN+"_train_loss_weights.pkl", 'wb') as file:
                pickle.dump(train_loss_weights, file)
            with open(config.DS_PATH_START+config.DATASET_TEST+"_test_loss_weights.pkl", 'wb') as file:
                pickle.dump(test_loss_weights, file)
        with open(config.DS_PATH_START+config.DATASET_TEST+"_test_dataset.pkl", 'wb') as file:
            pickle.dump(test_ds, file)
        with open(config.DS_PATH_START+config.DATASET_TEST+"_test_tok_annot.pkl", 'wb') as file:
            pickle.dump(test_tok_set_annot, file)
        loggit(config, "Dataset Save Done")

    train_dataloader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=config.BATCH_SHUFFLE)
    test_dataloader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=config.BATCH_SHUFFLE)
    loggit(config, "Dataset Prep Done")

    # Model
    if config.LOAD_MODEL:
        with open(config.MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        model = LongformerPairClassifier(config)
    model.to(config.DEVICE)
    loggit(config, "Model Done")

    # Optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE
    )
    loggit(config, "Optimizer Done")

    # Loss function
    if config.LOSS=='foc':
        loss_fn = FocalLoss(alpha=config.ALPHA, gamma=config.GAMMA)
    elif config.LOSS=='asl':
        loss_fn = AsymmetricLoss(
            gamma_neg=config.GAMMA_NEG,
            gamma_pos=config.GAMMA_POS,
            clip=config.CLIP,
            eps=config.EPS,
            disable_torch_grad_focal_loss=config.DTGFL
        )
    elif config.LOSS=='bce':
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    else:
        raise ValueError(f"Invalid loss function '{config.LOSS}'. Valid options are: {', '.join(config.VALID_LOSSES)}")
    loggit(config, "Loss Function Done")

    # Loss function post optimisation
    if config.LOSS_OPT=='foc':
        loss_fn_opt = FocalLoss(alpha=config.ALPHA, gamma=config.GAMMA)
    elif config.LOSS_OPT=='asl':
        loss_fn_opt = AsymmetricLoss(
            gamma_neg=config.GAMMA_NEG,
            gamma_pos=config.GAMMA_POS,
            clip=config.CLIP,
            eps=config.EPS,
            disable_torch_grad_focal_loss=config.DTGFL
        )
    elif config.LOSS_OPT=='bce':
        loss_fn_opt = nn.BCEWithLogitsLoss(reduction='none')
    else:
        raise ValueError(f"Invalid loss function '{config.LOSS}'. Valid options are: {', '.join(config.VALID_LOSSES)}")
    loggit(config, "Loss Function Done")

    
    # Attempt at saving gpu memory
    data_save((train_ds, test_ds, dataset))
    if not config.DATASET_LOAD:
        data_save((train_documents, test_documents))
    loggit(config, "Garbage Reference Del Done")

    # Attempt at updating the config 
    try:
        log_config(config)
        loggit(config, "Update Config Done")
    except:
        loggit(config, "Update Config Failed. Moving on...")
    
    ### Main training and evaluation loop
    for epoch in range(config.START_EPOCH, config.NUM_EPOCHS):
        epoch_str = '{:0>2}'.format(str(epoch))

        train_report, train_preds, train_golds, train_idx, train_inds = train(config, model, train_dataloader, train_tok_set_annot, optimizer, loss_fn, loss_fn_opt, train_loss_weights, epoch)
        loggit(config, f"Train Epoch {epoch} Done")

        # Log the classification report to a file 
        with open(f'{config.LOG_DIR}train_classification_report_{epoch_str}.txt', 'w') as file:
            file.write(train_report)
        loggit(config, f"Log Repport Epoch {epoch} Done")
        if config.TRAIN_PRED_SAVE:
            # Save the predictions for each epoch
            with open(config.TRAIN_PREDS_LOG+epoch_str+".pkl", 'wb') as file:
                pickle.dump(train_preds, file)
            with open(config.TRAIN_GOLDS_LOG+epoch_str+".pkl", 'wb') as file:
                pickle.dump(train_golds, file)
            with open(config.TRAIN_IDX_LOG+epoch_str+".pkl", 'wb') as file:
                pickle.dump(train_idx, file) 
            with open(config.TRAIN_IND_LOG+epoch_str+".pkl", 'wb') as file:
                pickle.dump(train_inds, file) 
        del train_preds, train_golds, train_idx, train_inds, train_report
        loggit(config, f"Train Preds Golds and Idx Save Done")

        # del optimizer
        # torch.cuda.empty_cache()

        report, eval_preds, eval_golds, eval_idx, eval_inds = test(config, model, test_dataloader, test_tok_set_annot, loss_fn, test_loss_weights, epoch)
        loggit(config, f"Test Epoch {epoch} Done")

        # Log the classification report to a file 
        with open(f'{config.LOG_DIR}classification_report_{epoch_str}.txt', 'w') as file:
            file.write(report)
        loggit(config, f"Log Repport Epoch {epoch} Done")
        if config.TEST_PRED_SAVE:
            # Save the predictions for each epoch
            with open(config.PREDS_LOG+epoch_str+".pkl", 'wb') as file:
                pickle.dump(eval_preds, file)
            with open(config.GOLDS_LOG+epoch_str+".pkl", 'wb') as file:
                pickle.dump(eval_golds, file)
            with open(config.IDX_LOG+epoch_str+".pkl", 'wb') as file:
                pickle.dump(eval_idx, file) 
            with open(config.IND_LOG+epoch_str+".pkl", 'wb') as file:
                pickle.dump(eval_inds, file) 
        del eval_preds, eval_golds, eval_idx, eval_inds, report	
        loggit(config, f"Eval Preds Golds and Idx Save Done")

        loggit(config, check_gpu_tensors())
        # torch.cuda.empty_cache()
        loggit(config, check_gpu_tensors())
    
    if not config.BEST_MODEL:
        with open(config.MODEL_SAVE, 'wb') as file:
            pickle.dump(model, file)
        loggit(config, f"Model Save Done")
    
    if config.PROFILE:
        # Run your code here
        profiler.disable()
        # Print profiling stats
        stats = pstats.Stats(profiler)
        loggit(config, stats.sort_stats(SortKey.TIME, SortKey.CUMULATIVE).print_stats(config.LINE_PROFILE))
        loggit(config, "Profile Finalized Done")
        stats.dump_stats(config.LOG_PROFILE)
    return


if __name__ == "__main__":
    __main__()
