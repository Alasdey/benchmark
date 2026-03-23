
import argparse  
import torch
import os
import numpy
import pickle
import math
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score, confusion_matrix


def thresh_log(floor, steps):
    """
    """
    for i in range(-steps, 1):
        thresh = math.exp(floor*i/steps)
        yield thresh

def thresh_lin(floor, steps):
    """
    """
    bot = math.exp(-floor)
    alp = abs(1-bot)/steps
    for i in range(steps):
        yield bot + alp*i


def thresh_find(eval_golds, eval_preds, path_pref, pref, suf, uniform=True, dataset='Nofing/Hievents-span', steps=100, floor=3, tfn=thresh_lin, delta=0.0):
    """
    """
    if dataset == 'Nofing/MAVEN-ERE-Causal-Events':
        LABEL_LIST = ['CAUSE', 'PRECONDITION']
        LABEL_GROUPS = {"CAUSAL": ["CAUSE", "PRECONDITION"]}
    if dataset == 'maven_em':
        LABEL_LIST = ['BEFORE', 'OVERLAP', 'CONTAINS', 'SIMULTANEOUS', 'ENDS-ON', 'BEGINS-ON', 'CAUSE', 'PRECONDITION', 'subevent', 'coreference']
        LABEL_GROUPS = {"TEMPORAL": ["BEFORE", "OVERLAP", "CONTAINS", "SIMULTANEOUS", "BEGINS-ON", "ENDS-ON"], "CAUSAL": ["CAUSE", "PRECONDITION"]}
    if dataset == 'maven':
        LABEL_LIST = ["BEFORE", "OVERLAP", "CONTAINS", "SIMULTANEOUS", "BEGINS-ON", "ENDS-ON", "CAUSE", "PRECONDITION", "SUBEVENT"]
        LABEL_GROUPS = {"TEMPORAL": ["BEFORE", "OVERLAP", "CONTAINS", "SIMULTANEOUS", "BEGINS-ON", "ENDS-ON"], "CAUSAL": ["CAUSE", "PRECONDITION"]}
    if dataset == 'matres':
        LABEL_LIST = ['AFTER', 'BEFORE', 'VAGUE', 'EQUAL']
    if dataset == 'Nofing/Hievents-span':
        LABEL_LIST = ['Coref', 'SubSuper', 'SuperSub']
        LABEL_GROUPS = {'rel': ['SubSuper', 'SuperSub']}
    if dataset == 'Nofing/MECI-v0.1-public-span':
        LABEL_LIST = ['CauseEffect', 'EffectCause', 'NoRel']
        LABEL_GROUPS = {'rel': ['CauseEffect', 'EffectCause']}
    
    # mask = [0, 1, 2, 3, 4, 5]
    # mask = [6, 7]
    # mask = [8]
    # mask = [9]
    # mask = [0, 1, 2]
    mask = [0, 1]
    # mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    print(pref+'classification_report'+suf)
    a = torch.stack([ep for eps in eval_preds for ep in eps])
    b = torch.stack([ep for eps in eval_golds for ep in eps])
    n_labels = b.size(dim=1)
    #start = 1 if n_labels > 9 else 0
    start = 0
    np_preds = a[:, start:].to("cpu").numpy()
    np_golds = b[:, start:].to("cpu").numpy()
    top = 0.0
    top_th = 0.0

    if len(mask)==1:
        f1_average = 'binary'
    else:
        f1_average = 'micro'

    if uniform: # Unique threshold for everyone
        for thresh in tfn(floor, steps):
            binary_preds = (np_preds[:, mask] + delta > thresh).astype(int)
            score = f1_score(np_golds[:, mask], binary_preds, average=f1_average)
            print(thresh, score)
            if score >= top:
                top_th = thresh
                top = score
    else: # A threshold each
        top = numpy.array([0.0]*n_labels)
        top_th = numpy.array([0.0]*n_labels)
        for i in range(n_labels):
            for thresh in tfn(floor, steps):
                binary_preds = (np_preds[:, i] > thresh).astype(int)
                tn, fp, fn, tp = confusion_matrix(np_golds[:, i], binary_preds).ravel()
                score = tp/(fp+fn)
                if score >= top[i]:
                    top_th[i] = thresh
                    top[i] = score
            binary_preds = (np_preds[:, i] > top_th[i]).astype(int)
            score = f1_score(np_golds[:, i], binary_preds)
            print(LABEL_LIST[i], top_th[i], score)
    binary_preds = (np_preds + delta > top_th).astype(int)
    MASKED_LABEL_LIST = [LABEL_LIST[:][i] for i in mask]
    
    if len(mask)==1:
        MASKED_LABEL_LIST = ["NONE"] + MASKED_LABEL_LIST
    
    report = classification_report(np_golds[:, mask], binary_preds[:, mask], target_names=MASKED_LABEL_LIST, zero_division=0.0, digits=4)
    dict_report = classification_report(np_golds[:, mask], binary_preds[:, mask], target_names=MASKED_LABEL_LIST, zero_division=0.0, digits=4, output_dict=True)
    dict_report["Threshold"] = top_th

    # Map labels to their indices
    label_to_index = {label: idx for idx, label in enumerate(LABEL_LIST)}
    # Calculate micro and macro F1 scores for each label group
    for group, members in LABEL_GROUPS.items():
        indices = [label_to_index[label] for label in members]
        group_golds = np_golds[:, indices]
        group_preds = binary_preds[:, indices]

        micro_f1 = f1_score(group_golds, group_preds, average='micro', zero_division=0.0)
        macro_f1 = f1_score(group_golds, group_preds, average='macro', zero_division=0.0)
        precision = precision_score(group_golds, group_preds, average='micro', zero_division=0.0)
        recall = recall_score(group_golds, group_preds, average='micro', zero_division=0.0)

        dict_report[group] = {"precision": precision, "recall": recall, "micro_f1": micro_f1, "macro_f1": macro_f1}
    
    print(report)
    print(path_pref+pref+'classification_report'+suf)
    with open(path_pref+pref+'classification_report'+suf, 'w') as file:
        file.write(report)
        file.write('\n'+str(top_th))
    with open(path_pref+'dict_'+pref+'classification_report'+suf, 'w') as file:
        file.write(str(dict_report))
    return

def directories(path='./logs'):
    """
    """
    direct = next(os.walk(path))[1]
    print("Before filter:", direct)
    for name in direct:
        if name[:3]!='202':
            direct.pop(direct.index(name))
    direct.sort()
    print("After filter:", direct)
    return direct

def main(args, uniform=True, path='./logs', dataset='maven_em'):  
    """
    """
    # Convert list of string arguments to list of integers  
    dir_idx = [int(arg) for arg in args]
    dir_list = directories(path)
    dirs = [dir_list[-i] for i in dir_idx]
    for log_dir in dirs:
        # try:
        print(log_dir)
        for name_pref in ['eval_', 'train_']:
            num_iter = list(range(20))
            for num_list in [[''], ['{:0>2}'.format(str(x)) for x in num_iter], [str(i) for i in num_iter]]:
                for num_str in num_list:
                    got = False
                    try:
                        with open(path+log_dir+'/'+name_pref+'golds'+num_str+'.pkl', 'rb') as file:
                            eval_golds = pickle.load(file)
                        with open(path+log_dir+'/'+name_pref+'preds'+num_str+'.pkl', 'rb') as file:
                            eval_preds = pickle.load(file)
                        path_pref = path+log_dir+'/'
                        pref = 'adjusted_'+name_pref
                        suf = '_'+num_str+'.txt' if len(num_str)>0 else '.txt'
                        got = True
                    except:
                        #print(path+log_dir+'/'+name_pref+'golds'+num_str+'.pkl')
                        continue
                    if got:
                        thresh_find(eval_golds, eval_preds, path_pref, pref, suf, uniform=uniform, dataset=dataset)
        # except Exception as e:
        #     print(e)
        #     continue
    return


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Process a list of integers.")  
    parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the list')  
    args = parser.parse_args()  
    # 'Nofing/MECI-v0.1-public-span' 'Nofing/Hievents-span'
    dataset_list = [
        'Nofing/MAVEN-ERE-Causal-Events', 
        'maven_em', 
        'maven', 
        'matres', 
        'Nofing/Hievents-span', 
        'Nofing/MECI-v0.1-public-span'
    ]
    dataset = 'Nofing/MECI-v0.1-public-span'
    main(args.integers, uniform=True, path='./encoder_baseline/logs/MatrixIEO/', dataset=dataset)
    #main(args.integers, path='./logs/')
