import re


RELATIONS = ["CAUSE", "PRECONDITION", "coreference", "subevent", "BEFORE", "BEGINS-ON", "CONTAINS", "ENDS-ON", "OVERLAP", "SIMULTANEOUS"]

def compute_f1(pred_list, gold_list, default_return=-1):
    #print(pred_list, gold_list, "\n")
    npred = sum([len(set(pred)) for pred in pred_list])
    ngold = sum([len(set(gold)) for gold in gold_list])
    if npred==0 and ngold==0:
        return default_return

    tp, fp, fn = 0, 0, 0
    for pred, gold in zip(pred_list, gold_list):
        tp += len(set(pred) & set(gold))
        fp += len(set(pred) - set(gold))
        fn += len(set(gold) - set(pred))

    # Compute precision, recall, and F1 score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1#, precision, recall

def rel_extract(text):
    """Extract best as posible the relations in the text"""
    pref = r""
    #pref = r"<answer>.*?"
    flag_pattern = r":((?:\s*?[et][0-9]{1,3})*|\s*?none);?"#.*?</answer>$"
    # relations = ["SIMULTANEOUS", "ENDS-ON", "BEGINS-ON", "OVERLAP", "CONTAINS", "BEFORE"]
    fps = [pref+r+flag_pattern for r in RELATIONS]
    res = []
    for pattern in fps:
        res.append([])
        rel_seg = re.findall(pattern, text)
        for segment in rel_seg:
            res[-1] = re.findall(r"[et][0-9]+", segment)
    #res = [re.findall(r"<[et][0-9]+ [^><]*?>", sec) for pattern in fps for sec in re.findall(pattern, text)]
    return res

def accuracy_reward(pred, gold):
    gold = rel_extract(gold)
    pred = rel_extract(pred)

    reward = compute_f1(pred, gold)
    return reward