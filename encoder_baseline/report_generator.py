import os
import json
import pickle
import datetime
import argparse
import platform
import sys
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report
from torch.utils.data import DataLoader

# Import your existing classes/functions needed for Model loading
# adjusting the import based on your actual file name
from MatrixIEOmask import Config, LongformerPairClassifier, PairDataset, collate_fn, data_prep, aggregate_rel
from datasets_interface.old_format import data_preped

class ResultsReporter:
    def __init__(self, config, run_id=None):
        self.config = config
        self.run_id = run_id if run_id else datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
    def _get_git_info(self):
        # Basic check if git is available (optional)
        try:
            import subprocess
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode('utf-8')
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
            return {"is_git_repo": True, "branch": branch, "commit": commit}
        except:
            return {"is_git_repo": False}

    def _get_runtime_info(self):
        return {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "cwd": os.getcwd()
        }

    def _safe_div(self, n, d):
        return n / d if d > 0 else 0.0

    def compute_metrics(self, preds_binary, golds_binary, label_list):
        # preds_binary: (N_samples, N_classes)
        # golds_binary: (N_samples, N_classes)
        
        # Remove "Nothing/NoRel" from meta-metrics calculations if necessary
        # usually class 0 is NoRel/Nothing.
        target_indices = [i for i, label in enumerate(label_list) if label not in ['NoRel', 'NOTHING']]
        target_names = [label_list[i] for i in target_indices]
        
        # 1. Micro/Macro Metrics
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            golds_binary[:, target_indices], 
            preds_binary[:, target_indices], 
            average='micro', 
            zero_division=0
        )
        
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            golds_binary[:, target_indices], 
            preds_binary[:, target_indices], 
            average='macro', 
            zero_division=0
        )

        # 2. Binary Metrics (Any Relation vs No Relation)
        # Assuming 0 is NoRel, if any index > 0 is 1, then binary is 1
        # Or simpler: Is the row vector equal to exact match of NoRel?
        # Let's assume standard binary definition: Positive class is any non-NoRel class
        
        # Flattening logic for binary:
        # If any significant relation exists in the vector space (taking max or specific indices)
        # Here we assume multi-label. If any target index is 1, it's a positive binary instance.
        bin_golds = np.max(golds_binary[:, target_indices], axis=1)
        bin_preds = np.max(preds_binary[:, target_indices], axis=1)
        
        bp, br, bf1, _ = precision_recall_fscore_support(bin_golds, bin_preds, average='binary', zero_division=0)
        support_pos = int(np.sum(bin_golds))

        # 3. Per Label Metrics
        per_label = {}
        # Calculate for ALL labels including NoRel for completeness
        p_all, r_all, f1_all, s_all = precision_recall_fscore_support(
            golds_binary, 
            preds_binary, 
            average=None, 
            zero_division=0
        )
        
        for i, label in enumerate(label_list):
            per_label[label] = {
                "precision": float(p_all[i]),
                "recall": float(r_all[i]),
                "f1": float(f1_all[i]),
                "support": int(s_all[i])
            }

        return {
            "micro_precision": float(p_micro),
            "micro_recall": float(r_micro),
            "micro_f1": float(f1_micro),
            "macro_f1": float(f1_macro),
            "binary": {
                "precision": float(bp),
                "recall": float(br),
                "f1": float(bf1),
                "support_pos": support_pos
            },
            "per_label": per_label
        }

    def generate_report(self, preds, golds, indices, inds_map, dataset_inter):
        """
        preds: (Num_Pairs, Num_Classes) - probability or logits (sigmoid applied later if logic requires, inputs here assumed probabilities)
        golds: (Num_Pairs, Num_Classes) - one-hot/multi-hot
        indices: List of objects saved in log (optional, used for reconstruction)
        inds_map: List of (doc_idx, set_pair) tuples corresponding to the rows in preds/golds
        """
        
        # Convert to numpy and Binary Threshold
        if isinstance(preds, torch.Tensor): preds = preds.detach().cpu().numpy()
        if isinstance(golds, torch.Tensor): golds = golds.detach().cpu().numpy()
        
        # Handle NOTHING label offset if necessary
        # In your main.py, you often slice [:, 1:] if NOTHING_LABEL is True.
        # We need to know the mapping.
        
        if self.config.NOTHING_LABEL:
            actual_labels = self.config.LABEL_LIST[1:] # remove NOTHING
            preds_sliced = preds[:, 1:]
            golds_sliced = golds[:, 1:]
        else:
            actual_labels = self.config.LABEL_LIST
            preds_sliced = preds
            golds_sliced = golds

        preds_binary = (preds_sliced > self.config.THRESHOLD).astype(int)
        golds_binary = golds_sliced.astype(int)

        # Basic aggregated metrics
        results_structure = self.compute_metrics(preds_binary, golds_binary, actual_labels)
        
        results_structure["total_pairs"] = len(preds)
        
        # Per Document & Per Pair Analysis
        per_doc_metrics = {} # intermediate storage
        per_pair_predictions = []

        # We assume inds_map aligns perfectly with row 0..N of preds
        # inds_map element: (doc_idx, ((idx1, idx2), (jdx1, jdx2))) roughly
        # based on aggregate_rel: inds.append((doc_idx, set_pair))
        
        for row_idx, (doc_idx, set_pair) in enumerate(inds_map):
            # Identify Document
            # We try to get doc ID from dataset interface if possible, else doc_idx
            try:
                # Assuming standard interface structure from your code
                doc_id = dataset_inter.data[doc_idx].doc_id 
            except:
                doc_id = str(doc_idx)

            # Get Pair Labels (Predicted vs Gold)
            row_p = preds_binary[row_idx]
            row_g = golds_binary[row_idx]
            
            p_labels = [actual_labels[i] for i, val in enumerate(row_p) if val == 1]
            g_labels = [actual_labels[i] for i, val in enumerate(row_g) if val == 1]
            
            if not p_labels: p_labels = ["NoRel"] # Or NOTHING
            if not g_labels: g_labels = ["NoRel"]

            # Store Pair Prediction
            # Assuming set_pair is something printable like frozenset of indices
            # converting to string for JSON serialization
            pair_str = f"{set_pair}" 
            
            pair_entry = {
                "doc_idx": doc_idx,
                "id": doc_id,
                "pair": pair_str,
                "gold": g_labels,
                "pred": p_labels,
                "scores": {l: float(preds_sliced[row_idx][i]) for i, l in enumerate(actual_labels)}
            }
            per_pair_predictions.append(pair_entry)

            # Accumulate for per-doc metrics
            if doc_id not in per_doc_metrics:
                per_doc_metrics[doc_id] = {"p": [], "g": []}
            per_doc_metrics[doc_id]["p"].append(row_p)
            per_doc_metrics[doc_id]["g"].append(row_g)

        # Calculate Per-Doc Metrics
        doc_metrics_list = []
        for doc_id, data in per_doc_metrics.items():
            dp = np.array(data["p"])
            dg = np.array(data["g"])
            
            dm = self.compute_metrics(dp, dg, actual_labels)
            
            # Simplify for report density
            simple_dm = {
                "id": doc_id,
                "pairs": len(dp),
                "micro_f1": dm["micro_f1"],
                "macro_f1": dm["macro_f1"],
                "binary_f1": dm["binary"]["f1"]
            }
            doc_metrics_list.append(simple_dm)

        results_structure["per_doc_metrics"] = doc_metrics_list
        results_structure["per_pair_predictions"] = per_pair_predictions # Warning: This can be huge

        # Assemble Final JSON
        final_report = {
            "run_id": self.run_id,
            "timestamp_utc": self.timestamp,
            "git": self._get_git_info(),
            "runtime": self._get_runtime_info(),
            "config": {
                "dataset": self.config.DATASET,
                "model": self.config.MODEL_NAME,
                "loss": self.config.LOSS,
                "threshold": self.config.THRESHOLD
            },
            "results": results_structure
        }
        
        return final_report

def run_from_pickle(config, path_preds, path_golds, path_inds):
    """ Load pickle files and generate report """
    print(f"Loading pickles from: {path_preds}")
    with open(path_preds, 'rb') as f:
        # These are usually lists of tensors, logic may strictly depend on how train() saved them
        # In main.py: preds are appended layers of sigmoid outputs
        raw_preds = pickle.load(f)
    with open(path_golds, 'rb') as f:
        raw_golds = pickle.load(f)
    with open(path_inds, 'rb') as f:
        # inds is list of lists of tuples
        raw_inds = pickle.load(f)

    # Flatten logic (similar to main.py test() concatenation)
    # pickle structure in main.py: List[Tensor(batch_size, n_labels)]
    
    if isinstance(raw_preds, list):
        preds_all = torch.cat(raw_preds, dim=0) if isinstance(raw_preds[0], torch.Tensor) else torch.tensor(np.concatenate(raw_preds))
    else:
        preds_all = raw_preds

    if isinstance(raw_golds, list):
        golds_all = torch.cat(raw_golds, dim=0) if isinstance(raw_golds[0], torch.Tensor) else torch.tensor(np.concatenate(raw_golds))
    else:
        golds_all = raw_golds

    # Flatten inds (triply nested list in main.py: batch -> doc -> pair)
    # aggregate_rel returns `inds` which is `List[Tuple(doc_idx, set_pair)]`
    # The train loop appends this list to `train_inds`. So `train_inds` is List[List[Tuple]].
    inds_flat = [item for sublist in raw_inds for item in sublist]

    return preds_all, golds_all, inds_flat

def run_inference(config, model_path, dataset_split):
    """ Load model, run test loop, generate tensors """
    print(f"Loading Model: {model_path} on {dataset_split}")
    
    # 1. Load Data
    dataset = data_preped(hf_path=config.DATASET)
    dataset.set_dataset(dataset_split)
    
    # Re-use main.py logic heavily
    documents, tok_set_annot, _, _ = data_prep(dataset, config, undsamp=False)
    ds = PairDataset(documents, config)
    dataloader = DataLoader(ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
    
    # 2. Load Model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    model.eval()
    model.to(config.DEVICE)

    # 3. Inference Loop
    all_preds_list = []
    all_golds_list = []
    all_inds_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0: print(f"Processing batch {batch_idx}...")
            
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            global_attention_mask = batch['global_attention_mask']
            if global_attention_mask is not None:
                global_attention_mask = global_attention_mask.to(config.DEVICE)
            pair_indices = batch['pair_indices']
            doc_indices = batch['doc_indices']

            logits, indices, tlabels = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                pair_indices=pair_indices,
                doc_indices=doc_indices,
                pair_labels=[[[0]*config.NUM_LABELS]*len(p) for p in pair_indices] # Dummy
            )

            agg_logits, golds, inds = aggregate_rel(tok_set_annot, indices, logits.to("cpu"), tlabels.to("cpu"), config)
            
            # Apply Sigmoid here as in main.py
            preds = torch.sigmoid(agg_logits)
            
            all_preds_list.append(preds)
            all_golds_list.append(golds)
            all_inds_list.extend(inds)

    final_preds = torch.cat(all_preds_list, dim=0)
    final_golds = torch.cat(all_golds_list, dim=0)
    
    return final_preds, final_golds, all_inds_list, dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pickle", "model"], required=True, help="Load from 'pickle' or run 'model'")
    parser.add_argument("--config_path", type=str, default=None, help="Path to a pickled config object if available, otherwise instantiates default")
    
    # Pickle Mode Args
    parser.add_argument("--preds_pkl", type=str, help="Path to preds.pkl")
    parser.add_argument("--golds_pkl", type=str, help="Path to golds.pkl")
    parser.add_argument("--inds_pkl", type=str, help="Path to inds.pkl")
    
    # Model Mode Args
    parser.add_argument("--model_path", type=str, help="Path to model.pkl")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    
    parser.add_argument("--out_file", type=str, default="evaluation_report.json")
    
    args = parser.parse_args()

    # Initialize Config
    # If you saved Config as a pickle (recommended in main.py but not explicitly done in global scope), load it.
    # Otherwise, create new instance.
    config = Config() 
    
    # IMPORTANT: Ensure Label List matches what the model was trained on
    # We load the dataset object just to get the label list if needed
    tmp_dataset = data_preped(hf_path=config.DATASET)
    config.LABEL_LIST = tmp_dataset.ere_types
    if config.NOTHING_LABEL:
        config.LABEL_LIST = ['NOTHING'] + config.LABEL_LIST
    
    dataset_inter = tmp_dataset

    if args.mode == 'pickle':
        if not (args.preds_pkl and args.golds_pkl and args.inds_pkl):
            print("Pickle mode requires --preds_pkl, --golds_pkl, and --inds_pkl")
            sys.exit(1)
        preds, golds, inds = run_from_pickle(config, args.preds_pkl, args.golds_pkl, args.inds_pkl)
        
    elif args.mode == 'model':
        if not args.model_path:
            print("Model mode requires --model_path")
            sys.exit(1)
        preds, golds, inds, dataset_inter = run_inference(config, args.model_path, args.split)

    # Generate Report
    reporter = ResultsReporter(config)
    report_json = reporter.generate_report(preds, golds, None, inds, dataset_inter)

    # Save
    with open(args.out_file, 'w') as f:
        json.dump(report_json, f, indent=4)
        print(f"Report saved to {args.out_file}")