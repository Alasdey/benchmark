
from datasets import load_dataset
from vllm import LLM

def main(args):
    """
    """
    ds = load_dataset(args.dataset)
    model = LLM(model=args.model)
    return

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a model from HF with a dataset from HF on ERE task with xml format"
    )
    p.add_argument("--full_annot", type=bool, default=False,
                   help="Full annotations: missing mention pairs==no_rel, otherwise only gold mention pairs evaluated\
                     (if a pair has at least a relation all missing relations are considered false in annotations)")
    p.add_argument("--HF_token", type=bool, default=False,
                   help="HF token required")
    p.add_argument("--dataset", type=str, default="Nofing/Hievents-span",
                   help="Name of the HF dataset")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                   help="Name of the HF model")
    p.add_argument("--split_name", type=str, default='test',
                   help="Name of the split to use for evaluation")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)