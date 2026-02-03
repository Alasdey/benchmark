
import argparse
from datasets import load_dataset

def regularize(dataset):
    return

def find_symmetry(dataset, rel_type_1, rel_type_2):
    for split in dataset:
        for doc in dataset[split]:
            for rel_type in doc['relations']:
    return


def cli():
    parser = argparse.ArgumentParser(description="Chat with WikiExplorer")
    parser.add_argument("--dataset", default="Nofing/EventStoryLine-1.5-span", type="string", help="User message")
    args = parser.parse_args()
    
    dataset = load_dataset(args.dataset)

    return

if __name__ == "__main__":
    cli()
