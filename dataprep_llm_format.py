
import argparse
from datasets import load_dataset


MECI_ERE_PROMPT = """"""


def rel_none_to_list(relations):
    for rel_type, rel_list in relations.items():
        if not rel_list:
            relations[rel_type] = []
    return relations

def rel_parse_dict(relations, spans, text):
    for rel_type in relations:
        annots += f"{rel_type}: "
        if not relations[rel_type]:
            annots += "None; "
            continue
        for rel in relations[rel_type]:
            annots += " ".join([text[i] for i in spans[rel[0]]])
            annots += " - "
            annots += " ".join([text[i] for i in spans[rel[1]]])
            annots += ", "
        annots = annots[:-2] + "; "
    return annots

def rel_parse_triplets(relations, spans, text):
    annots = ""
    for rel_type in relations:
        if not relations[rel_type]:
            continue
        for rel in relations[rel_type]:
            annots += " ".join([text[i] for i in spans[rel[0]]])
            annots += f" {rel_type} "
            annots += " ".join([text[i] for i in spans[rel[1]]])
            annots += ", "
    return annots

def text_parse(sample):
    tokens, spans, mentions, relations = sample['tokens'], sample['spans'], sample['mentions'], sample['relations']
    text = tokens.copy()
    annots = ""

    for span_idx, span in enumerate(spans):
        for i in span:
            if i-1 not in span:
                text[i] = f"<{str(mentions[span_idx])} {text[i]}"
            if i+1 not in span:
                text[i] = f"{text[i]}>"
    
    # annots = rel_parse_dict(relations, spans, text)
    annots = rel_parse_triplets(relations, spans, text)
    text = " ".join(text)
    relations = rel_none_to_list(relations)

    return {'text': text, 'annots': annots, 'relations': relations}

def main():
    parser = argparse.ArgumentParser(description="Add \"text\" and \"annots\" to the dataset.")
    parser.add_argument("--dataset", default="Nofing/EventStoryLine-1.5-Causal", help="Dataset name")
    # parser.add_argument("--dict_style", action="store_true", help="Use the dictionary style annotation instead of the triplets")
    args = parser.parse_args()
    
    dataset = load_dataset(args.dataset)

    new_dataset = dataset.map(text_parse)

    new_dataset.push_to_hub(args.dataset)

    dataset = load_dataset(args.dataset)
    print(dataset['train'][0])

    return

if __name__ == "__main__":
    main()
