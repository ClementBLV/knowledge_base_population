import preprocess
import pandas as pd
import nltk


import argparse
import json
import pathlib


parser = argparse.ArgumentParser(description="preprocess")
parser.add_argument(
    "--task", 
    default="wn18rr", 
    type=str, 
    metavar="N", 
    help="dataset name"
)
parser.add_argument(
    "--workers", 
    default=2, 
    type=int, 
    metavar="N", 
    help="number of workers"
)
parser.add_argument(
    "--train-path",
    default="./data/WN18RR/train.txt",
    type=str,
    metavar="N",
    help="path to training data",
)
parser.add_argument(
    "--valid-path",
    default="./data/WN18RR/valid.txt",
    type=str,
    metavar="N",
    help="path to valid data",
)
parser.add_argument(
    "--test-path",
    default="./data/WN18RR/test.txt",
    type=str,
    metavar="N",
    help="path to test data",
)

args = parser.parse_args()


### Global fonction
def generate_nli_data(path: str, dataset: str = args.task):
    assert (
        type(dataset) == type("str")
    ), "Must presise the dataset either 'wordnet', 'wn', 'wn18rr' or 'freebase', 'fb', 'fb15k237' in a string"
    
    if dataset.lower() in ["wordnet", "wn", "wn18rr"]:
        nltk.download("wordnet")
        from nltk.corpus import wordnet as wn
        return generate_nli_data_wordnet(path)
    
    if dataset.lower() in ["freebase", "fb", "fb15k237"]:
        return generate_nli_data_freebase(path)


### case of wordnet
def generate_nli_data_wordnet(path):
    head2tail_dict = preprocess.preprocess_wn18rr(path)
    head2tail_df = pd.DataFrame(head2tail_dict)
    hypos2premises = []
    for index in range(0, len(head2tail_df)):
        triplet = head2tail_df.iloc[index]
        hypo2premise = {}
        # get the synset object from NLTK using the id to have more info like lemmas
        s_head = get_synset(triplet["head"], triplet["head_id"])
        s_tail = get_synset(triplet["tail"], triplet["tail_id"])
        # process the retreived lemmas
        l_head = " or ".join(
            [lemma.name().replace("_", " ").strip() for lemma in s_head.lemmas()][0:2]
        )
        l_tail = " or ".join(
            [lemma.name().replace("_", " ").strip() for lemma in s_tail.lemmas()][0:2]
        )

        # dictionnary contruction
        hypo2premise["head_id"] = triplet["head_id"]
        hypo2premise["tail_id"] = triplet["tail_id"]
        hypo2premise["context"] = (
            l_head
            + " means "
            + s_head.definition()
            + ". "
            + l_tail
            + " means "
            + s_tail.definition()
        )
        hypo2premise["premise"] = (
            l_head + " <" + str(triplet["relation"]) + "> " + l_tail
        )
        hypo2premise["subj"] = l_tail
        hypo2premise["obj"] = l_head
        hypo2premise["relation"] = triplet["relation"]
        hypos2premises.append(hypo2premise)

    out_path = path.replace("txt", "json")
    json.dump(
        hypos2premises,
        open(out_path, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )
    print("Save {} hypothesis and premises to {}".format(len(hypos2premises), out_path))
    return hypos2premises


def get_synset(name, id):
    if "NN" in name:
        s = wn.synset_from_pos_and_offset("n", int(id))
        return s

    if "VB" in name:
        s = wn.synset_from_pos_and_offset("v", int(id))
        return s

    if "JJ" in name:
        s = wn.synset_from_pos_and_offset("a", int(id))
        return s
    if "RB" in name:
        s = wn.synset_from_pos_and_offset("r", int(id))
        return s

    else:
        print("+" + name)


### case of freebase
def generate_nli_data_freebase(path):
    head2tail_dict, desciptions = preprocess.preprocess_fb15k237(path)
    all_id = list(desciptions.keys())
    #print(head2tail_dict)
    head2tail_df = pd.DataFrame(head2tail_dict)
    hypos2premises = []
    for index in range(0, len(head2tail_df)):
        triplet = head2tail_df.iloc[index]
        # remove the case with no description availaible
        if triplet["head_id"] in all_id and triplet["tail_id"] in all_id:
            hypo2premise = {}
            hypo2premise["head_id"] = triplet["head_id"]
            hypo2premise["tail_id"] = triplet["tail_id"]
            hypo2premise["context"] = (
                desciptions[triplet["head_id"]]
                + ". "
                + desciptions[triplet["tail_id"]].replace("@en", "")
            )  # context formed with the definitions
            hypo2premise["premise"] = (
                triplet["head"]
                + " <"
                + str(triplet["relation"])
                + "> "
                + triplet["tail"]
            )
            hypo2premise["subj"] = triplet["tail"]
            hypo2premise["obj"] = triplet["head"]
            hypo2premise["relation"] = triplet["relation"]
            hypos2premises.append(hypo2premise)

    out_path = path.replace("txt", "json")
    json.dump(
        hypos2premises,
        open(out_path, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )
    print("Save {} hypothesis and premises to {}".format(len(hypos2premises), out_path))
    return hypos2premises

# TODO : changer ça pour le cas de wn ET fb puissent être géré 
def main():
    """path = "./data/WN18RR/test.txt"
    hypos2premises = generate_nli_data(path)
    print(pd.DataFrame(hypos2premises))"""

    generate_nli_data(args.train_path, args.task)
    generate_nli_data(args.test_path, args.task)
    generate_nli_data(args.valid_path, args.task)


if __name__ == "__main__":
    main()
