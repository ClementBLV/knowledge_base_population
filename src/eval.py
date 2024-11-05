from argparse import ArgumentParser
from collections import Counter  # TODO remoove useless libraiies 
from dataclasses import dataclass
from pathlib import Path
from typing import List
from transformers import (
    AutoTokenizer,
    DebertaForSequenceClassification,
    AutoModelForSequenceClassification,
)
from tqdm import tqdm
import json
import numpy as np
import torch
import sys
import os
from pprint import pprint

### device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device ", device)

parser = ArgumentParser()

parser.add_argument("--input_file", type=str, default="data/WN18RR/valid_eval.json")
parser.add_argument("--output_file", type=str, default="eval")
parser.add_argument("--model", type=str, default="")
parser.add_argument("--source_model", type=str, default="")
parser.add_argument("--name", type=str, default=None)

args = parser.parse_args()
print("=========== EVALUATION ============")

# Define the file name where you want to save the output and redirect the print
if args.name is None:
    name = args.model.split("/")[-1]
else:
    name = args.name
output_file = Path(f"{args.output_file}/eval_{name}.txt")
output_file.parent.mkdir(exist_ok=True, parents=True)

#output_file = (
#    f"{os.path.join( os.path.dirname(os.getcwd()),args.output_file)}/eval_{name}.txt"
#)
sys.stdout = open(output_file, "w")
# print("=========== EVALUATION ============")

# Initialition
@dataclass
class MNLIInputFeatures:
    premise: str  # context
    hypothesis_true: List[str]
    hypothesis_false: List[str]
    relation: str


mnli_data = []
# dict with key = True relation and the list of score to evaluate each realtion independantly
relation_score = {}


# load model
path = args.model
""" With the below writing some weight where randomly initialised
tokenizer = AutoTokenizer.from_pretrained(args.source_model)
model = DebertaForSequenceClassification.from_pretrained(path, ignore_mismatched_sizes=True)
model.to(device)"""

# model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)
model.to(device)


# read json
lines = json.load(open(args.input_file, "rt"))
for line in lines:
    mnli_data.append(
        MNLIInputFeatures(
            premise=line[
                "premise"
            ],  # should add the relation to have each hits for each relations
            hypothesis_true=line["hypothesis_true"],  # the true
            hypothesis_false=line["hypothesis_false"],  # all the false possible
            relation=line["relation"],
        )
    )

# evalute
def rank(mnli_input: MNLIInputFeatures, number_relation: int = 11):
    """
    Method to rank a feature, the entailement probability of the true premise is ranked against all the
    other premises probabilities. The output is a list of rank index, the position of the index corresponding to
    the rank and the index to the relation. The true relation is the element 0. If the 0 is place at the
    position 2 that would mean that the true premise only have the third highest rank.
    """

    # initialisation
    premise = mnli_input.premise

    # entail the true

    input = tokenizer(
        premise, mnli_input.hypothesis_true[0], truncation=True, return_tensors="pt"
    )
    output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    entailment = [prediction[0]]  # the True element is at index 0
    # entail the false
    for hf in mnli_input.hypothesis_false:
        input = tokenizer(premise, hf, truncation=True, return_tensors="pt")
        output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        entailment.append(prediction[0])  # [ proba entail , proba contradiction
    # rank the element, ind is the link of index of increasing probabilty
    return np.array(entailment).argsort()[-number_relation:][
        ::-1
    ]  # pour le truc global mais on va aussi return la relation


def compute_hits_at_k(shots, element=0, k_values=[1, 3, 10]):
    """
    shots : list of ranked index (the position correspond to the index)
    element : element in the list we want to compute the hit@
    k_values : the hit@k_values
    """
    hits = {k: 0 for k in k_values}

    for shot in shots:
        if element in shot[: k_values[-1]]:
            for k in k_values:
                if element in shot[:k]:
                    hits[k] += 1

    return hits


# Sample ranked lists for each shot
shots = []
relation2shots = {}
for data in tqdm(mnli_data):
    if len(data.hypothesis_true) > 0:  # TODO remove that in the future
        ranked_relations = rank(data)
        # all the shots to compute the global hit@
        shots.append(ranked_relations)
        # each relation to ist rank
        if data.relation in relation2shots.keys():
            relation2shots[data.relation].append(ranked_relations)
        else:
            relation2shots[data.relation] = [ranked_relations]

# Compute Hit at 1 and Hit at 3 for the element across shots
hits = compute_hits_at_k(shots)

# Display the Global results
for k, hit in hits.items():
    # print(f"Global Hit_at_{k}: {hit/len(shots)}")
    print(f"Hit_at_{k};Global;{hit/len(shots)}")
# print("------------------------------------")

# Display the results for each relation
for relation in relation2shots.keys():
    # Compute Hit at 1 and Hit at 3 for the element across shots
    hits = compute_hits_at_k(relation2shots[relation])

    # Display the results
    for k, hit in hits.items():
        # print(f"{relation} Hit_at_{k}: {hit/len(relation2shots[relation])}")
        print(f"Hit_at_{k};{relation};{hit/len(relation2shots[relation])}")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

sys.stdout.close()
sys.stdout = sys.__stdout__
