from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from typing import List
from transformers import AutoTokenizer, DebertaForSequenceClassification, AutoModelForSequenceClassification
from tqdm import tqdm
import json
import numpy as np 
import torch
import sys 
import os 
from pprint import pprint

### device 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device " ,device) 

parser = ArgumentParser()

parser.add_argument("--input_file", type=str, default="data/WN18RR/valid_eval.json")
parser.add_argument("--input_file_indirect", type=str, default="data/WN18RR/valid_eval_indirect.json")

parser.add_argument("--output_file", type=str, default="eval")
parser.add_argument("--model", type=str, default="")
parser.add_argument("--source_model", type=str, default="")
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--normalise", type=bool, default=True)

args = parser.parse_args()
print("=========== EVALUATION ============")

# Define the file name where you want to save the output and redirect the print
if args.name is None : 
    name = args.model.split("/")[-1]
else : 
    name = args.name
output_file = f"{os.path.join( os.path.dirname(os.getcwd()),args.output_file)}/eval_{name}.txt"
sys.stdout = open(output_file, "w")
#print("=========== EVALUATION ============")

# Initialition 
@dataclass
class MNLIInputFeatures:
    premise: str #context
    hypothesis_true:List[str]
    hypothesis_false:List[str]
    relation:str

mnli_data_direct = []
mnli_data_indirect = []

# dict with key = True relation and the list of score to evaluate each realtion independantly
relation_score = {} 


# load model
path = args.model
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)
model.to(device)


# read json DIRECT
lines = json.load(open(args.input_file, "rt"))
for line in lines :
    mnli_data_direct.append(
        MNLIInputFeatures(
            premise = line["premise"],                     # should add the relation to have each hits for each relations 
            hypothesis_true =line["hypothesis_true"],      # the true 
            hypothesis_false = line["hypothesis_false"],  # all the false possible 
            relation=line["relation"],
        )
    )
    
# read json INDIRECT
lines = json.load(open(args.input_file_indirect, "rt"))
for line in lines :
    mnli_data_indirect.append(
        MNLIInputFeatures(
            premise = line["premise"],                     # should add the relation to have each hits for each relations 
            hypothesis_true =line["hypothesis_true"],      # the true 
            hypothesis_false = line["hypothesis_false"],  # all the false possible 
            relation=line["relation"],
        )
    )
# normalisation 
def no_normalization(matrix): 
    return np.sum(matrix, axis=1) 
def post_normalization(matrix): 
    b = np.sum(matrix , axis=1)
    return b/sum(b)
def pre_normalization (matrix): 
    return matrix[:,0]/sum(matrix[:,0]) + matrix[:,1]/sum(matrix[:,1])

# evalute
def rank(mnli_input_direct : MNLIInputFeatures,
         mnli_input_indirect : MNLIInputFeatures, 
         normalize : bool, 
         number_relation:int =11):
    """ 
    Method to rank a feature, the entailement probability of the true premise is ranked against all the 
    other premises probabilities. The output is a list of rank index, the position of the index corresponding to 
    the rank and the index to the relation. The true relation is the element 0. If the 0 is place at the 
    position 2 that would mean that the true premise only have the third highest rank. 
    """

    ## initialisation 
    #direct
    premise_direct = mnli_input_direct.premise
    # indirect 
    premise_indirect = mnli_input_indirect.premise

    
    ## entail the true 
    #direct
    input = tokenizer(
                premise_direct,
                mnli_input_direct.hypothesis_true[0], 
                truncation=True, 
                return_tensors="pt"
                )
    output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    prediction_direct = torch.softmax(output["logits"][0], -1).tolist()
    #indirect
    input = tokenizer(
                premise_indirect,
                mnli_input_indirect.hypothesis_true[0], 
                truncation=True, 
                return_tensors="pt"
                )
    output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    prediction_indirect = torch.softmax(output["logits"][0], -1).tolist()
    entailment = [[prediction_direct[0], prediction_indirect[0]]]  # the True element is at index 0
    hypothesis = [mnli_input_direct.hypothesis_true[0]]
    # entail the false 
    for hf_direct , hf_indirect in zip( mnli_input_direct.hypothesis_false, mnli_input_indirect.hypothesis_false):
        input = tokenizer(
                        premise_direct ,
                        hf_direct, 
                        truncation=True, 
                        return_tensors="pt"
                        )
        output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
        prediction_direct = torch.softmax(output["logits"][0], -1).tolist()

        input = tokenizer(
                        premise_indirect ,
                        hf_indirect, 
                        truncation=True, 
                        return_tensors="pt"
                        )
        output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
        prediction_indirect = torch.softmax(output["logits"][0], -1).tolist()

        entailment.append([prediction_direct[0], prediction_indirect[0]])  # [ proba entail , proba contradiction
    hypothesis.append(mnli_input_direct.hypothesis_false)
    # rank the element, ind is the link of index of increasing probabilty    
    #pprint(hypothesis)
    #pprint(entailment)
    if normalize:
        entailment = pre_normalization(np.array(entailment))

    else : 
        entailment = no_normalization(np.array(entailment))
    #pprint(entailment)
    return  entailment.argsort()[-number_relation:][::-1]  # pour le truc global mais on va aussi return la relation


def compute_hits_at_k(shots, element=0, k_values=[1,3]):
    """
    shots : list of ranked index (the position correspond to the index)
    element : element in the list we want to compute the hit@
    k_values : the hit@k_values 
    """
    hits = {k: 0 for k in k_values}

    for shot in shots:
        if element in shot[:k_values[-1]]:
            for k in k_values:
                if element in shot[:k]:
                    hits[k] += 1

    return hits

# Sample ranked lists for each shot
shots = []
relation2shots = {}
for i, (data_direct,data_indirect ) in tqdm(enumerate(zip(mnli_data_direct, mnli_data_indirect)), total=len(mnli_data_direct)):
    if len(data_direct.hypothesis_true)>0: # TODO remove that in the future
        #print(data_direct.hypothesis_true)
        ranked_relations = rank(data_direct, data_indirect , args.normalise)
        # all the shots to compute the global hit@
        shots.append(ranked_relations)
        # each relation to ist rank 
        if data_direct.relation in relation2shots.keys():
            relation2shots[data_direct.relation].append(ranked_relations)
        else : 
            relation2shots[data_direct.relation] = [ranked_relations]
# Compute Hit at 1 and Hit at 3 for the element across shots
hits = compute_hits_at_k(shots)

# Display the Global results 
for k, hit in hits.items():
    #print(f"Global Hit_at_{k}: {hit/len(shots)}")
    print(f"Hit_at_{k};Global;{hit/len(shots)}")
#print("------------------------------------")

# Display the results for each relation
for relation in relation2shots.keys():
    # Compute Hit at 1 and Hit at 3 for the element across shots
    hits = compute_hits_at_k(relation2shots[relation])

    # Display the results
    for k, hit in hits.items():
        #print(f"{relation} Hit_at_{k}: {hit/len(relation2shots[relation])}")
        print(f"Hit_at_{k};{relation};{hit/len(relation2shots[relation])}")
    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

sys.stdout.close()
sys.stdout = sys.__stdout__