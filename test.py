from dataclasses import dataclass
from typing import List
from transformers import DebertaForSequenceClassification, AutoTokenizer
import torch
import json
import os
import pathlib

path = os.path.join( os.getcwd(),'tmp/MNLI/checkpoint-9000' )
print(path)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Replace 'path/to/your/checkpoint' with the actual path to your fine-tuned checkpoint
model = DebertaForSequenceClassification.from_pretrained(path, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

model.to(device)

@dataclass
class MNLIInputFeatures:
    premise: str #context
    hypothesis_true:List[str]
    hypothesis_false:List[str]
    relation:str


# read json
mnli_data = []
path_valid=os.path.join( os.getcwd(),"data/WN18RR/valid_eval.json")
lines = json.load(open(path_valid, "rt"))
for line in lines[0:10] :
    mnli_data.append(
        MNLIInputFeatures(
            premise = line["premise"],                     # should add the relation to have each hits for each relations 
            hypothesis_true =line["hypothesis_true"],      # the true 
            hypothesis_false = line["hypothesis_false"],  # all the false possible 
            relation=line["relation"],
        )
    )

for mnli_input in mnli_data:
    # entail the true 
    input = tokenizer(mnli_input.premise,mnli_input.hypothesis_true[0], truncation=True, return_tensors="pt")
    output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    entailment = [prediction[0]]  # the True element is at index 0
    print("======================")
    print(mnli_input.relation)
    print(mnli_input.premise)
    print(mnli_input.hypothesis_true[0])
    print(entailment, "0 is entail")
