from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import json
import numpy as np 
import torch

### device 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device " ,device) 

parser = ArgumentParser()

parser.add_argument("--input_file", type=str, default="/home/clement/Documents/Stage/knowledge_base_population/data/WN18RR/valid_eval.json")
parser.add_argument("--model", type=str, default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c")

args = parser.parse_args()
print("=========== EVALUATION ============")

@dataclass
class MNLIInputFeatures:
    premise: str #context
    hypothesis_true:List[str]
    hypothesis_false:List[str]

mnli_data = []

# load model
model_name = args.model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# read json 
lines = json.load(open(args.input_file, "rt"))
for line in lines :
    mnli_data.append(
        MNLIInputFeatures(
            premise = line["premise"],
            hypothesis_true =line["hypothesis_true"],
            hypothesis_false = line["hypothesis_false"],
        )
    )
# evalute
def hit_at(mnli_input : MNLIInputFeatures, number_relation =11):
    # initialisation 
    premise = mnli_input.premise
    
    # entail the true 
    input = tokenizer(premise,mnli_input.hypothesis_true[0], truncation=True, return_tensors="pt")
    output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    entailment = [prediction[0]]  # the True element is at index 0
    
    # entail the false 
    for hf in mnli_input.hypothesis_false:
        input = tokenizer(
                        premise ,
                        hf, 
                        truncation=True, 
                        return_tensors="pt"
                        )
        output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        entailment.append(prediction[0])  # [ proba entail , proba contradiction

    # rank the element, ind is the link of index of increasing probabilty    
    ind = np.array(entailment).argsort()[-number_relation:][::-1]
    return np.where(ind==0)[0][0]+1 # +1 because the index starts at 0
            
hits = []
for data in tqdm(mnli_data):
    hits.append(hit_at(data))

count = Counter(hits)
for elt in count: 
    print("Hit@"+str(elt)+" = "+str(count[elt]/len(hits)))