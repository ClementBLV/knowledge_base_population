"""
BINAIRY CLASSIFICATION

premise | prob 1 | prob 2 | prob 3 | prob 4 | real label | predicted output vote | predicted output meta 

"""

################ setup : parser ################
import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
from pprint import pprint
import random
import sys
from typing import List

import pandas as pd
from tqdm import tqdm
from data2mnli import MNLIInputFeatures
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from torch.multiprocessing import Pool, set_start_method
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

from templates import FB_LABEL_TEMPLATES, WN_LABEL_TEMPLATES, WN_LABELS
LABELS, LABEL_TEMPLATES = None , None

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True,
                    help="Input file obtained after the data_generator (should be a .json file)")
parser.add_argument("--output_folder", type=str, required=True, 
                    help="Folder were the output will be saved")
parser.add_argument("--saving_name", type=str, required=True, 
                    help="Name of the saved data usable by the meta model")
parser.add_argument("--config_file", type=str, required=True, 
                    help="Name if the config file of for the meta model")
parser.add_argument("--parallel", type=str2bool, default=False,
                    help="Whether to run the model evaluations in parallel")
parser.add_argument("--training_number", type=int, default=0,
                    help="Number of examples used for generation of the data, by default all file will be taken")
parser.add_argument("--task", type=str, required=True, 
                    help="Dataset name (e.g., 'wordnet' or 'freebase').")

args = parser.parse_args()


################ Setup: Logger ################
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.info("Program: data2meta_v2.py ****")

################ setup : device ################
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logger.info(f"Device: {device}")

################ setup : config ################
current_dir = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(current_dir), "configs", args.config_file)
with open(config_path, "r") as config_file:
    config = json.load(config_file)

################ Initialization: Data ################
# Load the input JSON
with open(args.input_file) as f:
    datas = json.load(f)

if args.training_number !=0 : 
    datas = datas[:args.training_number]

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    config["source_model"],
    use_fast=False,
)

################ Utility Functions ################
def create_templates(task: str):
    global LABELS
    global LABEL_TEMPLATES
    if task.lower() in ["wordnet", "wn", "wn18rr"]:
        LABELS = WN_LABELS
        LABEL_TEMPLATES = WN_LABEL_TEMPLATES
    elif task.lower() in ["freebase", "fb", "fb15k237"]:
        LABELS = list(FB_LABEL_TEMPLATES.keys())
        LABEL_TEMPLATES = FB_LABEL_TEMPLATES
    else:
        raise ValueError(f"Unknown task: {task}")
    
@dataclass
class Relation:
    """Represents a relation and its template."""
    name:str
    template: str
    label: int
    relation_direct : str
    relation_reverse : str


@dataclass
class MNLIInputFeatures:
    """Represents an MNLI input instance."""
    id : str
    context: str
    subj: str
    obj: str
    relations : List[Relation]


    
def sha1(text):
    """Generate SHA-1 hash for a given text."""
    return hashlib.sha1(text.encode('utf-8')).hexdigest()

def worker_init():
    global models
    models = {
        "model_1": AutoModelForSequenceClassification.from_pretrained(config["models_path"]["model_1way"]).to(device),
        "model_3": AutoModelForSequenceClassification.from_pretrained(config["models_path"]["model_2ways"]).to(device),
        "model_2": AutoModelForSequenceClassification.from_pretrained(config["models_path"]["model_1way_reverse"]).to(device),
        "model_4": AutoModelForSequenceClassification.from_pretrained(config["models_path"]["model_2ways"]).to(device),
    }

def format_relation(t: Relation, obj: str , subj:str , way: int)-> str:
    if way > 0 :
        return f"{t.template.format(subj=subj, obj=obj)}."
    else : 
        return f"{t.template.format(subj=obj, obj=obj)}."
    
def dataload (line):
    relations: List[Relation] = []
    template = LABEL_TEMPLATES[line["relation"]]
    true_relation = Relation(
        name=line["relation"],
        template=template, 
        relation_direct=format_relation(t=template[0], obj=line["obj"], subj=line["subj"], way=1),
        relation_reverse=format_relation(t=template[1], obj=line["obj"], subj=line["subj"], way=-1),
        label=config["label2id"]["entailment"], 
    )

    relations.append(true_relation)
    for r in LABELS: 
        if r != line["relation"]:
            template = LABEL_TEMPLATES[r] 
            relations.append(
                Relation(
                    name=r,
                    template=template, 
                    relation_direct=format_relation(t=template[0], obj=line["obj"], subj=line["subj"], way=1),
                    relation_reverse=format_relation(t=template[1], obj=line["obj"], subj=line["subj"], way=-1),
                    label=config["label2id"]["not_entailment"], 
                )
            )
    random.shuffle(relations)

    return MNLIInputFeatures(
        id = sha1(line["context"]),
        context=line["context"], 
        obj=line["obj"],
        subj=line["subj"],
        relations=relations
    )

################ sequential approach ################
def process_data(line):
    """Process a single data point to get predictions."""

    inputs = dataload(line)
    labels = []
    probas = []
    for relation in inputs.relations: 

        with torch.no_grad():
            # direct relation 
            encoded_context = tokenizer(
                inputs.context,
                relation.relation_direct,
                truncation="longest_first",
                max_length=config["max_length"],
                padding="max_length",
                return_tensors="pt"
            ).to(device)
            # compute [[p_entailment , p_contradiction]]
            p1 = models["model_1"](**encoded_context).logits.softmax(dim=-1).cpu().tolist()
            p3 = models["model_3"](**encoded_context).logits.softmax(dim=-1).cpu().tolist()

            # reverse relation 
            encoded_context = tokenizer(
                inputs.context,
                relation.relation_reverse,
                truncation="longest_first",
                max_length=config["max_length"],
                padding="max_length",
                return_tensors="pt"
            ).to(device)
            p2 = models["model_2"](**encoded_context).logits.softmax(dim=-1).cpu().tolist()
            p4 = models["model_4"](**encoded_context).logits.softmax(dim=-1).cpu().tolist()
        
        probas.extend([p1, p2, p3, p4])
        labels.append(relation.label)
    return id, probas, labels

################ parralel approach ################
def  process_relation(relations: List[Relation], context:str , model, model_name: str):
    """Process a single relation with the given model."""
    for relation in inputs.relations: 
        with torch.no_grad():
            # Direct relation
            if model_name in ["model_1", "model_3"]: 
                encoded_context = tokenizer(
                    context,
                    relation.relation_direct,
                    truncation="longest_first",
                    max_length=config["max_length"],
                    padding="max_length",
                    return_tensors="pt"
                ).to(device)
                direct_probs = model(**encoded_context).logits.softmax(dim=-1).cpu().tolist()
                return direct_probs

            if model_name in ["model_2", "model_4"]: 
                # Reverse relation
                encoded_context = tokenizer(
                    context,
                    relation.relation_reverse,
                    truncation="longest_first",
                    max_length=config["max_length"],
                    padding="max_length",
                    return_tensors="pt"
                ).to(device)
                reverse_probs = model(**encoded_context).logits.softmax(dim=-1).cpu().tolist()
                return direct_probs, reverse_probs

def process_data_for_model(lines, model_name):
    """Process all data for a specific model."""
    results = []
    model = models[model_name]
    for line in lines:
        inputs = dataload(line)
        labels = [re]
        probs = process_relation(inputs, model)
        results.append({"id": inputs.id, "proba": probs, "label": labels, "model_name": model_name})
    return results

def process_data_concurrently(datas):
    """Run the inference concurrently on four models using threads."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        # Submit a task for each model
        for model_name in ["model_1", "model_2", "model_3", "model_4"]:
            futures.append(executor.submit(process_data_for_model, datas, model_name))

        # Collect results from all threads
        all_results = []
        for future in futures:
            all_results.extend(future.result())

    return all_results



def fuse_results (all_results):
    """Fuse the probabilities obtain on the four threads"""
    for result in all results: 
        
    final_results.append({
            "id": id,
            "proba": proba for all relation
            "label": list of labels
        })

    return final_results

        
################ Main Processing ################
create_templates(args.task)
logger.info("Running in sequential mode...")
# Adding tqdm to monitor processing progress in sequential mode
final_results = []
labels = []
for line in tqdm(datas, desc="Processing Data Sequentially"):
    id , p, l = process_data(line)
    final_results.append({
            "id": id,
            "proba": p,
            "label": l
        })

logger.info(f"There are {len(final_results)} premisses (direct and reverse relation) each one with two examples [entailment - contradiction]")
logger.info(f"Example of the data  \n\n\t {final_results[0]} \n")
################ Save Results ################
df = pd.DataFrame(final_results)
output_file = Path(args.output_folder) / args.saving_name
output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_json(output_file, orient="records", lines=True)
logger.info(f"Save: Output saved to {output_file}")