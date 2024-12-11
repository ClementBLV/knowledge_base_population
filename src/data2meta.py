"""
BINAIRY CLASSIFICATION

premise | prob 1 | prob 2 | prob 3 | prob 4 | real label | predicted output vote | predicted output meta 

"""

################ setup : parser ################
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from pprint import pprint
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
                    help="Input file obtained ofter the mnli convertion (should be a .mnli.json file)")
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
args = parser.parse_args()


################ Setup: Logger ################
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.info("Program: data2meta.py ****")

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
def sha1(text):
    """Generate SHA-1 hash for a given text."""
    return hashlib.sha1(text.encode('utf-8')).hexdigest()

def worker_init():
    global models
    global model2prob
    models = {
        "model_1": AutoModelForSequenceClassification.from_pretrained(config["models_path"]["model_1way"]).to(device),
        "model_3": AutoModelForSequenceClassification.from_pretrained(config["models_path"]["model_2ways"]).to(device),
        "model_2": AutoModelForSequenceClassification.from_pretrained(config["models_path"]["model_1way_reverse"]).to(device),
        "model_4": AutoModelForSequenceClassification.from_pretrained(config["models_path"]["model_2ways"]).to(device),
    }
    model2prob = {
        "model_1": "p1",
        "model_3": "p2",
        "model_2": "p3",
        "model_4": "p4",
    }

def process_data(line):
    """Process a single data point to get predictions."""
    id = sha1(line["premise"])
    encoded_context = tokenizer(
        line["premise"],
        line["hypothesis"],
        truncation="longest_first",
        max_length=config["max_length"],
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    # Select models based on "way"
    if line["way"] > 0:
        with torch.no_grad():
            p1 = models["model_1"](**encoded_context).logits.softmax(dim=-1).cpu().tolist()
            p3 = models["model_3"](**encoded_context).logits.softmax(dim=-1).cpu().tolist()
            return {"id": id, "p1": p1, "p2": None, "p3": p3, "p4": None, "real_label": line["label"]}
    else:
        with torch.no_grad():
            p2 = models["model_2"](**encoded_context).logits.softmax(dim=-1).cpu().tolist()
            p4 = models["model_4"](**encoded_context).logits.softmax(dim=-1).cpu().tolist()
            return {"id": id, "p1": None, "p2": p2, "p3": None, "p4": p4, "real_label": line["label"]}


def process_data_line(line, model, model_name):
    """Process a single data point to get predictions using a specific model."""
    id = sha1(line["premise"])
    encoded_context = tokenizer(
        line["premise"],
        line["hypothesis"],
        truncation="longest_first",
        max_length=config["max_length"],
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**encoded_context).logits.softmax(dim=-1).cpu().tolist()
    return {"id": id, "logits": logits, "real_label": line["label"]}

def process_model_for_all_data(model_name, data_batch):
    """Process all data for a specific model."""
    results = []
    model = models[model_name]
    for line in data_batch:
        if ((line["way"] > 0 and model_name in ["model_1", "model_3"]) 
            or line["way"]< 0 and model_name in ["model_2", "model_4"]):
            result = process_data_line(line, model, model_name)
            # Store the result with the model's prediction
            result["model_name"] = model_name
            results.append(result)
        
    return results

def process_data_concurrently(datas):
    """Run the inference concurrently on four models using threads."""
    print("len(datas" , len(datas))
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        # Submit a task for each model, running inference on all data points
        for model_name in ["model_1", "model_2", "model_3", "model_4"]:
            futures.append(executor.submit(process_model_for_all_data, model_name, datas))

        # Wait for all threads to finish and collect results
        all_results = []
        for future in futures:
            all_results.extend(future.result())

    return all_results




################ Main Processing ################
worker_init()


if args.parallel:
    logger.info("Running in parallel mode...")
    results = process_data_concurrently(datas)
    
else:
    logger.info("Running in sequential mode...")
    # Adding tqdm to monitor processing progress in sequential mode
    results = [process_data(line) for line in tqdm(datas, desc="Processing Data Sequentially")]


################ Merge Probabilities ################
# Group results by ID and label and merge probabilities
merged_results = {
    result["id"]: {label_id: {"p1": None, "p2": None, "p3": None, "p4": None} for label_id in config["id2label"]}
    for result in tqdm(results, desc="Initializing Merged Results")
}

if args.parallel:
    for result in tqdm(results, desc="Merging Results Paraalel"):
        merged_results[result["id"]][str(result["real_label"])][model2prob[result["model_name"]]] = result['logits']
    
    pprint(merged_results, indent=4)
    
else:
    # Update merged_results with probabilities
    for result in tqdm(results, desc="Merging Probabilities Sequential"):
        for key in ["p1", "p3"]:
            if result[key] is not None:
                merged_results[result["id"]][str(result["real_label"])][key] = result[key]

        for key in ["p2", "p4"]:
            if result[key] is not None:
                merged_results[result["id"]][str(result["real_label"])][key] = result[key]




# Construct the final results
final_results = []
for id, label_probs in tqdm(merged_results.items(), desc="Building Final Results"):
    for label, probs in label_probs.items():
        final_results.append({
            "id": id,
            "p1": probs["p1"],
            "p2": probs["p2"],
            "p3": probs["p3"],
            "p4": probs["p4"],
            "label": label
        })
logger.info(f"There are {len(final_results)} premisses (direct and reverse relation) each one with two examples [entailment - contradiction]")
logger.info(f"Example of the data : \n\n\t {final_results[0]} \n")

################ Save Results ################
df = pd.DataFrame(final_results)
output_file = Path(args.output_folder) / args.saving_name
output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_json(output_file, orient="records", lines=True)
logger.info(f"Save: Output saved to {output_file}")