"""
BINARY CLASSIFICATION

premise | prob 1 | prob 2 | prob 3 | prob 4 | real label | predicted output vote | predicted output meta

"""

import argparse
import hashlib
import logging
from pprint import pprint
import sys
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from concurrent.futures import ThreadPoolExecutor
from src.utils.utils import str2bool


def setup_logger():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Program: data2_meta.py ****\n")
    return logger


def sha1(text):
    """Generate SHA-1 hash for a given text."""
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def worker_init(config, device):
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


def process_data_line(line, model, tokenizer, config, device):
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


def process_model_for_all_data(model_name, data_batch, tokenizer, config, device):
    """Process all data for a specific model."""
    results = []
    model = models[model_name]
    for line in data_batch:
        if ((line["way"] > 0 and model_name in ["model_1", "model_3"]) or
            line["way"] < 0 and model_name in ["model_2", "model_4"]):
            result = process_data_line(line, model, tokenizer, config, device)
            result["model_name"] = model_name
            results.append(result)
    return results


def process_data_concurrently(datas, tokenizer, config, device):
    """Run the inference concurrently on four models using threads."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for model_name in ["model_1", "model_2", "model_3", "model_4"]:
            futures.append(executor.submit(process_model_for_all_data, model_name, datas, tokenizer, config, device))

        all_results = []
        for future in futures:
            all_results.extend(future.result())

    return all_results


def process_data(datas, tokenizer, config, device):
    """Process data sequentially."""
    results = []
    for line in tqdm(datas, desc="Processing Data Sequentially"):
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
            if line["way"] > 0:
                p1 = models["model_1"](**encoded_context).logits.softmax(dim=-1).cpu().tolist()
                p3 = models["model_3"](**encoded_context).logits.softmax(dim=-1).cpu().tolist()
                results.append({"id": id, "p1": p1, "p2": None, "p3": p3, "p4": None, "real_label": line["label"]})
            else:
                p2 = models["model_2"](**encoded_context).logits.softmax(dim=-1).cpu().tolist()
                p4 = models["model_4"](**encoded_context).logits.softmax(dim=-1).cpu().tolist()
                results.append({"id": id, "p1": None, "p2": p2, "p3": None, "p4": p4, "real_label": line["label"]})
    return results



def main(args):
    ################ Setup: Logger ################
    logger = setup_logger()
    logger.info("Starting data2meta processing...")

    ################ setup : device ################
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Device: {device}")
    ################ setup : config ################
    config = args.config

    ################ setup : tokenizer ################
    tokenizer = AutoTokenizer.from_pretrained(
        config["source_model"], 
        use_fast=False
    )

    ################ setup : models ################
    worker_init(config, device)

    ################ Process data ################
    if args.parallel:
        results = process_data_concurrently(args.datas, tokenizer, config, device)
        print("results", results)
    else:
        results = process_data(args.datas, tokenizer, config, device)

    
    ################ Merge Probabilities ################
    # Group results by ID and label and merge probabilities
    merged_results = {
        result["id"]: {label_id: {"p1": None, "p2": None, "p3": None, "p4": None} for label_id in list(config["label2id"].values())}
        for result in results #tqdm(results, desc="Initializing Merged Results")
    }

    if args.parallel:
        for result in results :  #tqdm(results, desc="Merging Results Paraalel"):
            merged_results[result["id"]][result["real_label"]][model2prob[result["model_name"]]] = result['logits']
        
        pprint(merged_results, indent=4)
        
    else:
        # Update merged_results with probabilities
        for result in results : #tqdm(results, desc="Merging Probabilities Sequential"):
            for key in ["p1", "p3"]:
                if result[key] is not None:
                    merged_results[result["id"]][result["real_label"]][key] = result[key]

            for key in ["p2", "p4"]:
                if result[key] is not None:
                    merged_results[result["id"]][result["real_label"]][key] = result[key]


    # Construct the final results
    final_results = []
    for id, label_probs in tqdm(merged_results.items(), desc="Building Final Results"):
        for label, probs in label_probs.items():
            if not(None in probs.values()):
                final_results.append({
                    "id": id,
                    "p1": probs["p1"],
                    "p2": probs["p2"],
                    "p3": probs["p3"],
                    "p4": probs["p4"],
                    "label": label
                })
    logger.info(f"In the {len(args.datas)}, there are {len(final_results)} premisses (direct and reverse relation) each one with two examples [entailment - contradiction]")
    logger.info(f"Example of the data : \n\n\t {final_results[0]} \n")
        
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Config file for the meta model")
    parser.add_argument("--parallel", type=str2bool, default=False,
                        help="Whether to run the model evaluations in parallel")
    parser.add_argument("--datas_path", type=str)
    args = parser.parse_args()
    # TODO read the data and add them to args 

    main(args)
