import argparse
from dataclasses import dataclass
import logging
from setup import setup_logger
from pathlib import Path
from typing import List
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from tqdm import tqdm
import json
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
print("=========== EVALUATION ============")

################ setup : parser ################
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
                    help=" File with the eval json eg : data/WN18RR/valid_eval.json")
parser.add_argument("--output_file", type=str, required=True, default="eval")
parser.add_argument("--source_model", type=str, required=True, 
                    help="Hugging face ID of the model used")
parser.add_argument("--model", type=str, required=True, 
                    help="Path of the weight of the model")
parser.add_argument("--saving_name", type=str, default=None, 
                    help="Name to save the evaluation file")
parser.add_argument('-parallel', '--parallel', type=str2bool, default=True,
                    help='If true the evaluation will be done in batch and parallelise on GPU')
parser.add_argument('-batch_size', '--batch_size', type=int, default=32,
                    help='Batch size for the evaluation')
args = parser.parse_args()

################ initialisation : path ################
name = args.model.split("/")[-1] if args.saving_name is None else args.saving_name
output_file = Path(f"{args.output_file}/eval_{name}")
output_file.parent.mkdir(exist_ok=True, parents=True)


################ Setup: Logger ################

# Set up the logger
log_file = f"{args.output_file}.log"  # Save log to `eval.log` or as specified
logger = setup_logger(log_file)
logger.info("Program: eval.py ****")

################ setup : device ################
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logger.info(f"Device : {device}")
if not(torch.cuda.is_available()) and args.parallel: 
    args.parallel = False 
    logger.warning("CPU devise, no parralelization possible - evaluation will be done sequentially")



################ initialisation : data-objects ################
@dataclass
class MNLIInputFeatures:
    premise: str  # context
    hypothesis_true: List[str]
    hypothesis_false: List[str]
    relation: str


class MNLIDataset(Dataset):
    def __init__(self, mnli_data : List[MNLIInputFeatures], tokenizer : AutoTokenizer, number_relation=11):
        self.mnli_data = mnli_data
        self.tokenizer = tokenizer
        self.number_relation = number_relation

    def __len__(self):
        return len(self.mnli_data)

    def __getitem__(self, idx):
        data = self.mnli_data[idx]
        inputs = []
        true_hypothesis = data.hypothesis_true[0]
        inputs.append((data.premise, true_hypothesis, 1))  # True hypothesis marked as label 1

        for hf in data.hypothesis_false:
            inputs.append((data.premise, hf, 0))  # False hypothesis marked as label 0

        return inputs, data.relation  # Return relation for mapping

################ initialisation : data ################
mnli_data = []
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


################ code : ranking ################
def rank(
            mnli_input : MNLIInputFeatures, 
            tokenizer : AutoTokenizer,
            model: AutoModelForSequenceClassification, 
            device, 
            number_relation=11
        ):
    """
    Sequential computation for ranking entailment scores without parallelism.
    """
    premise = mnli_input.premise

    # Compute entailment for the true hypothesis
    input = tokenizer(
        premise, mnli_input.hypothesis_true[0], truncation=True, return_tensors="pt"
    )
    input = input.to(device)
    output = model(input["input_ids"])
    prediction = torch.softmax(output["logits"][0], dim=-1).tolist()
    entailment = [prediction[0]]  # True hypothesis score

    # Compute entailment for false hypotheses
    for hf in mnli_input.hypothesis_false:
        input = tokenizer(premise, hf, truncation=True, return_tensors="pt").to(device)
        output = model(input["input_ids"])
        prediction = torch.softmax(output["logits"][0], dim=-1).tolist()
        entailment.append(prediction[0])

    # Rank the scores
    return np.array(entailment).argsort()[-number_relation:][::-1]

def collate_fn(batch):
    premises, hypotheses, labels = [], [], []
    relations = []
    for inputs, relation in batch:
        for premise, hypothesis, label in inputs:
            premises.append(premise)
            hypotheses.append(hypothesis)
            labels.append(label)
        relations.append(relation)
    return premises, hypotheses, labels, relations

def rank_parallel(
            mnli_data : List[MNLIInputFeatures], 
            tokenizer : AutoTokenizer,
            model: AutoModelForSequenceClassification, 
            device, 
            batch_size=32,
            number_relation=11
            ):
    """
    Parallelized computation for ranking entailment scores using batching on GPU.
    """
    dataset = MNLIDataset(mnli_data, tokenizer, number_relation)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    ranked_results = []
    relation2shots = {}

    model.to(device)
    model.eval()

    for premises, hypotheses, _, relations in tqdm(dataloader, desc="Running Ranking :"):
        inputs = tokenizer(
            premises, hypotheses, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(inputs["input_ids"])
            predictions = F.softmax(outputs["logits"], dim=-1)[:, 0]  # Probability of entailment

        # Group predictions by relation and rank
        for i, relation in enumerate(relations):
            entailments = predictions[i * number_relation: (i + 1) * number_relation]
            ranked_indices = entailments.argsort(descending=True).tolist()

            # Store ranked indices
            ranked_results.append(ranked_indices)

            if relation not in relation2shots:
                relation2shots[relation] = []
            relation2shots[relation].append(ranked_indices)

    return ranked_results, relation2shots

################ code : score ################

def compute_hits_at_k(ranked_results, element=0, k_values=[1, 3, 10]):
    """
    shots : list of ranked index (the position correspond to the index)
    element : element in the list we want to compute the hit@
    k_values : the hit@k_values
    """
    hits = {k: 0 for k in k_values}

    for ranking in ranked_results:
        if element in ranking[: k_values[-1]]:
            for k in k_values:
                if element in ranking[:k]:
                    hits[k] += 1

    return hits

def compute_mrr(ranked_results, element=0):
    """
    Compute the Mean Reciprocal Rank (MRR) for a set of ranked results.
    
    Args:
        ranked_results: List of ranked indices for each evaluation example.
        element: The target element (e.g., the correct index) for which to compute the rank.

    Returns:
        float: The Mean Reciprocal Rank (MRR).
    """
    reciprocal_ranks = []
    for ranking in ranked_results:
        try:
            rank_position = ranking.index(element) + 1  # Rank positions are 1-based
            reciprocal_ranks.append(1 / rank_position)
        except ValueError:
            reciprocal_ranks.append(0.0)  # Element not found in ranking

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


################ code : evaluation ################

def evaluate(
        mnli_data: List [MNLIInputFeatures], 
        tokenizer : AutoTokenizer, 
        model : AutoModelForSequenceClassification, 
        parallel : bool, 
        device : str = "cpu", 
        batch_size : str = 32, 
        number_relation : int =11):
    """
    Main function to evaluate rankings, with an option for parallelism.
    """
    if parallel and torch.cuda.is_available():
        logger.info("Running in parallel mode on GPU...")
        device = torch.device("cuda")
        return rank_parallel(mnli_data, tokenizer, model, device, batch_size, number_relation)
    else:
        logger.info("Running in sequential mode on CPU...")
        ranked_results = []
        relation2shots = {}

        for data in tqdm(mnli_data, desc="Running Sequential Ranking :"):
            if len(data.hypothesis_true) > 0:  # Ensure valid input
                ranked_relations = rank(data, tokenizer, model, device, number_relation)
                ranked_results.append(ranked_relations)

                if data.relation in relation2shots.keys():
                    relation2shots[data.relation].append(ranked_relations)
                else:
                    relation2shots[data.relation] = [ranked_relations]

        return ranked_results, relation2shots


path = args.model
""" With the below writing some weight where randomly initialised
tokenizer = AutoTokenizer.from_pretrained(args.source_model)
model = DebertaForSequenceClassification.from_pretrained(path, ignore_mismatched_sizes=True)
model.to(device)"""
tokenizer = AutoTokenizer.from_pretrained(
        args.source_model, 
        use_fast=False, 
        model_max_length=512
    ) 
model = AutoModelForSequenceClassification.from_pretrained(path)
model.to(device)

# Compute Hit at 1 and Hit at 3 for the element across shots
ranked_results, relation2shots = evaluate(
            mnli_data=mnli_data, 
            tokenizer=tokenizer, 
            model=model, 
            parallel=args.parallel, 
            device=device, 
            batch_size = args.batch_size, 
            number_relation =11
        )
hits = compute_hits_at_k(ranked_results)
mrr = compute_mrr(ranked_results)
logger.info(f"Global MRR: {mrr}")
# Display the Global results
for k, hit in hits.items():
    logger.info(f"Hit_at_{k};Global;{hit/len(ranked_results)}")

# Display the results for each relation
for relation in relation2shots.keys():
    # Compute Hit at 1 and Hit at 3 for the element across shots
    mrr_relation = compute_mrr(relation2shots[relation])
    logger.info(f"MRR;{relation};{mrr_relation}")

    hits = compute_hits_at_k(relation2shots[relation])

    # Display the results
    for k, hit in hits.items():
        logger.info(f"Hit_at_{k};{relation};{hit/len(relation2shots[relation])}")
    
    
