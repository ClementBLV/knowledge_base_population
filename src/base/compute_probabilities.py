import argparse
import os
import pathlib
from typing import List, Dict, Union
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

from tqdm import tqdm
from src.meta.meta_models import DummyModel
from src.utils.utils import get_config, setup_logger_basic, str2bool
from src.base.mnli_dataclass import *

logger = setup_logger_basic()
logger.info("Program: compute_probabilities.py ****")


def parse_args():
    """Parse command-line arguments and return them as a Namespace."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, required=True, help="Input JSON file")
    parser.add_argument("--proba_file", type=str, required=True, help="File to save computed probabilities")
    parser.add_argument("--config_file", type=str, required=True, help="Config file for the model")
    parser.add_argument("--model_weight_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for probability computation")
    parser.add_argument("--parallel", type=str2bool, default=True, help="Parallel execution on GPU")
    parser.add_argument("--fast", type=str2bool, default=False, help="Use only 1000 examples for fast testing")
    parser.add_argument("--dummy", type=str2bool, default=False, help="Use the dummy model for testing testing")
    return parser.parse_args()


def collate_fn(batch):
    premises, hypotheses, labels, relations = [], [], [], []
    for inputs, relation in batch:
        for premise, hypothesis, label in inputs:
            premises.append(premise)
            hypotheses.append(hypothesis)
            labels.append(label)
        relations.append(relation)
    return premises, hypotheses, labels, relations


def compute_prediction(
    data_item : Union[MetaPredictionInputFeatures, PredictionInputFeatures]
) -> List[int]:
    return data_item.probabilities.argsort(descending=True).tolist()


def compute_probabilities(
    mnli_data: List[PredictionInputFeatures],
    tokenizer: AutoTokenizer,
    model,
    device,
    config,
    batch_size=32,
) -> List[PredictionInputFeatures]:
    """Compute probabilities for MNLI data using the given model."""
    dataset = MNLIDataset(mnli_data, tokenizer, config)
    dataloader = DataLoader(
                                dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                collate_fn=collate_fn, 
                                num_workers=4
                            )

    model.to(device)
    model.eval()

    idx = 0
    for premises, hypotheses, _, relations in tqdm(dataloader, desc="Computing Probabilities"):
        print("SIZE PREMISSES", len(premises))
        inputs = tokenizer(
                            premises,
                            hypotheses, 
                            padding=True, 
                            truncation='only_first', 
                            max_length=config["max_length"],
                            return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"])
            batch_probs = F.softmax(outputs["logits"], dim=-1)[:, 0].cpu()

        for i, relation in enumerate(relations):
            data_item = mnli_data[idx]
            num_hypotheses = len(data_item.hypothesis_false) + 1  
            start, end = i * num_hypotheses, (i + 1) * num_hypotheses

            if end > len(batch_probs):  
                raise ValueError(f"Indexing error: {end} exceeds batch_probs size {len(batch_probs)}")

            data_item.probabilities = batch_probs[start:end] 
            data_item.predictions = compute_prediction(data_item)

            idx += 1

    return mnli_data

## TODO here put compute without parralélisation 
def compute_probabilities_fully_sequential(
    mnli_data: List[PredictionInputFeatures],
    tokenizer: AutoTokenizer,
    model,
    device,
    config,
    batch_size=0,
) -> List[PredictionInputFeatures]:
    """Compute probabilities for MNLI data using the given model, processing hypotheses one by one."""

    model.to(device)
    model.eval()

    for data_item in tqdm(mnli_data, desc="Computing Probabilities (Fully Sequential)"):
        premise = data_item.premise
        all_hypotheses = [data_item.hypothesis_true] + data_item.hypothesis_false

        probs = []

        for hypothesis in all_hypotheses:
            inputs = tokenizer(
                premise,
                hypothesis,
                padding=True,
                truncation="only_first",
                max_length=config["max_length"],
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(inputs["input_ids"])
                prob = F.softmax(outputs["logits"], dim=-1)[0, 0].item()  # Scalar: P(entailment)

            probs.append(prob)

        data_item.probabilities = torch.tensor(probs)
        data_item.predictions = compute_prediction(data_item)

    return mnli_data

def run_probability_computation(
        eval_file, 
        proba_file, 
        config_file, 
        model_weight_path, 
        batch_size, 
        parallel, 
        fast, 
        dummy) -> List[Dict]:
    """Run the full pipeline: load data, compute probabilities, and return JSON."""
    global config
    config = get_config(config_file)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], 
        use_fast=False,
        max_length=config["max_length"],
        padding="max_length"
    ) 
    
    if dummy:
        logger.info("Using Dummy Model")
        model = DummyModel(num_labels=config.get("num_labels", 3))  # Use dummy model
    else:
        model = AutoModelForSequenceClassification.from_pretrained(pathlib.Path(model_weight_path))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    if not torch.cuda.is_available() and parallel:
        parallel = False
        logger.warning("Parallel execution disabled: No GPU detected, running sequentially.")

    if not parallel: 
        compute_probabilities_fn = compute_probabilities_fully_sequential
    else : 
        compute_probabilities_fn = compute_probabilities


    mnli_data = load_data(eval_file, fast, logger)
    mnli_data = compute_probabilities_fn(mnli_data, tokenizer, model, device, config, batch_size)

    mnli_data_serializable = [item.to_dict() for item in mnli_data]
    
    os.makedirs(os.path.dirname(proba_file), exist_ok=True)

    with open(proba_file, "w") as f:
        json.dump(mnli_data_serializable, f, indent=2)
    
    logger.info(f"Probabilities saved to {proba_file}")
    return mnli_data_serializable


if __name__ == "__main__":
    args = parse_args()
    run_probability_computation(
        args.eval_file,
        args.proba_file,
        args.config_file,
        args.model_weight_path,
        args.batch_size,
        args.parallel,
        args.fast,
        args.dummy
    )
