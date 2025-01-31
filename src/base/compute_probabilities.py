import argparse
from dataclasses import dataclass
import os
import pathlib
from pathlib import Path
from typing import List, Dict
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch.nn.functional as F
from src.utils.utils import get_config, setup_logger_basic, str2bool, sha1

logger = setup_logger_basic()
logger.info("Program: compute_probabilities.py ****")


@dataclass
class MNLIInputFeatures:
    id: str
    premise: str
    hypothesis_true: str
    hypothesis_false: List[str]
    relation: str
    probabilities: List[float]

    def to_dict(self) -> Dict:
        """Converts the instance to a dictionary for JSON serialization."""
        return {
            "id": self.id,
            "premise": self.premise,
            "hypothesis_true": self.hypothesis_true,
            "hypothesis_false": self.hypothesis_false,
            "relation": self.relation,
            "probabilities": self.probabilities,
        }


class MNLIDataset(Dataset):
    def __init__(self, mnli_data: List[MNLIInputFeatures], tokenizer: AutoTokenizer):
        self.mnli_data = mnli_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.mnli_data)

    def __getitem__(self, idx):
        data = self.mnli_data[idx]
        inputs = [(data.premise, data.hypothesis_true, config["label2id"]["entailment"])]

        for hf in data.hypothesis_false:
            inputs.append((data.premise, hf, config["label2id"]["not_entailment"]))

        return inputs, data.relation


def parse_args():
    """Parse command-line arguments and return them as a Namespace."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="File to save computed probabilities")
    parser.add_argument("--config_file", type=str, required=True, help="Config file for the model")
    parser.add_argument("--model", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for probability computation")
    parser.add_argument("--parallel", type=str2bool, default=True, help="Parallel execution on GPU")
    parser.add_argument("--fast", type=str2bool, default=False, help="Use only 1000 examples for fast testing")
    return parser.parse_args()


def load_data(input_file: str, fast: bool) -> List[MNLIInputFeatures]:
    """Load JSON data from the input file and convert it to MNLIInputFeatures."""
    with open(input_file, "r") as f:
        data = json.load(f)

    logger.info(f"Loading input file: {input_file}")
    
    if fast:
        logger.warning("Fast mode enabled: Using only 1000 samples for debugging.")
        data = data[:1000]

    mnli_data = [
        MNLIInputFeatures(
            id=sha1(line["premise"] + line["hypothesis_true"]),
            premise=line["premise"],
            hypothesis_true=line["hypothesis_true"],
            hypothesis_false=line["hypothesis_false"],
            relation=line["relation"],
            probabilities=[],
        )
        for line in data
    ]

    if not mnli_data:
        raise ValueError("Loaded data is empty!")

    return mnli_data


def collate_fn(batch):
    premises, hypotheses, labels, relations = [], [], [], []
    for inputs, relation in batch:
        for premise, hypothesis, label in inputs:
            premises.append(premise)
            hypotheses.append(hypothesis)
            labels.append(label)
        relations.append(relation)
    return premises, hypotheses, labels, relations


def compute_probabilities(
    mnli_data: List[MNLIInputFeatures],
    tokenizer: AutoTokenizer,
    model,
    device,
    batch_size=32,
) -> List[MNLIInputFeatures]:
    """Compute probabilities for MNLI data using the given model."""
    dataset = MNLIDataset(mnli_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model.to(device)
    model.eval()

    idx = 0
    for premises, hypotheses, _, relations in tqdm(dataloader, desc="Computing Probabilities"):
        inputs = tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"])
            batch_probs = F.softmax(outputs["logits"], dim=-1)[:, 0].cpu().tolist()

        for i, relation in enumerate(relations):
            data_item = mnli_data[idx]
            num_hypotheses = len(data_item.hypothesis_false) + 1
            data_item.probabilities = batch_probs[i * num_hypotheses: (i + 1) * num_hypotheses]
            idx += 1

    return mnli_data


def run_probability_computation(eval_file, output_file, config_file, model_path, batch_size, parallel, fast) -> List[Dict]:
    """Run the full pipeline: load data, compute probabilities, and return JSON."""
    global config
    config = get_config(config_file)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=False, model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(pathlib.Path(model_path))
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    if not torch.cuda.is_available() and parallel:
        parallel = False
        logger.warning("Parallel execution disabled: No GPU detected, running sequentially.")

    mnli_data = load_data(eval_file, fast)
    mnli_data = compute_probabilities(mnli_data, tokenizer, model, device, batch_size)

    mnli_data_serializable = [item.to_dict() for item in mnli_data]

    with open(output_file, "w") as f:
        json.dump(mnli_data_serializable, f, indent=2)
    
    logger.info(f"Probabilities saved to {output_file}")
    return mnli_data_serializable


if __name__ == "__main__":
    args = parse_args()
    run_probability_computation(
        args.eval_file,
        args.output_file,
        args.config_file,
        args.model,
        args.batch_size,
        args.parallel,
        args.fast,
    )
