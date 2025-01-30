import argparse
from dataclasses import dataclass
import os
import pathlib
from pathlib import Path
from typing import List
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch.nn.functional as F
from src.utils.utils import get_config, setup_logger_basic, str2bool, sha1

parser = argparse.ArgumentParser()
parser.add_argument("--eval_file", type=str, required=True, help="Input JSON file")
parser.add_argument("--output_file", type=str, required=True, help="File to save computed probabilities")
parser.add_argument("--config_file", type=str, required=True, help="Config file for the model")
parser.add_argument("--model", type=str, required=True, help="Path to the model weights")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for probability computation")
parser.add_argument("--parallel",    type=str2bool,    default=True,    help="If true the evaluation will be done in batch and parallelise on GPU",)
parser.add_argument("--fast", type=str2bool, default=False, help="Use only 1000 for debug and fast test")
args = parser.parse_args()

logger = setup_logger_basic()
logger.info("Program: eval.py ****")

@dataclass
class MNLIInputFeatures:
    id: str
    premise: str
    hypothesis_true: str
    hypothesis_false: List[str]
    relation: str
    probabilities: List[int]

    def to_dict(self):
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
    def __init__(
        self,
        mnli_data: List[MNLIInputFeatures],
        tokenizer: AutoTokenizer,
        number_relation=11,
    ):
        self.mnli_data = mnli_data
        self.tokenizer = tokenizer
        self.number_relation = number_relation

    def __len__(self):
        return len(self.mnli_data)

    def __getitem__(self, idx):
        data = self.mnli_data[idx]
        inputs = []
        true_hypothesis = data.hypothesis_true
        inputs.append((data.premise, true_hypothesis, config["label2id"]["entailment"]))
        
        for hf in data.hypothesis_false:
            inputs.append((data.premise, hf, config["label2id"]["not_entailment"]))
        
        return inputs, data.relation


def load(input_file: str, fast: bool) -> List[MNLIInputFeatures]:
    mnli_data = []
    data = json.load(open(input_file, "rt"))
    logger.info(f"Data : Input file : {input_file}")
    if fast:
        logger.warning(
            "\n!!! YOU ARE USING THE FAST TRAINING MODE ONLY 1000 WILL BE USED !!! (this mode is used for debug)\n"
        )
        data = data[0:10]

    mnli_data = [
        MNLIInputFeatures(
            id=sha1(line["premise"] + line["hypothesis_true"]),
            premise=line["premise"],
            hypothesis_true=line["hypothesis_true"],  # the true
            hypothesis_false=line["hypothesis_false"],  # all the false possible
            relation=line["relation"],
            probabilities=[],
        )
        for line in data
    ]
    assert len(mnli_data) > 0, "The data are empty"
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
    number_relation,
    device,
    batch_size=32,
):
    """
    Compute probabilities for each premise-hypothesis pair and map them back to their respective data.
    """
    dataset = MNLIDataset(mnli_data, tokenizer, number_relation)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    model.to(device)
    model.eval()

    idx = 0  # Index to track which MNLIInputFeatures we're processing

    for premises, hypotheses, _, relations in tqdm(
        dataloader, desc="Computing Probabilities:"
    ):
        inputs = tokenizer(
            premises, hypotheses, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"])
            batch_probs = (
                F.softmax(outputs["logits"], dim=-1)[:, 0].cpu().tolist()
            )  # Probability of entailment

        # Process batch probabilities
        batch_size = len(relations)  # Number of data items in this batch
        for i in range(batch_size):
            # Fetch the current MNLIInputFeatures item
            data_item = mnli_data[idx]

            num_hypotheses = (
                len(data_item.hypothesis_false) + 1
            )  # Include the true hypothesis
            item_probs = batch_probs[i * num_hypotheses : (i + 1) * num_hypotheses]

            mnli_data[idx].probabilities = item_probs
            idx += 1  # Increment index for the next item in mnli_data

    return mnli_data


if __name__ == "__main__":
    config = get_config(args.config_file)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], use_fast=False, model_max_length=512
    )
    model = AutoModelForSequenceClassification.from_pretrained(pathlib.Path(args.model))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Device : {device}")
    if not (torch.cuda.is_available()) and args.parallel:
        args.parallel = False
        logger.warning(
            "CPU devise, no parralelization possible - evaluation will be done sequentially"
        )

    mnli_data = load(args.input_file, args.fast)
    number_relation = 11
    mnli_data = compute_probabilities(
        mnli_data, tokenizer, model, number_relation, device, args.batch_size
    )

    # Convert `mnli_data` to a list of dictionaries
    mnli_data_serializable = [item.to_dict() for item in mnli_data]

    # Save to the output file
    name = args.model.split("/")[-1] if args.saving_name is None else args.saving_name
    output_file = (
        os.path.dirname(__file__) if args.output_file is None else args.output_file
    )
    output_file = Path(f"{output_file}/proba_{name}")
    with open(args.output_file, "w") as f:
        json.dump(mnli_data_serializable, f, indent=2)
    print(f"Probabilities saved to {output_file}")
