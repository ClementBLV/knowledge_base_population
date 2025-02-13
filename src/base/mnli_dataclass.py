
import json
from logging import Logger
from typing import Dict, List, Type, Union
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.utils.utils import sha1


@dataclass
class EvalInputFeatures:
    premise: str  # context
    hypothesis_true: str
    hypothesis_false: List[str]
    relation: str

@dataclass
class PredictionInputFeatures (EvalInputFeatures):
    id: str
    probabilities: torch.Tensor
    predictions: List[int]

    def to_dict(self) -> Dict:
        """Converts the instance to a dictionary for JSON serialization."""
        return {
            "id": self.id,
            "premise": self.premise,
            "hypothesis_true": self.hypothesis_true,
            "hypothesis_false": self.hypothesis_false,
            "relation": self.relation,
            "probabilities": self.probabilities.tolist(),
            "predictions":self.predictions
        }

@dataclass
class MetaPredictionInputFeatures (PredictionInputFeatures):
    fused_probability: List[List[float]]
    
    def to_dict(self) -> Dict:
        """Converts the instance to a dictionary for JSON serialization."""
        return {
            "id": self.id,
            "premise": self.premise,
            "hypothesis_true": self.hypothesis_true,
            "hypothesis_false": self.hypothesis_false,
            "relation": self.relation,
            "fused_probability": self.fused_probability,
            "probabilities": self.probabilities.tolist(),
            "predictions":self.predictions
        }


class MNLIDataset(Dataset):
    def __init__(self, mnli_data: List[PredictionInputFeatures], tokenizer: AutoTokenizer, config):
        self.mnli_data = mnli_data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.mnli_data)

    def __getitem__(self, idx):
        data = self.mnli_data[idx]
        inputs = [(data.premise, data.hypothesis_true, self.config["label2id"]["entailment"])]

        for hf in data.hypothesis_false:
            inputs.append((data.premise, hf, self.config["label2id"]["not_entailment"]))

        return inputs, data.relation




def load_data(input_file: str, fast: bool, logger) -> List[PredictionInputFeatures]:
    """Load JSON raw data from the input file of the evaluation 
    and convert it to MNLIInputFeatures for either direct or indirect eval."""
    with open(input_file, "r") as f:
        data = json.load(f)

    logger.info(f"Loading input file: {input_file}")
    
    if fast:
        logger.warning("Fast mode enabled: Using only 1000 samples for debugging.")
        data = data[:1000]

    mnli_data = [
        PredictionInputFeatures(
            id=sha1(line["premise"] + line["hypothesis_true"][0]),
            premise=line["premise"],
            hypothesis_true=line["hypothesis_true"][0],
            hypothesis_false=line["hypothesis_false"],
            relation=line["relation"],
            probabilities=torch.tensor([], dtype=torch.float),
            predictions=[]
        )
        for line in data
    ]

    if not mnli_data:
        raise ValueError("Loaded data is empty!")

    return mnli_data

def load_predictions(
    prediction_file: str, 
    logger: Logger, 
    type_: Type[Union[MetaPredictionInputFeatures, PredictionInputFeatures]]
) -> List[PredictionInputFeatures]:
    """Loads computed probabilities and predictions from a JSON file into objects."""
    
    logger.info(f"LOAD: Attempting to load file {prediction_file}")

    try:
        with open(prediction_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"ERROR: File {prediction_file} not found.")
        return []
    except json.JSONDecodeError:
        logger.error(f"ERROR: Failed to decode JSON from {prediction_file}.")
        return []

    required_keys = {"id", "premise", "hypothesis_true", "hypothesis_false", "relation", "probabilities", "predictions"}
    if type_ is MetaPredictionInputFeatures:
        required_keys.add("fused_probability")

    mnli_data = []
    for item in data:
        if not required_keys.issubset(item.keys()):
            logger.warning(f"WARNING: Skipping item due to missing keys: {item}")
            continue
        
        mnli_data.append(type_(**item))

    logger.info(f"LOAD: Successfully loaded {len(mnli_data)} predictions from {prediction_file}")
    
    return mnli_data
