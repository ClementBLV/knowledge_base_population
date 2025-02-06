import json
from logging import Logger
from typing import Dict, List

import torch
from src.base.compute_probabilities import compute_prediction
from src.base.mnli_dataclass import MetaPredictionInputFeatures, PredictionInputFeatures
from src.meta.meta_models import DummyMetaModelNN, MetaModelNN

USE_META_DUMMY = True  # Set to True to use the dummy meta model

def load_meta_model(config_meta, device, logger):
    """Loads either the real or dummy meta model based on the flag."""
    num_models = config_meta["num_models"]
    num_classes = config_meta["num_classes"]
    
    if USE_META_DUMMY:
        logger.info("Using Dummy Meta Model")
        return DummyMetaModelNN(num_models=num_models, num_classes=num_classes)
    else:
        logger.info("Using Real Meta Model")
        return  MetaModelNN.load_meta_model(config_meta, device=device)    


def compute_meta_probabilities(
        aggregated_prob : List[MetaPredictionInputFeatures], 
        config_meta: Dict,
        meta_proba_file: str, 
        logger: Logger,
    )-> List[MetaPredictionInputFeatures]: 
    
    # Load the meta model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta_model = load_meta_model(config_meta, device=device, logger=logger)
    
    for data_item in aggregated_prob:
        # Convert fused probabilities to a tensor
        input_tensor = torch.tensor(data_item.fused_probability, dtype=torch.float32)

        # Ensure correct shape: (batch_size, num_models * num_classes)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension if needed

        # Forward pass through the model
        with torch.no_grad():  # No need for gradients during inference
            probabilities = meta_model(input_tensor).squeeze().tolist()  # Convert back to list
        
        # Assign probabilities to the object
        data_item.probabilities = probabilities
        data_item.predictions = compute_prediction(data_item)

        mnli_data_serializable = [item.to_dict() for item in aggregated_prob]

        with open(meta_proba_file, "w") as f:
            json.dump(mnli_data_serializable, f, indent=2)
        
        logger.info(f"Probabilities saved to {meta_proba_file}")
    return mnli_data_serializable


def aggregate_probabilities(
        direct: List[PredictionInputFeatures],
        reverse: List[PredictionInputFeatures],
        both_direct: List[PredictionInputFeatures],
        both_reverse: List[PredictionInputFeatures]) -> List[MetaPredictionInputFeatures]:
    """Aggregate probabilities and return a list of MetaPredictionInputFeatures."""

    meta_predictions = []

    for p_direct, p_reverse, p_both_direct, p_both_reverse in zip(direct, reverse, both_direct, both_reverse):
        # Ensure all IDs match before proceeding
        if len({p_direct.id, p_reverse.id, p_both_direct.id, p_both_reverse.id}) == 1:
            fused_probabilities = list(zip(
                p_direct.probabilities,
                p_reverse.probabilities,
                p_both_direct.probabilities,
                p_both_reverse.probabilities
            ))  # [[p11, p12, p13, p14], [p21, p22, p23, p24], ...]

            # Create a MetaPredictionInputFeatures instance using attributes from p_direct
            meta_prediction = MetaPredictionInputFeatures(
                **p_direct.__dict__,  # Copy all attributes from p_direct
                fused_probability=fused_probabilities  # Add new attribute
            )

            meta_predictions.append(meta_prediction)

    return meta_predictions