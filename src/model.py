from argparse import ArgumentParser
import logging
import sys
import torch
import torch.nn as nn
from transformers import DebertaModel

from src.meta.meta_models import MetaModelNN, VotingModel

################ setup : logger ################
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.info("Progam : model.py ****")


class CombinedModel(nn.Module):
    def __init__(self, deberta_paths, meta_model_type:str):
        super(CombinedModel, self).__init__()
        
        # Load the four DeBERTa models from checkpoints
        self.deberta_models = nn.ModuleList([
            DebertaModel.from_pretrained(path) for path in deberta_paths
        ])
        
        # Freeze the DeBERTa models if you don’t want to fine-tune them further
        for model in self.deberta_models:
            for param in model.parameters():
                param.requires_grad = False
        
        # Load the meta-model as a final layer
        if meta_model_type == "meta":
            logger.info("Meta model is used")
            # Load the trained meta-model
            meta_model = MetaModelNN()
            meta_model.load_state_dict(torch.load("meta_model.pth"))
            self.top_model = meta_model
        elif meta_model_type == "vote":
            logger.info("Voting stategy is used")
            voting = VotingModel()
            self.top_model = voting
        else : 
            logger.warning("Unknown final model, must be either 'meta' or 'vote' default voting stategy will be used")
            voting = VotingModel()
            self.top_model = voting

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Assume input_ids, attention_mask, and token_type_ids are prepared for each DeBERTa model
        probabilities = []
        
        # Get predictions from each DeBERTa model
        for model in self.deberta_models:
            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # Assuming the model’s CLS token output can be converted to probability
            # Here we take the CLS token embedding and pass it through a linear layer + sigmoid
            cls_output = output.last_hidden_state[:, 0, :]  # CLS token output
            prob = torch.sigmoid(cls_output).mean(dim=-1)  # Averaging to get a single probability
            probabilities.append(prob)
        
        # Stack probabilities from all DeBERTa models
        probabilities = torch.stack(probabilities, dim=1)  # Shape: (batch_size, 4)
        
        # Pass through the meta-model
        final_output = self.top_model(probabilities).squeeze(-1)  # Final prediction

        return final_output


# Example forward pass with dummy data
# input_ids, attention_mask, and token_type_ids should be prepared for each input (h, p)
# You can prepare these using a tokenizer specific to DeBERTa
# output = combined_model(input_ids, attention_mask, token_type_ids)
