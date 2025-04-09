import torch
import torch.nn as nn
from src.utils.helper_functions import *

################ setup : models ################
class VotingModel(nn.Module):
    def __init__(self, num_models, num_classes, strategy="max_row"):
        """
        Args:
            num_models (int): Number of sub-models (e.g., p1, p2, p3, p4).
            num_classes (int): Number of classes for each sub-model (e.g., 2 for entail and contradict).
            strategy (str): Voting strategy - "average", "max_row", or "max_column".
        """
        super(VotingModel, self).__init__()
        self.num_models = num_models
        self.num_classes = num_classes
        self.strategy = strategy  # Store the strategy

    def forward(self, x):
        # x is expected to have shape (batch_size, num_models * num_classes), flattened probabilities
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_models, self.num_classes)  # Reshape to (batch_size, num_models, num_classes)

        if self.strategy == "average":
            x = row_means(x).t()[0]  # Average probabilities per class
            if edge_case(x):
              x = max_row(x).t()[0]
            votes = one_hot_max(x)

        elif self.strategy == "max_row":
            x = max_row(x).t()[0]  # Get max probability per row
            votes = one_hot_max(x)

        elif self.strategy == "max_column":
            t = max_column(x)  # Apply column-wise max voting
            if edge_case(t):
              t = row_means(x)
              if edge_case(t):
                t = max_row(x).t()[0]
            votes = one_hot_max(t)

        else:
            raise ValueError(f"Invalid strategy: {self.strategy}. Choose from 'average', 'max_row', or 'max_column'.")
        return votes.unsqueeze(-1)  # Keep shape (batch_size, 1)

class MetaModelNN(nn.Module):
    def __init__(self, num_models, num_classes, hidden_scale=2):
        """
        Args:
            num_models (int): Number of sub-models (e.g., p1, p2, p3, p4).
            num_classes (int): Number of classes for each sub-model (e.g., 2 for entail and contradict).
            hidden_scale (int): Multiplier for the size of the hidden layer.
        """
        super(MetaModelNN, self).__init__()
        
        input_size = num_models * num_classes
        hidden_size = hidden_scale * input_size
        self.num_models = num_models
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.fc3 = nn.Linear(input_size, 1)  # Output layer for binary classification
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # For binary probability output

    def forward(self, x, flattened=True):
        if not flattened: 
            batch_size = x.size(0)
            # x is is not already flattened, it is expected to have shape 
            # (batch_size, num_models * num_classes), flattened probabilities
            x = x.view(batch_size, self.num_models, self.num_classes)
            # Flatten back to (batch_size, input_size)
            x = x.view(batch_size, -1)
        #print(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


    @staticmethod
    def load_meta_model(config_meta, device):
        """
        Load a trained meta model.

        Args:
            config_meta: Object containing model parameters and path.
            device (str): Device to load the model on ("cpu" or "cuda").

        Returns:
            MetaModelNN: Loaded model.
        """
        model = MetaModelNN(num_models=config_meta["num_models"], num_classes=config_meta["num_classes"])
        model.load_state_dict(torch.load(config_meta["meta_model_path"], map_location=torch.device(device)))
        model.to(device)  # Move model to the specified device
        model.eval()  # Set model to evaluation mode
        return model
    



class GlobalRelationMetaModelNN(nn.Module):
    def __init__(self, num_relations=11, num_models=4, hidden_scale=2):
        """
        Args:
            num_relations (int): Number of relations (e.g., 11).
            num_models (int): Number of models per relation (e.g., 4).
            hidden_scale (int): Hidden layer multiplier.
        """
        super(GlobalRelationMetaModelNN, self).__init__()

        input_size = num_relations * num_models  # Whole (11x4) matrix as input
        hidden_size = hidden_scale * input_size

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.fc3 = nn.Linear(input_size, num_relations)  # Output size = 11

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Ensure only one "1" in output

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (batch_size, 11, 4) - Full matrix input.

        Returns:
            Tensor: Shape (batch_size, 11, 1) - One-hot vector for relation selection.
        """
        batch_size = x.size(0)

        # Flatten input from (batch_size, 11, 4) → (batch_size, 44)
        x = x.view(batch_size, -1)

        # Pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output logits

        # Apply softmax to enforce a single "1"
        x = self.softmax(x)

        # Reshape to (batch_size, 11, 1)
        x = x.unsqueeze(-1)

        return x  # One-hot vector: (batch_size, 11, 1)

    @staticmethod
    def load_meta_model(config_meta, device):
        """
        Load a trained meta model.

        Args:
            config_meta (dict): Contains model parameters and path.
            device (str): Device to load the model on ("cpu" or "cuda").

        Returns:
            GlobalRelationMetaModelNN: Loaded model.
        """
        model = GlobalRelationMetaModelNN(
            num_relations=config_meta["num_relations"],
            num_models=config_meta["num_models"],
            hidden_scale=config_meta.get("hidden_scale", 2)  # Default hidden_scale = 2
        )
        
        # Load model weights
        model.load_state_dict(torch.load(config_meta["meta_model_path"], map_location=torch.device(device)))
        
        # Move to device and set to eval mode
        model.to(device)
        model.eval()
        
        return model

class DummyModel:
    """A dummy model that generates random probabilities."""
    def __init__(self, num_labels=3, num_classes=11):  # Assume 3 classes (entailment, neutral, contradiction)
        self.num_labels = num_labels
        self.num_classes = num_classes

    def to(self, device):
        pass  # No-op for dummy model

    def eval(self):
        pass  # No-op for dummy model

    def __call__(self, input_ids):
        batch_size = input_ids.shape[0]
        #random_prediction = torch.randperm(self.num_classes).float()  # Get shuffled indices as floats
        #random_probs = (random_prediction / self.num_classes).unsqueeze(0).expand(batch_size, -1)
        random_probs = torch.rand((batch_size, self.num_labels))  # Random probabilities
        return {"logits": random_probs}

class DummyMetaModelNN(nn.Module):
    """A dummy version of MetaModelNN that outputs random probabilities."""
    def __init__(self, num_models, num_classes):
        super(DummyMetaModelNN, self).__init__()
        self.num_models = num_models
        self.num_classes = num_classes

    def forward(self, x, flattened=True):
        batch_size = x.size(0)
        # Random binary probabilities
        random_probs = torch.rand(batch_size, 1)
        return random_probs
    
    @staticmethod
    def load_meta_model(config_meta, device="cpu"):
        """
        Load a trained meta model.

        Args:
            config_meta: Object containing model parameters and path.
            device (str): Device to load the model on ("cpu" or "cuda").

        Returns:
            MetaModelNN: Loaded model.
        """
        model = DummyMetaModelNN(num_models=config_meta["num_models"], num_classes=config_meta["num_classes"])
        return model
# %%
