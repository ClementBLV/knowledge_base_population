import torch
import torch.nn as nn

################ setup : models ################
class VotingModel(nn.Module):
    def __init__(self, num_models, num_classes):
        """
        Args:
            num_models (int): Number of sub-models (e.g., p1, p2, p3, p4).
            num_classes (int): Number of classes for each sub-model (e.g., 2 for entail and contradict).
        """
        super(VotingModel, self).__init__()
        self.num_models = num_models
        self.num_classes = num_classes

    def forward(self, x):
        # x is expected to have shape (batch_size, num_models * num_classes), flattened probabilities
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_models, self.num_classes)  # Reshape to (batch_size, num_models, num_classes)
        
        # Perform voting
        votes = torch.argmax(x, dim=-1)  # Get the index of the max probability (0 for entail, 1 for contradict)
        # Find the most common label (entail or contradict) in each batch
        majority, _ = torch.mode(votes, dim=1)  # Mode gives the most frequent element
        return majority.unsqueeze(-1)  

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
    def load_meta_model(config_meta, device="cpu"):
        """
        Load a trained meta model.

        Args:
            config_meta: Object containing model parameters and path.
            device (str): Device to load the model on ("cpu" or "cuda").

        Returns:
            MetaModelNN: Loaded model.
        """
        model = MetaModelNN(num_models=config_meta["num_models"], num_classes=config_meta["num_classes"])
        model.load_state_dict(torch.load(config_meta.model_path, map_location=torch.device(device)))
        model.to(device)  # Move model to the specified device
        model.eval()  # Set model to evaluation mode
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
