from argparse import ArgumentParser
import logging
from pprint import pprint
import sys
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

################ setup : logger ################
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.info("Progam : trainer_meta_binary.py ****")

################ setup : parser ################
parser = ArgumentParser()
parser.add_argument("--input_file", type=str, required=True,
                    help="Path to the training data, must be a csv file with the four columns p{1..4} and a label column with the ground truth")
parser.add_argument("--num_epochs", type=int)
parser.add_argument('-output_dir', '--output_dir',type=str, required=True,
                    help='Directroy to save the outputs (log - weights)')
args = parser.parse_args()


################ setup : dataframe ################
df = pd.read_json(args.input_file, orient="records", lines=True)

# Extract labels
y = df['label'].values
X = []
filtered_y = []  # Use this to store corresponding labels

for i in range(len(df)):
    l = []
    for p in ['p1', 'p2', 'p3', 'p4']:
        if df.iloc[i][p] is not None:
            l.extend(df.iloc[i][p][0])
    if l:  # Append only if 'l' is non-empty
        X.append(l)
        filtered_y.append(y[i])  # Append the corresponding label

# Ensure resulting shapes are valid
assert len(X) == len(filtered_y), f"X and y lengths mismatch: {len(X)} != {len(filtered_y)}"
logger.info(f"Example of the training data: \n\n\tX vector: {X[0]} \n\ty vector: {filtered_y[0]}\n")

################ setup : tensor ################
X_tensor = torch.tensor(X, dtype=torch.float32)  # Shape: (n_samples, 8)
y_tensor = torch.tensor(filtered_y, dtype=torch.float32)  # Shape: (n_samples,)

print("Shape of X_tensor:", X_tensor.shape)  # Should be (500, 8)
print("Shape of y_tensor:", y_tensor.shape)  # Should be (500,)


################ setup : datastructure ################
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

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
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


################ training ################
# Parameters
num_models = 4
num_classes = 2

# Create instances of the models
voting_model = VotingModel(num_models=num_models, num_classes=num_classes)
meta_model = MetaModelNN(num_models=num_models, num_classes=num_classes)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(meta_model.parameters(), lr=0.001)

# Training loop
num_epochs = args.num_epochs
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        # Forward pass
        outputs = meta_model(inputs).squeeze()
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print average loss per epoch
    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

logger.info("Training complete.")
logger.info("Small test.")
torch.save(meta_model.state_dict(), "meta_model.pth")

# Example usage
dummy_x = torch.tensor([[[0.8, 0.2], [0.4, 0.6], [0.7, 0.3], [0.6, 0.4]],
                        [[0.2, 0.8], [0.5, 0.5], [0.3, 0.7], [0.1, 0.9]]])

voting_output = voting_model(dummy_x)
meta_model_output = meta_model(dummy_x,flattened=False)

logger.info(f"Input : \n\t{dummy_x}")
logger.info(f"Voting Model Output:\n\t{voting_output}")
logger.info(f"MetaModelNN Output: \n\t{meta_model_output}")


logger.info(f"Save : model saved at {args.output_dir}.")
