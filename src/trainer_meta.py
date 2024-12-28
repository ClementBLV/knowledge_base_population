from argparse import ArgumentParser
import json
import logging
import os
from pprint import pprint
import sys
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from meta_models import MetaModelNN, VotingModel

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
parser.add_argument("--config_file", type=str, required=True, 
                    help="Name if the config file of for the meta model")
args = parser.parse_args()

################ setup : config ################
current_dir = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(current_dir), "configs", args.config_file)
with open(config_path, "r") as config_file:
    config = json.load(config_file)
    
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
            ent_indx = config["label2id"]["entailment"]
            l.extend([df.iloc[i][p][0][ent_indx]])
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

################ training ################
# Parameters
num_models = 4
num_classes = 1

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
dummy_x = torch.tensor([[0.8, 0.4,0.7,0.6],
                        [0.2, 0.3, 0.1, 0.9]])
voting_output = voting_model(dummy_x)
meta_model_output = meta_model(dummy_x,flattened=False)

logger.info(f"Input : \n\t{dummy_x}")
logger.info(f"Processed Input : \n\t{dummy_x}")
logger.info(f"Voting Model Output:\n\t{voting_output}")
logger.info(f"MetaModelNN Output: \n\t{meta_model_output}")


logger.info(f"Save : model saved at {args.output_dir}.")
