from argparse import ArgumentParser
import logging
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

################ setup : logger ################
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.info("Progam : hf_trainer.py ****")

################ setup : parser ################
parser = ArgumentParser()
parser.add_argument("--input_file", type=str, 
                    help="Path to the training data, must be a csv file with the four columns p{1..4} and a label column with the ground truth")
parser.add_argument("--num_epochs", type=int)
parser.add_argument('-output_dir', '--output_dir',type=str, required=True,
                    help='Directroy to save the outputs (log - weights)')
args = parser.parse_args()


################ setup : dataframe ################
df = pd.read_csv(args.input_file)
X = df[['p1', 'p2', 'p3', 'p4']].values  # Features (probabilities)
y = df['label'].values                   # Labels

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create a DataLoader for batching
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

class VotingModel(): 
    def __init__(self):
        logger.warning("NOT IMPLEMEMENTED YET")
        pass
        

# Define a simple neural network
class MetaModelNN(nn.Module):
    def __init__(self):
        super(MetaModelNN, self).__init__()
        self.fc1 = nn.Linear(4, 8)   # 4 inputs (p1, p2, p3, p4), 8 hidden units
        self.fc2 = nn.Linear(8, 4)   # 4 units in the hidden layer
        self.fc3 = nn.Linear(4, 1)   # Output layer for binary classification
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # For binary probability output

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize model, loss function, and optimizer
model = MetaModelNN()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = args.num_epochs
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        # Forward pass
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print average loss per epoch
    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

logger.info("Training complete.")
torch.save(model.state_dict(), "meta_model.pth")
logger.info(f"Save : model saved at {args.output_dir}.")
