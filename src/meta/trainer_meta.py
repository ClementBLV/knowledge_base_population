import logging
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from meta_models import MetaModelNN, VotingModel, GlobalRelationMetaModelNN

def train_model(X_tensor , y_tensor, num_epochs, config_file, type_model):

    # Setup
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Program: trainer_meta.py ****")
    ################ setup : datastructure ################
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    ################ training ################
    # Initialize models and training tools
    if type_model=="global":
        meta_model = GlobalRelationMetaModelNN(num_relations=11, num_models=4, num_classes=1)
    else :
        meta_model = MetaModelNN(num_models=4, num_classes=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(meta_model.parameters(), lr=0.001)

    # Train
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            outputs = meta_model(inputs).squeeze()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    logger.info("Training complete.")
    return meta_model
