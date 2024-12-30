import torch
from torch.utils.data import DataLoader, TensorDataset

def evaluate_model(model, X_tensor, y_tensor):

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs).squeeze()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy
