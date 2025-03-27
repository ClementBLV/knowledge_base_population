import torch

# Helper functions
def row_means(tensor):
    return tensor.mean(dim=1)

def max_row(tensor):
    return tensor.max(dim=1).values  # Extract max values per row

def one_hot_max(tensor):
    max_indices = tensor.argmax(dim=0, keepdim=True)
    
    # Create a tensor of zeros with the same shape
    one_hot = torch.zeros_like(tensor)
    # Scatter 1s at the max indices
    one_hot[max_indices]+=1
    return one_hot


def max_column(tensor):
    c0 = one_hot_max(tensor[:, 0:1].flatten())  # Keep as (batch_size, 1) shape
    c1 = one_hot_max(tensor[:, 1:2].flatten())
    c2 = one_hot_max(tensor[:, 2:3].flatten())
    c3 = one_hot_max(tensor[:, 3:4].flatten())
    final = c0 + c1 + c2 + c3
    return final


def edge_case(tensor):
  max_value = tensor.max()
  max_count = (tensor == max_value).sum().item()
  return max_count > 1