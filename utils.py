import os
import math
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import scipy.io
import matplotlib.pyplot as plt

# --- Utility Functions ---

def get_max_and_min_arrays(list_of_seeds_array):
    """
    Compute max, min, and average variance across multiple runs.

    Args:
        list_of_seeds_array (List[List[float]]): A list of lists containing metric values (e.g., accuracy over epochs) from different seeds.

    Returns:
        Tuple[List[float], List[float], float]:
            - max_values: element-wise maximum across seeds
            - min_values: element-wise minimum across seeds
            - avg_variance: average variance across all points
    """
    max_values = []
    min_values = []
    variance_values = []

    for idx in range(len(list_of_seeds_array[0])):
        values_at_idx = [run[idx] for run in list_of_seeds_array]
        max_values.append(max(values_at_idx))
        min_values.append(min(values_at_idx))
        variance_values.append(np.var(values_at_idx))

    avg_variance = np.mean(variance_values)
    return max_values, min_values, avg_variance


def mask_topk(tensor, topk):
    """
    Generate a binary mask keeping top-k elements (row-wise) in the last dimension.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, rows, cols).
        topk (float): Proportion (0 < topk <= 1) of elements to keep.

    Returns:
        torch.Tensor: Mask tensor of same shape with 1s for top-k values and 0s elsewhere.
    """
    k = max(1, int(tensor.shape[-1] * topk))
    top_values, _ = tensor.topk(k, dim=-1)
    threshold = top_values[..., -1].unsqueeze(-1)
    mask = tensor >= threshold
    return mask.float()
