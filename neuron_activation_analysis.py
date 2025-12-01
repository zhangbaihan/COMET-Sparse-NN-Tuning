import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from tqdm import tqdm

# Import your model factories
from models.COMET import get_COMET
from models.COMET_Orth import get_Orthogonal
from loading_datasets import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(model_type, checkpoint_path, args):
    """
    Instantiates the model and loads weights from a checkpoint.
    """
    if model_type == 'COMET':
        model = get_COMET(args.dataset, args.neurons, args.neurons, args.neurons, 10, 
                          args.topk, args.norm, args.activation)
    elif model_type == 'Orthogonal':
        model = get_Orthogonal(args.dataset, args.neurons, args.neurons, args.neurons, 10, 
                               args.topk, args.norm, args.activation)
    else:
        raise ValueError("Unknown model type")

    print(f"Loading {model_type} from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def get_class_indices(loader, target_class_idx):
    """
    Returns indices of all samples in the loader that belong to target_class_idx.
    """
    indices = []
    current_idx = 0
    for _, y in loader:
        batch_indices = (y == target_class_idx).nonzero(as_tuple=True)[0]
        indices.append(batch_indices + current_idx)
        current_idx += y.size(0)
    return torch.cat(indices)

def collect_masks(model, loader, target_class_idx=None):
    """
    Runs inference and collects binary masks from Layer 1.
    If target_class_idx is set, only collects masks for that specific class (e.g., Ships).
    """
    masks = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Collecting Masks"):
            x = x.to(device)
            
            # Filter by class if requested
            if target_class_idx is not None:
                mask_indices = (y == target_class_idx).nonzero(as_tuple=True)[0]
                if len(mask_indices) == 0:
                    continue
                x = x[mask_indices]
            
            # Forward pass (activates hooks/internal state)
            _ = model(x)
            
            # Retrieve the mask from the model's internal state
            # (Assuming model saves self.layer_1_mask as in your COMET.py)
            batch_mask = model.layer_1_mask.cpu() # Shape: [Batch, Neurons]
            masks.append(batch_mask)
            
    return torch.cat(masks, dim=0)

def calculate_metrics(masks):
    """
    Computes utilization metrics for a tensor of masks [N_samples, N_neurons]
    """
    n_samples, n_neurons = masks.shape
    
    # 1. Activation Frequency per Neuron
    # How often did each neuron fire? (0.0 to 1.0)
    neuron_freqs = masks.mean(dim=0)
    
    # 2. Dead Neurons (Never fired)
    dead_count = (neuron_freqs == 0).sum().item()
    
    # 3. Always-On Neurons (Fired > 90% of the time)
    # These are likely "background detectors"
    always_on_count = (neuron_freqs > 0.9).sum().item()
    
    # 4. Pairwise Jaccard Similarity (Subset of data for speed)
    # Measures how similar the masks are between different images.
    # High Jaccard = Collapse (All ships look the same to the model)
    subset = masks[:1000] # Use first 1000 for speed
    intersection = torch.mm(subset, subset.t())
    union = subset.sum(dim=1).unsqueeze(1) + subset.sum(dim=1).unsqueeze(0) - intersection
    jaccard = intersection / (union + 1e-8)
    
    # Get upper triangle only (exclude diagonal)
    triu_indices = torch.triu_indices(len(subset), len(subset), offset=1)
    avg_jaccard = jaccard[triu_indices[0], triu_indices[1]].mean().item()
    
    return {
        'freqs': neuron_freqs.numpy(),
        'dead': dead_count,
        'always_on': always_on_count,
        'jaccard': avg_jaccard
    }

def plot_utilization(metrics_comet, metrics_orth, class_name):
    """
    Plots histograms comparing neuron utilization.
    """
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Histogram of Activation Frequencies
    plt.subplot(1, 2, 1)
    sns.histplot(metrics_comet['freqs'], color='red', alpha=0.5, label='COMET (Gaussian)', bins=50)
    sns.histplot(metrics_orth['freqs'], color='blue', alpha=0.5, label='COMET (Orthogonal)', bins=50)
    plt.xlabel("Activation Frequency (0=Dead, 1=Always On)")
    plt.ylabel("Count of Neurons")
    plt.title(f"Neuron Utilization Profile for '{class_name}'")
    plt.legend()
    
    # Plot 2: Text Stats
    plt.subplot(1, 2, 2)
    plt.axis('off')
    text_str = f"--- {class_name} Analysis ---\n\n"
    
    text_str += "COMET (Gaussian):\n"
    text_str += f"  Dead Neurons: {metrics_comet['dead']}\n"
    text_str += f"  'Always-On' (>90%): {metrics_comet['always_on']}\n"
    text_str += f"  Mask Similarity (Jaccard): {metrics_comet['jaccard']:.3f}\n\n"
    
    text_str += "COMET (Orthogonal):\n"
    text_str += f"  Dead Neurons: {metrics_orth['dead']}\n"
    text_str += f"  'Always-On' (>90%): {metrics_orth['always_on']}\n"
    text_str += f"  Mask Similarity (Jaccard): {metrics_orth['jaccard']:.3f}\n"
    
    plt.text(0.1, 0.5, text_str, fontsize=14, family='monospace', va='center')
    
    save_path = f"activations_{class_name}.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    # Paths to your saved .pth models
    parser.add_argument("--comet_path", type=str, required=True, help="Path to standard COMET .pth")
    parser.add_argument("--orth_path", type=str, required=True, help="Path to Orthogonal COMET .pth")
    
    # Model Config (Must match training!)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--neurons", type=int, default=3000)
    parser.add_argument("--topk", type=float, default=0.5)
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--activation", type=str, default="softplus")
    parser.add_argument("--batch_size", type=int, default=128)
    
    args = parser.parse_args()

    # 1. Load Data (Ships are Index 8 in CIFAR-10)
    _, val_loader, _, _ = get_data_loaders(args.dataset, args.batch_size)
    ship_index = 8 
    
    # 2. Load Models
    model_comet = load_trained_model('COMET', args.comet_path, args)
    model_orth = load_trained_model('Orthogonal', args.orth_path, args)
    
    # 3. Collect Masks for Ships
    print("\nCollecting masks for SHIP class...")
    masks_comet = collect_masks(model_comet, val_loader, ship_index)
    masks_orth = collect_masks(model_orth, val_loader, ship_index)
    
    # 4. Calculate Stats
    print("Calculating metrics...")
    metrics_comet = calculate_metrics(masks_comet)
    metrics_orth = calculate_metrics(masks_orth)
    
    # 5. Visualize
    plot_utilization(metrics_comet, metrics_orth, "Ship")

if __name__ == "__main__":
    main()