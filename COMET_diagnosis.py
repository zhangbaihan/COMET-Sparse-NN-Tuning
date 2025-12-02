import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr

# Import factories
try:
    from models.COMET import get_COMET
    from models.COMET_Orth import get_Orthogonal
except ImportError:
    from COMET import get_COMET
    from models.COMET_Orth import get_Orthogonal

from loading_datasets import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_type, checkpoint_path, args):
    if model_type == 'COMET':
        model = get_COMET(args.dataset, args.neurons, args.neurons, args.neurons, 10, args.topk, args.norm, args.activation)
    elif model_type == 'Orthogonal':
        model = get_Orthogonal(args.dataset, args.neurons, args.neurons, args.neurons, 10, args.topk, args.norm, args.activation)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def get_neuron_selectivity(model, loader, num_classes=10):
    """
    Analyzes which class each neuron 'prefers'.
    Returns: A matrix [Neurons, Classes] representing activation frequency per class.
    """
    # Hook to capture masks
    masks = []
    labels = []
    
    def hook_fn(module, input, output):
        # We need to compute the mask from the output projection
        # This assumes output is the projection c_l
        k = max(1, int(output.shape[-1] * model.top_k))
        topk_values, _ = torch.topk(output, k, dim=-1)
        threshold = topk_values[:, -1].unsqueeze(-1)
        mask = (output >= threshold).float()
        masks.append(mask.cpu())

    handle = model.spec_1.register_forward_hook(hook_fn)

    print("Analyzing Neuron Selectivity...")
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            _ = model(x)
            labels.append(y.cpu())
            
    handle.remove()
    
    all_masks = torch.cat(masks, dim=0) # [N, Neurons]
    all_labels = torch.cat(labels, dim=0) # [N]
    
    num_neurons = all_masks.shape[1]
    class_freqs = torch.zeros(num_neurons, num_classes)
    
    # Calculate P(Active | Class)
    for c in range(num_classes):
        class_indices = (all_labels == c).nonzero(as_tuple=True)[0]
        if len(class_indices) > 0:
            class_masks = all_masks[class_indices]
            # Average activation for this class
            class_freqs[:, c] = class_masks.mean(dim=0)
            
    return class_freqs

def analyze_background_correlation(model, loader):
    """
    Checks if mask similarity is correlated with background color similarity.
    Focuses on SHIP class (Index 8) vs PLANE class (Index 0).
    """
    print("Analyzing Background Correlation (Ship vs Plane)...")
    
    # Hook
    masks = []
    images = []
    targets = []
    
    def hook_fn(module, input, output):
        k = max(1, int(output.shape[-1] * model.top_k))
        vals, _ = torch.topk(output, k, dim=-1)
        thresh = vals[:, -1].unsqueeze(-1)
        masks.append((output >= thresh).float().cpu())

    handle = model.spec_1.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for x, y in loader:
            # Filter for Ships (8) and Planes (0)
            indices = ((y == 0) | (y == 8)).nonzero(as_tuple=True)[0]
            if len(indices) == 0: continue
            
            x_filt = x[indices].to(device)
            y_filt = y[indices]
            
            _ = model(x_filt)
            
            # Store raw images (for background intensity)
            images.append(x_filt.cpu())
            targets.append(y_filt.cpu())
            
    handle.remove()
    
    all_masks = torch.cat(masks, dim=0)
    all_imgs = torch.cat(images, dim=0) # [N, 3, 32, 32]
    all_targets = torch.cat(targets, dim=0)
    
    # 1. Calculate Image "Background" Stat (Mean Intensity)
    # Simple proxy: Average pixel value per image
    img_means = all_imgs.view(all_imgs.shape[0], -1).mean(dim=1)
    
    # 2. Pairwise Differences (Sample 1000 pairs)
    n_samples = min(1000, len(all_imgs))
    pairs_idx = torch.randperm(len(all_imgs))[:n_samples]
    
    # We want to see: Does |Mean_A - Mean_B| correlate with Mask_Overlap?
    # Ideally: NO. (Router should ignore intensity).
    # Expected (COMET): YES. (Similar intensity -> Similar mask).
    
    intensities = img_means[pairs_idx].numpy()
    mask_subset = all_masks[pairs_idx].numpy()
    
    # Compute similarity matrix for masks
    # (Dot product) / k
    # Since masks are binary, dot product is intersection size.
    # Normalizing by k gives 0..1 overlap
    k = model.top_k * all_masks.shape[1]
    
    # Compare Item i with Item j
    # Let's just compare neighbors in the shuffled list to avoid N^2
    diff_intensities = []
    mask_overlaps = []
    
    for i in range(n_samples - 1):
        # Intensity Diff
        diff = abs(intensities[i] - intensities[i+1])
        diff_intensities.append(diff)
        
        # Mask Overlap
        overlap = (mask_subset[i] * mask_subset[i+1]).sum() / k
        mask_overlaps.append(overlap)
        
    return diff_intensities, mask_overlaps

def plot_diagnostics(class_freqs, bg_diffs, mask_overlaps, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Neuron Selectivity (Heatmap)
    # X-axis: Classes, Y-axis: Neurons (Sorted by max selectivity)
    # We sort neurons by which class they prefer to make it readable
    max_class = class_freqs.argmax(dim=1)
    sorted_indices = torch.argsort(max_class)
    sorted_freqs = class_freqs[sorted_indices].numpy()
    
    sns.heatmap(sorted_freqs, ax=axes[0], cmap="viridis", cbar_kws={'label': 'Activation Prob'})
    axes[0].set_title(f"{model_name}: Neuron Selectivity (Sorted)")
    axes[0].set_xlabel("Classes (0-9)")
    axes[0].set_ylabel("Neurons (Sorted)")
    
    # Plot 2: Background Correlation
    axes[1].scatter(bg_diffs, mask_overlaps, alpha=0.3, s=10)
    axes[1].set_xlabel("Difference in Image Intensity (Background)")
    axes[1].set_ylabel("Mask Overlap (Similarity)")
    axes[1].set_title(f"{model_name}: Router Sensitivity to Intensity")
    
    # Calc correlation
    corr, _ = pearsonr(bg_diffs, mask_overlaps)
    axes[1].text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=axes[1].transAxes, 
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"diagnostics_{model_name}.png")
    print(f"Saved diagnostics_{model_name}.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="COMET")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--neurons", type=int, default=3000)
    parser.add_argument("--topk", type=float, default=0.5)
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--activation", type=str, default="softplus")
    parser.add_argument("--batch_size", type=int, default=128)
    
    args = parser.parse_args()
    
    _, val_loader, _, _ = get_data_loaders(args.dataset, args.batch_size)
    model = load_model(args.model, args.checkpoint, args)
    
    # Run Tests
    class_freqs = get_neuron_selectivity(model, val_loader)
    bg_diffs, mask_overlaps = analyze_background_correlation(model, val_loader)
    
    plot_diagnostics(class_freqs, bg_diffs, mask_overlaps, args.model)

if __name__ == "__main__":
    main()