import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# --- FIX: Added Missing Import ---
from loading_datasets import get_data_loaders

# Import Model Factories
try:
    from models.COMET import get_COMET
    from models.COMET_center import get_COMET_center
    from models.COMET_guided_center import get_guided_center
except ImportError:
    # Fallback for direct execution structure
    from COMET import get_COMET
    from models.COMET_center import get_COMET_center
    from models.COMET_guided_center import get_guided_center

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_model(model_name, checkpoint_path, args):
    """Factory to load any of the 3 narrative models."""
    if model_name == 'COMET_model':
        model = get_COMET(args.dataset, args.neurons, args.neurons, args.neurons, 10, args.topk, args.norm, args.activation)
    elif model_name == 'COMET_center':
        model = get_COMET_center(args.dataset, args.neurons, args.neurons, args.neurons, 10, args.topk, args.norm, args.activation)
    elif model_name == 'COMET_guided_center':
        # Guided center needs the bottleneck arg, defaulting to 32 as per our experiment
        model = get_guided_center(args.dataset, args.neurons, args.neurons, args.neurons, 10, args.topk, args.norm, args.activation, bottleneck_dim=32)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def get_layer_masks(model, x):
    """Runs forward pass and extracts masks from the model state."""
    with torch.no_grad():
        # Handle alpha for guided model if needed (default to 1.0 for evaluation)
        if hasattr(model, 'forward') and 'alpha' in model.forward.__code__.co_varnames:
             _ = model(x, alpha=1.0)
        else:
             _ = model(x)
    
    masks = {}
    if hasattr(model, 'layer_1_mask') and model.layer_1_mask is not None:
        masks['spec_1'] = model.layer_1_mask.clone()
    if hasattr(model, 'layer_2_mask') and model.layer_2_mask is not None:
        masks['spec_2'] = model.layer_2_mask.clone()
    if hasattr(model, 'layer_3_mask') and model.layer_3_mask is not None:
        masks['spec_3'] = model.layer_3_mask.clone()
    return masks

# ==============================================================================
# 1. Performance Diagnostics (Confusion Matrix & Accuracy)
# ==============================================================================
def analyze_performance(base_dir, save_cm, save_report):
    print("Generating Confusion Matrix & Performance Report...")
    all_y_true, all_y_pred = [], []
    
    # Load detailed predictions from training run
    seed_files = [f for f in os.listdir(base_dir) if f.endswith('_detailed.pkl')]
    if not seed_files:
        print("No detailed results found. Run experiment first.")
        return

    for f in seed_files:
        with open(os.path.join(base_dir, f), "rb") as pkl:
            res = pickle.load(pkl)
            all_y_true.append(res['y_true'])
            all_y_pred.append(res['y_pred'])
            
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    # Text Report
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=3)
    with open(save_report, "w") as f: f.write(report)
    print(report)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Aggregated Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_cm)
    plt.close()

    # Calculate Ship Sink Metric specifically
    plane_idx, ship_idx = 0, 8
    ship_sink_rate = (cm[plane_idx, ship_idx] / cm[plane_idx, :].sum()) * 100
    print(f"\n[DIAGNOSTIC] Ship Sink Rate: {ship_sink_rate:.2f}% of Planes misclassified as Ships")

# ==============================================================================
# 2. Mechanism Diagnostic: Sensitivity Analysis
# ==============================================================================
class PerturbationProbe:
    def __init__(self, device):
        self.device = device
        kernel = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]])
        self.kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(device)

    def perturb_background(self, x):
        # Mask out center 16x16
        B, C, H, W = x.shape
        mask = torch.ones_like(x)
        start_h, start_w = H//2 - 8, W//2 - 8
        mask[:, :, start_h:start_h+16, start_w:start_w+16] = 0
        return x * (1 - mask) + torch.randn_like(x) * mask

    def perturb_object(self, x):
        # Mask out background, keeping only center
        x_occ = x.clone()
        start_h, start_w = x.shape[2]//2 - 8, x.shape[3]//2 - 8
        x_occ[:, :, start_h:start_h+16, start_w:start_w+16] = 0.5 # Gray out object
        return x_occ

    def perturb_grayscale(self, x):
        weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(self.device)
        return (x * weights).sum(dim=1, keepdim=True).repeat(1, 3, 1, 1)

    def perturb_highpass(self, x):
        return F.conv2d(x, self.kernel, padding=1, groups=3)

def analyze_sensitivity(model, loader, layers, save_path):
    print("Running Sensitivity Analysis (Background vs Object Obsession)...")
    probe = PerturbationProbe(device)
    perts = {'Background': probe.perturb_background, 'Object': probe.perturb_object, 
             'Grayscale': probe.perturb_grayscale, 'HighPass': probe.perturb_highpass}
    
    x_orig, _ = next(iter(loader))
    x_orig = x_orig[:100].to(device)
    
    # Store cosine similarities
    results = {l: {p: 0 for p in perts} for l in layers}
    masks_orig = get_layer_masks(model, x_orig)

    for p_name, p_func in perts.items():
        x_pert = p_func(x_orig)
        masks_pert = get_layer_masks(model, x_pert)
        
        for layer_name in layers:
            if layer_name in masks_orig:
                # Cosine sim between original mask and perturbed mask
                # High Sim = Invariant (Router ignored the change)
                # Low Sim = Sensitive (Router reacted to the change)
                sim = F.cosine_similarity(masks_orig[layer_name], masks_pert[layer_name], dim=1).mean().item()
                results[layer_name][p_name] = sim

    # Plot
    fig, axes = plt.subplots(1, len(layers), figsize=(6*len(layers), 5), squeeze=False)
    colors = ['skyblue', 'salmon', 'lightgreen', 'orange']
    
    for i, layer in enumerate(layers):
        ax = axes[0, i]
        vals = results[layer]
        ax.bar(vals.keys(), vals.values(), color=colors)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"Sensitivity: {layer}")
        ax.set_ylabel("Mask Similarity (Invariant)")
        for j, v in enumerate(vals.values()):
            ax.text(j, v+0.02, f"{v:.2f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved Sensitivity Plot: {save_path}")

# ==============================================================================
# 3. Structural Diagnostic: Jaccard Redundancy Analysis (NEW)
# ==============================================================================
def analyze_redundancy(model, loader, save_path, num_classes=10):
    """
    Calculates Intra-Class Mask Overlap (Jaccard Similarity).
    High Overlap = Semantically Coherent Experts (Good)
    Low Overlap = Fragmented Experts (Bad/Redundant)
    """
    print("Running Redundancy (Jaccard) Analysis...")
    model.eval()
    
    # Collect masks by class
    masks_by_class = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Collecting Masks"):
            x = x.to(device)
            # Forward pass to generate masks
            if hasattr(model, 'forward') and 'alpha' in model.forward.__code__.co_varnames:
                _ = model(x, alpha=1.0)
            else:
                _ = model(x)
            
            # Grab Layer 1 mask (Primary Router)
            if model.layer_1_mask is None: continue
            
            m = model.layer_1_mask.cpu().numpy()
            
            for i in range(len(y)):
                label = y[i].item()
                if len(masks_by_class[label]) < 100: # Limit to 100 samples per class
                    masks_by_class[label].append(m[i])
            
            if all(len(v) >= 100 for v in masks_by_class.values()):
                break
    
    # Calculate Jaccard per class
    class_jaccards = []
    classes_present = []
    
    for c in range(num_classes):
        if len(masks_by_class[c]) < 2: continue
        
        stack = np.stack(masks_by_class[c]) # [N, Neurons]
        
        # Jaccard = (A & B) / (A | B)
        # Note: masks are binary, so matmul is intersection
        intersection = stack @ stack.T
        union = stack.sum(axis=1, keepdims=True) + stack.sum(axis=1).T - intersection
        iou = intersection / (union + 1e-8)
        
        # Average of upper triangle (pairwise comparisons)
        mean_iou = iou[np.triu_indices(len(stack), k=1)].mean()
        class_jaccards.append(mean_iou)
        classes_present.append(CLASSES[c])
        
    global_avg = np.mean(class_jaccards)
    print(f"\n[DIAGNOSTIC] Average Intra-Class Jaccard Similarity: {global_avg:.4f}")
    
    # Plot Per-Class Jaccard
    plt.figure(figsize=(10, 6))
    sns.barplot(x=classes_present, y=class_jaccards, palette="viridis")
    plt.axhline(global_avg, color='r', linestyle='--', label=f'Mean: {global_avg:.3f}')
    plt.title("Intra-Class Mask Coherence (Higher is Better)")
    plt.ylabel("Avg Jaccard Similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved Redundancy Plot: {save_path}")

# ==============================================================================
# Main Execution
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model type: COMET_model, COMET_center, COMET_guided_center")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to results folder (e.g. results/COMET_model/...)")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--neurons", type=int, default=3000)
    parser.add_argument("--topk", type=float, default=0.5)
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--activation", type=str, default="softplus")
    
    args = parser.parse_args()
    
    print(f"\n--- Analyzing {args.model} ---")
    
    # 1. Performance Diagnostics (Confusion Matrix)
    analyze_performance(
        args.base_dir, 
        os.path.join(args.base_dir, "analysis_confusion_matrix.png"),
        os.path.join(args.base_dir, "analysis_report.txt")
    )
    
    # 2. Deep Diagnostics (Load Model)
    # Try loading seed 0, or find the first available model file
    checkpoint = os.path.join(args.base_dir, "seed_0_model.pth")
    if not os.path.exists(checkpoint):
        # Fallback to seed 1 or 2 if 0 doesn't exist
        for i in range(3):
            cp = os.path.join(args.base_dir, f"seed_{i}_model.pth")
            if os.path.exists(cp):
                checkpoint = cp
                break
    
    if os.path.exists(checkpoint):
        print(f"Loading Checkpoint: {checkpoint}")
        model = load_model(args.model, checkpoint, args)
        
        # Get Data
        _, val_loader, _, _ = get_data_loaders(args.dataset, 128, False)
        
        # Target only the main routing layers
        target_layers = ['spec_1', 'spec_2', 'spec_3']
        
        # A. Sensitivity Analysis (Mechanism Check)
        analyze_sensitivity(model, val_loader, target_layers, os.path.join(args.base_dir, "analysis_sensitivity.png"))
        
        # B. Redundancy Analysis (Structural Check)
        analyze_redundancy(model, val_loader, os.path.join(args.base_dir, "analysis_redundancy.png"))
        
    else:
        print(f"CRITICAL: No model checkpoint found in {args.base_dir}. Cannot run Deep Diagnostics.")

if __name__ == "__main__":
    main()