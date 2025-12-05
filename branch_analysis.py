import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Import your model and data loader
from loading_datasets import get_data_loaders
from models.COMET_guided_center import get_guided_center

# --- CONFIG ---
MODEL_PATH = "downloaded_results/COMET_guided_center/cifar10_topk_0.5_neurons_3000/seed_0_model.pth"
DATASET = 'cifar10'
NEURONS = 3000
BOTTLENECK = 32 # Ensure this matches your training config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def analyze_branches():
    print(f"Loading model from {MODEL_PATH}...")
    # Initialize model structure
    model = get_guided_center(DATASET, NEURONS, NEURONS, NEURONS, 10, 0.5, None, 'softplus', BOTTLENECK)
    
    # Load weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # Get data
    _, val_loader, _, _ = get_data_loaders(DATASET, 128, False)
    
    # Storage for magnitudes
    stats = {
        'Layer 1': {'fixed': [], 'semantic': []},
        'Layer 2': {'fixed': [], 'semantic': []},
        'Layer 3': {'fixed': [], 'semantic': []}
    }

    print("Running Inference to measure branch magnitudes...")
    with torch.no_grad():
        for x, _ in tqdm(val_loader):
            x = x.to(DEVICE)
            x_flat = x.view(x.size(0), -1)
            
            # --- MANUALLY REPLICATE FORWARD PASS TO INSPECT INTERNALS ---
            
            # Layer 1 Router
            x_r1 = model.center(x_flat)
            fixed_1 = model.router_1.static(x_r1)
            semantic_1 = model.router_1.semantic(x_r1)
            
            stats['Layer 1']['fixed'].extend(fixed_1.norm(dim=1).cpu().numpy())
            stats['Layer 1']['semantic'].extend(semantic_1.norm(dim=1).cpu().numpy())
            
            # Compute mask to propagate to Layer 2
            scores_1 = fixed_1 + semantic_1 # Alpha=1.0 at end of training
            k = max(1, int(scores_1.shape[-1] * model.top_k))
            topk_vals, _ = torch.topk(scores_1, k, dim=-1)
            kth = topk_vals[:, -1].unsqueeze(-1)
            mask1 = (scores_1 >= kth).float()
            
            # Layer 1 Backbone Output
            x_nn = model.act(model.fc1(x_flat)) * mask1
            
            # Layer 2 Router
            x_r2 = model.center(x_nn)
            fixed_2 = model.router_2.static(x_r2)
            semantic_2 = model.router_2.semantic(x_r2)
            
            stats['Layer 2']['fixed'].extend(fixed_2.norm(dim=1).cpu().numpy())
            stats['Layer 2']['semantic'].extend(semantic_2.norm(dim=1).cpu().numpy())
            
            # Compute mask 2
            scores_2 = fixed_2 + semantic_2
            topk_vals, _ = torch.topk(scores_2, k, dim=-1)
            kth = topk_vals[:, -1].unsqueeze(-1)
            mask2 = (scores_2 >= kth).float()
            
            # Layer 2 Backbone Output
            x_nn = model.act(model.fc2(x_nn)) * mask2
            
            # Layer 3 Router
            x_r3 = model.center(x_nn)
            fixed_3 = model.router_3.static(x_r3)
            semantic_3 = model.router_3.semantic(x_r3)
            
            stats['Layer 3']['fixed'].extend(fixed_3.norm(dim=1).cpu().numpy())
            stats['Layer 3']['semantic'].extend(semantic_3.norm(dim=1).cpu().numpy())

    # --- PLOTTING ---
    plt.figure(figsize=(15, 5))
    
    for i, layer in enumerate(['Layer 1', 'Layer 2', 'Layer 3']):
        plt.subplot(1, 3, i+1)
        f = np.array(stats[layer]['fixed'])
        s = np.array(stats[layer]['semantic'])
        
        ratio = np.mean(s) / np.mean(f)
        
        plt.hist(f, bins=50, alpha=0.6, label='Fixed (Static)', color='blue')
        plt.hist(s, bins=50, alpha=0.6, label='Semantic (Learnable)', color='red')
        plt.title(f"{layer}\nSemantic/Fixed Ratio: {ratio:.3f}")
        plt.xlabel("Output Vector Magnitude (L2)")
        if i == 0: plt.legend()
        plt.grid(True, alpha=0.3)
        
    save_path = "analysis_branch_contribution.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    analyze_branches()