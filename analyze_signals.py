import sys
sys.dont_write_bytecode = True
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# Import your models
from models.COMET import get_COMET
from models.COMET_structure import get_COMET_structure
from models.COMET_center import get_COMET_center
# Add others as needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_type, weights_path, args):
    """Reconstructs the model and loads weights."""
    # Hardcoded architecture params matching your run_experiment defaults
    layer_sizes = [3000, 3000, 3000, 10] 
    topk = 0.5
    norm = None
    act = 'softplus'
    
    if model_type == 'COMET_model':
        model = get_COMET('cifar10', *layer_sizes, topk, norm, act)
    elif model_type == 'COMET_structure':
        model = get_COMET_structure('cifar10', *layer_sizes, topk, norm, act)
    elif model_type == 'COMET_highpass':
        model = get_COMET_highpass('cifar10', *layer_sizes, topk, norm, act)
    elif model_type == 'COMET_center':
        model = get_COMET_center('cifar10', *layer_sizes, topk, norm, act)
    else:
        raise ValueError(f"Unknown model: {model_type}")
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def hook_fn(module, input, output, name, storage):
    """
    Hooks into the mask generation. 
    Note: Since mask generation happens inside forward() without explicit modules in some implementations,
    we might need to rely on the 'self.layer_X_mask' attributes if your model class saves them.
    
    Fortunately, your COMET implementations DO save 'self.layer_X_mask'.
    We will read those directly after a forward pass.
    """
    pass

def analyze_model(model, dataloader, classes_to_compare=[0, 8]): # 0=Plane, 8=Ship
    """
    Runs analysis on a specific model.
    """
    
    # Store masks and representations per class
    # Structure: layer_idx -> class_idx -> list of vectors
    masks = defaultdict(lambda: defaultdict(list))
    
    print(f"Analyzing signal flow for classes: {classes_to_compare}...")
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            
            # Filter batch for target classes
            relevant_indices = [i for i, label in enumerate(y) if label.item() in classes_to_compare]
            if not relevant_indices: continue
            
            x_rel = x[relevant_indices]
            y_rel = y[relevant_indices]
            
            # Run forward pass to populate internal mask attributes
            # We handle the different forward signatures via try/except
            try:
                _ = model(x_rel)
            except TypeError:
                _ = model(x_rel, 'test')
            
            # Extract Masks (Saved in self.layer_X_mask by your class design)
            # Shapes are (Batch, Neurons)
            current_masks = {
                1: model.layer_1_mask,
                2: model.layer_2_mask,
                3: model.layer_3_mask
            }
            
            # Group by class
            for i, label in enumerate(y_rel):
                lbl = label.item()
                for layer_idx, m_tensor in current_masks.items():
                    if m_tensor is not None:
                        # Convert binary mask to numpy
                        masks[layer_idx][lbl].append(m_tensor[i].cpu().numpy())

    # --- METRIC 1: Mask Overlap (Cosine Similarity) ---
    # We want to know: Do Plane masks overlap with Ship masks?
    results = {}
    
    for layer in [1, 2, 3]:
        # Stack all vectors for class A and class B
        vecs_A = np.array(masks[layer][classes_to_compare[0]]) # Plane
        vecs_B = np.array(masks[layer][classes_to_compare[1]]) # Ship
        
        if len(vecs_A) == 0 or len(vecs_B) == 0: continue
            
        # 1. Intra-Class Similarity (Plane vs Plane)
        # Take a subset to speed up
        sim_AA = cosine_similarity(vecs_A[:100], vecs_A[:100]).mean()
        
        # 2. Inter-Class Similarity (Plane vs Ship)
        sim_AB = cosine_similarity(vecs_A[:100], vecs_B[:100]).mean()
        
        results[layer] = {
            "Intra-Class Sim": sim_AA,
            "Inter-Class Sim": sim_AB,
            "Separability": sim_AA - sim_AB  # Higher is better
        }
        
    return results

def plot_analysis(baseline_res, experiment_res, title):
    """
    Visualizes the layer-wise separability.
    """
    layers = [1, 2, 3]
    
    # Extract Separability Scores
    base_sep = [baseline_res[l]["Separability"] for l in layers]
    exp_sep = [experiment_res[l]["Separability"] for l in layers]
    
    base_inter = [baseline_res[l]["Inter-Class Sim"] for l in layers]
    exp_inter = [experiment_res[l]["Inter-Class Sim"] for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Expert Separation (The "Goal")
    x = np.arange(len(layers))
    width = 0.35
    
    ax1.bar(x - width/2, base_sep, width, label='Baseline (RGB)', color='gray')
    ax1.bar(x + width/2, exp_sep, width, label='Experiment (Structure)', color='blue')
    ax1.set_ylabel('Mask Separability (Intra - Inter)')
    ax1.set_title('Did Experts Specialize?')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Layer {l}' for l in layers])
    ax1.legend()
    
    # Plot 2: The "Confusion" (Inter-Class Similarity)
    ax2.plot(layers, base_inter, marker='o', label='Baseline (RGB)', color='gray', linestyle='--')
    ax2.plot(layers, exp_inter, marker='o', label='Experiment (Structure)', color='red')
    ax2.set_ylabel('Plane-Ship Mask Overlap')
    ax2.set_title('Are Planes triggering Ship Experts?')
    ax2.set_xticks(layers)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.savefig('signal_analysis.png')
    plt.show()

if __name__ == "__main__":
    from loading_datasets import get_data_loaders
    
    # Paths to your saved models (Update these!)
    BASELINE_PATH = "results/COMET_model/softplus/sgd/lr_0.0001/cifar10_topk_0.5_neurons_3000/seed_0_model.pth"
    EXP_PATH = "results/COMET_structure/softplus/sgd/lr_0.0001/cifar10_topk_0.5_neurons_3000/seed_0_model.pth"
    
    # Load Data
    _, val_loader, _, _ = get_data_loaders('cifar10', 32, False)
    
    # Run Analysis
    print("--- Analyzing Baseline (COMET_model) ---")
    base_model = load_model('COMET_model', BASELINE_PATH, None)
    base_stats = analyze_model(base_model, val_loader)
    
    print("\n--- Analyzing Experiment (COMET_structure) ---")
    exp_model = load_model('COMET_structure', EXP_PATH, None)
    exp_stats = analyze_model(exp_model, val_loader)
    
    # Plot
    plot_analysis(base_stats, exp_stats, "Plane vs. Ship: Expert Specialization Analysis")