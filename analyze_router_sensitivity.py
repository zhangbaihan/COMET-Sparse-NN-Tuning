import sys
sys.dont_write_bytecode = True
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from copy import deepcopy
import os

# Import your model loader
# Adjust path if COMET.py is in a subdirectory or root
try:
    from COMET import get_COMET
except ImportError:
    from models.COMET import get_COMET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_baseline(weights_path):
    print(f"Loading weights from {weights_path}...")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
    # Hardcoded params for your successful baseline run
    # Ensure these match exactly what you trained with!
    model = get_COMET('cifar10', 3000, 3000, 3000, 10, 0.5, None, 'softplus')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_mask(model, x):
    """
    Extracts the binary mask from Layer 1 for a given input batch.
    """
    with torch.no_grad():
        # Ensure input is contiguous to avoid 'view' errors in COMET.py
        x = x.contiguous()
        
        try:
            _ = model(x)
        except TypeError:
            _ = model(x, 'test')
        return model.layer_1_mask.float() # (B, 3000)

def perturb_background(x, mask_val=0):
    """
    Replaces the outer boundary (background) with a constant value.
    """
    x_out = x.clone()
    mask = torch.ones_like(x_out)
    mask[:, :, 8:24, 8:24] = 0 
    x_out = x_out * (1 - mask) + (mask * mask_val)
    return x_out.contiguous()

def perturb_object(x, mask_val=0):
    """
    Replaces the center (object) with a constant value.
    """
    x_out = x.clone()
    x_out[:, :, 8:24, 8:24] = mask_val
    return x_out.contiguous()

def perturb_grayscale(x):
    """
    Converts RGB to Grayscale.
    """
    x_gray = TF.rgb_to_grayscale(x, num_output_channels=3)
    return x_gray.contiguous()

def perturb_highpass(x):
    """
    Applies Laplacian filter to remove low-freq color.
    """
    kernel = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]).to(x.device)
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    
    x_pad = torch.nn.functional.pad(x, (1,1,1,1), mode='reflect')
    out = torch.nn.functional.conv2d(x_pad, kernel, groups=3)
    return out.contiguous()

def cosine_sim(a, b):
    # Add epsilon to avoid division by zero
    eps = 1e-8
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1, eps=eps)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1, eps=eps)
    return (a_norm * b_norm).sum(dim=1).mean().item()

def run_analysis(model, dataloader):
    print("Running Router Sensitivity Analysis...")
    
    similarities = {
        "Background Noise": [],
        "Object Occlusion": [],
        "Grayscale": [],
        "High-Pass": []
    }
    
    max_batches = 10  # Use more batches for robust stats
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches: break
            x = x.to(device)
            
            # 1. Original Mask
            mask_orig = get_mask(model, x)
            
            # 2. Perturbations
            x_bg = perturb_background(x, mask_val=torch.randn_like(x)) 
            mask_bg = get_mask(model, x_bg)
            similarities["Background Noise"].append(cosine_sim(mask_orig, mask_bg))
            
            x_obj = perturb_object(x, mask_val=-1.0)
            mask_obj = get_mask(model, x_obj)
            similarities["Object Occlusion"].append(cosine_sim(mask_orig, mask_obj))
            
            x_gray = perturb_grayscale(x)
            mask_gray = get_mask(model, x_gray)
            similarities["Grayscale"].append(cosine_sim(mask_orig, mask_gray))
            
            x_hp = perturb_highpass(x)
            mask_hp = get_mask(model, x_hp)
            similarities["High-Pass"].append(cosine_sim(mask_orig, mask_hp))

    results = {k: np.mean(v) for k, v in similarities.items()}
    return results

def plot_results(results):
    names = list(results.keys())
    values = list(results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=['skyblue', 'salmon', 'lightgreen', 'orange'])
    
    plt.ylabel('Mask Cosine Similarity (Higher = Router Ignored Change)')
    plt.title('Router Sensitivity Analysis\n(Which features determine the expert?)')
    plt.ylim(0, 1.1)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.3f}', ha='center', va='bottom')
        
    plt.savefig('router_sensitivity.png')
    print("Sensitivity plot saved to router_sensitivity.png")
    plt.show()

if __name__ == "__main__":
    from loading_datasets import get_data_loaders
    
    # ---------------------------------------------------------
    # IMPORTANT: Update this path to your actual trained weights
    # ---------------------------------------------------------
    MODEL_PATH = "results/COMET_model/softplus/sgd/lr_0.0001/cifar10_topk_0.5_neurons_3000/seed_0_model.pth"
    
    _, val_loader, _, _ = get_data_loaders('cifar10', 64, False)
    
    try:
        model = load_baseline(MODEL_PATH)
        results = run_analysis(model, val_loader)
        plot_results(results)
    except FileNotFoundError as e:
        print(e)
        print("Please check the MODEL_PATH variable in the script.")