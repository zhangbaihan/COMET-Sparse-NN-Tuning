import sys
sys.dont_write_bytecode = True
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from copy import deepcopy

# Import your model loader
from models.COMET import get_COMET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_baseline(weights_path):
    # Hardcoded params for your successful baseline run
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
        # We only need the routing path logic for Layer 1
        # Re-implementing just the routing logic to be safe/explicit
        # (Or we can just run forward() and read model.layer_1_mask)
        try:
            _ = model(x)
        except TypeError:
            _ = model(x, 'test')
        return model.layer_1_mask.float() # (B, 3000)

def perturb_background(x, mask_val=0):
    """
    Replaces the outer boundary (background) with a constant value.
    Assumes object is in center 16x16 of 32x32 image.
    """
    x_out = x.clone()
    # Mask out the outer frame
    # Center is [8:24, 8:24]
    mask = torch.ones_like(x_out)
    mask[:, :, 8:24, 8:24] = 0 
    
    # Apply change to background only
    x_out = x_out * (1 - mask) + (mask * mask_val)
    return x_out

def perturb_object(x, mask_val=0):
    """
    Replaces the center (object) with a constant value.
    """
    x_out = x.clone()
    x_out[:, :, 8:24, 8:24] = mask_val
    return x_out

def perturb_grayscale(x):
    """
    Converts RGB to Grayscale (replicated 3 times to match dim).
    """
    x_gray = TF.rgb_to_grayscale(x, num_output_channels=3)
    return x_gray

def perturb_highpass(x):
    """
    Applies Laplacian filter to remove low-freq color.
    """
    # Simple kernel
    kernel = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]).to(x.device)
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    
    x_pad = torch.nn.functional.pad(x, (1,1,1,1), mode='reflect')
    return torch.nn.functional.conv2d(x_pad, kernel, groups=3)

def cosine_sim(a, b):
    # a, b: (Batch, Dim)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return (a_norm * b_norm).sum(dim=1).mean().item()

def run_analysis(model, dataloader):
    print("Running Router Sensitivity Analysis...")
    
    similarities = {
        "Background Noise": [],
        "Object Occlusion": [],
        "Grayscale": [],
        "High-Pass": []
    }
    
    # Process a few batches
    max_batches = 5
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches: break
            x = x.to(device)
            
            # 1. Original Mask
            mask_orig = get_mask(model, x)
            
            # 2. Perturbation: Background Change (Replace bg with noise)
            x_bg = perturb_background(x, mask_val=torch.randn_like(x)) 
            mask_bg = get_mask(model, x_bg)
            similarities["Background Noise"].append(cosine_sim(mask_orig, mask_bg))
            
            # 3. Perturbation: Object Removal (Replace obj with black)
            x_obj = perturb_object(x, mask_val=-1.0) # Assuming normalized [-1, 1]
            mask_obj = get_mask(model, x_obj)
            similarities["Object Occlusion"].append(cosine_sim(mask_orig, mask_obj))
            
            # 4. Perturbation: Grayscale
            x_gray = perturb_grayscale(x)
            mask_gray = get_mask(model, x_gray)
            similarities["Grayscale"].append(cosine_sim(mask_orig, mask_gray))
            
            # 5. Perturbation: High-Pass
            x_hp = perturb_highpass(x)
            mask_hp = get_mask(model, x_hp)
            similarities["High-Pass"].append(cosine_sim(mask_orig, mask_hp))

    # Aggregate
    results = {k: np.mean(v) for k, v in similarities.items()}
    return results

def plot_results(results):
    names = list(results.keys())
    values = list(results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=['skyblue', 'salmon', 'lightgreen', 'orange'])
    
    plt.ylabel('Mask Cosine Similarity (Higher = More Invariant)')
    plt.title('What does the Router care about?\n(Sensitivity Analysis of Layer 1 Masks)')
    plt.ylim(0, 1.0)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom')
        
    plt.savefig('router_sensitivity.png')
    plt.show()

if __name__ == "__main__":
    from loading_datasets import get_data_loaders
    
    # Path to your BEST baseline model
    MODEL_PATH = "results/COMET_model/softplus/sgd/lr_0.0001/cifar10_topk_0.5_neurons_3000/seed_0_model.pth"
    
    _, val_loader, _, _ = get_data_loaders('cifar10', 64, False)
    model = load_baseline(MODEL_PATH)
    
    results = run_analysis(model, val_loader)
    plot_results(results)