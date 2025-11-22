import sys
sys.dont_write_bytecode = True
import os
import pickle
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from loading_datasets import get_data_loaders
from models.COMET import get_COMET
from models.topk_scheduler import get_TopK_Scheduler

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IndexedDataset(Dataset):
    """
    Wrapper that returns (sample, label, index).
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Model")
    
    # Model info
    parser.add_argument("--model_type", type=str, required=True, choices=["COMET", "scheduler"])
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth model state")
    
    # Dataset info
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    
    # Model Config (Must match training)
    parser.add_argument("--neurons", type=int, default=1000)
    parser.add_argument("--topk", type=float, default=0.1, help="For COMET or Scheduler (final)")
    parser.add_argument("--activation", type=str, default="softplus")
    parser.add_argument("--norm", type=str, default=None)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--visualize_misclassified", action="store_true", help="Save grid of errors")
    parser.add_argument("--num_visualize", type=int, default=16, help="Number of errors to visualize")
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data: {args.dataset}...")
    train_loader_orig, val_loader_orig, test_loader_orig, num_classes = get_data_loaders(args.dataset, args.batch_size, False)
    
    if args.split == 'train':
        target_dataset = train_loader_orig.dataset
    elif args.split == 'val':
        target_dataset = val_loader_orig.dataset
    else:
        target_dataset = test_loader_orig.dataset
        
    # Wrap dataset to get indices
    indexed_dataset = IndexedDataset(target_dataset)
    loader = DataLoader(indexed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Initialize Model
    print(f"Initializing {args.model_type} model...")
    layer_sizes = [args.neurons, args.neurons, args.neurons, num_classes]
    
    if args.model_type == "COMET":
        model = get_COMET(
            dataset_name=args.dataset,
            layer_1_neurons=layer_sizes[0],
            layer_2_neurons=layer_sizes[1],
            layer_3_neurons=layer_sizes[2],
            layer_4_neurons=layer_sizes[3],
            topk_rate=args.topk,
            norm=args.norm,
            activation=args.activation
        )
    else: # scheduler
        # For evaluation, we assume the scheduler is at its final state (min_pk=topk)
        # We initialize it and set epoch to a large number to ensure min_pk is used
        model = get_TopK_Scheduler(
            dataset_name=args.dataset,
            layer_1_neurons=layer_sizes[0],
            layer_2_neurons=layer_sizes[1],
            layer_3_neurons=layer_sizes[2],
            layer_4_neurons=layer_sizes[3],
            topk_rate=args.topk, # Target final rate
            norm=args.norm,
            activation=args.activation,
            step_every=1, # Doesn't matter for eval if we force epoch
            step_pk=-0.1
        )
        # Force model to final state
        model.set_epoch(10000) 
    
    model.to(device)
    
    # Load Weights
    print(f"Loading weights from {args.model_path}...")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Run Inference
    print("Running inference...")
    all_preds = []
    all_targets = []
    all_indices = []
    all_logits = []
    
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device)
            y = y.to(device)
            
            out = model(x)
            
            if args.dataset == 'SARCOS':
                 # Regression handling if needed, but user asked for classification metrics mostly
                 pass
            
            preds = out.argmax(dim=1)
            
            all_logits.append(out.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
            all_indices.append(idx)
            
    # Concatenate
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    logits = torch.cat(all_logits).numpy()
    indices = torch.cat(all_indices).numpy()
    
    # Calculate Accuracy
    acc = (y_pred == y_true).mean()
    print(f"Accuracy on {args.split} set: {acc:.4f}")
    
    # Save Results
    results = {
        "y_true": y_true,
        "y_pred": y_pred,
        "logits": logits,
        "indices": indices,
        "accuracy": acc
    }
    
    save_file = os.path.join(args.output_dir, f"predictions_{args.dataset}_{args.model_type}_{args.split}.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Predictions saved to {save_file}")
    
    # Visualize Misclassified
    if args.visualize_misclassified and args.dataset != 'SARCOS':
        print("Generating misclassified samples visualization...")
        misclassified_mask = (y_pred != y_true)
        mis_indices = indices[misclassified_mask]
        mis_preds = y_pred[misclassified_mask]
        mis_true = y_true[misclassified_mask]
        
        if len(mis_indices) > 0:
            num_to_show = min(args.num_visualize, len(mis_indices))
            # Randomly select if more than needed, or just take first N
            perm = torch.randperm(len(mis_indices))[:num_to_show]
            
            fig, axes = plt.subplots(int(num_to_show**0.5), int(num_to_show**0.5) + 1, figsize=(12, 12))
            axes = axes.flatten()
            
            # Get inverse transform for visualization (un-normalize)
            if args.dataset == 'cifar10':
                mean = np.array([0.5, 0.5, 0.5])
                std = np.array([0.5, 0.5, 0.5])
            elif args.dataset == 'cifar100':
                mean = np.array([0.5071, 0.4867, 0.4408])
                std = np.array([0.2675, 0.2565, 0.2761])
            elif args.dataset == 'svhn':
                mean = np.array([0.4377, 0.4438, 0.4728])
                std = np.array([0.1980, 0.2010, 0.1970])
            elif args.dataset == 'tiny_imagenet':
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
            else:
                mean = np.array([0.5, 0.5, 0.5]) # Default
                std = np.array([0.5, 0.5, 0.5])
            
            for i, idx_ptr in enumerate(perm):
                if i >= len(axes): break
                
                global_idx = mis_indices[idx_ptr]
                pred_lbl = mis_preds[idx_ptr]
                true_lbl = mis_true[idx_ptr]
                
                # Fetch image from original dataset (avoid tensor/gpu mess from loader)
                img_tensor, _ = target_dataset[global_idx]
                img_np = img_tensor.numpy() # (C, H, W)
                
                # Un-normalize
                img_np = img_np * std[:, None, None] + mean[:, None, None]
                
                # Permute to (H, W, C)
                img_np = img_np.transpose(1, 2, 0)
                img_np = np.clip(img_np, 0, 1)
                
                ax = axes[i]
                ax.imshow(img_np)
                ax.set_title(f"T:{true_lbl} P:{pred_lbl}")
                ax.axis('off')
                
            # Hide unused subplots
            for j in range(i+1, len(axes)):
                axes[j].axis('off')
                
            viz_path = os.path.join(args.output_dir, f"misclassified_{args.dataset}_{args.model_type}.png")
            plt.tight_layout()
            plt.savefig(viz_path)
            print(f"Visualization saved to {viz_path}")
        else:
            print("No misclassified samples found!")

if __name__ == "__main__":
    main()

