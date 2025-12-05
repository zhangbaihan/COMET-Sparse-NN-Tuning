import sys
sys.dont_write_bytecode = True

import os
import json
import pickle
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from loading_datasets import get_data_loaders

from models.COMET import get_COMET
from models.COMET_center import get_COMET_center
from models.COMET_foveal import get_COMET_foveal
from models.COMET_guided_center import get_guided_center

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. Helper Functions for Analysis & Logging
# ==============================================================================

def save_detailed_results(model, val_loader, save_path):
    """
    Runs inference on the validation set and saves detailed instance-level predictions.
    This is CRITICAL for the 'Ship Sink' error analysis and Confusion Matrices.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_logits = []
    all_indices = []
    current_idx = 0
    
    print("Generating detailed inference results...")
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0], batch[1]
            x, y = x.to(device), y.to(device)
            
            # Handle models with different forward signatures
            if hasattr(model, 'forward'):
                try:
                    logits = model(x)
                except TypeError:
                    logits = model(x, 'test')
            
            preds = logits.argmax(dim=1)
            
            all_logits.append(logits.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
            # Create indices for this batch (assumes shuffle=False for validation)
            batch_len = x.size(0)
            all_indices.append(np.arange(current_idx, current_idx + batch_len))
            current_idx += batch_len

    results = {
        "indices": np.concatenate(all_indices),
        "y_true": np.concatenate(all_targets),
        "y_pred": np.concatenate(all_preds),
        "logits": np.concatenate(all_logits, axis=0)
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"-> Detailed results saved to: {save_path}")

def plot_metrics(train_data, val_data, metric_name, save_path):
    """
    Generates and saves a simple plot for Loss or Accuracy over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_data, label=f'Train {metric_name}')
    plt.plot(val_data, label=f'Val {metric_name}')
    plt.title(f'{metric_name} over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    print(f"-> Plot saved to: {save_path}")

def evaluate_test_set(model, test_loader):
    """
    Runs evaluation on the held-out Test Set for final reporting.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            # Forward pass compatibility check
            try:
                out = model(x)
            except TypeError: 
                out = model(x, 'test')
                
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# ==============================================================================
# 2. Model Initialization
# ==============================================================================

def init_model(model_name, seed, dataset_name, layer_sizes, topk_rate, norm, activation):
    torch.manual_seed(seed)
    layer_1, layer_2, layer_3, layer_4 = layer_sizes

    # Dictionary to hold model factories. 
    model_map = {
        'COMET_model': lambda: get_COMET(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'Orthogonal': lambda: get_Orthogonal(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'COMET_edge': lambda: get_COMET_edge(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'COMET_highpass': lambda: get_COMET_highpass(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'COMET_structure': lambda: get_COMET_structure(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'COMET_center': lambda: get_COMET_center(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'COMET_conv_router': lambda: get_COMET_conv_router(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'COMET_normalized_model': lambda: get_COMET_normalized(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'COMET_Orth': lambda: get_Orthogonal(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'COMET_Hebbian': lambda: get_Oja_Robust(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'COMET_foveal': lambda: get_COMET_foveal(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'COMET_guided_center': lambda: get_guided_center(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
    }

    if model_name not in model_map:
        raise ValueError(f"Model {model_name} not found. Available: {list(model_map.keys())}")

    model = model_map[model_name]().to(device)
    return model

# ==============================================================================
# 3. Main Training Loop
# ==============================================================================

def train_and_evaluate(model_name, layer_sizes, topk_rate, train_loader, val_loader, test_loader, criterion, args):
    model_results = {
        "train_loss": [], "val_loss": [], 
        "train_acc": [], "val_acc": [], 
        "test_acc": []
    }

    for seed_i in range(args.seeds):
        print(f"\n{'='*40}")
        print(f"Starting Seed {seed_i+1}/{args.seeds}")
        print(f"{'='*40}")
        
        # 1. Initialize Model & Optimizer
        seed = torch.randint(1, 10000, (1,)).item()
        model = init_model(model_name, seed, args.dataset, layer_sizes, topk_rate, args.norm, args.activation)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model initialized. Trainable Params: {param_count/1e6:.2f}M")

        if hasattr(model, 'pretrain_router'):
            print(f"Model has pretrain_router method. Starting pre-training...")
            model.pretrain_router(train_loader, epochs=5, lr=0.01, device=device)

        optimizer = {
            'sgd': optim.SGD(model.parameters(), lr=args.lr),
            'sgd_momentum': optim.SGD(model.parameters(), lr=args.lr, momentum=0.9),
            'adam': optim.Adam(model.parameters(), lr=args.lr),
            'adamW': optim.AdamW(model.parameters(), lr=args.lr),
        }[args.optimizer]

        # 2. Training Loop
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(args.epochs):
            # --- ALPHA WARMUP SCHEDULE ---
            # Linear ramp from 0.0 to 1.0 over the first 20 epochs
            current_alpha = None
            if model_name == 'COMET_guided_center':
                if epoch < 20:
                    current_alpha = epoch / 20.0
                else:
                    current_alpha = 1.0

            # Train
            model.train()
            batch_losses, batch_accs_list = [], []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                # Forward with Alpha check
                if model_name == 'COMET_guided_center':
                    out = model(x, alpha=current_alpha)
                else:
                    out = model(x)
                
                loss = criterion(out, y)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clipping:
                    if args.clip_type == 'norm':
                        nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
                    elif args.clip_type == 'value':
                        nn.utils.clip_grad_value_(model.parameters(), args.clip_value)
                optimizer.step()
                
                # Metrics
                acc = (out.argmax(1) == y).float().mean().item()
                batch_losses.append(loss.item())
                batch_accs_list.append(acc)

            train_losses.append(np.mean(batch_losses))
            train_accs.append(np.mean(batch_accs_list))

            # Validate
            model.eval()
            with torch.no_grad():
                v_losses, v_accs = [], []
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    # Forward Compatibility
                    if model_name == 'COMET_guided_center':
                        out = model(x, alpha=current_alpha)
                    else:
                        try: out = model(x)
                        except TypeError: out = model(x, 'test')
                        
                    loss = criterion(out, y)
                    acc = (out.argmax(1) == y).float().mean().item()
                    v_losses.append(loss.item())
                    v_accs.append(acc)
                
                val_losses.append(np.mean(v_losses))
                val_accs.append(np.mean(v_accs))
                
            if (epoch + 1) % 10 == 0 or epoch == 0:
                alpha_msg = f" | Alpha: {current_alpha:.2f}" if current_alpha is not None else ""
                print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_losses[-1]:.4f} | Val Acc: {val_accs[-1]:.4f}{alpha_msg}")

        # 3. Post-Training Analysis & Saving
        print(f"\n--- Training Finished for Seed {seed_i+1} ---")
        
        base_dir = f"results/{model_name}/{args.activation}/{args.optimizer}/lr_{args.lr}/{args.dataset}_topk_{topk_rate}_neurons_{layer_sizes[0]}"
        os.makedirs(base_dir, exist_ok=True)

        # A. Save Detailed Predictions
        detailed_path = f"{base_dir}/seed_{seed_i}_detailed.pkl"
        save_detailed_results(model, val_loader, detailed_path)

        # B. Save Model Weights
        weights_path = f"{base_dir}/seed_{seed_i}_model.pth"
        torch.save(model.state_dict(), weights_path)
        print(f"-> Model weights saved to: {weights_path}")

        # C. Save Plots
        plot_metrics(train_losses, val_losses, "Loss", f"{base_dir}/seed_{seed_i}_loss.png")
        plot_metrics(train_accs, val_accs, "Accuracy", f"{base_dir}/seed_{seed_i}_accuracy.png")

        # D. Test Set Evaluation
        if test_loader:
            test_accuracy = evaluate_test_set(model, test_loader)
            print(f"-> Final Test Set Accuracy: {test_accuracy:.4f}")
            model_results["test_acc"].append(test_accuracy)
        
        # E. Save Config
        with open(f"{base_dir}/config.json", "w") as f:
            json.dump(vars(args), f, indent=4)

        model_results["train_loss"].append(train_losses)
        model_results["val_loss"].append(val_losses)
        model_results["train_acc"].append(train_accs)
        model_results["val_acc"].append(val_accs)

    agg_save_path = f"{base_dir}/aggregated_metrics.pkl"
    with open(agg_save_path, "wb") as f:
        pickle.dump(model_results, f)
        
    return model_results

# ==============================================================================
# 4. Entry Point & Args
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run COMET Experiments")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "tiny_imagenet", "svhn", "SARCOS"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    # Defaulting to 3000 for better COMET performance
    parser.add_argument("--neurons_list", nargs="+", type=int, default=[3000])
    # Defaulting to 0.5 
    parser.add_argument("--topks", nargs="+", type=float, default=[0.5])
    parser.add_argument("--activation", type=str, default="softplus")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "sgd_momentum", "adam", "adamW"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clipping", action="store_true")
    parser.add_argument("--clip_value", type=float, default=1.0)
    parser.add_argument("--clip_type", type=str, default="norm", choices=["norm", "value"])
    parser.add_argument("--norm", type=str, default=None, choices=[None, "batch", "layer"])
    parser.add_argument("--models", nargs="+", type=str, default=["COMET_model"], 
                        help="List of models to run. Options: COMET_model, Orthogonal")
    return parser.parse_args()

def get_loss_function(dataset):
    return nn.MSELoss() if dataset == 'SARCOS' else nn.CrossEntropyLoss()

def main():
    args = parse_args()
    print(f"Running experiment on {device}")
    
    criterion = get_loss_function(args.dataset)
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(args.dataset, args.batch_size, False)

    for neurons in args.neurons_list:
        layer_sizes = [neurons, neurons, neurons, num_classes]
        
        for topk in args.topks:
            print(f"\n>>> CONFIG: Neurons={neurons}, TopK={topk} <<<")
            
            for model_name in args.models:
                print(f"Training {model_name}...")
                train_and_evaluate(
                    model_name=model_name,
                    layer_sizes=layer_sizes,
                    topk_rate=topk,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    criterion=criterion,
                    args=args
                )

if __name__ == "__main__":
    main()