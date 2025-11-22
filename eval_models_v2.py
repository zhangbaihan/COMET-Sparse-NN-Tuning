import sys
sys.dont_write_bytecode = True

import os
import pickle
import argparse
import torch
import torch.nn as nn
import numpy as np

from loading_datasets_v2 import get_data_loaders
from models.COMET import get_COMET
from models.topk_scheduler import get_TopK_Scheduler
from models.fc_standard import get_standard_model 
# Import others if needed, but we focus on COMET/TopK

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_model_for_eval(model_name, dataset_name, layer_sizes, topk_rate, norm, activation):
    layer_1, layer_2, layer_3, layer_4 = layer_sizes
    
    if model_name == 'COMET_model':
        model = get_COMET(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation)
    elif model_name == 'topk_scheduler':
        # specific args for scheduler might be default in init, but we need to ensure they match
        model = get_TopK_Scheduler(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation)
    else:
        raise ValueError(f"Model {model_name} not supported in this eval script yet.")
    
    return model.to(device)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--neurons_list", nargs="+", type=int, default=[100, 500, 1000, 3000])
    parser.add_argument("--topks", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    parser.add_argument("--activation", type=str, default="softplus")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--models", nargs="+", type=str, default=["COMET_model", "topk_scheduler"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--use_test_set", action="store_true", help="Use test set instead of validation set")
    return parser.parse_args()

def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    all_indices = []
    all_logits = []

    with torch.no_grad():
        for data in loader:
            if len(data) == 3:
                x, y, idx = data
            else:
                x, y = data
                idx = torch.zeros_like(y) - 1 # Placeholder if no indices

            x, y = x.to(device), y.to(device)
            out = model(x)
            
            # Softmax for probabilities
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_indices.append(idx.cpu().numpy())
            all_logits.append(probs.cpu().numpy())

    return {
        'y_pred': np.concatenate(all_preds),
        'y_true': np.concatenate(all_targets),
        'indices': np.concatenate(all_indices),
        'y_prob': np.concatenate(all_logits)
    }

def main():
    args = parse_args()
    
    # Get data loader with indices
    # Note: SARCOS has no classes, check compatibility if using SARCOS
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(args.dataset, args.batch_size, False, return_indices=True)
    
    eval_loader = test_loader if args.use_test_set and test_loader is not None else val_loader
    loader_name = "test" if args.use_test_set and test_loader is not None else "val"
    print(f"Evaluating on {loader_name} set...")

    for model_name in args.models:
        for neurons in args.neurons_list:
            layer_sizes = [neurons, neurons, neurons, num_classes]
            
            for topk in args.topks:
                base_path = f"results/{model_name}/{args.activation}/{args.optimizer}/lr_{args.lr}"
                
                # Iterate over seeds
                for seed in range(args.seeds):
                    checkpoint_name = f"{args.dataset}_topk_{topk}_neurons_{neurons}_seed_{seed}.pth"
                    checkpoint_path = os.path.join(base_path, checkpoint_name)
                    
                    if not os.path.exists(checkpoint_path):
                        print(f"Checkpoint not found: {checkpoint_path}")
                        continue
                        
                    print(f"Evaluating {model_name} | Neurons: {neurons} | TopK: {topk} | Seed: {seed}")
                    
                    # Init model
                    model = init_model_for_eval(model_name, args.dataset, layer_sizes, topk, norm=None, activation=args.activation)
                    
                    # Load weights
                    try:
                        state_dict = torch.load(checkpoint_path, map_location=device)
                        model.load_state_dict(state_dict)
                    except Exception as e:
                        print(f"Error loading checkpoint {checkpoint_path}: {e}")
                        continue

                    # Run evaluation
                    results = evaluate_model(model, eval_loader)
                    
                    # Save results
                    save_name = f"{args.dataset}_topk_{topk}_neurons_{neurons}_seed_{seed}_eval.pkl"
                    save_path = os.path.join(base_path, save_name)
                    
                    with open(save_path, "wb") as f:
                        pickle.dump(results, f)
                    print(f"Saved eval results to {save_path}")

if __name__ == "__main__":
    main()

