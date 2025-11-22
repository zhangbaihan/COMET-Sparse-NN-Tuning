import sys
sys.dont_write_bytecode = True

import os
import pickle
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from loading_datasets_v2 import get_data_loaders
from models.fc_standard import get_standard_model
from models.dropout_standard_fc import get_dropout_standard_fc
from models.smaller_fc_standard import get_smaller_standard_fc
from models.COMET import get_COMET
from models.topk_fc_standard import get_Top_k_FC_model
from models.moe import get_moe_model
from models.layer_wise_routing import get_layer_wise_routing
from models.bernoulli_masking import get_bernoulli_masking
from models.example_tied_dropout import get_example_tied_dropout
from models.COMET_affine import get_COMET_affine
from models.topk_scheduler import get_TopK_Scheduler

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

def init_model(model_name, seed, dataset_name, layer_sizes, topk_rate, norm, activation,
               layer_1_vector_dict=None, layer_2_vector_dict=None, layer_3_vector_dict=None):
    torch.manual_seed(seed)
    layer_1, layer_2, layer_3, layer_4 = layer_sizes

    # We only focus on COMET and topk_scheduler, but keeping others for compatibility if needed
    model_map = {
        'standard_model': lambda: get_standard_model(dataset_name, layer_1, layer_2, layer_3, layer_4, norm, activation),
        'standard_model_l1': lambda: get_standard_model(dataset_name, layer_1, layer_2, layer_3, layer_4, norm, activation),
        'smaller_model': lambda: get_smaller_standard_fc(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'moe_trainable': lambda: get_moe_model(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation, int(1/topk_rate), True),
        'moe_non_trainable': lambda: get_moe_model(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation, int(1/topk_rate), False),
        'dropout_model': lambda: get_dropout_standard_fc(dataset_name, layer_1, layer_2, layer_3, layer_4, norm, activation, dropout_rate=topk_rate),
        'COMET_model': lambda: get_COMET(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'top_k_model': lambda: get_Top_k_FC_model(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'layer_wise_routing': lambda: get_layer_wise_routing(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'bernoulli_masking': lambda: get_bernoulli_masking(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation, layer_1_vector_dict, layer_2_vector_dict, layer_3_vector_dict),
        'example_tied_dropout': lambda: get_example_tied_dropout(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation, layer_1_vector_dict, layer_2_vector_dict, layer_3_vector_dict),
        'COMET_affine': lambda: get_COMET_affine(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation, freeze_backbone=False),
        'topk_scheduler': lambda: get_TopK_Scheduler(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation)
    }
    
    return_model = model_map[model_name]().to(device)
    total_params = sum(p.numel() for p in return_model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    return return_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train COMET and TopK Scheduler models")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "tiny_imagenet", "svhn", "SARCOS"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--neurons_list", nargs="+", type=int, default=[100, 500, 1000, 3000])
    parser.add_argument("--topks", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    parser.add_argument("--activation", type=str, default="softplus")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "sgd_momentum", "adam", "adamW"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clipping", action="store_true")
    parser.add_argument("--clip_value", type=float, default=1.0)
    parser.add_argument("--clip_type", type=str, default="norm", choices=["norm", "value"])
    parser.add_argument("--norm", type=str, default=None, choices=[None, "batch", "layer"])
    parser.add_argument("--models", nargs="+", type=str, default=["COMET_model", "topk_scheduler"], help="List of models to train")
    return parser.parse_args()


def get_loss_function(dataset):
    return nn.MSELoss() if dataset == 'SARCOS' else nn.CrossEntropyLoss()


def main():
    args = parse_args()
    criterion = get_loss_function(args.dataset)
    # Use default loading without indices for training
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(args.dataset, args.batch_size, False, return_indices=False)

    # Filter to ensure we only use supported models if user doesn't specify
    default_models = ['COMET_model', 'topk_scheduler']
    model_names = args.models if args.models else default_models

    for neurons in args.neurons_list:
        layer_sizes = [neurons, neurons, neurons, num_classes]

        for topk in args.topks:
            performances = {}

            for model_name in model_names:
                print(f"Training {model_name} with {neurons} neurons and topk {topk}")

                # Prepare masking vectors if needed (legacy support for other models)
                layer_1_vector_dict = layer_2_vector_dict = layer_3_vector_dict = None
                
                # Run one model across seeds
                model_results = train_and_evaluate(
                    model_name=model_name,
                    layer_sizes=layer_sizes,
                    topk_rate=topk,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    args=args,
                    vector_dicts=(layer_1_vector_dict, layer_2_vector_dict, layer_3_vector_dict),
                    neurons=neurons # passed for file naming inside train_and_evaluate if needed
                )

                performances.update(model_results)

                # Save Results Pickle
                save_dir = f"results/{model_name}/{args.activation}/{args.optimizer}/lr_{args.lr}"
                os.makedirs(save_dir, exist_ok=True)
                save_path = f"{save_dir}/{args.dataset}_topk_{topk}_neurons_{neurons}"
                
                with open(save_path, "wb") as f:
                    pickle.dump(model_results, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved results to {save_path}")


def train_and_evaluate(model_name, layer_sizes, topk_rate, train_loader, val_loader, criterion, args, vector_dicts, neurons):
    layer_1_vector_dict, layer_2_vector_dict, layer_3_vector_dict = vector_dicts
    
    # Lists to store results for all seeds
    all_seeds_train_losses = []
    all_seeds_val_losses = []
    all_seeds_train_accs = []
    all_seeds_val_accs = []
    all_seeds_topks = [] # To store top_k per epoch

    param_amount = "0M"

    for seed_epoch in range(args.seeds):
        seed = torch.randint(1, 1000, (1,)).item()
        model = init_model(model_name, seed, args.dataset, layer_sizes, topk_rate, args.norm, args.activation,
                           layer_1_vector_dict, layer_2_vector_dict, layer_3_vector_dict)
        
        param_amount = f'{round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)}M'
        print(f"Seed {seed_epoch+1}/{args.seeds}, Parameters: {param_amount}")

        optimizer = {
            'sgd': optim.SGD(model.parameters(), lr=args.lr),
            'sgd_momentum': optim.SGD(model.parameters(), lr=args.lr, momentum=0.9),
            'adam': optim.Adam(model.parameters(), lr=args.lr),
            'adamW': optim.AdamW(model.parameters(), lr=args.lr),
        }[args.optimizer]

        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []
        epoch_topks = []

        for epoch in range(args.epochs):
            # Handle TopK Scheduler updates
            if hasattr(model, 'set_epoch'):
                model.set_epoch(epoch)
            
            # Log current top_k
            current_topk = getattr(model, 'top_k', topk_rate)
            epoch_topks.append(current_topk)

            model.train()
            batch_losses, batch_accs = [], []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                loss = criterion(out, y)
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clipping:
                    if args.clip_type == 'norm':
                        nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
                    elif args.clip_type == 'value':
                        nn.utils.clip_grad_value_(model.parameters(), args.clip_value)
                optimizer.step()
                
                acc = (out.argmax(1) == y).float().mean().item()
                batch_losses.append(loss.item())
                batch_accs.append(acc)

            train_losses.append(np.mean(batch_losses))
            train_accuracies.append(np.mean(batch_accs))

            model.eval()
            with torch.no_grad():
                batch_losses, batch_accs = [], []
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x) # Simplified call, removed special casing for bernoulli/tied as we focus on COMET
                    loss = criterion(out, y)
                    acc = (out.argmax(1) == y).float().mean().item()
                    batch_losses.append(loss.item())
                    batch_accs.append(acc)
                val_losses.append(np.mean(batch_losses))
                val_accuracies.append(np.mean(batch_accs))
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1}/{args.epochs}, Val Acc: {val_accuracies[-1]:.4f}, TopK: {current_topk:.2f}")

        # Save model state_dict for the last seed (or we could save all, but usually one representative or the last is enough for analysis)
        # We will save the last seed's model.
        save_dir = f"results/{model_name}/{args.activation}/{args.optimizer}/lr_{args.lr}"
        os.makedirs(save_dir, exist_ok=True)
        save_path_model = f"{save_dir}/{args.dataset}_topk_{topk_rate}_neurons_{neurons}_seed_{seed_epoch}.pth"
        torch.save(model.state_dict(), save_path_model)
        print(f"Saved model checkpoint to {save_path_model}")

        all_seeds_train_losses.append(train_losses)
        all_seeds_val_losses.append(val_losses)
        all_seeds_train_accs.append(train_accuracies)
        all_seeds_val_accs.append(val_accuracies)
        all_seeds_topks.append(epoch_topks)

    return {
        f"{model_name}_train_loss": all_seeds_train_losses,
        f"{model_name}_val_loss": all_seeds_val_losses,
        f"{model_name}_train_acc": all_seeds_train_accs,
        f"{model_name}_val_acc": all_seeds_val_accs,
        f"{model_name}_top_k_history": all_seeds_topks,
        f"{model_name}_param_amount": param_amount
    }


if __name__ == "__main__":
    main()

