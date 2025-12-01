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
from loading_datasets import get_data_loaders
from models.fc_standard import get_standard_model
from models.dropout_standard_fc import get_dropout_standard_fc
from models.smaller_fc_standard import get_smaller_standard_fc
from models.COMET import get_COMET
from models.COMET_normalized import get_COMET_normalized
from models.topk_fc_standard import get_Top_k_FC_model
from models.moe import get_moe_model
from models.layer_wise_routing import get_layer_wise_routing
from models.bernoulli_masking import get_bernoulli_masking
from models.example_tied_dropout import get_example_tied_dropout

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(model_name, seed, dataset_name, layer_sizes, topk_rate, norm, activation,
               layer_1_vector_dict=None, layer_2_vector_dict=None, layer_3_vector_dict=None):
    torch.manual_seed(seed)
    layer_1, layer_2, layer_3, layer_4 = layer_sizes

    model_map = {
        'standard_model': lambda: get_standard_model(dataset_name, layer_1, layer_2, layer_3, layer_4, norm, activation),
        'standard_model_l1': lambda: get_standard_model(dataset_name, layer_1, layer_2, layer_3, layer_4, norm, activation),
        'smaller_model': lambda: get_smaller_standard_fc(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'moe_trainable': lambda: get_moe_model(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation, int(1/topk_rate), True),
        'moe_non_trainable': lambda: get_moe_model(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation, int(1/topk_rate), False),
        'dropout_model': lambda: get_dropout_standard_fc(dataset_name, layer_1, layer_2, layer_3, layer_4, norm, activation, dropout_rate=topk_rate),
        'COMET_model': lambda: get_COMET(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'COMET_normalized_model': lambda: get_COMET_normalized(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'top_k_model': lambda: get_Top_k_FC_model(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'layer_wise_routing': lambda: get_layer_wise_routing(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation),
        'bernoulli_masking': lambda: get_bernoulli_masking(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation, layer_1_vector_dict, layer_2_vector_dict, layer_3_vector_dict),
        'example_tied_dropout': lambda: get_example_tied_dropout(dataset_name, layer_1, layer_2, layer_3, layer_4, topk_rate, norm, activation, layer_1_vector_dict, layer_2_vector_dict, layer_3_vector_dict),
    }

    return model_map[model_name]().to(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Train different neural models with varying sparsity and architecture")
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
    return parser.parse_args()


def get_loss_function(dataset):
    return nn.MSELoss() if dataset == 'SARCOS' else nn.CrossEntropyLoss()


def main():
    args = parse_args()
    criterion = get_loss_function(args.dataset)
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(args.dataset, args.batch_size, False)

    model_names = [
        'standard_model', 'smaller_model', 'dropout_model', 'COMET_model', 'standard_model_l1',
        'top_k_model', 'moe_trainable', 'moe_non_trainable', 'layer_wise_routing', 'bernoulli_masking', 'example_tied_dropout'
    ]

    for neurons in args.neurons_list:
        layer_sizes = [neurons, neurons, neurons, num_classes]

        for topk in args.topks:
            performances = {}

            for model_name in model_names:
                print(f"Training {model_name} with {neurons} neurons and topk {topk}")

                # Prepare masking vectors if needed
                layer_1_vector_dict = layer_2_vector_dict = layer_3_vector_dict = None
                if model_name in ['bernoulli_masking', 'example_tied_dropout']:
                    p = 0.2 if model_name == 'example_tied_dropout' else 0.5
                    layer_1_vector_dict, layer_2_vector_dict, layer_3_vector_dict = {}, {}, {}
                    for batch_idx, (train_x, _) in enumerate(train_loader):
                        for x in train_x:
                            key = f'{x[0, 0, 0].item()}{x[1, 1, 1].item()}'
                            if key not in layer_1_vector_dict:
                                layer_1_vector_dict[key] = torch.bernoulli(torch.tensor(p).expand(neurons))
                                layer_2_vector_dict[key] = torch.bernoulli(torch.tensor(p).expand(neurons))
                                layer_3_vector_dict[key] = torch.bernoulli(torch.tensor(p).expand(neurons))
                                if model_name == 'example_tied_dropout':
                                    k = int(topk * neurons)
                                    layer_1_vector_dict[key][:k] = 1
                                    layer_2_vector_dict[key][:k] = 1
                                    layer_3_vector_dict[key][:k] = 1

                # Run one model across seeds
                model_results = train_and_evaluate(
                    model_name=model_name,
                    layer_sizes=layer_sizes,
                    topk_rate=topk,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    args=args,
                    vector_dicts=(layer_1_vector_dict, layer_2_vector_dict, layer_3_vector_dict)
                )

                performances.update(model_results)

                # Save
                save_path = f"results/{args.activation}/{args.optimizer}/lr_{args.lr}/{args.dataset}_topk_{topk}_neurons_{neurons}"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    pickle.dump(performances, f, protocol=pickle.HIGHEST_PROTOCOL)


def train_and_evaluate(model_name, layer_sizes, topk_rate, train_loader, val_loader, criterion, args, vector_dicts):
    layer_1_vector_dict, layer_2_vector_dict, layer_3_vector_dict = vector_dicts
    model_train_losses, model_val_losses = [], []
    model_train_accuracies, model_val_accuracies = [], []

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

        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

        for epoch in range(args.epochs):
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
                    out = model(x, 'test') if model_name == 'bernoulli_masking' or model_name == 'example_tied_dropout' else model(x)
                    loss = criterion(out, y)
                    acc = (out.argmax(1) == y).float().mean().item()
                    batch_losses.append(loss.item())
                    batch_accs.append(acc)
                val_losses.append(np.mean(batch_losses))
                val_accuracies.append(np.mean(batch_accs))
                print(f"Epoch {epoch+1}/{args.epochs}, Val Acc: {val_accuracies[-1]:.4f}")

        model_train_losses.append(train_losses)
        model_val_losses.append(val_losses)
        model_train_accuracies.append(train_accuracies)
        model_val_accuracies.append(val_accuracies)

    return {
        f"{model_name}_train_loss": model_train_losses,
        f"{model_name}_val_loss": model_val_losses,
        f"{model_name}_train_acc": model_train_accuracies,
        f"{model_name}_val_acc": model_val_accuracies,
        f"{model_name}_param_amount": param_amount
    }


if __name__ == "__main__":
    main()
