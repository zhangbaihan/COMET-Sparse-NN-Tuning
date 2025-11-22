import sys
sys.dont_write_bytecode = True
import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from loading_datasets import get_data_loaders
from models.COMET import get_COMET

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Train COMET Model")
    
    # Data and Training settings
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "tiny_imagenet", "svhn", "SARCOS"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamW"])
    parser.add_argument("--seed", type=int, default=42)
    
    # Model Architecture
    parser.add_argument("--neurons", type=int, default=1000, help="Number of neurons per hidden layer")
    parser.add_argument("--topk", type=float, default=0.1, help="Sparsity rate (fraction of neurons kept)")
    parser.add_argument("--activation", type=str, default="softplus")
    parser.add_argument("--norm", type=str, default=None, choices=[None, "batch", "layer"])
    
    # Saving
    parser.add_argument("--save_dir", type=str, default="experiments_COMET", help="Directory to save results")
    
    return parser.parse_args()

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        
        # Accuracy
        if out.shape[1] > 1: # Classification
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        else: # Regression (SARCOS)
             total += x.size(0)

    avg_loss = total_loss / total
    acc = correct / total if total > 0 and out.shape[1] > 1 else 0.0
    return avg_loss, acc

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            
            total_loss += loss.item() * x.size(0)
            
            if out.shape[1] > 1:
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)
            else:
                 total += x.size(0)
                 
    avg_loss = total_loss / total
    acc = correct / total if total > 0 and out.shape[1] > 1 else 0.0
    return avg_loss, acc

def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    print(f"Starting COMET training on {args.dataset}...")
    print(f"Device: {device}")
    
    # Create save directory
    run_name = f"{args.dataset}_neurons{args.neurons}_topk{args.topk}_lr{args.lr}_seed{args.seed}"
    save_path = os.path.join(args.save_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Data Loaders
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(args.dataset, args.batch_size, False)
    
    # Define Layer Sizes (Simple 4-layer structure matching original code)
    layer_sizes = [args.neurons, args.neurons, args.neurons, num_classes]
    
    # Initialize Model
    model = get_COMET(
        dataset_name=args.dataset,
        layer_1_neurons=layer_sizes[0],
        layer_2_neurons=layer_sizes[1],
        layer_3_neurons=layer_sizes[2],
        layer_4_neurons=layer_sizes[3],
        topk_rate=args.topk,
        norm=args.norm,
        activation=args.activation
    ).to(device)
    
    # Loss and Optimizer
    criterion = nn.MSELoss() if args.dataset == 'SARCOS' else nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr) # Default
        
    # Logging storage
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'top_k': [],
        'epoch_times': []
    }
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = validate(model, val_loader, criterion)
        
        epoch_end = time.time()
        duration = epoch_end - epoch_start
        
        # Log metrics
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        history['top_k'].append(args.topk) # Constant for standard COMET
        history['epoch_times'].append(duration)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} | "
              f"TopK: {args.topk} | Time: {duration:.2f}s")
              
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    
    # Save Model State
    model_save_path = os.path.join(save_path, "model_state.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save History
    history_save_path = os.path.join(save_path, "training_logs.pkl")
    with open(history_save_path, "wb") as f:
        pickle.dump(history, f)
    print(f"Logs saved to {history_save_path}")

if __name__ == "__main__":
    main()

