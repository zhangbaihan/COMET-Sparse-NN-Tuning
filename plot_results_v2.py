import sys
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import argparse
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import itertools
import torch

# Try to import dataset loader for visualization
try:
    from loading_datasets_v2 import get_data_loaders
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Plot results and analyze errors")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--neurons_list", nargs="+", type=int, default=[100, 500, 1000, 3000])
    parser.add_argument("--topks", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    parser.add_argument("--activation", type=str, default="softplus")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--models", nargs="+", type=str, default=["COMET_model", "topk_scheduler"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--visualize_errors", action="store_true", help="Visualize misclassified images (requires dataset)")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()

def get_mean_std(data_list):
    # data_list is list of lists (seeds x epochs)
    arr = np.array(data_list)
    return np.mean(arr, axis=0), np.std(arr, axis=0)

def plot_training_metrics(args):
    # Plot Loss and Accuracy vs Epoch
    for model_name in args.models:
        for neurons in args.neurons_list:
            for topk in args.topks:
                base_path = f"results/{model_name}/{args.activation}/{args.optimizer}/lr_{args.lr}"
                file_name = f"{args.dataset}_topk_{topk}_neurons_{neurons}"
                full_path = os.path.join(base_path, file_name)
                
                if not os.path.exists(full_path):
                    print(f"Warning: {full_path} not found")
                    continue

                with open(full_path, 'rb') as f:
                    results = pickle.load(f)
                
                # Extract metrics
                train_loss = results[f"{model_name}_train_loss"]
                val_loss = results[f"{model_name}_val_loss"]
                train_acc = results[f"{model_name}_train_acc"]
                val_acc = results[f"{model_name}_val_acc"]
                
                # Optional: TopK history
                topk_hist = results.get(f"{model_name}_top_k_history", None)

                epochs = range(1, len(train_loss[0]) + 1)
                
                # Setup figure
                fig, axs = plt.subplots(1, 3 if topk_hist else 2, figsize=(18, 5))
                
                # Plot Loss
                mean_tl, std_tl = get_mean_std(train_loss)
                mean_vl, std_vl = get_mean_std(val_loss)
                
                axs[0].plot(epochs, mean_tl, label='Train Loss')
                axs[0].fill_between(epochs, mean_tl-std_tl, mean_tl+std_tl, alpha=0.2)
                axs[0].plot(epochs, mean_vl, label='Val Loss')
                axs[0].fill_between(epochs, mean_vl-std_vl, mean_vl+std_vl, alpha=0.2)
                axs[0].set_title(f'Loss (N={neurons}, k={topk})')
                axs[0].set_xlabel('Epoch')
                axs[0].set_ylabel('Loss')
                axs[0].legend()

                # Plot Acc
                mean_ta, std_ta = get_mean_std(train_acc)
                mean_va, std_va = get_mean_std(val_acc)
                
                axs[1].plot(epochs, mean_ta, label='Train Acc')
                axs[1].fill_between(epochs, mean_ta-std_ta, mean_ta+std_ta, alpha=0.2)
                axs[1].plot(epochs, mean_va, label='Val Acc')
                axs[1].fill_between(epochs, mean_va-std_va, mean_va+std_va, alpha=0.2)
                axs[1].set_title(f'Accuracy (N={neurons}, k={topk})')
                axs[1].set_xlabel('Epoch')
                axs[1].set_ylabel('Accuracy')
                axs[1].legend()

                # Plot TopK if available
                if topk_hist:
                    mean_tk, std_tk = get_mean_std(topk_hist)
                    axs[2].plot(epochs, mean_tk, label='Sparsity (p_k)')
                    axs[2].fill_between(epochs, mean_tk-std_tk, mean_tk+std_tk, alpha=0.2)
                    axs[2].set_title('Top-K Schedule')
                    axs[2].set_xlabel('Epoch')
                    axs[2].set_ylabel('p_k')
                    axs[2].set_ylim(0, 1.1)

                plt.tight_layout()
                plot_filename = f"{base_path}/plot_{file_name}.png"
                plt.savefig(plot_filename)
                plt.close()
                print(f"Saved plot to {plot_filename}")


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def analyze_eval_results(args):
    # Helper to get class names
    if args.dataset == 'cifar10':
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif args.dataset == 'cifar100':
        classes = [str(i) for i in range(100)]
    else:
        classes = [str(i) for i in range(10)] # Placeholder

    for model_name in args.models:
        for neurons in args.neurons_list:
            for topk in args.topks:
                base_path = f"results/{model_name}/{args.activation}/{args.optimizer}/lr_{args.lr}"
                
                # We analyze the first seed for detailed plots (CM, Images)
                seed = 0
                eval_file = f"{args.dataset}_topk_{topk}_neurons_{neurons}_seed_{seed}_eval.pkl"
                full_path = os.path.join(base_path, eval_file)
                
                if not os.path.exists(full_path):
                    continue
                
                with open(full_path, 'rb') as f:
                    results = pickle.load(f)
                
                y_true = results['y_true']
                y_pred = results['y_pred']
                
                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(10, 10))
                plot_confusion_matrix(cm, classes, normalize=True, title=f'CM: {model_name} N={neurons} k={topk}')
                cm_filename = f"{base_path}/cm_{args.dataset}_{model_name}_N{neurons}_k{topk}.png"
                plt.savefig(cm_filename)
                plt.close()
                
                # Per-class metrics
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
                
                plt.figure(figsize=(12, 5))
                x = np.arange(len(classes))
                width = 0.25
                plt.bar(x - width, precision, width, label='Precision')
                plt.bar(x, recall, width, label='Recall')
                plt.bar(x + width, f1, width, label='F1')
                plt.xticks(x, classes, rotation=45)
                plt.legend()
                plt.title(f'Per-class Metrics: {model_name} N={neurons} k={topk}')
                metrics_filename = f"{base_path}/metrics_{args.dataset}_{model_name}_N{neurons}_k{topk}.png"
                plt.savefig(metrics_filename)
                plt.close()

                # Visualize Errors
                if args.visualize_errors and DATASET_AVAILABLE:
                    visualize_misclassified(args, results, classes, base_path, f"{model_name}_N{neurons}_k{topk}")


def visualize_misclassified(args, results, classes, save_dir, suffix):
    y_true = results['y_true']
    y_pred = results['y_pred']
    indices = results['indices']
    
    misclassified_mask = y_true != y_pred
    bad_indices = indices[misclassified_mask]
    bad_preds = y_pred[misclassified_mask]
    bad_true = y_true[misclassified_mask]
    
    if len(bad_indices) == 0:
        return

    # Load dataset
    # We need the validation/test set
    _, val_loader, test_loader, _ = get_data_loaders(args.dataset, args.batch_size, False, return_indices=True)
    dataset = val_loader.dataset # Assuming val set was used. If test set, need to switch. 
    # Since we don't pass use_test_set here easily, let's just guess or assume Val for now.
    # Ideally we should save which set was used in the pickle.
    
    # Sample 16 random errors
    num_show = min(16, len(bad_indices))
    perm = np.random.choice(len(bad_indices), num_show, replace=False)
    
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    axs = axs.flatten()
    
    for i, idx in enumerate(perm):
        dataset_idx = bad_indices[idx]
        pred_cls = classes[int(bad_preds[idx])]
        true_cls = classes[int(bad_true[idx])]
        
        # Fetch image
        # Note: dataset[i] returns (img, label, idx)
        # We need to find the item in the dataset with the matching index, 
        # OR if the dataset is just indexed 0..N, we can access directly.
        # Since we used Sequential/Random Samplers, dataset[dataset_idx] should work if dataset is indexable
        
        try:
            img, _, _ = dataset[int(dataset_idx)]
            # Denormalize for visualization
            img = img.permute(1, 2, 0).numpy()
            img = (img * 0.5 + 0.5) # Un-normalize (assuming standard cifar norm)
            img = np.clip(img, 0, 1)
            
            axs[i].imshow(img)
            axs[i].set_title(f"T: {true_cls}\nP: {pred_cls}", color='red', fontsize=8)
            axs[i].axis('off')
        except Exception as e:
            print(f"Error visualizing index {dataset_idx}: {e}")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/errors_{suffix}.png")
    plt.close()


def main():
    args = parse_args()
    plot_training_metrics(args)
    analyze_eval_results(args)

if __name__ == "__main__":
    main()

