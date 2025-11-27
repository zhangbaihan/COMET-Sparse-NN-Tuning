import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import os
import argparse

# CIFAR-10 Classes
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_results(filepath):
    """Load the detailed result dictionary."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def get_validation_dataset(root='./data', dataset_name='cifar10'):
    """
    Loads the validation dataset to retrieve actual images for visualization.
    """
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        try:
            # We use train=False for test/val set in standard PyTorch datasets
            val_set = torchvision.datasets.CIFAR10(
                root=root, train=False, download=True, transform=transform
            )
            return val_set
        except Exception as e:
            print(f"Could not load dataset: {e}")
            return None
    else:
        print(f"Dataset {dataset_name} visualization not yet supported in this script.")
        return None

def plot_confusion_matrix(y_true, y_pred, title, ax=None, save_path=None):
    """Draws a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")

def visualize_errors(results, val_set, num_examples=5, save_path=None):
    """
    Visualizes specific misclassified examples.
    """
    if val_set is None: return

    indices = results['indices']
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    # Find all errors
    errors_mask = y_pred != y_true
    error_indices = indices[errors_mask]
    
    if len(error_indices) == 0:
        print("No errors found! Perfect accuracy?")
        return

    # Pick random errors to show
    chosen_idxs = np.random.choice(error_indices, size=min(len(error_indices), num_examples), replace=False)
    
    fig, axes = plt.subplots(1, len(chosen_idxs), figsize=(15, 4))
    if len(chosen_idxs) == 1: axes = [axes]
    
    fig.suptitle("Random Misclassified Examples", fontsize=14)
    
    for ax, idx in zip(axes, chosen_idxs):
        # We need to map the global dataset index back to our local results arrays
        # (Assuming indices in results map 1:1 to dataset if we didn't shuffle val loader)
        # But to be safe, we use the raw index 'idx' to fetch the image from dataset
        img_tensor, _ = val_set[idx]
        img = img_tensor.permute(1, 2, 0).numpy()
        
        # Un-normalize for display if needed (assuming roughly [0,1] or [-1,1])
        if img.min() < 0: img = (img * 0.5) + 0.5
            
        ax.imshow(np.clip(img, 0, 1))
        
        # Find the prediction for this index
        # We look up where in our results arrays this index sits
        loc = np.where(indices == idx)[0][0]
        
        true_lab = CLASSES[y_true[loc]]
        pred_lab = CLASSES[y_pred[loc]]
        
        ax.set_title(f"True: {true_lab}\nPred: {pred_lab}", color='red', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved error examples to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze detailed results from COMET experiments")
    parser.add_argument("file", type=str, help="Path to the detailed .pkl file (e.g., results/.../seed_0_detailed.pkl)")
    args = parser.parse_args()

    print(f"Loading results from {args.file}...")
    res = load_results(args.file)
    if res is None: return

    base_dir = os.path.dirname(args.file)
    base_name = os.path.splitext(os.path.basename(args.file))[0]

    # 1. Print Metrics
    print("\n" + "="*40)
    print("Classification Report")
    print("="*40)
    print(classification_report(res['y_true'], res['y_pred'], target_names=CLASSES, digits=3))

    # 2. Confusion Matrix
    print("Generating Confusion Matrix...")
    cm_path = os.path.join(base_dir, f"{base_name}_confusion_matrix.png")
    plot_confusion_matrix(res['y_true'], res['y_pred'], title="Confusion Matrix", save_path=cm_path)

    # 3. Visualize Errors
    print("Visualizing Error Examples...")
    val_set = get_validation_dataset()
    errors_path = os.path.join(base_dir, f"{base_name}_error_examples.png")
    visualize_errors(res, val_set, save_path=errors_path)

if __name__ == "__main__":
    main()