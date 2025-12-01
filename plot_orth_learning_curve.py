import sys
sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from matplotlib.lines import Line2D

def get_max_and_min_arrays(arrays):
    """
    Given a list of arrays (one per seed), returns:
    - max_arr: element-wise max across seeds
    - min_arr: element-wise min across seeds
    - avg_var: average variance (scalar)
    """
    if not arrays:
        return [], [], 0
    
    # Truncate to min length
    min_len = min(len(a) for a in arrays)
    arrays = [a[:min_len] for a in arrays]
    
    arr_stack = np.stack(arrays, axis=0)
    max_arr = np.max(arr_stack, axis=0)
    min_arr = np.min(arr_stack, axis=0)
    mean_arr = np.mean(arr_stack, axis=0)
    var_arr = np.var(arr_stack, axis=0)
    
    return max_arr, min_arr, np.mean(var_arr)

# Configuration
activation = 'softplus'
optimizer_name = 'sgd'
lr = '0.0001'
dataset_name = 'cifar10'

# Models to compare
# Note: Check the folder names in 'downloaded_results/results/'
model_names = ['COMET_Orth'] 
# You can add others if you have them, e.g. ['COMET_Orth', 'COMET_model']

topks = [0.5]
neurons_list = [3000]

# Colors
colors = plt.cm.tab10(np.linspace(0, 1, max(len(model_names), 10)))

fig, axs = plt.subplots(len(topks), len(neurons_list), figsize=(10 * len(neurons_list), 6 * len(topks)), squeeze=False)

for j, neurons in enumerate(neurons_list):
    for i, topk_rate in enumerate(topks):
        ax = axs[i, j]
        handles = []
        labels = []
        
        for k, model_name in enumerate(model_names):
            # Construct path
            # We look in 'downloaded_results/results/' based on your file structure
            # Path: downloaded_results/results/<model_name>/<activation>/<opt>/<lr>/<dataset>_topk_<k>_neurons_<n>/aggregated_metrics.pkl
            # OR sometimes: downloaded_results/results/<model_name>/<dataset>_topk_<k>_neurons_<n>/ if flattened.
            # Based on your listing:
            # downloaded_results/results/COMET_Orth/cifar10_topk_0.5_neurons_3000/aggregated_metrics.pkl
            # Wait, the listing showed:
            # downloaded_results/results/COMET_Orth/cifar10_topk_0.5_neurons_3000/aggregated_metrics.pkl
            
            # Let's try to construct the path flexibly
            # Try 1: Flat structure inside results/ModelName/
            path1 = f'COMET/downloaded_results/results/{model_name}/{dataset_name}_topk_{topk_rate}_neurons_{neurons}/aggregated_metrics.pkl'
            # Try 2: Nested
            path2 = f'COMET/downloaded_results/results/{model_name}/{activation}/{optimizer_name}/lr_{lr}/{dataset_name}_topk_{topk_rate}_neurons_{neurons}/aggregated_metrics.pkl'
            
            if os.path.exists(path1):
                pkl_path = path1
            elif os.path.exists(path2):
                pkl_path = path2
            else:
                print(f"Warning: Could not find results for {model_name} at {path1} or {path2}")
                continue
                
            try:
                with open(pkl_path, 'rb') as fp:
                    performances = pickle.load(fp)
                    
                # Check keys - newer code uses 'val_acc' list of lists
                # Some older code might have prefixed keys
                key = 'val_acc'
                if key not in performances:
                     # Try looking for prefixed key
                     key = f'{model_name}_val_acc'
                
                if key in performances:
                    val_accs = performances[key]
                    max_arr, min_arr, _ = get_max_and_min_arrays(val_accs)
                    
                    epochs = range(1, len(max_arr) + 1)
                    ax.fill_between(epochs, max_arr, min_arr, alpha=.3, color=colors[k])
                    handle, = ax.plot(epochs, (max_arr + min_arr)/2, linewidth=2, color=colors[k], label=model_name)
                    handles.append(handle)
                    labels.append(model_name)
                    
                    print(f"Loaded {model_name}: Final Avg Acc = {(max_arr[-1] + min_arr[-1])/2:.4f}")
            except Exception as e:
                print(f"Error loading {pkl_path}: {e}")

        ax.set_title(f'Neurons={neurons}, TopK={topk_rate}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Validation Accuracy')
        ax.grid(True, alpha=0.3)
        if handles:
            ax.legend()

plt.tight_layout()
save_name = 'COMET_Orth_learning_curve.png'
plt.savefig(save_name, dpi=300)
print(f"Saved learning curve to {save_name}")

