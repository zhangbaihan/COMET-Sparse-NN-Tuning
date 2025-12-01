import sys
sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.lines import Line2D
from utils import get_max_and_min_arrays

# one big plot
model_names = ['standard_model', 'smaller_model', 'dropout_model', 'COMET_model', 'standard_model_l1', 'top_k_model', 'moe_trainable', 'moe_non_trainable', 'layer_wise_routing', 'bernoulli_masking', 'example_tied_dropout']

topks = [0.1, 0.5, 0.9]
neurons_list = [100, 500, 1000, 3000]

activation = 'softplus' # gelu / softplus / leaky / tanh
optimizer_name = 'sgd' # sgd / sgd_momentum / adam / adamW
dataset_name = 'cifar10'

# Define a custom set of colors
colors = plt.cm.tab20(np.linspace(0, 1, len(model_names)))

fig, axs = plt.subplots(len(topks), len(neurons_list), figsize=(40, 20))
for j, neurons in enumerate(neurons_list):
    for i, topk_rate in enumerate(topks):
        with open(f'results/{activation}/{optimizer_name}/lr_0.001/{dataset_name}_topk_{topk_rate}_neurons_{neurons}', 'rb') as fp:
            performances = pickle.load(fp)
            ax = axs[i, j]
            handles = []
            labels = []
            for k, model_name in enumerate(model_names):
                max_seed, min_seed, avg_variance = get_max_and_min_arrays(performances[f'{model_name}_val_acc']) # get lists with max and min values per seed
                ax.fill_between([i+1 for i in range(len(min_seed))], np.array(max_seed), np.array(min_seed), alpha=.3, linewidth=2.0, color=colors[k])
                handle, = ax.plot([i+1 for i in range(len(min_seed))], (np.array(max_seed) + np.array(min_seed))/2, linewidth=2, color=colors[k])
                handles.append(handle)
                labels.append(model_name)
            if j == 0:  # Set y-axis label for the first column (topk values)
                ax.set_ylabel(f'p_k = {topk_rate}', fontsize=30)
            if i == 0:  # Set title only for the first row (neurons values)
                ax.set_title(f'Neurons={neurons}', fontsize=30)

for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

fig.text(0.04, 0.5, 'Validation Accuracy', va='center', rotation='vertical', fontsize=30)
fig.text(0.5, 0.08, 'Epochs', ha='center', fontsize=30)

# Add legend at the bottom of the figure
fig.legend(handles=[Line2D([0], [0], color=h.get_color(), lw=5) for h in handles], handlelength=4, labels=labels, loc='lower center', ncol=len(model_names)//2, prop={'size': 30}, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=(0.05, 0.1, 1, 1))
plt.savefig('standard_mlp_cifar10.pdf', bbox_inches='tight', dpi=300)