from utils import mask_topk
import torch
import torch.nn as nn

def get_COMET(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons, layer_4_neurons,
                      topk_rate, norm, activation):
    """
    Returns a model implementing COMET routing (conditionally-overlapping routing function).
    """

    class SpecModel(nn.Module):
        def __init__(self):
            super(SpecModel, self).__init__()

            # Define fully connected layers with trainable weights
            if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                input_dim = 32 * 32 * 3
            elif dataset_name == 'SARCOS':
                input_dim = 21
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            self.fc1 = nn.Linear(input_dim, layer_1_neurons, bias=True)
            self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=True)
            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=True)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=True)

            # Define specificity routing weights (non-trainable)
            self.spec_1 = nn.Linear(input_dim, layer_1_neurons, bias=False)
            self.spec_2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=False)
            self.spec_3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=False)
            self.spec_4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=False)

            # Freeze specificity routing weights
            for spec_layer in [self.spec_1, self.spec_2, self.spec_3, self.spec_4]:
                spec_layer.weight.requires_grad = False

            self.top_k = topk_rate  # Fraction of neurons to keep via top-k masking

            # Normalization layers
            if norm == 'batch':
                self.norm_init = nn.BatchNorm1d(input_dim)
                self.norm_fc1 = nn.BatchNorm1d(layer_1_neurons)
                self.norm_fc2 = nn.BatchNorm1d(layer_2_neurons)
                self.norm_fc3 = nn.BatchNorm1d(layer_3_neurons)
            elif norm == 'layer':
                self.norm_init = nn.LayerNorm(input_dim)
                self.norm_fc1 = nn.LayerNorm(layer_1_neurons)
                self.norm_fc2 = nn.LayerNorm(layer_2_neurons)
                self.norm_fc3 = nn.LayerNorm(layer_3_neurons)
            else:
                self.norm_init = None
                self.norm_fc1 = None
                self.norm_fc2 = None
                self.norm_fc3 = None

            # Activation function lookup
            activations = {
                'gelu': nn.GELU(),
                'softplus': nn.Softplus(),
                'relu': nn.ReLU(),
                'leaky': nn.LeakyReLU(0.1),
                'tanh': nn.Tanh(),
                'selu': nn.SELU(),
                'sigmoid': nn.Sigmoid(),
                'silu': nn.SiLU(),
                'elu': nn.ELU(),
                'mish': nn.Mish()
            }
            if activation not in activations:
                raise ValueError(f"Unsupported activation: {activation}")
            self.act = activations[activation]

            # Masks saved for inspection if needed
            self.layer_1_mask = None
            self.layer_2_mask = None
            self.layer_3_mask = None

        def forward(self, x):
            # Flatten input depending on dataset
            if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                x = x.view(-1, 32 * 32 * 3)
            elif dataset_name == 'tiny_imagenet':
                x = x.view(-1, 224 * 224 * 3)
            elif dataset_name == 'SARCOS':
                x = x.view(-1, 21)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            # Initial normalization if specified
            if self.norm_init is not None:
                x = self.norm_init(x)

            # Layer 1
            x_nn = self.fc1(x)
            x_nn = self.act(x_nn)
            with torch.no_grad():
                x_spec1 = self.spec_1(x)
                mask1 = mask_topk(x_spec1, self.top_k)
                self.layer_1_mask = mask1
            x = x_nn * mask1

            if self.norm_fc1 is not None:
                x = self.norm_fc1(x)

            # Layer 2
            x_nn = self.fc2(x)
            x_nn = self.act(x_nn)
            with torch.no_grad():
                if self.norm_fc1 is not None:
                    x_spec1 = self.norm_fc1(x_spec1)
                x_spec2 = self.spec_2(x_spec1 * mask1)
                mask2 = mask_topk(x_spec2, self.top_k)
                self.layer_2_mask = mask2
            x = x_nn * mask2

            if self.norm_fc2 is not None:
                x = self.norm_fc2(x)

            # Layer 3
            x_nn = self.fc3(x)
            x_nn = self.act(x_nn)
            with torch.no_grad():
                if self.norm_fc2 is not None:
                    x_spec2 = self.norm_fc2(x_spec2)
                x_spec3 = self.spec_3(x_spec2 * mask2)
                mask3 = mask_topk(x_spec3, self.top_k)
                self.layer_3_mask = mask3
            x = x_nn * mask3

            if self.norm_fc3 is not None:
                x = self.norm_fc3(x)

            # Output layer (Layer 4)
            x = self.fc4(x)

            return x

    return SpecModel()
