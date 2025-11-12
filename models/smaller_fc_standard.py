import torch
import torch.nn as nn

def get_smaller_standard_fc(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons, layer_4_neurons,
                            topk_rate, norm, activation):
    """
    A smaller model with a reduced number of neurons, specifically pkNℓ where Nℓ is the width of the standard model
    """

    class SmallerFCModel(nn.Module):
        def __init__(self):
            super(SmallerFCModel, self).__init__()

            input_dim = 32 * 32 * 3 if dataset_name in ['cifar10', 'cifar100'] else 224 * 224 * 3

            # Layer sizes scaled by topk_rate for first three layers
            scaled_l1 = int(topk_rate * layer_1_neurons)
            scaled_l2 = int(topk_rate * layer_2_neurons)
            scaled_l3 = int(topk_rate * layer_3_neurons)

            # Define fully connected layers
            self.fc1 = nn.Linear(input_dim, scaled_l1, bias=True)
            self.fc2 = nn.Linear(scaled_l1, scaled_l2, bias=True)
            self.fc3 = nn.Linear(scaled_l2, scaled_l3, bias=True)
            self.fc4 = nn.Linear(scaled_l3, layer_4_neurons, bias=True)

            # Define normalization layers
            if norm == 'batch':
                self.norm_init = nn.BatchNorm1d(input_dim)
                self.norm_fc1 = nn.BatchNorm1d(scaled_l1)
                self.norm_fc2 = nn.BatchNorm1d(scaled_l2)
                self.norm_fc3 = nn.BatchNorm1d(scaled_l3)
            elif norm == 'layer':
                self.norm_init = nn.LayerNorm(input_dim)
                self.norm_fc1 = nn.LayerNorm(scaled_l1)
                self.norm_fc2 = nn.LayerNorm(scaled_l2)
                self.norm_fc3 = nn.LayerNorm(scaled_l3)
            else:
                self.norm_init = None
                self.norm_fc1 = None
                self.norm_fc2 = None
                self.norm_fc3 = None

            # Define activation function
            if activation == 'gelu':
                self.act = nn.GELU()
            elif activation == 'softplus':
                self.act = nn.Softplus()
            elif activation == 'leaky':
                self.act = nn.LeakyReLU(0.1)
            elif activation == 'tanh':
                self.act = nn.Tanh()
            else:
                raise ValueError(f"Unsupported activation: {activation}")

        def forward(self, x):
            # Flatten input based on dataset
            if dataset_name in ['cifar10', 'cifar100']:
                x = x.view(-1, 32 * 32 * 3)
            elif dataset_name == 'tiny_imagenet':
                x = x.view(-1, 224 * 224 * 3)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            # Apply initial normalization if specified
            if self.norm_init is not None:
                x = self.norm_init(x)

            # Layer 1
            x = self.fc1(x)
            x = self.act(x)
            if self.norm_fc1 is not None:
                x = self.norm_fc1(x)

            # Layer 2
            x = self.fc2(x)
            x = self.act(x)
            if self.norm_fc2 is not None:
                x = self.norm_fc2(x)

            # Layer 3
            x = self.fc3(x)
            x = self.act(x)
            if self.norm_fc3 is not None:
                x = self.norm_fc3(x)

            # Layer 4 (output layer)
            x = self.fc4(x)

            return x

    return SmallerFCModel()
