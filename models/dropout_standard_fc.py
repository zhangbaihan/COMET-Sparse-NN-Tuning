import torch
import torch.nn as nn

def get_dropout_standard_fc(dataset_name, layer_1_neurons, layer_2_neurons,
                             layer_3_neurons, layer_4_neurons, norm,
                             activation, dropout_rate):
    """
    A standard model with a dropout rate equal to 1 − pk
    """

    class DropoutFCModel(nn.Module):
        def __init__(self):
            super().__init__()

            if dataset_name in ['cifar10', 'cifar100']:
                input_dim = 32 * 32 * 3
            elif dataset_name == 'tiny_imagenet':
                input_dim = 224 * 224 * 3
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            # Fully connected layers
            self.fc1 = nn.Linear(input_dim, layer_1_neurons)
            self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons)
            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons)

            # Normalization
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
                self.norm_init = self.norm_fc1 = self.norm_fc2 = self.norm_fc3 = None

            # Activation
            self.act = {
                'gelu': nn.GELU(),
                'softplus': nn.Softplus(),
                'leaky': nn.LeakyReLU(0.1),
                'tanh': nn.Tanh()
            }.get(activation, nn.ReLU())  # fallback to ReLU if not specified

            # Dropout
            self.dropout = nn.Dropout(1 - dropout_rate)

        def forward(self, x):
            if dataset_name in ['cifar10', 'cifar100']:
                x = x.view(-1, 32 * 32 * 3)
            elif dataset_name == 'tiny_imagenet':
                x = x.view(-1, 224 * 224 * 3)

            if self.norm_init:
                x = self.norm_init(x)

            x = self.act(self.fc1(x))
            x = self.dropout(x)
            if self.norm_fc1:
                x = self.norm_fc1(x)

            x = self.act(self.fc2(x))
            x = self.dropout(x)
            if self.norm_fc2:
                x = self.norm_fc2(x)

            x = self.act(self.fc3(x))
            x = self.dropout(x)
            if self.norm_fc3:
                x = self.norm_fc3(x)

            x = self.fc4(x)
            return x

    return DropoutFCModel()
