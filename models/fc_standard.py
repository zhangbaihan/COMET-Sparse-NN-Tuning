import torch.nn as nn

def get_standard_model(dataset_name, layer_1_neurons, layer_2_neurons,
                       layer_3_neurons, layer_4_neurons, norm, activation):
    """
    A standard MLP model with the same number of neurons and no sparsity
    """

    class FCModel(nn.Module):
        def __init__(self):
            super().__init__()

            if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                input_dim = 32 * 32 * 3
                self.fc1 = nn.Linear(input_dim, layer_1_neurons)
                self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons)
            elif dataset_name == 'SARCOS':
                input_dim = 21
                self.fc1 = nn.Linear(input_dim, layer_1_neurons)
                self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons)

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

            # Activation function mapping with fallback
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
            self.act = activations.get(activation, nn.ReLU())

        def forward(self, x):
            batch_size = x.size(0)

            if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                x = x.view(batch_size, -1)
            elif dataset_name == 'tiny_imagenet':
                x = x.view(batch_size, -1)
            # SARCOS likely already flat

            if self.norm_init is not None:
                x = self.norm_init(x)

            x = self.fc1(x)
            x = self.act(x)

            if self.norm_fc1 is not None:
                x = self.norm_fc1(x)

            x = self.fc2(x)
            x = self.act(x)

            if self.norm_fc2 is not None:
                x = self.norm_fc2(x)

            x = self.fc3(x)
            x = self.act(x)

            if self.norm_fc3 is not None:
                x = self.norm_fc3(x)

            x = self.fc4(x)
            return x

    return FCModel()
