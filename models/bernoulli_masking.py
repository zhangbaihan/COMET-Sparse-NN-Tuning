import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_bernoulli_masking(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons,
                          layer_4_neurons, topk_rate, norm, activation,
                          layer_1_vector_dict, layer_2_vector_dict, layer_3_vector_dict):
    """
    An MLP where each training example is associated with a fixed binary mask drawn from a Bernoulli distribution, with probability equal to pk. 
    Thus the relationship between inputs and their masks is arbitrary, rather than being mediated by the routing network in COMET
    """
    
    class BernoulliModel(nn.Module):
        def __init__(self):
            super().__init__()

            # Input layer setup
            if dataset_name in ['cifar10', 'cifar100']:
                input_dim = 32 * 32 * 3
            elif dataset_name == 'tiny_imagenet':
                input_dim = 224 * 224 * 3
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            self.fc1 = nn.Linear(input_dim, layer_1_neurons)
            self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons)
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
                self.norm_init = self.norm_fc1 = self.norm_fc2 = self.norm_fc3 = None

            # Activation function
            self.act = {
                'gelu': nn.GELU(),
                'softplus': nn.Softplus(),
                'leaky': nn.LeakyReLU(0.1),
                'tanh': nn.Tanh()
            }.get(activation, nn.ReLU())  # default fallback

            # Vector dictionaries for training masking
            self.layer_1_vector_dict = layer_1_vector_dict
            self.layer_2_vector_dict = layer_2_vector_dict
            self.layer_3_vector_dict = layer_3_vector_dict

        def forward(self, x, setup='train'):
            if dataset_name in ['cifar10', 'cifar100']:
                x = x.view(-1, 32 * 32 * 3)
            elif dataset_name == 'tiny_imagenet':
                x = x.view(-1, 224 * 224 * 3)

            if self.norm_init:
                x = self.norm_init(x)

            if setup == 'train':
                keys = [f'{tensor[0, 0, 0].item()}{tensor[1, 1, 1].item()}' for tensor in x.view(-1, 3, 32, 32)]
                layer_1_vecs = torch.stack([self.layer_1_vector_dict[k] for k in keys]).to(device)
                layer_2_vecs = torch.stack([self.layer_2_vector_dict[k] for k in keys]).to(device)
                layer_3_vecs = torch.stack([self.layer_3_vector_dict[k] for k in keys]).to(device)

            # Layer 1
            x = self.act(self.fc1(x))
            if setup == 'train':
                x = x * layer_1_vecs
            if self.norm_fc1:
                x = self.norm_fc1(x)

            # Layer 2
            x = self.act(self.fc2(x))
            if setup == 'train':
                x = x * layer_2_vecs
            if self.norm_fc2:
                x = self.norm_fc2(x)

            # Layer 3
            x = self.act(self.fc3(x))
            if setup == 'train':
                x = x * layer_3_vecs
            if self.norm_fc3:
                x = self.norm_fc3(x)

            # Final layer
            x = self.fc4(x)
            return x

    return BernoulliModel().to(device)
