from utils import mask_topk
import torch
import torch.nn as nn

def get_Top_k_FC_model(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons, layer_4_neurons,
                       topk_rate, norm, activation):
    """
    An MLP with a trainable routing function. The cap operation is applied directly to the backbone network by replacing eq. (5) with mℓ = Ckℓ (aℓ), so that the routing function selects the highest k values and masks the remaining ones..
    """

    class Top_k_FC_model(nn.Module):
        def __init__(self):
            super(Top_k_FC_model, self).__init__()
            self.top_k = topk_rate  # fraction of neurons to keep (e.g. 0.2 = top 20%)

            if dataset_name in ['cifar10', 'cifar100']:
                input_dim = 32 * 32 * 3
            elif dataset_name == 'tiny_imagenet':
                input_dim = 224 * 224 * 3
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            self.fc1 = nn.Linear(input_dim, layer_1_neurons, bias=True)
            self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=True)
            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=True)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=True)

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

            # Activation function selection
            activations = {
                'gelu': nn.GELU(),
                'softplus': nn.Softplus(),
                'leaky': nn.LeakyReLU(0.1),
                'tanh': nn.Tanh()
            }
            if activation not in activations:
                raise ValueError(f"Unsupported activation: {activation}")
            self.act = activations[activation]

        def forward(self, x):
            if dataset_name in ['cifar10', 'cifar100']:
                x = x.view(-1, 32 * 32 * 3)
            elif dataset_name == 'tiny_imagenet':
                x = x.view(-1, 224 * 224 * 3)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            if self.norm_init is not None:
                x = self.norm_init(x)

            x = self.fc1(x)
            x = self.act(x)
            with torch.no_grad():
                mask = mask_topk(x, self.top_k)
            x = x * mask

            if self.norm_fc1 is not None:
                x = self.norm_fc1(x)

            x = self.fc2(x)
            x = self.act(x)
            with torch.no_grad():
                mask = mask_topk(x, self.top_k)
            x = x * mask

            if self.norm_fc2 is not None:
                x = self.norm_fc2(x)

            x = self.fc3(x)
            x = self.act(x)
            with torch.no_grad():
                mask = mask_topk(x, self.top_k)
            x = x * mask

            if self.norm_fc3 is not None:
                x = self.norm_fc3(x)

            x = self.fc4(x)  # logits output for loss like CrossEntropyLoss
            return x

    return Top_k_FC_model()
