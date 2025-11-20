import torch
import torch.nn as nn
from utils import mask_topk

def get_COMET_affine(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons, layer_4_neurons,
                     topk_rate, norm, activation, freeze_backbone=False):
    """
    COMET + per-neuron affine after masking: x = mask ∘ (act(fc) ∘ gamma + beta).
    Routing remains fixed & non-trainable.
    """
    class SpecModel(nn.Module):
        def __init__(self):
            super().__init__()

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

            # routing (fixed)
            self.spec_1 = nn.Linear(input_dim, layer_1_neurons, bias=False)
            self.spec_2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=False)
            self.spec_3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=False)
            self.spec_4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=False)
            for spec_layer in [self.spec_1, self.spec_2, self.spec_3, self.spec_4]:
                spec_layer.weight.requires_grad = False

            self.top_k = topk_rate

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

            # small trainable affine per hidden layer
            self.gamma1 = nn.Parameter(torch.ones(layer_1_neurons))
            self.beta1  = nn.Parameter(torch.zeros(layer_1_neurons))
            self.gamma2 = nn.Parameter(torch.ones(layer_2_neurons))
            self.beta2  = nn.Parameter(torch.zeros(layer_2_neurons))
            self.gamma3 = nn.Parameter(torch.ones(layer_3_neurons))
            self.beta3  = nn.Parameter(torch.zeros(layer_3_neurons))

            if freeze_backbone:
                for m in [self.fc1, self.fc2, self.fc3]:
                    for p in m.parameters():
                        p.requires_grad = False

        def forward(self, x):
            if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                x = x.view(-1, 32 * 32 * 3)
            elif dataset_name == 'SARCOS':
                x = x.view(-1, 21)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            if self.norm_init is not None:
                x = self.norm_init(x)

            # L1
            x_nn = self.act(self.fc1(x))
            with torch.no_grad():
                x_spec1 = self.spec_1(x)
                mask1 = mask_topk(x_spec1, self.top_k)
            x = mask1 * (x_nn * self.gamma1 + self.beta1)
            if self.norm_fc1 is not None:
                x = self.norm_fc1(x)

            # L2
            x_nn = self.act(self.fc2(x))
            with torch.no_grad():
                if self.norm_fc1 is not None:
                    x_spec1 = self.norm_fc1(x_spec1)
                x_spec2 = self.spec_2(x_spec1 * mask1)
                mask2 = mask_topk(x_spec2, self.top_k)
            x = mask2 * (x_nn * self.gamma2 + self.beta2)
            if self.norm_fc2 is not None:
                x = self.norm_fc2(x)

            # L3
            x_nn = self.act(self.fc3(x))
            with torch.no_grad():
                if self.norm_fc2 is not None:
                    x_spec2 = self.norm_fc2(x_spec2)
                x_spec3 = self.spec_3(x_spec2 * mask2)
                mask3 = mask_topk(x_spec3, self.top_k)
            x = mask3 * (x_nn * self.gamma3 + self.beta3)
            if self.norm_fc3 is not None:
                x = self.norm_fc3(x)

            # L4
            x = self.fc4(x)
            return x

    return SpecModel()