from utils import mask_topk
import torch
import torch.nn as nn
import torch.nn.init as init

def get_Orthogonal(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons, layer_4_neurons,
                      topk_rate, norm, activation):
    """
    Returns the 'Orthogonal-Centered COMET'.
    
    Innovations over Baseline:
    1. Orthogonal Initialization of Routing Weights (spec_layers).
    2. Input Centering (Mean Subtraction) for Routing Path only.
    """

    class SpecModel(nn.Module):
        def __init__(self):
            super(SpecModel, self).__init__()

            # Define input dimensions
            if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                input_dim = 32 * 32 * 3
            elif dataset_name == 'SARCOS':
                input_dim = 21
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            # --- 1. Backbone Network (Identical to Baseline) ---
            self.fc1 = nn.Linear(input_dim, layer_1_neurons, bias=True)
            self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=True)
            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=True)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=True)

            # --- 2. Routing Network (The Innovation) ---
            self.spec_1 = nn.Linear(input_dim, layer_1_neurons, bias=False)
            self.spec_2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=False)
            self.spec_3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=False)
            self.spec_4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=False)

            # INNOVATION A: Orthogonal Initialization
            # Forces routing vectors to be maximally distinct, reducing expert overlap collision.
            print("Applying Orthogonal Initialization to Routing Network...")
            init.orthogonal_(self.spec_1.weight)
            init.orthogonal_(self.spec_2.weight)
            init.orthogonal_(self.spec_3.weight)
            init.orthogonal_(self.spec_4.weight)

            # Freeze routing weights (Standard COMET behavior)
            for spec_layer in [self.spec_1, self.spec_2, self.spec_3, self.spec_4]:
                spec_layer.weight.requires_grad = False

            self.top_k = topk_rate

            # Normalization (Standard behavior - we likely won't use this if 'norm' is None)
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

            # Activation
            activations = {'softplus': nn.Softplus(), 'relu': nn.ReLU(), 'tanh': nn.Tanh()}
            self.act = activations.get(activation, nn.Softplus())

        def _get_centered_input(self, x):
            # INNOVATION B: Input Centering for Router
            # Subtracts the mean of the input vector. 
            # Removes "background brightness/color" bias (The Ship Sink killer).
            mean = x.mean(dim=1, keepdim=True)
            return x - mean

        def forward(self, x):
            # Flatten
            if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                x = x.view(-1, 32 * 32 * 3)
            elif dataset_name == 'SARCOS':
                x = x.view(-1, 21)

            # Initial Norm (Backbone only)
            if self.norm_init is not None:
                x = self.norm_init(x)

            # --- Layer 1 ---
            # Backbone Path
            x_nn = self.fc1(x)
            x_nn = self.act(x_nn)
            
            # Routing Path (Enhanced)
            with torch.no_grad():
                # Apply centering ONLY to the router input
                x_router = self._get_centered_input(x)
                x_spec1 = self.spec_1(x_router)
                mask1 = mask_topk(x_spec1, self.top_k)
            x = x_nn * mask1

            if self.norm_fc1 is not None: x = self.norm_fc1(x)

            # --- Layer 2 ---
            x_nn = self.fc2(x)
            x_nn = self.act(x_nn)
            with torch.no_grad():
                if self.norm_fc1 is not None: x_spec1 = self.norm_fc1(x_spec1)
                # Note: We continue projecting the previous spec state, 
                # but the Orthogonal Init in spec_2 helps here too.
                x_spec2 = self.spec_2(x_spec1 * mask1)
                mask2 = mask_topk(x_spec2, self.top_k)
            x = x_nn * mask2

            if self.norm_fc2 is not None: x = self.norm_fc2(x)

            # --- Layer 3 ---
            x_nn = self.fc3(x)
            x_nn = self.act(x_nn)
            with torch.no_grad():
                if self.norm_fc2 is not None: x_spec2 = self.norm_fc2(x_spec2)
                x_spec3 = self.spec_3(x_spec2 * mask2)
                mask3 = mask_topk(x_spec3, self.top_k)
            x = x_nn * mask3

            if self.norm_fc3 is not None: x = self.norm_fc3(x)

            # --- Output Layer ---
            x = self.fc4(x)

            return x

    return SpecModel()