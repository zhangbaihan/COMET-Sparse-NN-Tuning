import torch
import torch.nn as nn
import torch.nn.init as init

def mask_topk(x, k_rate):
    if k_rate >= 1.0: return torch.ones_like(x)
    k = max(1, int(x.shape[-1] * k_rate))
    topk_values, _ = torch.topk(x, k, dim=-1)
    kth_value = topk_values[:, -1].unsqueeze(-1)
    return (x >= kth_value).float()

def orthogonal_init_(tensor, gain=1.0):
    if tensor.ndimension() < 2: return
    rows, cols = tensor.size(0), tensor.size(1)
    if rows <= cols:
        init.orthogonal_(tensor, gain=gain)
    else:
        with torch.no_grad():
            num_blocks = rows // cols
            remainder = rows % cols
            for i in range(num_blocks):
                init.orthogonal_(tensor[i*cols : (i+1)*cols, :], gain=gain)
            if remainder > 0:
                block = torch.empty(cols, cols, device=tensor.device)
                init.orthogonal_(block, gain=gain)
                tensor[num_blocks*cols :, :] = block[:remainder, :]

class CenteringLayer(nn.Module):
    """
    A domain-agnostic layer that centers the input vector by subtracting its mean.
    This removes the 'DC component' (e.g. background intensity) effectively acting
    as a high-pass filter in the statistical domain.
    """
    def forward(self, x):
        # x shape: [Batch, D]
        # Calculate mean across the feature dimension D
        mean = x.mean(dim=1, keepdim=True)
        return x - mean

def get_COMET_center(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons, layer_4_neurons,
                     topk_rate, norm, activation):
    """
    COMET Model with Input Centering for the Router.
    This is a general-purpose solution to the 'Background Bias' problem.
    """

    class CenterModel(nn.Module):
        def __init__(self):
            super(CenterModel, self).__init__()

            # 1. Dimensions
            if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                input_dim = 32 * 32 * 3
            elif dataset_name == 'tiny_imagenet':
                input_dim = 224 * 224 * 3
            elif dataset_name == 'SARCOS':
                input_dim = 21
            elif dataset_name == 'mnist':
                input_dim = 784
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            # 2. Backbone Layers (Trainable)
            self.fc1 = nn.Linear(input_dim, layer_1_neurons)
            self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons)
            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons)

            # 3. Centering Pre-Processor (Router Only)
            self.center = CenteringLayer()

            # 4. Routing Layers (Fixed)
            self.spec_1 = nn.Linear(input_dim, layer_1_neurons, bias=False)
            self.spec_2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=False)
            self.spec_3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=False)
            self.spec_4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=False)

            for spec in [self.spec_1, self.spec_2, self.spec_3, self.spec_4]:
                spec.weight.requires_grad = False

            self.top_k = topk_rate

            # 5. Norm & Act
            self.norm_init = nn.LayerNorm(input_dim) if norm == 'layer' else None
            
            activations = {'gelu': nn.GELU(), 'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'softplus': nn.Softplus()}
            self.act = activations.get(activation, nn.ReLU())

            # Masks saved for inspection
            self.layer_1_mask = None
            self.layer_2_mask = None
            self.layer_3_mask = None

            # 6. Init (Orthogonal)
            self.apply_orthogonal_init(activation)

        def apply_orthogonal_init(self, act_name):
            try: gain = init.calculate_gain(act_name)
            except: gain = 1.0
            
            for m in [self.fc1, self.fc2, self.fc3]:
                orthogonal_init_(m.weight, gain=gain)
                if m.bias is not None: init.constant_(m.bias, 0)
            
            orthogonal_init_(self.fc4.weight, gain=1.0)
            
            # Router Gain
            for m in [self.spec_1, self.spec_2, self.spec_3, self.spec_4]:
                orthogonal_init_(m.weight, gain=1.41)

        def forward(self, x):
            # Flatten
            if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                x_flat = x.view(-1, 32 * 32 * 3)
            elif dataset_name == 'mnist':
                x_flat = x.view(-1, 784)
            else:
                x_flat = x.view(-1, 21)

            # Normalization (Backbone)
            if self.norm_init: x_backbone = self.norm_init(x_flat)
            else: x_backbone = x_flat

            # --- ROUTER INPUT PRE-PROCESSING ---
            # Center the input specifically for the router
            with torch.no_grad():
                x_router = self.center(x_flat)
                
            # Layer 1
            x_nn = self.act(self.fc1(x_backbone))
            with torch.no_grad():
                # Router sees CENTERED input
                # Removes global intensity bias (background)
                x_spec1 = self.spec_1(x_router)
                mask1 = mask_topk(x_spec1, self.top_k)
                self.layer_1_mask = mask1
            x = x_nn * mask1

            # Layer 2
            x_nn = self.act(self.fc2(x))
            with torch.no_grad():
                # For deeper layers, we can center the previous activations too!
                # This ensures consistent behavior deep in the network.
                x_spec2_input = self.center(x_spec1 * mask1)
                x_spec2 = self.spec_2(x_spec2_input) 
                mask2 = mask_topk(x_spec2, self.top_k)
                self.layer_2_mask = mask2
            x = x_nn * mask2

            # Layer 3
            x_nn = self.act(self.fc3(x))
            with torch.no_grad():
                x_spec3_input = self.center(x_spec2 * mask2)
                x_spec3 = self.spec_3(x_spec3_input)
                mask3 = mask_topk(x_spec3, self.top_k)
                self.layer_3_mask = mask3
            x = x_nn * mask3

            # Output
            x = self.fc4(x)
            return x

    return CenterModel()