from utils import mask_topk
import torch
import torch.nn as nn

def get_COMET_structure(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons, layer_4_neurons,
                      topk_rate, norm, activation):
    """
    Returns 'Structure-Routed COMET'.
    
    Innovation:
    - Router sees Grayscale + Instance Normalized input.
    - Reduces router input dim (1024 vs 3072), creating an Overcomplete Projection (1024->3000).
    - Removes Color Bias (Ship/Plane blue confusion) and Lighting Bias.
    """

    class StructureSpecModel(nn.Module):
        def __init__(self):
            super(StructureSpecModel, self).__init__()

            if dataset_name != 'cifar10':
                raise ValueError("COMET_structure currently hardcoded for CIFAR-10 dims")

            self.input_dim_backbone = 32 * 32 * 3
            self.input_dim_router = 32 * 32 * 1  # Reduced dimension (Grayscale)

            # --- 1. Backbone Network (Standard MLP) ---
            self.fc1 = nn.Linear(self.input_dim_backbone, layer_1_neurons, bias=True)
            self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=True)
            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=True)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=True)

            # --- 2. Routing Network (The Innovation) ---
            # Input dimension is now 1024 (Grayscale) instead of 3072
            # This creates a 1:3 Expansion Ratio to the hidden layer (3000),
            # improving the separability of the random projection.
            self.spec_1 = nn.Linear(self.input_dim_router, layer_1_neurons, bias=False)
            self.spec_2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=False)
            self.spec_3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=False)
            self.spec_4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=False)

            # Freeze routing weights
            for spec_layer in [self.spec_1, self.spec_2, self.spec_3, self.spec_4]:
                spec_layer.weight.requires_grad = False

            self.top_k = topk_rate

            # Normalization (Backbone)
            if norm == 'layer':
                self.norm_fc1 = nn.LayerNorm(layer_1_neurons)
                self.norm_fc2 = nn.LayerNorm(layer_2_neurons)
                self.norm_fc3 = nn.LayerNorm(layer_3_neurons)
            else:
                self.norm_fc1 = None; self.norm_fc2 = None; self.norm_fc3 = None

            # Activation
            activations = {'softplus': nn.Softplus(), 'relu': nn.ReLU(), 'tanh': nn.Tanh()}
            self.act = activations.get(activation, nn.Softplus())

            # Instance Norm for the Router
            # We treat the image as having 1 channel and 1024 length.
            # num_features=1 ensures it normalizes that single channel using stats from the length dim.
            self.router_norm = nn.InstanceNorm1d(1, affine=False)
            
            # Internal masks storage for analysis
            self.layer_1_mask = None
            self.layer_2_mask = None
            self.layer_3_mask = None

        def _get_structure_map(self, x):
            """
            Transforms (B, 3072) RGB -> (B, 1024) Normalized Grayscale
            """
            # Reshape to Image: (B, 3, 32, 32)
            x_img = x.view(-1, 3, 32, 32)
            
            # 1. Grayscale (Luminance)
            # 0.299 R + 0.587 G + 0.114 B
            x_gray = 0.299 * x_img[:, 0:1] + 0.587 * x_img[:, 1:2] + 0.114 * x_img[:, 2:3]
            
            # Flatten: (B, 1024)
            x_flat = x_gray.view(-1, self.input_dim_router)
            
            # 2. Instance Normalization
            # Input to InstanceNorm1d must be (Batch, Channels, Length) -> (B, 1, 1024)
            x_norm = self.router_norm(x_flat.unsqueeze(1)).squeeze(1)
            
            return x_norm

        def forward(self, x):
            x_flat = x.view(-1, self.input_dim_backbone)

            # --- Layer 1 ---
            # Backbone Path
            x_nn = self.fc1(x_flat)
            x_nn = self.act(x_nn)
            
            # Routing Path
            with torch.no_grad():
                x_struct = self._get_structure_map(x_flat) 
                x_spec1 = self.spec_1(x_struct)
                mask1 = mask_topk(x_spec1, self.top_k)
                self.layer_1_mask = mask1 # Save for analysis!
            x = x_nn * mask1

            if self.norm_fc1 is not None: x = self.norm_fc1(x)

            # --- Layer 2 ---
            x_nn = self.fc2(x)
            x_nn = self.act(x_nn)
            with torch.no_grad():
                if self.norm_fc1 is not None: x_spec1 = self.norm_fc1(x_spec1)
                x_spec2 = self.spec_2(x_spec1 * mask1)
                mask2 = mask_topk(x_spec2, self.top_k)
                self.layer_2_mask = mask2 # Save for analysis!
            x = x_nn * mask2

            if self.norm_fc2 is not None: x = self.norm_fc2(x)

            # --- Layer 3 ---
            x_nn = self.fc3(x)
            x_nn = self.act(x_nn)
            with torch.no_grad():
                if self.norm_fc2 is not None: x_spec2 = self.norm_fc2(x_spec2)
                x_spec3 = self.spec_3(x_spec2 * mask2)
                mask3 = mask_topk(x_spec3, self.top_k)
                self.layer_3_mask = mask3 # Save for analysis!
            x = x_nn * mask3

            if self.norm_fc3 is not None: x = self.norm_fc3(x)

            # --- Output Layer ---
            x = self.fc4(x)

            return x

    return StructureSpecModel()