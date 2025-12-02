import torch
import torch.nn as nn
from utils import mask_topk

def get_COMET_foveal(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons, layer_4_neurons,
                      topk_rate, norm, activation):
    """
    Returns 'Foveated COMET'.
    
    The 'Ship-Sink' Solution:
    - Previous failures (Center, Structure) removed statistical bias but left Spatial Bias.
    - The router was still 'seeing' the background pixels and grouping Smooth Sky with Smooth Water.
    - This model applies a fixed Gaussian Spatial Mask (Foveation) to the Router's input.
    - This mechanically forces the router to ignore peripheral pixels (background) 
      and index experts solely based on the central object.
    """

    class FovealSpecModel(nn.Module):
        def __init__(self):
            super(FovealSpecModel, self).__init__()

            if dataset_name == 'cifar10':
                self.img_dim = 32
                self.input_dim = 32 * 32 * 3
                # Standard Deviation for the Gaussian mask
                # 32px image -> sigma=8 ensures the corners are decayed effectively
                self.sigma = 8.0 
            else:
                # Default fallback
                self.img_dim = int(layer_1_neurons**0.5) # Rough guess if not provided
                self.input_dim = layer_1_neurons
                self.sigma = self.img_dim / 4

            # --- 1. Fixed Foveal Mask ---
            # We create this once and register it as a buffer (saved with model, not trained)
            mask = self._create_gaussian_mask(self.img_dim, self.sigma)
            # Expand to (1, 3, H, W) for broadcasting
            self.register_buffer('foveal_mask', mask.view(1, 1, self.img_dim, self.img_dim).repeat(1, 3, 1, 1))

            # --- 2. Backbone (Standard) ---
            self.fc1 = nn.Linear(self.input_dim, layer_1_neurons)
            self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons)
            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons)

            # --- 3. Routing Network (Standard Dimensions) ---
            self.spec_1 = nn.Linear(self.input_dim, layer_1_neurons, bias=False)
            self.spec_2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=False)
            self.spec_3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=False)
            self.spec_4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=False)

            # Freeze routing weights
            for spec in [self.spec_1, self.spec_2, self.spec_3, self.spec_4]:
                spec.weight.requires_grad = False

            self.top_k = topk_rate

            # Norm & Act
            self.norm_init = nn.LayerNorm(self.input_dim) if norm == 'layer' else None
            activations = {'softplus': nn.Softplus(), 'relu': nn.ReLU(), 'tanh': nn.Tanh()}
            self.act = activations.get(activation, nn.Softplus())

            # Analysis hooks
            self.layer_1_mask = None
            self.layer_2_mask = None
            self.layer_3_mask = None

        def _create_gaussian_mask(self, size, sigma):
            """Generates a 2D Gaussian mask centered on the image."""
            # Coordinate grid
            x = torch.arange(size).float()
            y = torch.arange(size).float()
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            
            # Center coordinates
            center = (size - 1) / 2.0
            
            # Distance from center
            dist_sq = (xx - center)**2 + (yy - center)**2
            
            # Gaussian
            mask = torch.exp(-dist_sq / (2 * sigma**2))
            
            # Normalize peak to 1.0 (we want to preserve the center, not scale it down)
            mask = mask / mask.max()
            return mask

        def forward(self, x):
            # x shape: [Batch, Dim] or [Batch, C, H, W]
            # Ensure flattening for FC layers
            x_flat = x.view(-1, self.input_dim)
            
            # View as image for Foveation
            x_img = x.view(-1, 3, self.img_dim, self.img_dim)

            if self.norm_init: x_backbone = self.norm_init(x_flat)
            else: x_backbone = x_flat

            # --- Layer 1 ---
            x_nn = self.act(self.fc1(x_backbone))
            
            # ROUTER SEES FOVEATED INPUT
            with torch.no_grad():
                # Apply mask: Corners -> 0, Center -> Original
                x_foveal = x_img * self.foveal_mask
                x_foveal_flat = x_foveal.view(-1, self.input_dim)
                
                x_spec1 = self.spec_1(x_foveal_flat)
                mask1 = mask_topk(x_spec1, self.top_k)
                self.layer_1_mask = mask1
            x = x_nn * mask1

            # --- Layer 2 ---
            x_nn = self.act(self.fc2(x))
            with torch.no_grad():
                # Note: Deep layers route on activations, which are already
                # "filtered" by the first layer's mask. We don't need to mask again.
                x_spec2 = self.spec_2(x_spec1 * mask1)
                mask2 = mask_topk(x_spec2, self.top_k)
                self.layer_2_mask = mask2
            x = x_nn * mask2

            # --- Layer 3 ---
            x_nn = self.act(self.fc3(x))
            with torch.no_grad():
                x_spec3 = self.spec_3(x_spec2 * mask2)
                mask3 = mask_topk(x_spec3, self.top_k)
                self.layer_3_mask = mask3
            x = x_nn * mask3

            # --- Output ---
            x = self.fc4(x)
            return x

    return FovealSpecModel()