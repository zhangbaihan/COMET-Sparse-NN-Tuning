from utils import mask_topk
import torch
import torch.nn as nn

def get_COMET_center(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons, layer_4_neurons,
                      topk_rate, norm, activation):
    """
    Returns 'Center-Focus COMET'.
    
    Innovation:
    - Applies a Fixed Gaussian Spatial Mask to the Router's input.
    - Preserves RGB Color (crucial for distinguishing objects).
    - Suppresses Background (crucial for fixing Ship/Plane bias).
    - Router focuses only on the object in the middle of the 32x32 frame.
    """

    class CenterSpecModel(nn.Module):
        def __init__(self):
            super(CenterSpecModel, self).__init__()

            if dataset_name != 'cifar10':
                raise ValueError("Center-Focus COMET currently hardcoded for CIFAR-10")

            self.img_size = 32
            self.input_dim = 32 * 32 * 3

            # --- 1. Fixed Spatial Mask (The Innovation) ---
            # Create a 2D Gaussian heatmap centered on the image
            sigma = 8.0 # Focus on the inner 50% of the image
            x = torch.arange(self.img_size).float() - (self.img_size - 1) / 2
            gauss_1d = torch.exp(-(x**2) / (2 * sigma**2))
            gauss_2d = (gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0))
            
            # Normalize so center is 1.0 (keep original brightness)
            gauss_2d = gauss_2d / gauss_2d.max()
            
            # Expand to (1, 3, 32, 32) to mask all channels equally
            self.register_buffer('center_mask', gauss_2d.view(1, 1, self.img_size, self.img_size).repeat(1, 3, 1, 1))

            # --- 2. Backbone Network (Standard) ---
            self.fc1 = nn.Linear(self.input_dim, layer_1_neurons, bias=True)
            self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=True)
            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=True)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=True)

            # --- 3. Routing Network ---
            self.spec_1 = nn.Linear(self.input_dim, layer_1_neurons, bias=False)
            self.spec_2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=False)
            self.spec_3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=False)
            self.spec_4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=False)

            # Freeze routing weights
            for spec_layer in [self.spec_1, self.spec_2, self.spec_3, self.spec_4]:
                spec_layer.weight.requires_grad = False

            self.top_k = topk_rate

            # Normalization
            if norm == 'layer':
                self.norm_fc1 = nn.LayerNorm(layer_1_neurons)
                self.norm_fc2 = nn.LayerNorm(layer_2_neurons)
                self.norm_fc3 = nn.LayerNorm(layer_3_neurons)
            else:
                self.norm_fc1 = None; self.norm_fc2 = None; self.norm_fc3 = None

            # Activation
            activations = {'softplus': nn.Softplus(), 'relu': nn.ReLU(), 'tanh': nn.Tanh()}
            self.act = activations.get(activation, nn.Softplus())

        def _apply_center_mask(self, x):
            # Reshape (B, 3072) -> (B, 3, 32, 32)
            x_img = x.view(-1, 3, self.img_size, self.img_size)
            
            # Apply Mask             x_masked = x_img * self.center_mask
            
            # Flatten
            return x_masked.view(-1, self.input_dim)

        def forward(self, x):
            x_flat = x.view(-1, self.input_dim)

            # --- Layer 1 ---
            # Backbone Path (Sees Everything)
            x_nn = self.fc1(x_flat)
            x_nn = self.act(x_nn)
            
            # Routing Path (Sees Center Only)
            with torch.no_grad():
                x_center = self._apply_center_mask(x_flat)
                x_spec1 = self.spec_1(x_center)
                mask1 = mask_topk(x_spec1, self.top_k)
            x = x_nn * mask1

            if self.norm_fc1 is not None: x = self.norm_fc1(x)

            # --- Layer 2 ---
            x_nn = self.fc2(x)
            x_nn = self.act(x_nn)
            with torch.no_grad():
                if self.norm_fc1 is not None: x_spec1 = self.norm_fc1(x_spec1)
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

    return CenterSpecModel()