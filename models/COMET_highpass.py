from utils import mask_topk
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_COMET_highpass(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons, layer_4_neurons,
                      topk_rate, norm, activation):
    """
    Returns 'High-Pass Routed COMET'.
    
    Innovation:
    - Preprocesses Router input using Unsharp Masking (High-Pass Filter).
    - Subtracts the smoothed (low-freq) version of the image from the original.
    - Removes dominant background color bias (Blue Sky/Water) while preserving texture/shape.
    """

    class HighPassSpecModel(nn.Module):
        def __init__(self):
            super(HighPassSpecModel, self).__init__()

            if dataset_name != 'cifar10':
                raise ValueError("High-Pass COMET currently hardcoded for CIFAR-10 dims")

            self.input_dim = 32 * 32 * 3

            # --- 1. Fixed Preprocessing (Gaussian Blur) ---
            # We create a depthwise convolution to blur each RGB channel independently
            self.blur_kernel_size = 5
            self.blur_sigma = 1.0
            # Create a fixed Gaussian kernel
            kernel = self._get_gaussian_kernel(self.blur_kernel_size, self.blur_sigma)
            # Expand to 3 channels (Depthwise conv: groups=3)
            self.blur_weight = kernel.view(1, 1, self.blur_kernel_size, self.blur_kernel_size).repeat(3, 1, 1, 1)
            
            # --- 2. Backbone Network (Standard) ---
            self.fc1 = nn.Linear(self.input_dim, layer_1_neurons, bias=True)
            self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=True)
            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=True)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=True)

            # --- 3. Routing Network ---
            # Takes the same dimension input, but preprocessed
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

        def _get_gaussian_kernel(self, kernel_size, sigma):
            # Create a 1D Gaussian distribution
            x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
            gauss = torch.exp(-(x**2) / (2 * sigma**2))
            gauss = gauss / gauss.sum()
            # Outer product to make 2D
            return (gauss.unsqueeze(1) @ gauss.unsqueeze(0))

        def _high_pass_filter(self, x):
            """
            Applies Unsharp Masking: Output = Original - Blurred
            """
            b, c, h, w = x.shape
            
            # Pad to maintain dimensions
            pad = self.blur_kernel_size // 2
            
            # Apply fixed Gaussian Blur (Depthwise)
            # Use functional conv2d with the fixed weight we created in init
            # We need to move the weight to the correct device/dtype during forward
            weight = self.blur_weight.to(x.device).type(x.dtype)
            
            x_blurred = F.conv2d(x, weight, padding=pad, groups=3)
            
            # High Pass = Original - Low Pass
            return x - x_blurred

        def forward(self, x):
            # Flatten for Linear Layers
            x_flat = x.view(-1, self.input_dim)
            
            # Reshape for Convolution
            x_img = x.view(-1, 3, 32, 32)

            # --- Layer 1 ---
            # Backbone Path (Sees Raw RGB)
            x_nn = self.fc1(x_flat)
            x_nn = self.act(x_nn)
            
            # Routing Path (Sees High-Pass Texture)
            with torch.no_grad():
                x_high = self._high_pass_filter(x_img)
                x_high_flat = x_high.view(-1, self.input_dim)
                
                x_spec1 = self.spec_1(x_high_flat)
                mask1 = mask_topk(x_spec1, self.top_k)
            x = x_nn * mask1

            if self.norm_fc1 is not None: x = self.norm_fc1(x)

            # --- Standard COMET Propagation for deeper layers ---
            x_nn = self.fc2(x)
            x_nn = self.act(x_nn)
            with torch.no_grad():
                if self.norm_fc1 is not None: x_spec1 = self.norm_fc1(x_spec1)
                x_spec2 = self.spec_2(x_spec1 * mask1)
                mask2 = mask_topk(x_spec2, self.top_k)
            x = x_nn * mask2

            if self.norm_fc2 is not None: x = self.norm_fc2(x)

            x_nn = self.fc3(x)
            x_nn = self.act(x_nn)
            with torch.no_grad():
                if self.norm_fc2 is not None: x_spec2 = self.norm_fc2(x_spec2)
                x_spec3 = self.spec_3(x_spec2 * mask2)
                mask3 = mask_topk(x_spec3, self.top_k)
            x = x_nn * mask3

            if self.norm_fc3 is not None: x = self.norm_fc3(x)

            x = self.fc4(x)
            return x

    return HighPassSpecModel()