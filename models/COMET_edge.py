from utils import mask_topk
import torch
import torch.nn as nn

def get_COMET_edge(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons, layer_4_neurons,
                      topk_rate, norm, activation):
    """
    Returns 'Edge-Routed COMET'.
    
    Innovation:
    - The Backbone sees the full RGB input.
    - The Router sees only a Grayscale Laplacian Edge Map.
    - This forces expert selection to be color-invariant and shape-dependent.
    """

    class EdgeSpecModel(nn.Module):
        def __init__(self):
            super(EdgeSpecModel, self).__init__()

            if dataset_name != 'cifar10':
                raise ValueError("Edge-Routed COMET currently hardcoded for CIFAR-10/Image Data")

            # Input Dimensions
            self.img_size = 32
            self.input_dim_backbone = 32 * 32 * 3
            self.input_dim_router = 32 * 32 * 1  # 1 Channel after Grayscale/Edge

            # --- 1. Fixed Preprocessing (The Edge Detector) ---
            # We use a standard Laplacian Kernel for edge detection
            self.edge_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
            laplacian_kernel = torch.tensor([[[[0, 1, 0], 
                                               [1, -4, 1], 
                                               [0, 1, 0]]]], dtype=torch.float32)
            self.edge_conv.weight = nn.Parameter(laplacian_kernel, requires_grad=False)

            # --- 2. Backbone Network (Standard MLP) ---
            self.fc1 = nn.Linear(self.input_dim_backbone, layer_1_neurons, bias=True)
            self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=True)
            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=True)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=True)

            # --- 3. Routing Network ---
            # NOTE: spec_1 now accepts the smaller Edge Map dimension!
            self.spec_1 = nn.Linear(self.input_dim_router, layer_1_neurons, bias=False)
            self.spec_2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=False)
            self.spec_3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=False)
            self.spec_4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=False)

            # Freeze routing weights
            for spec_layer in [self.spec_1, self.spec_2, self.spec_3, self.spec_4]:
                spec_layer.weight.requires_grad = False

            self.top_k = topk_rate

            # Normalization (Backbone only)
            if norm == 'layer':
                self.norm_fc1 = nn.LayerNorm(layer_1_neurons)
                self.norm_fc2 = nn.LayerNorm(layer_2_neurons)
                self.norm_fc3 = nn.LayerNorm(layer_3_neurons)
            else:
                self.norm_fc1 = None; self.norm_fc2 = None; self.norm_fc3 = None

            # Activation
            activations = {'softplus': nn.Softplus(), 'relu': nn.ReLU(), 'tanh': nn.Tanh()}
            self.act = activations.get(activation, nn.Softplus())

        def _get_edge_map(self, x):
            """
            Transforms (B, 3072) RGB -> (B, 1024) Edge Map
            """
            # Reshape to Image: (B, 3, 32, 32)
            x_img = x.view(-1, 3, self.img_size, self.img_size)
            
            # Convert to Grayscale (Standard luminance weights)
            # 0.299 R + 0.587 G + 0.114 B
            x_gray = 0.299 * x_img[:, 0:1] + 0.587 * x_img[:, 1:2] + 0.114 * x_img[:, 2:3]
            
            # Apply Laplacian Edge Detection
            x_edge = self.edge_conv(x_gray)
            
            # Flatten back to vector
            return x_edge.view(-1, self.input_dim_router)

        def forward(self, x):
            # Flatten backbone input if needed
            x_flat = x.view(-1, self.input_dim_backbone)

            # --- Layer 1 ---
            # Backbone path sees COLOR
            x_nn = self.fc1(x_flat)
            x_nn = self.act(x_nn)
            
            # Routing path sees EDGES 
            with torch.no_grad():
                x_edges = self._get_edge_map(x_flat) # Innovation happens here
                x_spec1 = self.spec_1(x_edges)       # Router projects the Edges
                mask1 = mask_topk(x_spec1, self.top_k)
            x = x_nn * mask1

            if self.norm_fc1 is not None: x = self.norm_fc1(x)

            # --- Layer 2 (Standard COMET Propagation) ---
            # Once we are in hidden space, we route based on the previous active neurons
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

    return EdgeSpecModel()