from utils import mask_topk
import torch
import torch.nn as nn

def get_COMET_conv_router(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons, layer_4_neurons,
                      topk_rate, norm, activation):
    """
    Returns 'Conv-Routed COMET'.
    
    Innovation:
    - Replaces the first Dense Projection of the Router with a Random Convolutional Projection.
    - Preserves the "Fixed/Random" philosophy (no training of router).
    - Respects data topology (Locality) to break background dominance.
    - Backbone remains a standard MLP.
    """

    class ConvRouterModel(nn.Module):
        def __init__(self):
            super(ConvRouterModel, self).__init__()

            if dataset_name != 'cifar10':
                raise ValueError("Conv-Router COMET currently hardcoded for CIFAR-10")

            self.input_dim_backbone = 32 * 32 * 3
            # We will project to a feature dimension that matches the layer width for simplicity, 
            # or a smaller 'feature' dim. Let's use a modest number of filters.
            self.num_filters = 128 

            # --- 1. Backbone Network (Standard MLP) ---
            self.fc1 = nn.Linear(self.input_dim_backbone, layer_1_neurons, bias=True)
            self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=True)
            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=True)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=True)

            # --- 2. Routing Network (The Innovation) ---
            # Layer 1 Router: Random Conv -> Pool -> TopK
            # We use a large-ish kernel (patch) to capture local structure
            self.router_conv = nn.Conv2d(3, self.num_filters, kernel_size=5, stride=2, padding=2, bias=False)
            
            # Project the pooled features to the mask size
            # Output of conv (stride 2) on 32x32 is 16x16. 
            # We will Global Average Pool to get a (B, Num_Filters) vector.
            # This makes the router Translation Invariant (fixes object shifting).
            self.router_fc1 = nn.Linear(self.num_filters, layer_1_neurons, bias=False)
            
            # Subsequent layers are standard Dense Routers (projecting previous state)
            self.spec_2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=False)
            self.spec_3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=False)
            self.spec_4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=False)

            # Freeze routing weights
            self.router_conv.weight.requires_grad = False
            self.router_fc1.weight.requires_grad = False
            for spec_layer in [self.spec_2, self.spec_3, self.spec_4]:
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
            
            # Internal masks storage for analysis
            self.layer_1_mask = None
            self.layer_2_mask = None
            self.layer_3_mask = None

        def _get_conv_features(self, x):
            # Reshape
            x_img = x.view(-1, 3, 32, 32)
            
            # 1. Random Conv (Extract Texture/Features)
            feat = self.router_conv(x_img) # (B, 128, 16, 16)
            
            # 2. Global Average Pooling (Spatial Invariance)
            # This kills the "Background is in the corner" bias because we average everything.
            # It forces the router to ask "Is there a wing feature ANYWHERE?" rather than "Is pixel 0,0 blue?"
            feat = feat.mean(dim=[2, 3]) # (B, 128)
            
            return feat

        def forward(self, x):
            x_flat = x.view(-1, self.input_dim_backbone)

            # --- Layer 1 ---
            # Backbone (Dense)
            x_nn = self.fc1(x_flat)
            x_nn = self.act(x_nn)
            
            # Router (Conv) [Image of convolution operation]
            with torch.no_grad():
                x_feat = self._get_conv_features(x_flat) 
                x_spec1 = self.router_fc1(x_feat)
                mask1 = mask_topk(x_spec1, self.top_k)
                self.layer_1_mask = mask1
            x = x_nn * mask1

            if self.norm_fc1 is not None: x = self.norm_fc1(x)

            # --- Layer 2 ---
            x_nn = self.fc2(x)
            x_nn = self.act(x_nn)
            with torch.no_grad():
                if self.norm_fc1 is not None: x_spec1 = self.norm_fc1(x_spec1)
                x_spec2 = self.spec_2(x_spec1 * mask1)
                mask2 = mask_topk(x_spec2, self.top_k)
                self.layer_2_mask = mask2
            x = x_nn * mask2

            if self.norm_fc2 is not None: x = self.norm_fc2(x)

            # --- Layer 3 ---
            x_nn = self.fc3(x)
            x_nn = self.act(x_nn)
            with torch.no_grad():
                if self.norm_fc2 is not None: x_spec2 = self.norm_fc2(x_spec2)
                x_spec3 = self.spec_3(x_spec2 * mask2)
                mask3 = mask_topk(x_spec3, self.top_k)
                self.layer_3_mask = mask3
            x = x_nn * mask3

            if self.norm_fc3 is not None: x = self.norm_fc3(x)

            # --- Output Layer ---
            x = self.fc4(x)

            return x

    return ConvRouterModel()