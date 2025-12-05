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
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        return x - mean

class HybridRouter(nn.Module):
    def __init__(self, input_dim, output_dim, bottleneck_dim=32, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
        # 1. Fixed Branch (The Anchor)
        self.static = nn.Linear(input_dim, output_dim, bias=False)
        self.static.weight.requires_grad = False 
        
        # 2. Semantic Branch (The Guide)
        self.semantic = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim, bias=True),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, output_dim, bias=False)
        )
        
        # Init semantic branch to near-zero to start as pure COMET
        nn.init.normal_(self.semantic[0].weight, std=0.01)
        nn.init.normal_(self.semantic[2].weight, std=0.01)

    def forward(self, x):
        # We assume x is ALREADY centered by the model before being passed here
        return self.static(x) + self.alpha * self.semantic(x)

def get_guided_center(dataset_name, layer_1, layer_2, layer_3, layer_4, 
                      topk_rate, norm, activation, bottleneck_dim=32, alpha=1.0):
    
    class GuidedCenterModel(nn.Module):
        def __init__(self):
            super().__init__()

            # 1. Dimensions
            if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                input_dim = 32 * 32 * 3
            elif dataset_name == 'mnist':
                input_dim = 784
            elif dataset_name == 'SARCOS':
                input_dim = 21
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            # 2. Backbone
            self.fc1 = nn.Linear(input_dim, layer_1)
            self.fc2 = nn.Linear(layer_1, layer_2)
            self.fc3 = nn.Linear(layer_2, layer_3)
            self.fc4 = nn.Linear(layer_3, layer_4)

            # 3. Pre-processor
            self.center = CenteringLayer()

            # 4. Hybrid Routers (Replacing Fixed Specs)
            # Note: We pass bottleneck_dim and alpha
            self.router_1 = HybridRouter(input_dim, layer_1, bottleneck_dim, alpha)
            self.router_2 = HybridRouter(layer_1, layer_2, bottleneck_dim, alpha)
            self.router_3 = HybridRouter(layer_2, layer_3, bottleneck_dim, alpha)

            self.top_k = topk_rate
            self.norm_init = nn.LayerNorm(input_dim) if norm == 'layer' else None
            
            activations = {'gelu': nn.GELU(), 'relu': nn.ReLU(), 'softplus': nn.Softplus()}
            self.act = activations.get(activation, nn.ReLU())

            # Masks
            self.layer_1_mask = None
            self.layer_2_mask = None
            self.layer_3_mask = None

            # 5. Orthogonal Init
            self.apply_orthogonal_init(activation)

        def apply_orthogonal_init(self, act_name):
            try: gain = init.calculate_gain(act_name)
            except: gain = 1.0
            
            for m in [self.fc1, self.fc2, self.fc3]:
                orthogonal_init_(m.weight, gain=gain)
                if m.bias is not None: init.constant_(m.bias, 0)
            
            orthogonal_init_(self.fc4.weight, gain=1.0)
            
            # Init the FIXED part of the routers orthogonally
            # (The learnable part is already init to near-zero in HybridRouter class)
            for router in [self.router_1, self.router_2, self.router_3]:
                orthogonal_init_(router.static.weight, gain=1.41)

        def forward(self, x):
            if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                x_flat = x.view(-1, 32 * 32 * 3)
            else:
                x_flat = x.view(-1, 21)

            # Backbone Norm
            if self.norm_init: x_backbone = self.norm_init(x_flat)
            else: x_backbone = x_flat

            # --- ROUTER PRE-PROCESSING ---
            # We center the input ONCE. 
            # Both the Fixed and Learnable branches of HybridRouter will see this centered input.
            with torch.no_grad():
                x_router = self.center(x_flat)
                
            # Layer 1
            x_nn = self.act(self.fc1(x_backbone))
            
            # Hybrid Routing on Centered Input
            # Note: We allow gradients to flow into the learnable part of the router!
            # So we do NOT use torch.no_grad() for the router call itself, 
            # only for the centering (which has no params anyway).
            scores_1 = self.router_1(x_router) 
            mask1 = mask_topk(scores_1, self.top_k)
            self.layer_1_mask = mask1
            x = x_nn * mask1

            # Layer 2
            x_nn = self.act(self.fc2(x))
            
            # Center previous activation for router
            with torch.no_grad():
                x_router_2 = self.center(x_nn.detach()) # Detach backbone grad? 
                # Decision: We usually DON'T want the router to shape the backbone features 
                # purely to make routing easier. So detaching x_router inputs is standard practice.
            
            scores_2 = self.router_2(x_router_2)
            mask2 = mask_topk(scores_2, self.top_k)
            self.layer_2_mask = mask2
            x = x_nn * mask2

            # Layer 3
            x_nn = self.act(self.fc3(x))
            with torch.no_grad():
                x_router_3 = self.center(x_nn.detach())
                
            scores_3 = self.router_3(x_router_3)
            mask3 = mask_topk(scores_3, self.top_k)
            self.layer_3_mask = mask3
            x = x_nn * mask3

            # Output
            x = self.fc4(x)
            return x

    return GuidedCenterModel()