import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from tqdm import tqdm

def mask_topk(x, k_rate):
    if k_rate >= 1.0:
        return torch.ones_like(x)
    k = max(1, int(x.shape[-1] * k_rate))
    topk_values, _ = torch.topk(x, k, dim=-1)
    kth_value = topk_values[:, -1].unsqueeze(-1)
    mask = (x >= kth_value).float()
    return mask

def oja_update_step(layer, x, y, mask, lr=1e-3):
    """
    Robust Competitive Oja's Rule.
    Includes explicit normalization to prevent weight explosion.
    """
    with torch.no_grad():
        batch_size = x.shape[0]
        y_masked = y * mask # Only winners learn [B, Out]
        
        # Hebbian Term: y * x
        # [Out, B] @ [B, In] -> [Out, In]
        hebbian_term = torch.matmul(y_masked.t(), x) / batch_size
        
        # Oja Decay Term (Standard): y^2 * w
        # y_squared_sum = (y_masked ** 2).sum(dim=0).unsqueeze(1) / batch_size
        # decay_term = y_squared_sum * layer.weight
        # delta = lr * (hebbian_term - decay_term)
        
        # SIMPLIFIED ROBUST UPDATE:
        # Just move W towards X (Hebbian) and then hard-normalize W.
        # This is equivalent to Oja but numerically stable.
        delta = lr * hebbian_term
        layer.weight.add_(delta)
        
        # Hard constraint: Keep weights on the hypersphere
        # This prevents NaN explosion which was killing your previous run.
        layer.weight.data = F.normalize(layer.weight.data, p=2, dim=1)

def get_Oja_Robust(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons, layer_4_neurons,
                   topk_rate, norm, activation):

    class OjaRobustModel(nn.Module):
        def __init__(self):
            super(OjaRobustModel, self).__init__()
            
            # --- 1. Dimensions ---
            if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                self.input_dim = 32 * 32 * 3
            elif dataset_name == 'tiny_imagenet':
                self.input_dim = 224 * 224 * 3
            elif dataset_name == 'SARCOS':
                self.input_dim = 21
            elif dataset_name == 'mnist':
                self.input_dim = 784
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            # --- 2. Trainable Backbone ---
            self.fc1 = nn.Linear(self.input_dim, layer_1_neurons)
            self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons)
            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons)

            # --- 3. Shadow Router (Oja-Trainable) ---
            self.spec_1 = nn.Linear(self.input_dim, layer_1_neurons, bias=False)
            self.spec_2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=False)
            self.spec_3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=False)
            self.spec_4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=False)
            
            # Init Router
            for layer in [self.spec_1, self.spec_2, self.spec_3, self.spec_4]:
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                # Normalize initially
                layer.weight.data = F.normalize(layer.weight.data, p=2, dim=1)
                layer.weight.requires_grad = False

            # --- 4. Whitening ---
            # Using affine=False to strictly learn statistics (Mean/Var)
            self.white_0 = nn.BatchNorm1d(self.input_dim, affine=False)
            self.white_1 = nn.BatchNorm1d(layer_1_neurons, affine=False)
            self.white_2 = nn.BatchNorm1d(layer_2_neurons, affine=False)

            self.top_k = topk_rate
            
            # Activation & Norm
            activations_dict = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'tanh': nn.Tanh()}
            self.act = activations_dict.get(activation, nn.ReLU())
            
            if norm == 'batch':
                self.norm_init = nn.BatchNorm1d(self.input_dim)
                self.norm_fc1 = nn.BatchNorm1d(layer_1_neurons)
                self.norm_fc2 = nn.BatchNorm1d(layer_2_neurons)
                self.norm_fc3 = nn.BatchNorm1d(layer_3_neurons)
            else:
                self.norm_init = None
                self.norm_fc1 = None
                self.norm_fc2 = None
                self.norm_fc3 = None

        def pretrain_router(self, dataloader, epochs=5, lr=0.005, device='cuda'):
            """
            Robust Pre-training Loop
            """
            print(f"\n[Robust-Hebbian] Starting Router Pre-training ({epochs} epochs)...")
            self.to(device)
            
            # CRITICAL: Put Whitening layers in TRAIN mode so they track stats
            # But keep Backbone in EVAL mode so Dropout/BN don't fluctuate
            self.eval() 
            self.white_0.train()
            self.white_1.train()
            self.white_2.train()
            
            for epoch in range(epochs):
                pbar = tqdm(dataloader, desc=f"Oja Epoch {epoch+1}/{epochs}")
                for inputs, _ in pbar:
                    inputs = inputs.to(device)
                    x = inputs.view(inputs.size(0), -1)
                    
                    # --- Layer 1 ---
                    # 1. Update Whitening Stats
                    x_w = self.white_0(x)
                    
                    # 2. Compute Router
                    scores1 = self.spec_1(x_w)
                    mask1 = mask_topk(scores1, self.top_k)
                    
                    # 3. Update Weights
                    oja_update_step(self.spec_1, x_w, scores1, mask1, lr)
                    
                    # 4. Prepare next input (Sparse)
                    x_spec_next = scores1 * mask1
                    
                    # --- Layer 2 ---
                    x_w2 = self.white_1(x_spec_next)
                    scores2 = self.spec_2(x_w2)
                    mask2 = mask_topk(scores2, self.top_k)
                    oja_update_step(self.spec_2, x_w2, scores2, mask2, lr)
                    x_spec_next = scores2 * mask2
                    
                    # --- Layer 3 ---
                    x_w3 = self.white_2(x_spec_next)
                    scores3 = self.spec_3(x_w3)
                    mask3 = mask_topk(scores3, self.top_k)
                    oja_update_step(self.spec_3, x_w3, scores3, mask3, lr)
            
            # Check for NaNs
            if torch.isnan(self.spec_1.weight).any():
                print("!!! WARNING: NaNs detected in Router weights! Training failed. !!!")
            else:
                print("[Robust-Hebbian] Pre-training Complete. Weights valid.")

            # Freeze whitening stats for main training
            self.white_0.eval()
            self.white_1.eval()
            self.white_2.eval()

        def forward(self, x):
            x = x.view(x.size(0), -1)
            
            # --- Main Training Forward ---
            if self.norm_init: x_in = self.norm_init(x)
            else: x_in = x
            
            # Layer 1
            # Router uses its own Whitening stats (frozen from pre-training)
            x_w = self.white_0(x) 
            spec_out1 = self.spec_1(x_w)
            mask1 = mask_topk(spec_out1, self.top_k)
            
            x_nn = self.fc1(x_in)
            x_nn = self.act(x_nn)
            x_nn = x_nn * mask1
            if self.norm_fc1: x_nn = self.norm_fc1(x_nn)

            # Layer 2
            router_in2 = spec_out1 * mask1
            x_w2 = self.white_1(router_in2)
            spec_out2 = self.spec_2(x_w2)
            mask2 = mask_topk(spec_out2, self.top_k)
            
            x_nn2 = self.fc2(x_nn)
            x_nn2 = self.act(x_nn2)
            x_nn2 = x_nn2 * mask2
            if self.norm_fc2: x_nn2 = self.norm_fc2(x_nn2)

            # Layer 3
            router_in3 = spec_out2 * mask2
            x_w3 = self.white_2(router_in3)
            spec_out3 = self.spec_3(x_w3)
            mask3 = mask_topk(spec_out3, self.top_k)
            
            x_nn3 = self.fc3(x_nn2)
            x_nn3 = self.act(x_nn3)
            x_nn3 = x_nn3 * mask3
            if self.norm_fc3: x_nn3 = self.norm_fc3(x_nn3)

            # Output
            out = self.fc4(x_nn3)
            return out

    return OjaRobustModel()