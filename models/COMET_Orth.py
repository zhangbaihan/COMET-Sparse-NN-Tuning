import torch
import torch.nn as nn
import torch.nn.init as init

def mask_topk(x, k_rate):
    """
    Generates a binary mask where the top k fraction of elements are 1, others 0.
    """
    if k_rate >= 1.0:
        return torch.ones_like(x)
    
    # shape of x: [batch_size, num_neurons]
    k = max(1, int(x.shape[-1] * k_rate))
    
    topk_values, _ = torch.topk(x, k, dim=-1)
    kth_value = topk_values[:, -1].unsqueeze(-1)
    
    mask = (x >= kth_value).float()
    return mask

def orthogonal_init_(tensor, gain=1.0):
    """
    Fills the input tensor with (Block) Orthogonal initialization.
    
    If tensor.shape[0] > tensor.shape[1] (Expansion Layer), it stacks 
    multiple independent orthogonal matrices to ensure maximal feature diversity.
    
    Args:
        tensor: A torch tensor (weight matrix)
        gain: Scaling factor
    """
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor.size(1)

    if rows <= cols:
        # Standard case: fewer output neurons than inputs (or square)
        # Rows can be mutually orthogonal.
        init.orthogonal_(tensor, gain=gain)
    else:
        # Expansion case: more output neurons than inputs (e.g., 21 -> 3000)
        # We cannot make 3000 vectors orthogonal in 21-dim space.
        # Strategy: Stack multiple independent orthogonal sets to guarantee coverage.
        with torch.no_grad():
            # How many full blocks of 'cols' can we fit?
            num_blocks = rows // cols
            remainder = rows % cols
            
            # Fill full blocks
            for i in range(num_blocks):
                # Create a temporary square matrix of size (cols, cols)
                weight_block = torch.empty(cols, cols, device=tensor.device)
                init.orthogonal_(weight_block, gain=gain)
                
                # Assign to the tensor
                tensor[i*cols : (i+1)*cols, :] = weight_block
            
            # Fill remainder
            if remainder > 0:
                weight_block = torch.empty(cols, cols, device=tensor.device)
                init.orthogonal_(weight_block, gain=gain)
                tensor[num_blocks*cols :, :] = weight_block[:remainder, :]

def get_Orthogonal(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons, layer_4_neurons,
                   topk_rate, norm, activation):
    """
    Returns a COMET model variant that uses Block Orthogonal Initialization.
    """

    class OrthogonalModel(nn.Module):
        def __init__(self):
            super(OrthogonalModel, self).__init__()

            # ------------------------------------------------------------------
            # 1. Determine Input Dimensions
            # ------------------------------------------------------------------
            if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                input_dim = 32 * 32 * 3
            elif dataset_name == 'tiny_imagenet':
                input_dim = 224 * 224 * 3
            elif dataset_name == 'SARCOS':
                input_dim = 21
            elif dataset_name == 'mnist':
                input_dim = 784
            else:
                # Fallback or error, though typical use cases are covered above
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            # ------------------------------------------------------------------
            # 2. Define Backbone Layers (Trainable)
            # ------------------------------------------------------------------
            self.fc1 = nn.Linear(input_dim, layer_1_neurons, bias=True)
            self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=True)
            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=True)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=True)

            # ------------------------------------------------------------------
            # 3. Define Routing Layers (Fixed)
            # ------------------------------------------------------------------
            self.spec_1 = nn.Linear(input_dim, layer_1_neurons, bias=False)
            self.spec_2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=False)
            self.spec_3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=False)
            self.spec_4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=False)

            # Freeze routing weights immediately
            for spec_layer in [self.spec_1, self.spec_2, self.spec_3, self.spec_4]:
                spec_layer.weight.requires_grad = False

            self.top_k = topk_rate

            # ------------------------------------------------------------------
            # 4. Normalization Layers
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # 5. Activation Function
            # ------------------------------------------------------------------
            activations_dict = {
                'gelu': nn.GELU(),
                'softplus': nn.Softplus(),
                'relu': nn.ReLU(),
                'leaky': nn.LeakyReLU(0.1),
                'tanh': nn.Tanh(),
                'selu': nn.SELU(),
                'sigmoid': nn.Sigmoid(),
                'silu': nn.SiLU(),
                'elu': nn.ELU(),
                'mish': nn.Mish()
            }
            if activation not in activations_dict:
                raise ValueError(f"Unsupported activation: {activation}")
            self.act = activations_dict[activation]

            # Masks placeholder for analysis
            self.layer_1_mask = None
            self.layer_2_mask = None
            self.layer_3_mask = None
            
            # ------------------------------------------------------------------
            # 6. Apply Orthogonal Initialization
            # ------------------------------------------------------------------
            self.apply_orthogonal_init(activation)

        def apply_orthogonal_init(self, act_name):
            """
            Applies orthogonal initialization to backbone and routing layers.
            Uses Block Orthogonalization for expansion layers.
            """
            # Calculate gain for backbone based on activation function
            valid_act_names = ['relu', 'leaky', 'tanh', 'sigmoid', 'selu']
            init_act = act_name if act_name in valid_act_names else 'linear'
            
            try:
                gain_backbone = init.calculate_gain(init_act)
            except:
                gain_backbone = 1.0

            # A. Initialize Trainable Backbone
            # Use custom orthogonal_init_ to handle potential expansion layers
            for layer in [self.fc1, self.fc2, self.fc3]:
                orthogonal_init_(layer.weight, gain=gain_backbone)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
            
            # Output layer (fc4)
            orthogonal_init_(self.fc4.weight, gain=1.0)
            if self.fc4.bias is not None:
                init.constant_(self.fc4.bias, 0)

            # B. Initialize Fixed Router
            # Gain ~ sqrt(2) for maintaining variance through sparsity
            gain_routing = 1.41 
            
            for layer in [self.spec_1, self.spec_2, self.spec_3, self.spec_4]:
                orthogonal_init_(layer.weight, gain=gain_routing)

        def forward(self, x):
            # ------------------------------------------------------------------
            # 1. Input Flattening
            # ------------------------------------------------------------------
            if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                x = x.view(-1, 32 * 32 * 3)
            elif dataset_name == 'tiny_imagenet':
                x = x.view(-1, 224 * 224 * 3)
            elif dataset_name == 'SARCOS':
                x = x.view(-1, 21)
            elif dataset_name == 'mnist':
                x = x.view(-1, 784)
            
            # Initial Norm
            if self.norm_init is not None:
                x = self.norm_init(x)

            # ------------------------------------------------------------------
            # 2. Layer 1
            # ------------------------------------------------------------------
            # Backbone
            x_nn = self.fc1(x)
            x_nn = self.act(x_nn)
            
            # Routing
            with torch.no_grad():
                x_spec1 = self.spec_1(x)
                mask1 = mask_topk(x_spec1, self.top_k)
                self.layer_1_mask = mask1
            
            # Apply Mask
            x = x_nn * mask1
            
            if self.norm_fc1 is not None:
                x = self.norm_fc1(x)

            # ------------------------------------------------------------------
            # 3. Layer 2
            # ------------------------------------------------------------------
            x_nn = self.fc2(x)
            x_nn = self.act(x_nn)
            
            with torch.no_grad():
                if self.norm_fc1 is not None:
                    x_spec1 = self.norm_fc1(x_spec1)
                
                x_spec2 = self.spec_2(x_spec1 * mask1)
                mask2 = mask_topk(x_spec2, self.top_k)
                self.layer_2_mask = mask2
            
            x = x_nn * mask2
            
            if self.norm_fc2 is not None:
                x = self.norm_fc2(x)

            # ------------------------------------------------------------------
            # 4. Layer 3
            # ------------------------------------------------------------------
            x_nn = self.fc3(x)
            x_nn = self.act(x_nn)
            
            with torch.no_grad():
                if self.norm_fc2 is not None:
                    x_spec2 = self.norm_fc2(x_spec2)
                
                x_spec3 = self.spec_3(x_spec2 * mask2)
                mask3 = mask_topk(x_spec3, self.top_k)
                self.layer_3_mask = mask3
            
            x = x_nn * mask3
            
            if self.norm_fc3 is not None:
                x = self.norm_fc3(x)

            # ------------------------------------------------------------------
            # 5. Output Layer
            # ------------------------------------------------------------------
            x = self.fc4(x)

            return x

    return OrthogonalModel()