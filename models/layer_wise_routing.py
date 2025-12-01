import torch
import torch.nn as nn
from utils import mask_topk

def get_layer_wise_routing(dataset_name, layer_1_neurons, layer_2_neurons,
                          layer_3_neurons, layer_4_neurons, topk_rate, norm, activation):
    """
    An MLP where each backbone hidden layer representation is projected using a fixed random matrix, which is then used to develop the binary mask for the next layer. This is done by replacing eq. (4) with cℓ = Vℓxℓ−1.
    
    Basically, instead of the specificity routing, we take layer L, project it, and that is the masking for layer L+1.
    The spec routing is independent of the trainable weights. This will be a part of them. 
    We pass input to layer 1, then use nontrainable weights to project and select mask for layer 2.
    """

    class LayerWiseRouting(nn.Module):
        def __init__(self):
            super().__init__()

            # regular fully connected layers
            if dataset_name in ['cifar10', 'cifar100']:
                input_dim = 32 * 32 * 3
                self.fc1 = nn.Linear(input_dim, layer_1_neurons)
                self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons)
                spec_input_dim = input_dim
            elif dataset_name == 'tiny_imagenet':
                input_dim = 224 * 224 * 3
                self.fc1 = nn.Linear(input_dim, 224)
                self.fc2 = nn.Linear(224, layer_2_neurons)
                spec_input_dim = input_dim
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons)

            # weights for specificity (routing)
            # these are non-trainable weights used to generate masks that select top-k neurons for the next layer
            self.spec_1 = nn.Linear(spec_input_dim, layer_1_neurons, bias=False)
            self.spec_2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=False)
            self.spec_3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=False)
            self.spec_4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=False)

            # make sure the routing function is not trainable
            for spec_layer in [self.spec_1, self.spec_2, self.spec_3, self.spec_4]:
                spec_layer.weight.requires_grad = False

            self.top_k = topk_rate  # what percentage of neurons to take (0.X = X%)

            # normalization layers for input and hidden layers
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

            # activation function selection
            activations = {
                'gelu': nn.GELU(),
                'softplus': nn.Softplus(),
                'leaky': nn.LeakyReLU(0.1),
                'tanh': nn.Tanh(),
                'relu': nn.ReLU()
            }
            self.act = activations.get(activation, nn.ReLU())

            # store mask of layer 3 if needed for inspection
            self.layer_3_mask = None

        def forward(self, x):
            batch_size = x.size(0)

            # flatten input for fully connected layers
            if dataset_name in ['cifar10', 'cifar100']:
                x = x.view(batch_size, -1)
            elif dataset_name == 'tiny_imagenet':
                x = x.view(batch_size, -1)

            if self.norm_init is not None:
                x = self.norm_init(x)

            # Layer 1 forward pass
            x_nn = self.fc1(x)
            x_nn = self.act(x_nn)
            with torch.no_grad():
                # we use the input projected through spec_1 to create a mask for layer 1 activations
                mask1 = mask_topk(self.spec_1(x), self.top_k)
            x = x_nn * mask1

            if self.norm_fc1 is not None:
                x = self.norm_fc1(x)

            # Layer 2 forward pass
            x_nn = self.fc2(x)
            x_nn = self.act(x_nn)
            with torch.no_grad():
                # use layer 1 activations projected through spec_2 to mask layer 2 activations
                mask2 = mask_topk(self.spec_2(x), self.top_k)
            x = x_nn * mask2

            if self.norm_fc2 is not None:
                x = self.norm_fc2(x)

            # Layer 3 forward pass
            x_nn = self.fc3(x)
            x_nn = self.act(x_nn)
            with torch.no_grad():
                # use layer 2 activations projected through spec_3 to mask layer 3 activations
                mask3 = mask_topk(self.spec_3(x), self.top_k)
                self.layer_3_mask = mask3
            x = x_nn * mask3

            if self.norm_fc3 is not None:
                x = self.norm_fc3(x)

            # Layer 4 forward pass (final output)
            x = self.fc4(x)

            return x

    return LayerWiseRouting()
