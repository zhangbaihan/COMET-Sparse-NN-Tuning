import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_example_tied_dropout(dataset_name, layer_1_neurons, layer_2_neurons,
                              layer_3_neurons, layer_4_neurons, topk_rate,
                              norm, activation,
                              layer_1_vector_dict, layer_2_vector_dict, layer_3_vector_dict):
    """
    Example-tied dropout (Maini et al., 2023), where each example in the training data is associated with a fixed binary mask drawn from a Bernoulli distribution, with probability equal to pk, and a fixed number of “generalization neurons” are active for all examples
    """

    class ExampleTiedDropout(nn.Module):
        def __init__(self):
            super().__init__()

            if dataset_name in ['cifar10', 'cifar100']:
                input_dim = 32 * 32 * 3
                fc1_out = layer_1_neurons
            elif dataset_name == 'tiny_imagenet':
                input_dim = 224 * 224 * 3
                fc1_out = 224
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            self.fc1 = nn.Linear(input_dim, fc1_out)
            self.fc2 = nn.Linear(fc1_out, layer_2_neurons)
            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons)

            # Normalization
            if norm == 'batch':
                self.norm_init = nn.BatchNorm1d(input_dim)
                self.norm_fc1 = nn.BatchNorm1d(fc1_out)
                self.norm_fc2 = nn.BatchNorm1d(layer_2_neurons)
                self.norm_fc3 = nn.BatchNorm1d(layer_3_neurons)
            elif norm == 'layer':
                self.norm_init = nn.LayerNorm(input_dim)
                self.norm_fc1 = nn.LayerNorm(fc1_out)
                self.norm_fc2 = nn.LayerNorm(layer_2_neurons)
                self.norm_fc3 = nn.LayerNorm(layer_3_neurons)
            else:
                self.norm_init = self.norm_fc1 = self.norm_fc2 = self.norm_fc3 = None

            # Activation
            self.act = {
                'gelu': nn.GELU(),
                'softplus': nn.Softplus(),
                'leaky': nn.LeakyReLU(0.1),
                'tanh': nn.Tanh()
            }.get(activation, nn.ReLU())  # default to ReLU if unspecified

            self.topk_rate = topk_rate
            self.layer_1_vector_dict = layer_1_vector_dict
            self.layer_2_vector_dict = layer_2_vector_dict
            self.layer_3_vector_dict = layer_3_vector_dict
            self.layer_1_neurons = fc1_out

        def forward(self, x, setup='train'):
            batch_size = x.size(0)

            # Select vectors depending on setup
            if setup == 'train':
                keys = [
                    f'{tensor[0, 0, 0].item()}{tensor[1, 1, 1].item()}'
                    for tensor in x
                ]
                layer_1_vecs = torch.stack([self.layer_1_vector_dict[k] for k in keys])
                layer_2_vecs = torch.stack([self.layer_2_vector_dict[k] for k in keys])
                layer_3_vecs = torch.stack([self.layer_3_vector_dict[k] for k in keys])
            else:
                generalization_vector = torch.ones(self.layer_1_neurons)
                generalization_vector[int(self.topk_rate * self.layer_1_neurons):] = 0
                layer_1_vecs = layer_2_vecs = layer_3_vecs = \
                    generalization_vector.unsqueeze(0).repeat(batch_size, 1)

            layer_1_vecs = layer_1_vecs.to(device)
            layer_2_vecs = layer_2_vecs.to(device)
            layer_3_vecs = layer_3_vecs.to(device)

            # Flatten input
            if dataset_name in ['cifar10', 'cifar100']:
                x = x.view(batch_size, -1)
            elif dataset_name == 'tiny_imagenet':
                x = x.view(batch_size, -1)

            if self.norm_init:
                x = self.norm_init(x)

            x = self.act(self.fc1(x)) * layer_1_vecs
            if self.norm_fc1:
                x = self.norm_fc1(x)

            x = self.act(self.fc2(x)) * layer_2_vecs
            if self.norm_fc2:
                x = self.norm_fc2(x)

            x = self.act(self.fc3(x)) * layer_3_vecs
            if self.norm_fc3:
                x = self.norm_fc3(x)

            x = self.fc4(x)
            return x

    return ExampleTiedDropout().to(device)
