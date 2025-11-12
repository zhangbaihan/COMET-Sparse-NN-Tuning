from models.smaller_fc_standard import get_smaller_standard_fc
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_moe_model(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons, layer_4_neurons,
                  topk_rate, norm, activation, num_experts, gating_trainable):
    """
    A MoE model with ⌊1/pk⌋ experts, each having pkNℓ neurons in each layer. The routing network is either trainable pr not depending on the config, with one hidden layer and a sparse ⌊1/pk⌋-dimensional output.
    """

    class MoEModel(nn.Module):
        def __init__(self):
            super(MoEModel, self).__init__()

            class GatingNetwork(nn.Module):
                def __init__(self):
                    super(GatingNetwork, self).__init__()
                    if dataset_name in ['cifar10', 'cifar100']:
                        self.fc1 = nn.Linear(32 * 32 * 3, num_experts, bias=True)
                    elif dataset_name == 'tiny_imagenet':
                        self.fc1 = nn.Linear(224 * 224 * 3, num_experts, bias=True)
                    else:
                        raise ValueError(f"Unsupported dataset: {dataset_name}")

                def forward(self, x):
                    # Softmax produces gating weights over experts for each input
                    gating_weights = torch.softmax(self.fc1(x), dim=1)
                    return gating_weights

            # Initialize the experts — a list of smaller fully connected models
            self.experts = nn.ModuleList([
                get_smaller_standard_fc(dataset_name, layer_1_neurons, layer_2_neurons, layer_3_neurons,
                                        layer_4_neurons, topk_rate, norm, activation)
                for _ in range(num_experts)
            ])

            # Initialize gating network
            self.gating_network = GatingNetwork()

            # Freeze gating network weights if specified
            if not gating_trainable:
                for param in self.gating_network.parameters():
                    param.requires_grad = False

        def forward(self, x):
            # Flatten input depending on dataset
            if dataset_name in ['cifar10', 'cifar100']:
                x = x.view(-1, 32 * 32 * 3)
            elif dataset_name == 'tiny_imagenet':
                x = x.view(-1, 224 * 224 * 3)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            # Prepare output tensor to store expert outputs for the batch
            batch_output = torch.zeros(x.size(0), layer_4_neurons, device=x.device)

            # Get gating weights and pick expert with highest weight (hard routing)
            gating_output = self.gating_network(x)
            expert_indices = torch.argmax(gating_output, dim=1)

            # Forward input samples to their assigned experts
            for expert_index, expert in enumerate(self.experts):
                mask = (expert_indices == expert_index)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = expert(expert_input)
                    batch_output[mask] = expert_output

            return batch_output

    return MoEModel()
