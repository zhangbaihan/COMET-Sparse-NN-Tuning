from utils import mask_topk
import torch
import torch.nn as nn


class TopK_Scheduler:
    """
    Piecewise-constant sparsity schedule.

    Default behavior:
        - pk = 1.0 for first `step_every` epochs (e.g. 10)
        - then pk -= 0.04 every `step_every` epochs
        - clamped at min_pk = final topk_rate (e.g. 0.1, 0.5, 0.9)

    Example for topk_rate = 0.1, step_every = 10:
        epochs  0–9   -> pk = 1.0
        epochs 10–19  -> pk = 0.9
        epochs 20–29  -> pk = 0.8
        ...
        epochs 90–99  -> pk = 0.1
        epochs >=100  -> pk = 0.1
    """
    def __init__(self,
                 start_pk: float = 1.0,
                 step_pk: float = -0.04,
                 step_every: int = 10,
                 min_pk: float = 0.8):
        assert 0.0 < min_pk <= 1.0
        assert 0.0 < start_pk <= 1.0
        assert step_every > 0

        self.start_pk = start_pk
        self.step_pk = step_pk      # negative for decreasing sparsity
        self.step_every = step_every
        self.min_pk = min_pk

    def __call__(self, epoch: int) -> float:
        """
        Return pk for a given epoch (0-based).
        """
        if epoch < 0:
            epoch = 0

        steps = epoch // self.step_every
        pk = self.start_pk + steps * self.step_pk

        # clamp to [min_pk, 1.0]
        pk = max(self.min_pk, min(1.0, pk))
        return pk


def get_TopK_Scheduler(dataset_name,
                       layer_1_neurons,
                       layer_2_neurons,
                       layer_3_neurons,
                       layer_4_neurons,
                       topk_rate,
                       norm,
                       activation,
                       step_every: int = 10,
                       step_pk: float = -0.1):
    """
    COMET with epoch-dependent top-k (sparsity annealing).

    - `topk_rate` is the *final* target pk (minimum sparsity level), e.g. 0.1 / 0.5 / 0.9.
    - During training, call `model.set_epoch(epoch)` at the start of each epoch.
      That will update `model.top_k` according to the schedule:

        pk(epoch) = clamp(1.0 + floor(epoch / step_every) * step_pk, min_pk=topk_rate, max_pk=1.0)

    All other behavior (routing, masking, normalization) matches the original COMET.py.
    """

    class SpecModel(nn.Module):
        def __init__(self):
            super(SpecModel, self).__init__()

            # -----------------------
            # Input dimension
            # -----------------------
            if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                input_dim = 32 * 32 * 3
            elif dataset_name == 'SARCOS':
                input_dim = 21
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            # -----------------------
            # Backbone (trainable)
            # -----------------------
            self.fc1 = nn.Linear(input_dim, layer_1_neurons, bias=True)
            self.fc2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=True)
            self.fc3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=True)
            self.fc4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=True)

            # -----------------------
            # Specificity routing (non-trainable)
            # -----------------------
            self.spec_1 = nn.Linear(input_dim,       layer_1_neurons, bias=False)
            self.spec_2 = nn.Linear(layer_1_neurons, layer_2_neurons, bias=False)
            self.spec_3 = nn.Linear(layer_2_neurons, layer_3_neurons, bias=False)
            self.spec_4 = nn.Linear(layer_3_neurons, layer_4_neurons, bias=False)

            for spec_layer in [self.spec_1, self.spec_2, self.spec_3, self.spec_4]:
                spec_layer.weight.requires_grad = False

            # top_k will be dynamically updated by the scheduler via set_epoch()
            self.top_k_final = topk_rate
            self.top_k = 1.0  # start dense by default
            self._epoch = 0

            # Annealing scheduler: from 1.0 down to topk_rate
            self.topk_scheduler = TopK_Scheduler(
                start_pk=1.0,
                step_pk=step_pk,
                step_every=step_every,
                min_pk=topk_rate,
            )

            # -----------------------
            # Normalization layers
            # -----------------------
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

            # -----------------------
            # Activation function
            # -----------------------
            activations = {
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
            if activation not in activations:
                raise ValueError(f"Unsupported activation: {activation}")
            self.act = activations[activation]

            # -----------------------
            # For inspection
            # -----------------------
            self.layer_1_mask = None
            self.layer_2_mask = None
            self.layer_3_mask = None

            # Initialize top_k for epoch 0
            self._update_topk_for_epoch(0)

        def set_epoch(self, epoch: int):
            """
            Should be called from the training loop at the start of each epoch:

                for epoch in range(num_epochs):
                    model.set_epoch(epoch)
                    ...

            This updates self.top_k according to the annealing schedule.
            """
            self._epoch = int(epoch)
            self._update_topk_for_epoch(self._epoch)

        def _update_topk_for_epoch(self, epoch: int):
            new_pk = float(self.topk_scheduler(epoch))
            self.top_k = new_pk

        def forward(self, x):
            # Flatten input depending on dataset
            if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                x = x.view(-1, 32 * 32 * 3)
            elif dataset_name == 'tiny_imagenet':
                x = x.view(-1, 224 * 224 * 3)
            elif dataset_name == 'SARCOS':
                x = x.view(-1, 21)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            # Initial normalization if specified
            if self.norm_init is not None:
                x = self.norm_init(x)

            # -----------------------
            # Layer 1
            # -----------------------
            x_nn = self.fc1(x)
            x_nn = self.act(x_nn)
            with torch.no_grad():
                x_spec1 = self.spec_1(x)
                mask1 = mask_topk(x_spec1, self.top_k)
                self.layer_1_mask = mask1
            x = x_nn * mask1

            if self.norm_fc1 is not None:
                x = self.norm_fc1(x)

            # -----------------------
            # Layer 2
            # -----------------------
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

            # -----------------------
            # Layer 3
            # -----------------------
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

            # -----------------------
            # Output layer (no masking)
            # -----------------------
            x = self.fc4(x)

            return x

    return SpecModel()
