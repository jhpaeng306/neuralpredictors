import torch
import torch.nn as nn


class SlotAttention(nn.Module):
    def __init__(
        self,
        num_iterations,
        num_slots,
        slot_size,
        input_size,
        mlp_hidden_size,
        epsilon=1e-8,
        draw_slots=True,
        use_slot_gru=True,
        use_weighted_mean=True,
        full_skip=False,
    ):
        """Builds the Slot Attention module.

        Args:
          num_iterations: Number of iterations.
          num_slots: Number of slots.
          slot_size: Dimensionality of slot feature vectors.
          mlp_hidden_size: Hidden layer size of MLP.
          epsilon: Offset for attention coefficients before normalization.
        """
        super().__init__()

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.input_size = input_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(self.input_size)
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)
        self.draw_slots = draw_slots
        self.use_slot_gru = use_slot_gru
        self.use_weighted_mean = use_weighted_mean
        self.full_skip = full_skip

        # change from orignial:
        # self.slots_mu = nn.Parameter(torch.randn(1, 1, self.slot_size))
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, self.slot_size))
        if self.draw_slots:
            self.slots_log_sigma = nn.Parameter(torch.randn(1, num_slots, self.slot_size))

        self.q_proj = nn.Linear(self.input_size, self.slot_size, bias=False)
        self.k_proj = nn.Linear(self.input_size, self.slot_size, bias=False)
        self.v_proj = nn.Linear(self.input_size, self.slot_size, bias=False)

        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

    def forward(self, x, output_attn_weights=None):
        batch_size, _, _ = x.size()  # Shape: (batch_size x num_inputs x input_size)

        inputs = self.norm_inputs(x)
        keys, values = self.k_proj(inputs), self.v_proj(inputs)  # Shape: (batch_size x num_inputs x slot_size)

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        # Change to learned initial query embeddings -> representing one neuron type
        if self.draw_slots:
            slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(
                [batch_size, self.num_slots, self.slot_size], device=x.device
            )
        else:
            # for learned parameters
            slots = self.slots_mu

        for i in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            queries = self.q_proj(slots)  # Shape: [batch_size, num_slots, slot_size].
            queries *= self.slot_size ** -0.5  # Normalization.

            # Attention.
            attn = torch.einsum("bid,bjd->bij", keys, queries)  # Shape: [batch_size, num_inputs, num_slots].
            attn = attn.softmax(dim=2) + self.epsilon

            # Weighted mean.
            if self.use_weighted_mean:
                attn = attn / attn.sum(dim=1, keepdim=True)

            updates = torch.einsum("bdi,bdj->bij", attn, values)  # Shape: [batch_size, num_slots, slot_size].

            if self.use_slot_gru:
                # Slot update.
                updates = self.gru(updates.view(-1, self.slot_size), slots_prev.view(-1, self.slot_size))
                updates = updates.view(batch_size, -1, self.slot_size)

            if self.full_skip:
                slots = slots + self.mlp(self.norm_mlp(updates))
            else:
                slots = updates + self.mlp(self.norm_mlp(updates))

        if output_attn_weights:
            return slots, attn
        else:
            return slots
