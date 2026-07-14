import os
import torch
import torch.nn as nn

MIN_WEIGHT = -0.05
MAX_WEIGHT =  0.20

# Soft-v2 remediation (results_log 10f/12b diagnosis), opt-in via SOFT_V2=1:
#   1. Two-layer expert heads. A posterior-weighted blend of LINEAR heads is
#      itself one linear map of the shared trunk features (sum_k p_k W_k h =
#      (sum_k p_k W_k) h), so K linear heads add no expressivity over a single
#      head. A hidden layer per head breaks that ceiling: the blend of
#      nonlinear experts is no longer representable by any single head.
#   2. Inverse-occupancy gradient rescaling. d(loss)/d(head_k) is scaled by
#      p_k, so heads for rare regimes train ~occupancy-times slower and decay
#      toward clones of the common-regime head. A straight-through rescale
#      equalises the training signal across heads without changing the
#      forward output (eval is untouched).
SOFT_V2 = os.environ.get("SOFT_V2", "0") == "1"

class ActorMoE(nn.Module):
    """
    Mixture-of-Experts Actor: Blends num_experts specialised sub-networks
    based on HMM regime probabilities (one expert head per HMM regime).
    Returns (mean, std) for a Gaussian policy that supports negative weights.
    """
    def __init__(self, input_dim, num_assets, hidden=256, num_experts=4,
                 occupancy=None):
        super().__init__()
        self.num_experts = num_experts
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim - num_experts, hidden),  # regime probs excluded from extractor
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        # Expert heads, ordered Bear -> Bull (K=4: Bear, Sideways Down, Sideways Up, Bull)
        if SOFT_V2:
            head_hidden = 64
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden, head_hidden),
                    nn.ReLU(),
                    nn.Linear(head_hidden, num_assets),
                ) for _ in range(num_experts)
            ])
        else:
            self.experts = nn.ModuleList([
                nn.Linear(hidden, num_assets) for _ in range(num_experts)
            ])
        # occupancy: mean posterior mass per regime over the TRAIN period
        # (no test information). Mean-normalised so overall gradient scale is
        # unchanged; capped so a near-empty regime can't dominate a batch.
        if SOFT_V2 and occupancy is not None:
            occ = torch.as_tensor(occupancy, dtype=torch.float32).clamp(min=1e-4)
            self.register_buffer("occ_scale", (occ.mean() / occ).clamp(max=20.0))
        else:
            self.occ_scale = None
        # Learned per-asset log-std (shared across experts)
        self.log_std = nn.Parameter(torch.zeros(num_assets))

    def forward(self, x):
        """
        x: [batch, obs_dim]  (last num_experts elements are regime probabilities)
        Returns (mean, std), each [batch, num_assets].
        mean is tanh-scaled to land inside [MIN_WEIGHT, MAX_WEIGHT].
        """
        K = self.num_experts
        regime_probs = x[:, -K:]  # [batch, K]
        features = self.feature_extractor(x[:, :-K])  # [batch, hidden] — exclude regime probs

        # Blend expert logits via soft regime gating
        expert_outputs = torch.stack(
            [head(features) for head in self.experts], dim=1
        )  # [batch, K, num_assets]
        combined_logits = torch.sum(
            expert_outputs * regime_probs.unsqueeze(-1), dim=1
        )  # [batch, num_assets]

        if self.occ_scale is not None and self.training:
            # Straight-through: value identical to combined_logits, but
            # gradients flow through the occupancy-rescaled blend.
            rescaled = torch.sum(
                expert_outputs * (regime_probs * self.occ_scale).unsqueeze(-1),
                dim=1
            )
            combined_logits = combined_logits.detach() + rescaled - rescaled.detach()

        # REMOVING TANH saturating bottleneck. Tanh at the end of the head forces the
        # weights to polarize into 'extreme binary' groups when passed to Softmax.
        # Now using Linear logs with 0.1 scale to maintain smooth continuous sensitivity.
        mean = combined_logits * 0.1

        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        std = torch.clamp(std, 1e-3, 1.0) # Allow proper exploration
        return mean, std

class Critic(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
