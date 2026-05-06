import torch
import torch.nn as nn

MIN_WEIGHT = -0.05
MAX_WEIGHT =  0.20

class ActorMoE(nn.Module):
    """
    Mixture-of-Experts Actor: Blends 4 specialised sub-networks
    based on HMM regime probabilities.
    Returns (mean, std) for a Gaussian policy that supports negative weights.
    """
    def __init__(self, input_dim, num_assets, hidden=256):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        # Expert heads (Bear, Sideways Down, Sideways Up, Bull)
        self.experts = nn.ModuleList([
            nn.Linear(hidden, num_assets) for _ in range(4)
        ])
        # Learned per-asset log-std (shared across experts)
        self.log_std = nn.Parameter(torch.zeros(num_assets))

    def forward(self, x):
        """
        x: [batch, obs_dim]  (last 4 elements are regime probabilities)
        Returns (mean, std), each [batch, num_assets].
        mean is tanh-scaled to land inside [MIN_WEIGHT, MAX_WEIGHT].
        """
        regime_probs = x[:, -4:]  # [batch, 4]
        features = self.feature_extractor(x)  # [batch, hidden]

        # Blend expert logits via soft regime gating
        expert_outputs = torch.stack(
            [head(features) for head in self.experts], dim=1
        )  # [batch, 4, num_assets]
        combined_logits = torch.sum(
            expert_outputs * regime_probs.unsqueeze(-1), dim=1
        )  # [batch, num_assets]

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
