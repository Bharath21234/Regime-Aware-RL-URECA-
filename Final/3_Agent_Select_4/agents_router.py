import torch
import torch.nn as nn

MIN_WEIGHT = -0.05
MAX_WEIGHT =  0.20


class ActorLearnedRouter(nn.Module):
    """
    Learned-Router MoE Actor.

    Ablation position in the paper:
      Hard Routing (argmax) → Soft MoE (HMM probs directly) → Learned Router (this)

    Difference from Soft MoE: instead of using raw HMM probabilities as gating
    weights, a small MLP jointly conditions on extracted state features AND HMM
    probs to produce routing weights. This lets the network learn when to trust
    or override the HMM signal — useful because HMM labels typically lag real
    regime transitions by several days.
    """

    def __init__(self, input_dim, num_assets, hidden=256, router_hidden=64):
        super().__init__()
        # Shared feature extractor — regime probs excluded (last 4 dims)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim - 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Learned router: state features + HMM probs → 4 routing weights
        self.router = nn.Sequential(
            nn.Linear(hidden + 4, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, 4),
            nn.Softmax(dim=-1),
        )
        # Expert heads — one per regime (Bear, Sideways Down, Sideways Up, Bull)
        self.experts = nn.ModuleList([
            nn.Linear(hidden, num_assets) for _ in range(4)
        ])
        # Shared learned per-asset log-std
        self.log_std = nn.Parameter(torch.zeros(num_assets))

    def forward(self, x):
        """
        x : [batch, obs_dim]  — last 4 elements are HMM regime probabilities
        Returns (mean, std), each [batch, num_assets].
        """
        regime_probs = x[:, -4:]                      # [batch, 4]
        features     = self.feature_extractor(x[:, :-4])  # [batch, hidden]

        # Router conditions on both learned features and raw HMM signal
        router_input    = torch.cat([features, regime_probs], dim=-1)  # [batch, hidden+4]
        routing_weights = self.router(router_input)                    # [batch, 4]

        # Soft blend of expert logits
        expert_outputs = torch.stack(
            [head(features) for head in self.experts], dim=1
        )  # [batch, 4, num_assets]
        combined_logits = torch.sum(
            expert_outputs * routing_weights.unsqueeze(-1), dim=1
        )  # [batch, num_assets]

        mean = combined_logits * 0.1
        std  = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        std  = torch.clamp(std, 1e-3, 1.0)
        return mean, std


class Critic(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
