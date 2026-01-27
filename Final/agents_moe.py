import torch
import torch.nn as nn

class ActorMoE(nn.Module):
    """
    Mixture-of-Experts Actor: Blends 3 specialized sub-networks 
    based on HMM regime probabilities.
    """
    def __init__(self, input_dim, num_assets, hidden=256):
        super().__init__()
        # Common feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        
        # Expert heads (Bull, Sideways, Bear)
        self.experts = nn.ModuleList([
            nn.Linear(hidden, num_assets) for _ in range(3)
        ])

    def forward(self, x):
        # x: [batch, obs_dim]
        # In our env, we'll append the 3 regime probabilities to the end of the observation
        # obs_dim = (n_assets * n_indicators) + covariance_matrix_elements + 3
        
        # Probabilities are the last 3 elements
        regime_probs = x[:, -3:] # [batch, 3]
        
        # Extract features
        features = self.feature_extractor(x) # [batch, hidden]
        
        # Get expert outputs
        expert_outputs = torch.stack([head(features) for head in self.experts], dim=1) # [batch, 3, num_assets]
        
        # Weighted sum based on probabilities (MoE)
        # regime_probs view is [batch, 3, 1] to multiply with [batch, 3, num_assets]
        combined_logits = torch.sum(expert_outputs * regime_probs.unsqueeze(-1), dim=1) # [batch, num_assets]
        
        # Dirichlet concentrations
        alpha = torch.nn.functional.softplus(combined_logits) + 1.0
        return alpha

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
