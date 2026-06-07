"""
Policy + value network. Asymmetric actor-critic: critic optionally sees opp hand during training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import STATE_DIM, ACTION_DIM


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        h = self.shared(state)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def masked_policy(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
    """Apply legal-action mask: illegal actions get -inf logit, then softmax."""
    masked_logits = logits.masked_fill(~mask, float('-inf'))
    return torch.distributions.Categorical(logits=masked_logits)


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
