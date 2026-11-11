"""GRU actor-critic that carries a single hidden state over time."""

import torch
from torch import nn


class GRUActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_count: int, hidden_size: int = 128):
        super().__init__()
        self.gru = nn.GRU(observation_size, hidden_size, batch_first=True)
        self.policy_head = nn.Linear(hidden_size, action_count)
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.gru.hidden_size, device=device)

    def step(self, observation: torch.Tensor, hidden: torch.Tensor):
        # observation shape: [batch, observation_size]
        features, next_hidden = self.gru(observation.unsqueeze(1), hidden)
        features = features[:, 0, :]
        return self.policy_head(features), self.value_head(features).squeeze(-1), next_hidden

    def sequence(self, observations: torch.Tensor, episode_starts: torch.Tensor, hidden: torch.Tensor):
        """Unroll a rollout, clearing recurrent state at episode starts."""
        logits, values = [], []
        for index in range(len(observations)):
            keep = (~episode_starts[index]).to(observations.dtype).view(1, 1, 1)
            hidden = hidden * keep
            step_logits, step_value, hidden = self.step(observations[index:index + 1], hidden)
            logits.append(step_logits.squeeze(0))
            values.append(step_value.squeeze(0))
        return torch.stack(logits), torch.stack(values)
