"""Shared card and history encoder for the initial neural policies."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def batch_observations(observations, device="cpu"):
    if not observations:
        raise ValueError("Cannot batch zero observations")
    lengths = [max(1, len(o.history)) for o in observations]
    histories = torch.zeros((len(observations), max(lengths), 12), dtype=torch.float32)
    for i, o in enumerate(observations):
        if o.history:
            histories[i, :len(o.history)] = torch.tensor(o.history, dtype=torch.float32)
    return {"cards": torch.tensor([o.cards for o in observations], dtype=torch.long, device=device),
            "scalars": torch.tensor([o.scalars for o in observations], dtype=torch.float32, device=device),
            "history": histories.to(device), "lengths": torch.tensor(lengths, dtype=torch.long),
            "mask": torch.tensor([o.legal_mask for o in observations], dtype=torch.float32, device=device)}


def masked_probs(values, mask, regret=False):
    legal = mask.to(dtype=torch.bool)
    if torch.any(legal.sum(dim=-1) == 0):
        raise ValueError("No legal action in a nonterminal policy query")
    if regret:
        positive = torch.where(legal, values.clamp_min(0), 0.)
        total = positive.sum(dim=-1, keepdim=True)
        uniform = legal.to(values.dtype) / legal.sum(dim=-1, keepdim=True)
        return torch.where(total > 0, positive / total.clamp_min(1e-20), uniform)
    return torch.softmax(values.masked_fill(~legal, -torch.inf), dim=-1)


class PokerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cards = nn.Embedding(53, 16, padding_idx=52)
        self.history = nn.GRU(12, 128, batch_first=True)
        self.body = nn.Sequential(nn.Linear(7*16 + 128 + 12, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU())
        self.policy = nn.Linear(256, 5)
        self.value = nn.Linear(256, 1)

    def forward(self, batch):
        cards = self.cards(batch["cards"]).flatten(1)
        packed = pack_padded_sequence(batch["history"], batch["lengths"].cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, hidden = self.history(packed)
        features = self.body(torch.cat((cards, hidden[-1], batch["scalars"]), dim=-1))
        return self.policy(features), self.value(features).squeeze(-1)
