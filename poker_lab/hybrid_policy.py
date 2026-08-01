"""PPO with hand features and separate hole, flop, turn, and river cards."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .agents import PPOAgent, RandomAgent
from .game import canonical_cards
from . import semantic_policy as semantic

ARCHITECTURE = 'hybrid_poker_v1'
CARD_SLOTS = ('hole_0', 'hole_1', 'flop_0', 'flop_1', 'flop_2', 'turn', 'river')


def architecture_metadata():
    return {
        'name': ARCHITECTURE, 'version': 1,
        'semantic_features': semantic.architecture_metadata(),
        'card_slots': list(CARD_SLOTS), 'card_embedding': [53, 16], 'padding_card': 52,
        'feature_projection': [len(semantic.FEATURE_NAMES), 128],
        'history_gru': [12, 128], 'public_scalar_count': 12,
        'policy_body_widths': [256, 256], 'actions': 5,
        'symmetry': 'all_suit_renamings_and_within_hole_or_flop_permutations',
        'turn_and_river_order': 'preserved',
        'raw_card_embeddings': True, 'raw_postflop_hole_ranks': True,
        'monte_carlo_equity': False, 'action_overrides': False,
    }


def _architecture(config):
    if config is not None and config != architecture_metadata():
        raise ValueError('Hybrid architecture metadata does not match hybrid_poker_v1')


def _canonical_slots(cards):
    rows = [canonical_cards(tuple(row)) for row in cards.detach().cpu().tolist()]
    return torch.tensor(rows, dtype=torch.long, device=cards.device)


def canonical_card_slots(batch):
    """Validate visible cards and keep hole, flop, turn, and river positions."""
    semantic.semantic_features(batch)
    return _canonical_slots(batch['cards'])


class HybridPokerNetwork(nn.Module):
    def __init__(self, config=None):
        _architecture(config)
        super().__init__()
        self.cards = nn.Embedding(53, 16, padding_idx=52)
        self.semantic_projection = nn.Sequential(
            nn.Linear(len(semantic.FEATURE_NAMES), 128), nn.ReLU())
        self.history = nn.GRU(12, 128, batch_first=True)
        self.body = nn.Sequential(
            nn.Linear(7*16+128+128+12, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU())
        self.policy, self.value = nn.Linear(256, 5), nn.Linear(256, 1)

    @staticmethod
    def architecture_metadata():
        return architecture_metadata()

    def forward(self, batch):
        features = self.semantic_projection(semantic.semantic_features(batch))
        cards = self.cards(_canonical_slots(batch['cards'])).flatten(1)
        packed = pack_padded_sequence(
            batch['history'], batch['lengths'].cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.history(packed)
        body = self.body(torch.cat((cards, features, hidden[-1], batch['scalars']), dim=-1))
        return self.policy(body), self.value(body).squeeze(-1)


class HybridPPOAgent(PPOAgent):
    name = 'PPO with semantic features and canonical card slots'

    def __init__(self, model_state, device='cpu', seed=0, architecture=None):
        _architecture(architecture)
        self.device = device
        with torch.random.fork_rng():
            self.model = HybridPokerNetwork(architecture).to(device)
        self.model.load_state_dict(model_state, strict=True)
        self.model.eval().requires_grad_(False)
        RandomAgent.__init__(self, seed)

    @staticmethod
    def architecture_metadata():
        return architecture_metadata()

    def probabilities(self, observation):
        mask = np.asarray(observation.legal_mask)
        if mask.shape != (5,) or not np.isin(mask, (0, 1)).all() or not mask.any():
            raise ValueError('Expected a nonempty binary five-slot legal mask')
        probabilities = super().probabilities(observation)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise FloatingPointError('Hybrid policy produced invalid probabilities')
        return probabilities
