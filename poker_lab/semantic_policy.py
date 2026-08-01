"""Visible-card hand ranks, draws, board texture, and blocker features."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from pokerkit import Card, StandardHighHand
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .agents import PPOAgent, RandomAgent

ARCHITECTURE = 'semantic_poker_v1'
SEMANTIC_DEFAULT_CONFIG = {}
CATEGORIES = ('High card', 'One pair', 'Two pair', 'Three of a kind', 'Straight',
              'Flush', 'Full house', 'Four of a kind', 'Straight flush')
PREFLOP_FEATURE_NAMES = ('preflop_pocket_pair', 'preflop_suited', 'preflop_high_rank',
                         'preflop_low_rank', 'preflop_rank_gap')
FEATURE_NAMES = (
    'street_preflop', 'street_flop', 'street_turn', 'street_river',
    *PREFLOP_FEATURE_NAMES, 'made_hand_available',
    *(f'made_{name.lower().replace(" ", "_")}' for name in CATEGORIES), 'made_ordinal',
    'board_hand_available',
    *(f'board_{name.lower().replace(" ", "_")}' for name in CATEGORIES), 'board_ordinal',
    'plays_board', 'improves_board', 'board_improvement_ordinal',
    'board_paired', 'board_pair_rank_fraction', 'board_trips', 'board_quads',
    'board_distinct_rank_fraction', 'board_max_suit_fraction', 'board_second_suit_fraction',
    'board_straight_window_coverage', 'visible_max_suit_fraction',
    'private_flush_draw', 'private_straight_draw', 'straight_out_card_fraction',
    'flush_out_card_fraction', 'private_completed_straight',
    'nut_flush_rank_blocker', 'second_flush_rank_blocker', 'max_hole_cards_in_board_flush_suit',
    'board_straight_completion_rank_blocker_fraction',
)
_RANKS, _SUITS = '23456789TJQKA', 'cdhs'
_CARD_OBJECTS = tuple(next(Card.parse(_RANKS[c//4]+_SUITS[c%4])) for c in range(52))
_STRAIGHTS = tuple(frozenset(range(start, start+5)) for start in range(9)) + (
    frozenset((12, 0, 1, 2, 3)),)


def architecture_metadata():
    return {'name': ARCHITECTURE, 'version': 1, 'feature_schema': 1,
            'feature_names': list(FEATURE_NAMES), 'preflop_only_features': list(PREFLOP_FEATURE_NAMES),
            'feature_projection': [len(FEATURE_NAMES), 128], 'history_gru': [12, 128],
            'public_scalar_count': 12, 'policy_body_widths': [256, 256], 'actions': 5,
            'made_hand_evaluator': 'PokerKit StandardHighHand exact best five of visible cards',
            'ordinal_divisor': 7461, 'board_relative_features': 'river_only',
            'draw_definition': 'private contribution; one-card completion before river; no backdoors',
            'symmetry': 'all_suit_renamings_and_within_hole_or_board_permutations',
            'raw_card_embeddings': False, 'raw_postflop_hole_ranks': False,
            'monte_carlo_equity': False, 'action_overrides': False}


def _architecture(config):
    if config is not None and config != architecture_metadata():
        raise ValueError('Semantic architecture metadata does not match semantic_poker_v1')


def _visible(cards):
    cards = tuple(cards)
    if (len(cards) != 7 or any(isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer))
                              or c < 0 or c > 52 for c in cards) or 52 in cards[:2]):
        raise ValueError('Expected two private cards and five board/padding slots')
    hole = tuple(int(c) for c in cards[:2])
    board = tuple(int(c) for c in cards[2:] if c != 52)
    if len(board) not in (0, 3, 4, 5) or cards[2:] != board+(52,)*(5-len(board)):
        raise ValueError('Board must be a visible prefix followed by padding')
    if len(set(hole+board)) != len(hole+board):
        raise ValueError('Visible cards must be distinct')
    return tuple(sorted(hole)), tuple(sorted(board))


@lru_cache(maxsize=32768)
def _made(cards):
    hand = StandardHighHand.from_game(tuple(_CARD_OBJECTS[c] for c in cards))
    return hand.entry.index, hand.entry.label.value


@lru_cache(maxsize=16384)
def _cached_features(hole, board):
    own_ranks = [c//4 for c in hole]
    own_suits = [c%4 for c in hole]
    n = len(board)
    result = [float(n == count) for count in (0, 3, 4, 5)]
    low, high = sorted(own_ranks)
    result += ([float(low == high), float(own_suits[0] == own_suits[1]),
                high/12, low/12, (high-low)/12] if not n else [0.]*len(PREFLOP_FEATURE_NAMES))
    if not n:
        return tuple(result+[0.]*(len(FEATURE_NAMES)-len(result)))
    visible = hole+board
    rank, category = _made(tuple(sorted(visible)))
    result += [1.] + [float(category == name) for name in CATEGORIES] + [rank/7461.]
    board_rank, board_category = _made(board) if n == 5 else (0, None)
    result += [float(n == 5)] + [float(board_category == name) for name in CATEGORIES]
    result += [board_rank/7461., float(n == 5 and rank == board_rank),
               float(n == 5 and rank > board_rank), (rank-board_rank)/7461. if n == 5 else 0.]
    board_ranks = [c//4 for c in board]
    board_suits = [c%4 for c in board]
    counts = [board_ranks.count(rank) for rank in range(13)]
    suit_counts = sorted([board_suits.count(suit) for suit in range(4)], reverse=True)
    all_suits = own_suits+board_suits
    ranks, board_rank_set = set(own_ranks+board_ranks), set(board_ranks)
    result += [float(max(counts) >= 2), sum(count == 2 for count in counts)/2,
               float(max(counts) == 3), float(max(counts) == 4), len(board_rank_set)/5,
               suit_counts[0]/5, suit_counts[1]/5,
               max(len(sequence & board_rank_set) for sequence in _STRAIGHTS)/5,
               max(all_suits.count(suit) for suit in range(4))/7]
    private_ranks = set(own_ranks)-board_rank_set
    completed = [sequence for sequence in _STRAIGHTS if sequence <= ranks]
    missing = set()
    if n < 5 and not completed:
        for sequence in _STRAIGHTS:
            absent = sequence-ranks
            if len(absent) == 1 and sequence & private_ranks:
                missing.update(absent)
    flush_suits = [s for s in range(4) if n < 5 and all_suits.count(s) == 4 and s in own_suits]
    # Count possible next cards, not winning odds.
    straight_outs = sum(4-sum(c//4 == rank for c in visible) for rank in missing)
    flush_outs = sum(13-all_suits.count(suit) for suit in flush_suits)
    result += [float(bool(flush_suits)), float(bool(missing)), straight_outs/52, flush_outs/52,
               float(any(not sequence <= board_rank_set for sequence in completed))]
    nut_blocker = second_blocker = 0.
    max_hole_in_suit = 0
    for suit in range(4):
        if board_suits.count(suit) < 2:
            continue
        # Flush blockers use the visible ranks of that suit.
        # They do not guarantee the best hand on a paired board.
        remaining = [4*rank+suit for rank in reversed(range(13)) if 4*rank+suit not in board]
        nut_blocker = max(nut_blocker, float(remaining[0] in hole))
        second_blocker = max(second_blocker, float(remaining[1] in hole))
        max_hole_in_suit = max(max_hole_in_suit, own_suits.count(suit))
    board_completion = set()
    for sequence in _STRAIGHTS:
        if len(sequence-board_rank_set) == 1:
            board_completion.update(sequence-board_rank_set)
    result += [nut_blocker, second_blocker, max_hole_in_suit/2,
               len(set(own_ranks) & board_completion)/13]
    if len(result) != len(FEATURE_NAMES):
        raise AssertionError('Semantic feature schema and implementation differ')
    return tuple(result)


def visible_card_features(cards):
    """Compute features from the observation, using only visible cards."""
    return _cached_features(*_visible(cards))


def semantic_features(batch):
    cards, scalars = batch['cards'], batch['scalars']
    if cards.ndim != 2 or cards.shape[1] != 7 or cards.shape[0] < 1 or scalars.shape != (len(cards), 12):
        raise ValueError('Expected nonempty batched seven cards and twelve public scalars')
    if cards.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('Card IDs must have integer dtype')
    if not scalars.is_floating_point() or not torch.isfinite(scalars).all():
        raise ValueError('Public scalars must be finite floating-point values')
    return scalars.new_tensor([visible_card_features(row) for row in cards.detach().cpu().tolist()])


class SemanticPokerNetwork(nn.Module):
    def __init__(self, config=None):
        _architecture(config)
        super().__init__()
        self.semantic_projection = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 128), nn.ReLU())
        self.history = nn.GRU(12, 128, batch_first=True)
        self.body = nn.Sequential(nn.Linear(128+128+12, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.policy, self.value = nn.Linear(256, 5), nn.Linear(256, 1)

    @staticmethod
    def architecture_metadata():
        return architecture_metadata()

    def forward(self, batch):
        cards = self.semantic_projection(semantic_features(batch))
        packed = pack_padded_sequence(batch['history'], batch['lengths'].cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, hidden = self.history(packed)
        body = self.body(torch.cat((cards, hidden[-1], batch['scalars']), dim=-1))
        return self.policy(body), self.value(body).squeeze(-1)


class SemanticPPOAgent(PPOAgent):
    name = 'PPO with deterministic visible-card semantic encoder'

    def __init__(self, model_state, device='cpu', seed=0, architecture=None):
        _architecture(architecture)
        self.device = device
        with torch.random.fork_rng():
            self.model = SemanticPokerNetwork(architecture).to(device)
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
            raise FloatingPointError('Semantic policy produced invalid probabilities')
        return probabilities
