"""Hand ranks, draws, and board features."""

import numpy as np
import pytest
import torch

from poker_lab import semantic_policy as semantic
from poker_lab.game import Observation
from poker_lab.networks import batch_observations


@pytest.fixture(autouse=True)
def deterministic_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(150319)
    yield
    torch.set_num_threads(previous)


def ids(text):
    return tuple(4*'23456789TJQKA'.index(text[i])+'cdhs'.index(text[i+1]) for i in range(0, len(text), 2))


def observation(hole='AsKd', board=''):
    cards = ids(board)
    return Observation(ids(hole)+cards+(52,)*(5-len(cards)),
                       (1., 0., {0:0., 3:1/3, 4:2/3, 5:1.}[len(cards)], .15, .9, .9, 0., .1, .1, .2, .9, .005),
                       (), (1.,)*5)


def features(hole='AsKd', board=''):
    return dict(zip(semantic.FEATURE_NAMES, semantic.visible_card_features(observation(hole, board).cards)))


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('hole,board,category', [
    ('AsKd', '3c4s5h9dJc', 'high_card'), ('AsAd', '3c4s5h9dJc', 'one_pair'),
    ('AsAd', '3c3s5h9dJc', 'two_pair'), ('AsAd', 'Ac4s5h9dJc', 'three_of_a_kind'),
    ('Ah2d', '3c4s5h9dJc', 'straight'), ('As8s', '2s4s7s9dJc', 'flush'),
    ('AsAd', 'Ac4s4h9dJc', 'full_house'), ('AsAd', 'AcAh5h9dJc', 'four_of_a_kind'),
    ('AsKs', 'QsJsTs9dJc', 'straight_flush')])
def test_exact_made_hand_categories_and_rank_bounds(hole, board, category):
    row = features(hole, board)
    assert row['made_'+category] == 1
    assert sum(row['made_'+name.lower().replace(' ', '_')] for name in semantic.CATEGORIES) == 1
    assert all(np.isfinite(list(row.values())))
    assert 0 <= row['board_ordinal'] <= row['made_ordinal'] <= 1


def test_board_only_royal_flush_discards_irrelevant_postflop_raw_hole_ranks():
    first, second = observation('2c3d', 'AsKsQsJsTs'), observation('6c7d', 'AsKsQsJsTs')
    assert semantic.visible_card_features(first.cards) == semantic.visible_card_features(second.cards)
    row = features('2c3d', 'AsKsQsJsTs')
    assert row['plays_board'] == 1 and row['improves_board'] == 0 and row['made_ordinal'] == 1
    model = semantic.SemanticPokerNetwork().eval()
    logits, values = model(batch_observations([first, second]))
    assert torch.equal(logits[0], logits[1]) and values[0] == values[1]


def test_draws_blockers_and_board_relative_availability():
    flop = features('AsKs', 'QsJs2d')
    assert flop['private_flush_draw'] == flop['private_straight_draw'] == 1
    assert flop['straight_out_card_fraction'] == 4/52 and flop['flush_out_card_fraction'] == 9/52
    assert flop['nut_flush_rank_blocker'] == flop['second_flush_rank_blocker'] == 1
    assert flop['board_hand_available'] == flop['board_ordinal'] == flop['plays_board'] == 0
    river = features('AsKs', 'QsJs2d7c8d')
    assert river['private_flush_draw'] == river['private_straight_draw'] == 0
    assert river['straight_out_card_fraction'] == river['flush_out_card_fraction'] == 0
    board_draw_only = features('AsKd', '2c3d4h5s')
    assert board_draw_only['made_straight'] == 1  # Ace completes the wheel.
    assert board_draw_only['private_straight_draw'] == 0
    assert features('9sKd', '2c3d4h5s')['private_straight_draw'] == 0
