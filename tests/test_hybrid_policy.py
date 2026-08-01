"""Card ownership, symmetry, and checkpoint identity for the hybrid encoder."""
import copy
from dataclasses import replace
from itertools import permutations
import random

import numpy as np
import pytest
import torch

from poker_lab import hybrid_policy as hybrid
from poker_lab import semantic_policy as semantic
from poker_lab.agents import cpu_state
from poker_lab.game import HoldemState, Observation
from poker_lab.networks import batch_observations


@pytest.fixture(autouse=True)
def deterministic_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(190926)
    yield
    torch.set_num_threads(previous)


def ids(text):
    return tuple(4*'23456789TJQKA'.index(text[i])+'cdhs'.index(text[i+1])
                 for i in range(0, len(text), 2))


def observation(hole='AsKd', board=''):
    cards = ids(board)
    street = {0: 0., 3: 1/3, 4: 2/3, 5: 1.}[len(cards)]
    return Observation(ids(hole)+cards+(52,)*(5-len(cards)),
                       (1., 0., street, .15, .9, .9, 0., .1, .1, .2, .9, .005),
                       (), (1.,)*5)


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


@pytest.mark.parametrize('board', ['', 'QhJh2c', 'QhJh2c8d', 'QhJh2c8d9s'])
def test_external_observations_preserve_suit_and_hole_flop_symmetry(board):
    obs = observation('AhKh', board)
    model = hybrid.HybridPokerNetwork().eval()
    expected_slots = hybrid.canonical_card_slots(batch_observations([obs]))
    expected_output = model(batch_observations([obs]))
    for suit_map in permutations(range(4)):
        cards = tuple(52 if card == 52 else 4*(card//4)+suit_map[card % 4]
                      for card in obs.cards)
        for flop in set(permutations(cards[2:5])):
            changed = replace(obs, cards=cards[1::-1]+flop+cards[5:])
            same(hybrid.canonical_card_slots(batch_observations([changed])), expected_slots)
        same(model(batch_observations([changed])), expected_output)


def test_turn_and_river_identity_and_reveal_order_are_preserved():
    first = observation('AhKh', 'QhJh2c8d9s')
    switched = replace(first, cards=first.cards[:5]+first.cards[6:]+first.cards[5:6])
    batch = batch_observations([first, switched])
    semantic_rows = semantic.semantic_features(batch)
    assert torch.equal(semantic_rows[0], semantic_rows[1])
    slots = hybrid.canonical_card_slots(batch)
    assert torch.equal(slots[0, :5], slots[1, :5])
    assert not torch.equal(slots[0, 5:], slots[1, 5:])
    logits, values = hybrid.HybridPokerNetwork().eval()(batch)
    assert not torch.equal(logits[0], logits[1]) and values[0] != values[1]


def test_hidden_opponent_cards_and_future_deck_do_not_change_policy():
    first = HoldemState.new(134)
    changed = list(first.deck)
    movable = [i for i in range(52) if i not in {0, 1, 4, 5, 6}]
    for index, value in zip(movable, reversed([changed[i] for i in movable])):
        changed[index] = value
    second = HoldemState.new(735, deck=changed)
    for state in (first, second):
        while state.street == 0:
            state.apply_slot(1)
    assert first.deck[2:4] != second.deck[2:4]
    assert first.deck[7:9] != second.deck[7:9]
    assert first.observe() == second.observe()
    agent = hybrid.HybridPPOAgent(cpu_state(hybrid.HybridPokerNetwork()))
    np.testing.assert_array_equal(agent.probabilities(first.observe()),
                                  agent.probabilities(second.observe()))


def test_deployment_and_queries_preserve_rng_and_clones_reproduce_actions():
    state = cpu_state(hybrid.HybridPokerNetwork())
    before_py, before_np, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
    agent = hybrid.HybridPPOAgent(state, seed=901)
    own_rng = copy.deepcopy(agent.rng.bit_generator.state)
    agent.probabilities(observation('AsKd', '3c4s5h'))
    assert random.getstate() == before_py
    same(np.random.get_state(), before_np)
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert own_rng == agent.rng.bit_generator.state
    clone = agent.clone()
    actions = [agent.act(observation()) for _ in range(20)]
    assert actions == [clone.act(observation()) for _ in range(20)]
    agent.reset(901)
    assert actions == [agent.act(observation()) for _ in range(20)]
    assert all(not parameter.requires_grad for parameter in agent.model.parameters())


@pytest.mark.parametrize('mask', [(0, 0, 0, 0, 0), (1, .5, 0, 0, 0), (1, 1), (1, float('nan'), 0, 0, 0)])
def test_invalid_masks_rejected(mask):
    agent = hybrid.HybridPPOAgent(cpu_state(hybrid.HybridPokerNetwork()))
    with pytest.raises(ValueError):
        agent.probabilities(replace(observation(), legal_mask=mask))
