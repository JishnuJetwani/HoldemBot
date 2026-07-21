"""Independent PokerKit oracle tests for the OpenSpiel-backed adapter."""
from dataclasses import asdict
from itertools import permutations

import numpy as np
import pytest

from poker_lab.game import Action, GameSpec, HoldemState, canonical_cards, card_string


def fixed_deck(prefix):
    cards = ["23456789TJQKA".index(card[0]) * 4 + "cdhs".index(card[1]) for card in prefix]
    assert len(cards) == len(set(cards))
    return cards + [card for card in range(52) if card not in cards]


class Oracle:
    def __init__(self, deck, spec):
        pokerkit = pytest.importorskip("pokerkit")
        A = pokerkit.Automation
        self.state = pokerkit.NoLimitTexasHoldem.create_state(
            (A.ANTE_POSTING, A.BET_COLLECTION, A.BLIND_OR_STRADDLE_POSTING,
             A.RUNOUT_COUNT_SELECTION, A.HOLE_CARDS_SHOWING_OR_MUCKING,
             A.HAND_KILLING, A.CHIPS_PUSHING, A.CHIPS_PULLING),
            True, 0, (spec.small_blind, spec.big_blind), spec.big_blind,
            (spec.stack, spec.stack), 2, mode=pokerkit.Mode.CASH_GAME)
        self.spec = spec
        self.board = [card_string(card) for card in deck[4:9]]
        self.board_count = 0
        for seat in range(2):
            self.state.deal_hole("".join(card_string(c) for c in deck[2 * seat:2 * seat + 2]), seat)

    def advance(self):
        while self.state.status and self.state.actor_index is None:
            if self.state.card_burning_status:
                self.state.burn_card("??")
            elif self.state.board_dealing_count:
                count = self.state.board_dealing_count
                self.state.deal_board("".join(self.board[self.board_count:self.board_count + count]))
                self.board_count += count
            else:
                raise AssertionError("unhandled oracle automatic state")

    def apply(self, action):
        if action.kind == "fold":
            self.state.fold()
        elif action.kind == "call":
            self.state.check_or_call()
        else:
            self.state.complete_bet_or_raise_to(action.amount)
        self.advance()

    def assert_matches(self, game):
        assert game.is_terminal() == (not self.state.status)
        if game.is_terminal():
            assert game.returns() == pytest.approx([x - self.spec.stack for x in self.state.stacks])
            assert sum(game.returns()) == pytest.approx(0)
        else:
            view = game.public_view(game.current_player())
            assert game.current_player() == self.state.actor_index
            assert view["stacks"] == self.state.stacks
            assert view["street_contributions"] == self.state.bets
            assert view["pot"] == self.state.total_pot_amount
            for action in game.action_menu():
                if action is not None and action.kind == "raise_to":
                    assert self.state.can_complete_bet_or_raise_to(action.amount)


def play_oracle(deck, actions, spec=None):
    spec = spec or GameSpec()
    game, oracle = HoldemState.new(seed=0, spec=spec, deck=deck), Oracle(deck, spec)
    oracle.assert_matches(game)
    for action in actions:
        game.apply_action(action)
        oracle.apply(action)
        oracle.assert_matches(game)
    return game


def test_headsup_order_limp_check_all_streets_and_tie():
    # Board is a royal flush, so neither private hand can break the tie.
    deck = fixed_deck(["2c", "3d", "4h", "5s", "Tc", "Jc", "Qc", "Kc", "Ac"])
    game = play_oracle(deck, [Action("call")] * 8)
    assert game.is_terminal()
    assert game.returns() == [0, 0]
    assert [row["player"] for row in game.history] == [1, 0, 0, 1, 0, 1, 0, 1]


def test_fold_uncalled_raise_payout_and_hidden_cards():
    game = play_oracle(fixed_deck(["Ac", "Ad", "Kc", "Kd"]),
                       [Action("raise_to", 15000), Action("fold")])
    assert game.returns() == [-100, 100]
    assert game.public_view(0)["opponent_cards"] == []
    assert game.public_view(1)["opponent_cards"] == []


def test_allin_runout_uses_correct_private_cards_and_board():
    deck = fixed_deck(["Ac", "Ad", "Kc", "Kd", "2c", "3h", "7s", "8d", "9c"])
    game = play_oracle(deck, [Action("raise_to", 20000), Action("call")])
    assert game.returns() == [20000, -20000]
    assert len(game.public_view(0)["board"]) == 5
    assert set(game.public_view(0)["opponent_cards"]) == {"Kc", "Kd"}


def test_short_allin_raise_is_allowed_below_full_raise_size():
    spec = GameSpec(stack=1000, small_blind=50, big_blind=100)
    deck = fixed_deck(["Ac", "Ad", "Kc", "Kd", "2c", "3h", "7s", "8d", "9c"])
    game = play_oracle(deck, [Action("raise_to", 700), Action("raise_to", 1000), Action("call")], spec)
    assert game.returns() == [1000, -1000]


def test_arbitrary_integer_raises_and_street_total_conversion():
    deck = fixed_deck(["Ac", "Ad", "Kc", "Kd", "2c", "3h", "7s", "8d", "9c"])
    actions = [Action("raise_to", 237), Action("call"), Action("raise_to", 311),
               Action("raise_to", 700), Action("call"), Action("call"), Action("call"),
               Action("call"), Action("call")]
    game = play_oracle(deck, actions)
    assert game.is_terminal()
    assert game.returns() == [937, -937]


def test_illegal_raise_does_not_mutate_game_and_menu_deduplicates():
    game = HoldemState.new(1, spec=GameSpec(stack=250))
    before = game.to_dict()
    with pytest.raises(ValueError):
        game.apply_action(Action("raise_to", 199))
    assert game.to_dict() == before
    menu = game.action_menu()
    raises = [action.amount for action in menu if action is not None and action.kind == "raise_to"]
    assert len(raises) == len(set(raises))
    assert all(amount <= 250 for amount in raises)


@pytest.mark.parametrize('stack', [2000, 20000])
def test_random_legal_trajectories_match_independent_oracle(stack):
    rng = np.random.default_rng(20260918)
    for seed in range(30):
        deck = rng.permutation(52).tolist()
        spec = GameSpec(stack=stack)
        game, oracle = HoldemState.new(seed, spec, deck), Oracle(deck, spec)
        while not game.is_terminal():
            oracle.assert_matches(game)
            actions = [action for action in game.action_menu() if action is not None]
            action = actions[int(rng.integers(len(actions)))]
            game.apply_action(action)
            oracle.apply(action)
        oracle.assert_matches(game)


def test_hidden_cards_and_future_deck_never_change_observation():
    deck = fixed_deck(["Ac", "Ad", "Kc", "Kd", "2c", "3h", "7s", "8d", "9c"])
    changed = deck.copy()
    changed[0], changed[10] = changed[10], changed[0]  # opponent card for acting SB
    changed[4], changed[11] = changed[11], changed[4]  # future board
    left, right = HoldemState.new(3, deck=deck), HoldemState.new(9, deck=changed)
    assert asdict(left.observe()) == asdict(right.observe())
    view = left.public_view(1)
    assert "deck" not in view and "seed" not in view
    assert view["opponent_cards"] == []


def test_suit_isomorphism_is_exact_and_replay_and_clone_are_independent():
    visible = (48, 45, 8, 13, 22, 52, 52)
    expected = canonical_cards(visible)
    for perm in permutations(range(4)):
        changed = tuple(52 if c == 52 else c // 4 * 4 + perm[c % 4] for c in visible)
        assert canonical_cards(changed) == expected
    game = HoldemState.new(8)
    child = game.child(1)
    assert game.history == []
    assert len(child.history) == 1
    reconstructed = HoldemState.from_dict(child.to_dict())
    assert reconstructed.public_view(0) == child.public_view(0)
    assert reconstructed.observe() == child.observe()
