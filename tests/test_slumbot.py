import pytest

from poker_lab.game import Action
from poker_lab.slumbot import ENDPOINTS, choose_action, encode_action, parse_action_history, state_from_response


def test_protocol_uses_street_totals_and_current_endpoint():
    actions = parse_action_history("b200c/kb400")
    assert [(a.street, a.code, a.amount) for a in actions] == [(0, "b", 200), (0, "c", None), (1, "k", None), (1, "b", 400)]
    assert ENDPOINTS["act"] == "https://slumbot.com/slumbot/api/act"
    assert encode_action(Action("raise_to", 400), facing_bet=False) == "b400"
    assert encode_action(Action("call"), facing_bet=False) == "k"
    assert encode_action(Action("call"), facing_bet=True) == "c"
    assert len(parse_action_history("b20000c///")) == 2


def response(**kwargs):
    return {"client_pos": 0, "action": "b200", "hole_cards": ["Ac", "9d"], "board": [], "token": "offline-test", **kwargs}


def test_reconstructs_preflop_position_and_raise_amount():
    state = state_from_response(response())
    view = state.public_view(0)
    assert state.current_player() == 0
    assert view["contributions"] == [100, 200]
    assert set(view["hole_cards"]) == {"Ac", "9d"}


def test_reconstructs_flop_contributions_separately_from_hand_total():
    state = state_from_response(response(action="b200c/kb400", board=["2c", "3h", "7s"]))
    view = state.public_view(0)
    assert view["street_contributions"] == [0, 400]
    assert view["contributions"] == [200, 600]
    assert view["stacks"] == [19800, 19400]


def test_rejects_bad_sequences_cards_and_terminal_scoring():
    for bad in [response(action="k"), response(action="cc"), response(hole_cards=["Ac", "Ac"]),
                response(board=["2c"]), response(winnings=0), response(action="b20000c///"),
                response(action="b200c/kb400", board=[]), response(action="b200c"),
                response(action="b200c/cc", board=["2c", "3h", "7s"])]:
        with pytest.raises(ValueError):
            state_from_response(bad)


def test_choose_action_only_constructs_payload():
    class Agent:
        def reset(self, seed):
            self.seed = seed

        def act(self, observation):
            return 1

    agent = Agent()
    assert choose_action(response(), agent, reset_seed=9) == {"token": "offline-test", "incr": "c"}
    assert agent.seed == 9
