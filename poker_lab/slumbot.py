"""Offline Slumbot protocol adapter; no network requests.

Protocol source, inspected September 18, 2026:
https://www.slumbot.com/sample_api.py
Endpoints are /slumbot/api/{new_hand,act,login}; bX is street-total chips.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from poker_lab.game import Action, GameSpec, HoldemState

SLUMBOT_SPEC = GameSpec(stack=20000, small_blind=50, big_blind=100)
ENDPOINTS = {name: f"https://slumbot.com/slumbot/api/{name}" for name in ("new_hand", "act", "login")}
_TOKEN = re.compile(r"[kcf]|b[0-9]+")


@dataclass(frozen=True)
class ProtocolAction:
    street: int
    code: str
    amount: int | None = None


def parse_action_history(history: str) -> list[ProtocolAction]:
    """Parse action tokens; the engine checks whether the actions are legal."""
    if not isinstance(history, str):
        raise ValueError("action history must be a string")
    streets = history.split("/")
    if len(streets) > 4:
        raise ValueError("too many streets")
    result = []
    for street, sequence in enumerate(streets):
        offset = 0
        while offset < len(sequence):
            match = _TOKEN.match(sequence, offset)
            if match is None:
                raise ValueError(f"invalid action token at offset {offset} on street {street}")
            code = match.group()
            result.append(ProtocolAction(street, code[0], int(code[1:]) if code[0] == "b" else None))
            offset = match.end()
    return result


def _card_id(card: str) -> int:
    if not isinstance(card, str) or len(card) != 2 or card[0] not in "23456789TJQKA" or card[1] not in "cdhs":
        raise ValueError(f"invalid card: {card!r}")
    return "23456789TJQKA".index(card[0]) * 4 + "cdhs".index(card[1])


def state_from_response(response: dict[str, Any]) -> HoldemState:
    """Rebuild the decision state, filling unknown cards with placeholders.

    The policy sees only its hole cards and the board. Use Slumbot's winnings
    to score the hand; placeholder cards cannot determine the real outcome.
    """
    if "error_msg" in response:
        raise ValueError(f"Slumbot response error: {response['error_msg']}")
    position = response.get("client_pos")
    if type(position) is not int or position not in (0, 1):
        raise ValueError("client_pos must be 0 (BB) or 1 (SB)")
    if response.get("winnings") is not None:
        raise ValueError("terminal response: use server winnings, do not request an action")
    hole, board = response.get("hole_cards"), response.get("board")
    if not isinstance(hole, list) or len(hole) != 2 or not isinstance(board, list) or len(board) not in (0, 3, 4, 5):
        raise ValueError("expected two private cards and a board of length 0, 3, 4, or 5")
    known = [_card_id(card) for card in hole + board]
    if len(set(known)) != len(known):
        raise ValueError("duplicate known cards")
    available = iter(card for card in range(52) if card not in known)
    deck: list[int | None] = [None] * 52
    deck[position * 2:position * 2 + 2] = known[:2]
    deck[4:4 + len(board)] = known[2:]
    for index, card in enumerate(deck):
        if card is None:
            deck[index] = next(available)
    state = HoldemState.new(seed=0, spec=SLUMBOT_SPEC, deck=[int(card) for card in deck])
    history = response.get("action", "")
    for action in parse_action_history(history):
        if state.is_terminal():
            raise ValueError("actions after the hand ended")
        actor = state.current_player()
        view = state.public_view(actor)
        if view["street"] != action.street:
            raise ValueError("street separators do not match betting completion")
        street_contributions = view["street_contributions"]
        facing_bet = street_contributions[1 - actor] > street_contributions[actor]
        if action.code == "k" and facing_bet:
            raise ValueError("check facing a bet")
        if action.code == "c" and not facing_bet:
            raise ValueError("call without a bet; protocol requires k")
        state.apply_action(Action("raise_to", action.amount) if action.code == "b"
                           else Action("fold") if action.code == "f" else Action("call"))
    if state.is_terminal():
        raise ValueError("terminal action history must be scored using server winnings")
    view = state.public_view(position)
    separators = history.count("/")
    # The official parser permits a just-completed street without its final
    # slash (e.g. "ck"). Extra empty streets cannot advance a live hand.
    valid_separator_counts = {view["street"]}
    if history and not history.endswith("/"):
        valid_separator_counts.add(view["street"] - 1)
    if separators not in valid_separator_counts:
        raise ValueError("trailing street separators do not match the live street")
    if view["current_player"] != position:
        raise ValueError("response does not describe the client's turn")
    if len(view["board"]) != len(board):
        raise ValueError("public board does not match the betting street")
    return state


def encode_action(action: Action, *, facing_bet: bool) -> str:
    if action.kind == "fold":
        return "f"
    if action.kind == "call":
        return "c" if facing_bet else "k"
    if action.kind == "raise_to" and isinstance(action.amount, int) and action.amount > 0:
        return f"b{action.amount}"
    raise ValueError("invalid protocol action")


def choose_action(response: dict[str, Any], agent: Any, *, reset_seed: int | None = None) -> dict[str, Any]:
    """Build an action payload without sending it. Reset only on new hands."""
    state = state_from_response(response)
    if reset_seed is not None:
        agent.reset(reset_seed)
    slot = agent.act(state.observe())
    menu = state.action_menu()
    if isinstance(slot, bool) or not isinstance(slot, Integral) or not 0 <= slot < len(menu) or menu[slot] is None:
        raise ValueError("agent selected an illegal action")
    actor = state.current_player()
    contributions = state.public_view(actor)["street_contributions"]
    return {"token": response.get("token"),
            "incr": encode_action(menu[slot], facing_bet=contributions[1 - actor] > contributions[actor])}
