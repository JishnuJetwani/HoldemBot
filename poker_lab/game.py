"""Full no-limit Hold'em backed by OpenSpiel's ACPC engine.

The engine measures raises by total chips put into the hand; this adapter
uses the total for the current street. Observations contain only visible cards.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import permutations
from typing import Any

import numpy as np
import pyspiel

RANKS = "23456789TJQKA"
SUITS = "cdhs"
ACTION_NAMES = ("fold", "check/call", "half-pot", "pot", "all-in")


@dataclass(frozen=True)
class GameSpec:
    stack: int = 20000
    small_blind: int = 50
    big_blind: int = 100

    def __post_init__(self):
        if not (0 < self.small_blind <= self.big_blind < self.stack):
            raise ValueError("Require 0 < small blind <= big blind < stack")
        if any(type(x) is not int for x in (self.stack, self.small_blind, self.big_blind)):
            raise ValueError("Chip amounts must be integers")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Action:
    kind: str
    amount: int | None = None

    def __post_init__(self):
        if self.kind not in ("fold", "call", "raise_to"):
            raise ValueError(f"Unknown action: {self.kind}")
        if self.kind == "raise_to" and (type(self.amount) is not int or self.amount < 1):
            raise ValueError("raise_to requires a positive integer street total")
        if self.kind != "raise_to" and self.amount is not None:
            raise ValueError("Only raise_to has an amount")


@dataclass(frozen=True)
class Observation:
    cards: tuple[int, ...]
    scalars: tuple[float, ...]
    history: tuple[tuple[float, ...], ...]
    legal_mask: tuple[float, ...]
    player: int = 0


def card_string(card: int) -> str:
    return RANKS[card // 4] + SUITS[card % 4]


@lru_cache(maxsize=1 << 16)
def canonical_cards(cards: tuple[int, ...]) -> tuple[int, ...]:
    """Map equivalent suit patterns to the same cards, ignoring hole/flop order."""
    candidates = []
    for perm in permutations(range(4)):
        c = [52 if x == 52 else (x // 4) * 4 + perm[x % 4] for x in cards]
        candidates.append(tuple(sorted(c[:2]) + sorted(c[2:5]) + c[5:]))
    return min(candidates)


@lru_cache(maxsize=16)
def _engine(spec: GameSpec):
    return pyspiel.load_game("universal_poker", {
        "numPlayers": 2, "betting": "nolimit", "stack": f"{spec.stack} {spec.stack}",
        "blind": f"{spec.big_blind} {spec.small_blind}", "numRounds": 4,
        "firstPlayer": "2 1 1 1", "numSuits": 4, "numRanks": 13,
        "numHoleCards": 2, "numBoardCards": "0 3 1 1", "bettingAbstraction": "fullgame",
    })


class HoldemState:
    @classmethod
    def new(cls, seed: int, spec: GameSpec | None = None, deck: list[int] | None = None):
        self = cls()
        self.spec = spec or GameSpec()
        self.seed = int(seed)
        self.deck = tuple(int(x) for x in (np.random.default_rng(seed).permutation(52) if deck is None else deck))
        if len(self.deck) != 52 or set(self.deck) != set(range(52)):
            raise ValueError("Deck must be a permutation of card IDs 0..51")
        self.raw = _engine(self.spec).new_initial_state()
        self._dealt = 0
        self._contributions = [self.spec.big_blind, self.spec.small_blind]
        self._street_start = [0, 0]
        self.street = 0
        self.history: list[dict[str, Any]] = []
        self._actions: list[Action] = []
        self._legal_cache = None
        self._advance_chance()
        return self

    def clone(self):
        other = object.__new__(HoldemState)
        other.spec, other.seed, other.deck = self.spec, self.seed, self.deck
        other.raw = self.raw.clone()
        other._dealt, other.street = self._dealt, self.street
        other._contributions = self._contributions.copy()
        other._street_start = self._street_start.copy()
        other.history = [dict(event) for event in self.history]
        other._actions = self._actions.copy()
        other._legal_cache = self._legal_cache
        return other

    def _advance_chance(self):
        if self.raw.is_chance_node() and self._dealt >= 4:
            self._street_start = self._contributions.copy()
        while self.raw.is_chance_node():
            self.raw.apply_action(self.deck[self._dealt])
            self._dealt += 1
        board_count = max(0, self._dealt - 4)
        self.street = {0: 0, 3: 1, 4: 2, 5: 3}[board_count]
        self._legal_cache = None

    def is_terminal(self) -> bool:
        return self.raw.is_terminal()

    def current_player(self) -> int:
        return self.raw.current_player()

    def returns(self) -> list[float]:
        return list(self.raw.returns())

    def _legal(self):
        if self._legal_cache is None:
            # fullgame exposes every integer raise; only keep bounds plus fold/call.
            legal = self.raw.legal_actions()
            raises = [x for x in legal if x >= 2]
            self._legal_cache = (0 in legal, 1 in legal, raises[0] if raises else None, raises[-1] if raises else None)
        return self._legal_cache

    def action_menu(self) -> list[Action | None]:
        if self.is_terminal():
            return [None] * 5
        fold, call, low, high = self._legal()
        menu = [Action("fold") if fold else None, Action("call") if call else None]
        seen = set()
        player = self.current_player()
        amount_to_call = max(self._contributions) - self._contributions[player]
        pot_after_call = sum(self._contributions) + amount_to_call
        for frac in (0.5, 1.0, None):
            if low is None:
                menu.append(None)
                continue
            total = high if frac is None else round(max(self._contributions) + frac * pot_after_call)
            total = max(low, min(high, total))
            if total in seen:
                menu.append(None)
            else:
                seen.add(total)
                menu.append(Action("raise_to", total - self._street_start[player]))
        return menu

    def legal_mask(self) -> np.ndarray:
        return np.array([a is not None for a in self.action_menu()], dtype=np.float32)

    def child(self, slot: int):
        child = self.clone()
        child.apply_slot(slot)
        return child

    def apply_slot(self, slot: int):
        if not isinstance(slot, (int, np.integer)) or not 0 <= slot < 5:
            raise ValueError("Invalid action slot")
        action = self.action_menu()[slot]
        if action is None:
            raise ValueError("Action slot is masked")
        self.apply_action(action)

    def apply_action(self, action: Action):
        if self.is_terminal():
            raise ValueError("Hand is already over")
        player = self.current_player()
        fold, call, low, high = self._legal()
        if action.kind == "fold":
            if not fold:
                raise ValueError("Cannot fold without facing a bet")
            engine_action = 0
        elif action.kind == "call":
            if not call:
                raise ValueError("Cannot check/call here")
            engine_action = 1
        else:
            engine_action = action.amount + self._street_start[player]
            if low is None or not low <= engine_action <= high:
                raise ValueError(f"Illegal raise_to {action.amount}; legal street totals: "
                                 f"{None if low is None else low-self._street_start[player]}.."
                                 f"{None if high is None else high-self._street_start[player]}")
        before = self._contributions[player]
        pot = sum(self._contributions)
        old_street = self.street
        was_check = action.kind == "call" and before == max(self._contributions)
        self.raw.apply_action(engine_action)
        if action.kind == "call":
            self._contributions[player] = min(self.spec.stack, max(self._contributions))
        elif action.kind == "raise_to":
            self._contributions[player] = engine_action
        self.history.append({"player": player, "street": old_street, "kind": action.kind,
                             "amount": action.amount, "paid": self._contributions[player] - before,
                             "street_total": self._contributions[player] - self._street_start[player],
                             "pot_before": pot, "check": was_check,
                             "all_in": self._contributions[player] == self.spec.stack})
        self._actions.append(action)
        self._advance_chance()

    def observe(self, player: int | None = None) -> Observation:
        player = self.current_player() if player is None else player
        if player not in (0, 1):
            raise ValueError("Observation requires player 0 or 1")
        board = self.deck[4:self._dealt]
        cards = self.deck[player * 2:player * 2 + 2] + board + (52,) * (5 - len(board))
        stack = self.spec.stack
        totals = self._contributions
        street = [totals[i] - self._street_start[i] for i in (0, 1)]
        low, high = self._legal()[2:] if not self.is_terminal() else (None, None)
        scalars = (float(player == 0), float(player == 1), self.street / 3, sum(totals) / (2 * stack),
                   (stack - totals[player]) / stack, (stack - totals[1 - player]) / stack,
                   street[player] / stack, street[1 - player] / stack,
                   (max(totals) - totals[player]) / stack,
                   0. if low is None else (low - self._street_start[player]) / stack,
                   0. if high is None else (high - self._street_start[player]) / stack,
                   self.spec.big_blind / stack)
        events = tuple(tuple(float(e["street"] == r) for r in range(4)) +
                       (float(e["player"] == player), float(e["player"] != player)) +
                       tuple(float(e["kind"] == k) for k in ("fold", "call", "raise_to")) +
                       (e["street_total"] / stack, e["pot_before"] / (2 * stack), float(e["all_in"]))
                       for e in self.history)
        return Observation(canonical_cards(cards), scalars, events, tuple(self.legal_mask()), player)

    def public_view(self, player: int) -> dict:
        if player not in (0, 1):
            raise ValueError("Invalid seat")
        terminal = self.is_terminal()
        showdown = terminal and self.history[-1]["kind"] != "fold"
        menu = self.action_menu()
        legal = []
        for slot, action in enumerate(menu):
            if action is None:
                continue
            label = ACTION_NAMES[slot]
            if slot == 1:
                label = "check" if self._contributions[self.current_player()] == max(self._contributions) else "call"
            legal.append({"slot": slot, **asdict(action), "label": label})
        return {"current_player": self.current_player(), "terminal": terminal, "street": self.street,
                "pot": 0 if terminal else sum(self._contributions),
                "last_pot": 2 * min(self._contributions) if terminal else None,
                "stacks": [self.spec.stack + x for x in self.returns()] if terminal else
                          [self.spec.stack - x for x in self._contributions],
                "contributions": self._contributions.copy(),
                "street_contributions": [x-y for x, y in zip(self._contributions, self._street_start)],
                "hole_cards": [card_string(c) for c in self.deck[player*2:player*2+2]],
                "board": [card_string(c) for c in self.deck[4:self._dealt]],
                "opponent_cards": [card_string(c) for c in self.deck[(1-player)*2:(1-player)*2+2]] if showdown else [],
                "legal_actions": legal, "history": [dict(e) for e in self.history],
                "returns": self.returns() if terminal else None, "game_spec": self.spec.to_dict()}

    def to_dict(self) -> dict:
        """Save the full state, including hidden cards, for replay."""
        return {"seed": self.seed, "deck": list(self.deck), "game_spec": self.spec.to_dict(),
                "actions": [asdict(a) for a in self._actions]}

    @classmethod
    def from_dict(cls, data: dict):
        self = cls.new(data["seed"], GameSpec(**data["game_spec"]), data["deck"])
        for action in data["actions"]:
            self.apply_action(Action(**action))
        return self
