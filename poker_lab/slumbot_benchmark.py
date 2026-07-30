"""Play a fixed number of hands against Slumbot anonymously.

Send each POST once. If its outcome is unclear, stop and keep completed hands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import t

from poker_lab.evaluation import _seed, validate_probabilities
from poker_lab.game import Action, HoldemState
from poker_lab.slumbot import (ENDPOINTS, SLUMBOT_SPEC, ProtocolAction, _card_id,
                               choose_action, parse_action_history, state_from_response)


class ProtocolError(ValueError):
    """Protocol errors without raw server responses."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post_json(endpoint: str, payload: dict, *, timeout: float = 20.0) -> dict:
    """Send one HTTPS request without redirects or retries."""
    if endpoint not in (ENDPOINTS["new_hand"], ENDPOINTS["act"]):
        raise ValueError("only anonymous Slumbot play endpoints are allowed")
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
        if response.status != 200:
            raise ProtocolError("unexpected HTTP status")
        data = response.read(2_000_001)
        if len(data) > 2_000_000:
            raise ProtocolError("response exceeds size limit")
        result = json.loads(data)
    if not isinstance(result, dict):
        raise ProtocolError("response must be an object")
    return result


def _validate_response(response: dict, previous: dict | None = None,
                       sent: str | None = None, sent_street: int | None = None) -> HoldemState:
    if not isinstance(response, dict) or "error_msg" in response:
        raise ProtocolError("server returned an error or invalid object")
    if "token" in response and (not isinstance(response["token"], str) or not response["token"]):
        raise ProtocolError("invalid session token")
    position = response.get("client_pos")
    history, holes, board = response.get("action"), response.get("hole_cards"), response.get("board")
    if type(position) is not int or position not in (0, 1) or not isinstance(history, str):
        raise ProtocolError("invalid position or history")
    if not isinstance(holes, list) or len(holes) != 2 or not isinstance(board, list) or len(board) not in (0, 3, 4, 5):
        raise ProtocolError("invalid private or public card count")
    try:
        known = [_card_id(card) for card in holes + board]
        actions = parse_action_history(history)
    except (TypeError, ValueError) as error:
        raise ProtocolError("invalid cards or action syntax") from error
    if len(set(known)) != len(known):
        raise ProtocolError("duplicate known cards")
    if previous is not None:
        if (position != previous["client_pos"] or set(holes) != set(previous["hole_cards"]) or
                board[:len(previous["board"])] != previous["board"]):
            raise ProtocolError("seat or visible cards changed within the hand")
        prior_actions = parse_action_history(previous["action"])
        increment = parse_action_history(sent or "")
        if len(increment) != 1 or sent_street is None:
            raise ProtocolError("invalid recorded client action")
        expected = prior_actions + [ProtocolAction(sent_street, increment[0].code, increment[0].amount)]
        if actions[:len(expected)] != expected:
            raise ProtocolError("server history does not extend the recorded client action")

    terminal = response.get("winnings") is not None
    if not terminal:
        try:
            return state_from_response(response)
        except (TypeError, ValueError) as error:
            raise ProtocolError("invalid live decision state") from error
    winnings = response["winnings"]
    if (isinstance(winnings, bool) or not isinstance(winnings, (int, float)) or
            not math.isfinite(winnings) or abs(winnings) > SLUMBOT_SPEC.stack):
        raise ProtocolError("invalid server winnings")

    # Placeholder cards let us check betting, not the real showdown result.
    deck: list[int | None] = [None] * 52
    deck[position * 2:position * 2 + 2] = known[:2]
    deck[4:4 + len(board)] = known[2:]
    unused = iter(card for card in range(52) if card not in known)
    deck = [next(unused) if card is None else card for card in deck]
    state = HoldemState.new(seed=0, spec=SLUMBOT_SPEC, deck=deck)
    try:
        for action in actions:
            if state.is_terminal() or state.street != action.street:
                raise ProtocolError("terminal history has invalid street sequencing")
            actor = state.current_player()
            view = state.public_view(actor)
            contributions = view["street_contributions"]
            facing = contributions[1 - actor] > contributions[actor]
            if (action.code == "k" and facing) or (action.code == "c" and not facing):
                raise ProtocolError("check/call token disagrees with betting state")
            state.apply_action(Action("raise_to", action.amount) if action.code == "b" else
                               Action("fold") if action.code == "f" else Action("call"))
    except (TypeError, ValueError) as error:
        raise ProtocolError("invalid terminal betting history") from error
    if not state.is_terminal() or len(state.public_view(position)["board"]) != len(board):
        raise ProtocolError("terminal result disagrees with history or public board")
    last = actions[-1]
    valid_slashes = {last.street}
    if last.code == "c" and max(state.public_view(position)["contributions"]) == SLUMBOT_SPEC.stack:
        valid_slashes.add(3)  # An all-in runout may omit trailing empty streets.
    if history.count("/") not in valid_slashes:
        raise ProtocolError("terminal street separators are inconsistent")
    return state


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path: Path, report: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _statistics(hands: list[dict]) -> dict:
    values = np.asarray([hand["winnings_chips"] for hand in hands], dtype=float) * 100 / SLUMBOT_SPEC.big_blind
    n = len(values)
    mean = float(values.mean()) if n else None
    se = float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    margin = float(t.ppf(.975, n - 1) * se) if n > 1 else None
    return {"completed_hands": n, "bb_per_100": mean,
            "ci95_bb_per_100": [mean - margin, mean + margin] if margin is not None else None,
            "standard_error_bb_per_100": se,
            "seat_counts": {"big_blind": sum(h["client_pos"] == 0 for h in hands),
                            "small_blind": sum(h["client_pos"] == 1 for h in hands)},
            "ci_method": "Student t over server-dealt hand returns, assuming independent hands; not duplicate-deal or AIVAT",
            "qualification": "Stopping early can bias the result; incomplete sessions are descriptive only."}


def run_benchmark(checkpoint: str | Path, *, expected_sha256: str, num_hands: int,
                  output: str | Path, seed: int = 83001, min_interval_seconds: float = 0.25,
                  request_timeout_seconds: float = 20.0, max_actions_per_hand: int = 10000,
                  transport: Callable[[str, dict], dict] | None = None,
                  agent_loader: Callable[[str], Any] | None = None,
                  clock: Callable[[], float] = time.monotonic,
                  sleep: Callable[[float], None] = time.sleep) -> dict:
    """Play one hand at a time; stop if a hand's outcome is unclear."""
    if type(num_hands) is not int or num_hands < 2 or num_hands % 2:
        raise ValueError("a positive even hand count of at least two is required")
    if type(seed) is not int or seed < 0 or type(max_actions_per_hand) is not int or max_actions_per_hand < 1:
        raise ValueError("invalid seed or action limit")
    if any(isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in
           (min_interval_seconds, request_timeout_seconds)):
        raise ValueError("request interval and timeout must be positive and finite")
    checkpoint, output = Path(checkpoint).resolve(), Path(output)
    if output.exists() or output.with_suffix(output.suffix + ".tmp").exists():
        raise FileExistsError("refusing to overwrite or resume a remote session artifact")
    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    actual = _hash(checkpoint)
    if actual != expected_sha256 or metadata.get("sha256") != actual:
        raise ValueError("checkpoint differs from its frozen identity")
    if metadata.get("game_spec") != SLUMBOT_SPEC.to_dict() or metadata.get("artifact_kind") != "deployment":
        raise ValueError("requires a deployed 200bb 50/100 checkpoint")
    if agent_loader is None:
        from poker_lab.agents import load_agent
        agent_loader = load_agent
    agent = agent_loader(str(checkpoint))
    if transport is None:
        transport = lambda endpoint, payload: post_json(endpoint, payload, timeout=request_timeout_seconds)
    protocol = {"checkpoint": str(checkpoint), "checkpoint_sha256": actual,
                "checkpoint_algorithm": metadata.get("algorithm"), "game_spec": SLUMBOT_SPEC.to_dict(),
                "requested_hands": num_hands, "policy_seed": seed,
                "min_interval_seconds": min_interval_seconds, "request_timeout_seconds": request_timeout_seconds,
                "max_actions_per_hand": max_actions_per_hand, "access": "anonymous; no account login",
                "source_sha256": {name: _hash(Path(__file__).parent / name) for name in
                                  ("slumbot_benchmark.py", "slumbot.py", "game.py", "agents.py")}}
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "protocol": protocol, "status": "running", "raw_hands": [], "statistics": _statistics([]),
              "request_attempts": 0, "incomplete_hand": None,
              "privacy": "Anonymous access does not establish private or unpublished results.",
              "claim_scope": "Frozen-policy external matchup; no duplicate deals, equilibrium or universal-strength claim."}
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, report)
    start, last_request = clock(), None
    token = None
    hand_index = 0
    stage = "new_hand"

    def request(endpoint, payload):
        nonlocal last_request
        if last_request is not None:
            delay = min_interval_seconds - (clock() - last_request)
            if delay > 0:
                sleep(delay)
        last_request = clock()
        report["request_attempts"] += 1
        # Payload/token are never retained in the report or diagnostic text.
        return transport(endpoint, payload)

    try:
        for hand_index in range(num_hands):
            stage = "new_hand"
            agent.reset(_seed(seed, hand_index, 1))
            response = request(ENDPOINTS["new_hand"], {"token": token} if token else {})
            previous, sent, sent_street = None, None, None
            action_count = 0
            while True:
                stage = "validate_response"
                state = _validate_response(response, previous, sent, sent_street)
                token = response.get("token", token)
                if not token:
                    raise ProtocolError("server did not establish a session token")
                if response.get("winnings") is not None:
                    report["raw_hands"].append({"hand": hand_index, "client_pos": response["client_pos"],
                                                "hole_cards": response["hole_cards"], "board": response["board"],
                                                "action": response["action"], "winnings_chips": response["winnings"],
                                                "policy_seed": _seed(seed, hand_index, 1), "client_actions": action_count})
                    report["statistics"] = _statistics(report["raw_hands"])
                    report["elapsed_seconds"] = clock() - start
                    _write(output, report)
                    break
                if action_count >= max_actions_per_hand:
                    raise ProtocolError("client action limit reached")
                stage = "policy_action"
                validate_probabilities(agent.probabilities(state.observe()), state.observe().legal_mask)
                payload = choose_action(response, agent)
                payload["token"] = token  # Carry forward a token omitted from a later response.
                previous, sent, sent_street = response, payload["incr"], state.street
                stage = "act"
                response = request(ENDPOINTS["act"], payload)
                action_count += 1
        report["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "incomplete"
        report["incomplete_hand"] = {"hand": hand_index, "stage": stage,
                                     "error_type": type(error).__name__,
                                     "reason": str(error) if isinstance(error, ProtocolError) else
                                               "request, validation or policy failure; detail suppressed to protect session data",
                                     "retry_attempted": False,
                                     "server_outcome_may_be_unknown": stage in ("new_hand", "act")}
    report["elapsed_seconds"] = clock() - start
    _write(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--hands", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=83001)
    parser.add_argument("--request-interval", type=float, default=.25)
    parser.add_argument("--request-timeout", type=float, default=20)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    report = run_benchmark(args.checkpoint, expected_sha256=args.sha256, num_hands=args.hands,
                           output=args.output, seed=args.seed, min_interval_seconds=args.request_interval,
                           request_timeout_seconds=args.request_timeout)
    print(json.dumps({"output": args.output, "status": report["status"],
                      "statistics": report["statistics"], "incomplete_hand": report["incomplete_hand"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
