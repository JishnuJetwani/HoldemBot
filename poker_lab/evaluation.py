"""Duplicate-deal evaluation with paired confidence intervals."""
from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Callable

import numpy as np
from scipy.stats import t


def paired_statistics(paired_chips: list[float], big_blind: float) -> dict[str, Any]:
    """Compute confidence intervals from the mean return of each deal pair."""
    if big_blind <= 0 or not math.isfinite(big_blind):
        raise ValueError("big_blind must be finite and positive")
    values = np.asarray(paired_chips, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("paired returns must be a finite one-dimensional sequence")
    n = len(values)
    scale = 100.0 / big_blind
    mean = float(values.mean() * scale) if n else None
    se = float(values.std(ddof=1) / np.sqrt(n) * scale) if n >= 2 else None
    margin = float(t.ppf(0.975, n - 1) * se) if n >= 2 else None
    return {
        "num_pairs": n,
        "num_hands": 2 * n,
        "bb_per_100": mean,
        "standard_error_bb_per_100": se,
        "ci95_bb_per_100": [mean - margin, mean + margin] if margin is not None else None,
        "ci_method": "Student t interval over independent duplicate-deal pair means",
    }


def _seed(base: int, pair: int, stream: int) -> int:
    digest = hashlib.sha256(f"{base}:{pair}:{stream}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def validate_probabilities(probabilities: Any, legal_mask: Any) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    mask = np.asarray(legal_mask, dtype=bool)
    if probabilities.shape != (5,) or mask.shape != (5,) or not mask.any():
        raise ValueError("expected a five-slot probability vector and nonempty legal mask")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("policy probabilities must be finite and nonnegative")
    if np.any(probabilities[~mask] != 0) or not np.isclose(probabilities.sum(), 1, atol=1e-6):
        raise ValueError("policy probabilities must normalize over legal actions only")
    return probabilities


def play_hand(agent0: Any, agent1: Any, *, seed: int, deck: list[int] | None = None,
              spec: Any = None, policy_seeds: tuple[int, int] | None = None,
              max_actions: int = 10000) -> dict[str, Any]:
    from poker_lab.game import HoldemState

    agents = (agent0, agent1)
    policy_seeds = policy_seeds or (_seed(seed, 0, 1), _seed(seed, 0, 2))
    for agent, agent_seed in zip(agents, policy_seeds):
        agent.reset(agent_seed)
    state = HoldemState.new(seed=seed, spec=spec, deck=deck)
    decisions = 0
    action_seconds = [0.0, 0.0]
    policy_checks_seconds = [0.0, 0.0]
    seat_decisions = [0, 0]
    while not state.is_terminal():
        if decisions >= max_actions:
            raise RuntimeError("hand exceeded action limit; no score recorded")
        actor = state.current_player()
        observation = state.observe(actor)
        # Check the policy used to choose the action.
        check_start = time.perf_counter()
        validate_probabilities(agents[actor].probabilities(observation), observation.legal_mask)
        policy_checks_seconds[actor] += time.perf_counter() - check_start
        action_start = time.perf_counter()
        slot = agents[actor].act(observation)
        action_seconds[actor] += time.perf_counter() - action_start
        seat_decisions[actor] += 1
        if not isinstance(slot, (int, np.integer)) or not 0 <= slot < 5 or not observation.legal_mask[slot]:
            raise ValueError("agent selected an illegal action; no score recorded")
        state.apply_slot(int(slot))
        decisions += 1
    returns = [float(x) for x in state.returns()]
    if len(returns) != 2 or not np.all(np.isfinite(returns)) or not np.isclose(sum(returns), 0):
        raise ValueError("invalid zero-sum terminal returns")
    return {"returns": returns, "decisions": decisions, "seat_decisions": seat_decisions,
            "action_seconds": action_seconds, "policy_checks_seconds": policy_checks_seconds}


def evaluate_matchup(agent_a: Any, agent_b: Any, *, num_pairs: int = 1000,
                     seed: int = 20260918, spec: Any = None,
                     max_seconds: float | None = None,
                     name_a: str = "candidate", name_b: str = "opponent",
                     previous_result: dict | None = None,
                     progress_callback: Callable[[dict], None] | None = None) -> dict[str, Any]:
    """Evaluate frozen policies with seat-swapped deals.

    Time limits apply between complete pairs. For fixed-count confidence
    intervals, leave max_seconds unset.
    """
    from poker_lab.game import GameSpec

    if num_pairs < 0 or (max_seconds is not None and max_seconds < 0):
        raise ValueError("counts and duration must be nonnegative")
    spec = spec or GameSpec()
    a, b = agent_a.clone(), agent_b.clone()
    start = time.monotonic()
    rows = []
    costs = {name: {"decisions": 0, "action_seconds": 0.0, "policy_checks_seconds": 0.0}
             for name in ("agent_a", "agent_b")}
    previous_elapsed = 0.0
    if previous_result is not None:
        import copy
        expected = {"agent_a": name_a, "agent_b": name_b, "game_spec": spec.to_dict(),
                    "seed": seed, "requested_pairs": num_pairs}
        if any(previous_result.get(key) != value for key, value in expected.items()):
            raise ValueError("resume requires the same matchup, game, seed and requested pair count")
        rows = copy.deepcopy(previous_result["raw_pairs"])
        if len(rows) > num_pairs or any(row["pair"] != index or row["deal_seed"] != _seed(seed, index, 0)
                                         for index, row in enumerate(rows)):
            raise ValueError("resume rows must be a contiguous prefix of the specified deal stream")
        costs = copy.deepcopy(previous_result["inference_costs"])
        previous_elapsed = float(previous_result["elapsed_seconds"])

    def snapshot() -> dict:
        result = paired_statistics([row["pair_mean_chips"] for row in rows], spec.big_blind)
        result.update({"schema_version": 1, "evaluation": "duplicate_deals", "agent_a": name_a,
                       "agent_b": name_b, "game_spec": spec.to_dict(), "seed": seed,
                       "requested_pairs": num_pairs, "max_seconds": max_seconds,
                       "elapsed_seconds": previous_elapsed + time.monotonic() - start,
                       "complete": len(rows) == num_pairs, "raw_pairs": list(rows),
                       "inference_costs": {name: dict(values) for name, values in costs.items()},
                       "scope": "One frozen-policy matchup; not exact exploitability or an algorithm-level result.",
                       "limitations": ["Confidence interval covers deal/action randomness, not training-seed variation.",
                                       "Time-limited intervals are descriptive; use fixed pair counts for inference."]})
        return result

    for pair in range(len(rows), num_pairs):
        if max_seconds is not None and time.monotonic() - start >= max_seconds:
            break
        deal_seed = _seed(seed, pair, 0)
        deck = np.random.default_rng(deal_seed).permutation(52).tolist()
        first = play_hand(a, b, seed=deal_seed, deck=deck, spec=spec,
                          policy_seeds=(_seed(seed, pair, 1), _seed(seed, pair, 2)))
        second = play_hand(b, a, seed=deal_seed, deck=deck, spec=spec,
                           policy_seeds=(_seed(seed, pair, 3), _seed(seed, pair, 4)))
        as_bb, as_sb = first["returns"][0], second["returns"][1]
        for name, assignments in (("agent_a", ((first, 0), (second, 1))),
                                  ("agent_b", ((first, 1), (second, 0)))):
            for hand, seat in assignments:
                costs[name]["decisions"] += hand["seat_decisions"][seat]
                for key in ("action_seconds", "policy_checks_seconds"):
                    costs[name][key] += hand[key][seat]
        rows.append({"pair": pair, "deal_seed": deal_seed,
                     "a_as_bb_chips": as_bb, "a_as_sb_chips": as_sb,
                     "pair_mean_chips": (as_bb + as_sb) / 2,
                     "decisions": first["decisions"] + second["decisions"]})
        if progress_callback is not None and (len(rows) % 32 == 0 or len(rows) == num_pairs):
            progress_callback(snapshot())
    return snapshot()
