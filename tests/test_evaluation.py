import numpy as np
import pytest

from poker_lab.evaluation import paired_statistics


def test_paired_units_and_uncertainty():
    stats = paired_statistics([100, 200, 300], 100)
    assert stats["num_hands"] == 6
    assert stats["bb_per_100"] == 200
    assert stats["standard_error_bb_per_100"] == pytest.approx(100 / np.sqrt(3))
    lo, hi = stats["ci95_bb_per_100"]
    assert lo < 100 < 300 < hi


def test_empty_and_single_pair_have_no_invented_confidence():
    assert paired_statistics([], 100)["bb_per_100"] is None
    assert paired_statistics([25], 100)["ci95_bb_per_100"] is None
    assert paired_statistics([0, 0, 0], 100)["ci95_bb_per_100"] == [0, 0]
    with pytest.raises(ValueError):
        paired_statistics([float("nan")], 100)


class CheckCallAgent:
    def reset(self, seed):
        pass

    def clone(self):
        return CheckCallAgent()

    def probabilities(self, observation):
        return np.array([0, 1, 0, 0, 0], dtype=float)

    def act(self, observation):
        return 1


def test_duplicate_deals_cancel_identical_deterministic_agents():
    from poker_lab.evaluation import evaluate_matchup

    result = evaluate_matchup(CheckCallAgent(), CheckCallAgent(), num_pairs=10, seed=91)
    assert result["num_hands"] == 20
    assert result["bb_per_100"] == 0
    assert all(row["pair_mean_chips"] == 0 for row in result["raw_pairs"])
    repeated = evaluate_matchup(CheckCallAgent(), CheckCallAgent(), num_pairs=10, seed=91)
    assert repeated["raw_pairs"] == result["raw_pairs"]


def test_zero_time_budget_does_not_score_incomplete_pairs():
    from poker_lab.evaluation import evaluate_matchup

    result = evaluate_matchup(CheckCallAgent(), CheckCallAgent(), num_pairs=10, max_seconds=0)
    assert result["raw_pairs"] == []
    assert not result["complete"]
