
import numpy as np
import pytest

from poker_lab.game import GameSpec, Observation
from poker_lab.parallel_rollout import RolloutPool
from poker_lab.ppo import generalized_advantages


OPPONENTS = [{"policy_id": "frozen", "kind": "neural", "weight": 1.}]


def inference(requests):
    results = []
    for request in requests:
        assert set(request) == {"policy_id", "observation"}
        observation = request["observation"]
        assert isinstance(observation, Observation)
        assert len(observation.cards) == 7
        board = [card for card in observation.cards[2:] if card < 52]
        assert len(board) in (0, 3, 4, 5)
        assert observation.cards[2:] == tuple(board) + (52,) * (5 - len(board))
        probabilities = np.array(observation.legal_mask, dtype=float) * [.6, 2., 1., .5, .1]
        probabilities /= probabilities.sum()
        results.append({"probabilities": probabilities, "value": observation.scalars[3]})
    return results


def calling_inference(requests):
    return [{"probabilities": np.array([0., 1., 0., 0., 0.]), "value": .25} for _ in requests]


def comparable(result):
    return {key: result[key] for key in ("samples", "counters", "hands", "per_opponent")}


def test_complete_hands_rewards_masks_and_gae():
    with RolloutPool(GameSpec(), 821, envs_per_worker=8, opponent_spec=OPPONENTS,
                     reward_scale=.005) as pool:
        result = pool.collect(inference, 128)
        state = pool.state_dict()
    assert len(result["samples"]) >= 128
    assert result["counters"]["steps"] == len(result["samples"])
    assert result["counters"]["hands"] == len(result["hands"])
    assert sum(result["counters"]["street_samples"]) == len(result["samples"])
    assert abs(np.diff(result["counters"]["learner_seats"])[0]) <= 1
    assert state["workers"][0]["hands_started"] == len(result["hands"])
    assert set(state["workers"][0]) == {"rng", "hands_started"}
    assert result["timing"]["mean_inference_batch"] > 1
    for hand in result["hands"]:
        samples = result["samples"][hand["sample_start"]:hand["sample_start"] + hand["steps"]]
        if not samples:
            continue
        rewards = np.zeros(len(samples), dtype=np.float32)
        rewards[-1] = hand["payoff_bb"] * .005
        advantages, returns = generalized_advantages(rewards, [sample[5] for sample in samples])
        np.testing.assert_array_equal(advantages, [sample[3] for sample in samples])
        np.testing.assert_array_equal(returns, [sample[4] for sample in samples])
        for observation, action, log_prob, advantage, return_, value in samples:
            assert observation.player == hand["seat"]
            assert observation.legal_mask[action]
            expected = inference([{"policy_id": "learner", "observation": observation}])[0]
            assert log_prob == pytest.approx(np.log(expected["probabilities"][action]))
            assert np.isfinite([log_prob, advantage, return_, value]).all()


@pytest.mark.parametrize("workers", [0, 2])
def test_rollout_boundary_resume_is_exact(workers):
    config = dict(spec=GameSpec(), seed=44, workers=workers, envs_per_worker=3,
                  opponent_spec=OPPONENTS)
    with RolloutPool(**config) as pool:
        pool.collect(inference, 32)
        state = pool.state_dict()
        expected = comparable(pool.collect(inference, 64))
        final_state = pool.state_dict()
    with RolloutPool(**config, state=state) as restored:
        actual = comparable(restored.collect(inference, 64))
        assert restored.state_dict() == final_state
    assert actual == expected


def test_deadline_drains_every_started_hand(monkeypatch):
    now = [0.]
    monkeypatch.setattr("poker_lab.parallel_rollout.time.monotonic", lambda: now[0])
    batches = []

    def expire(requests):
        batches.append(requests)
        now[0] = 10.
        return calling_inference(requests)

    with RolloutPool(GameSpec(), 3, envs_per_worker=4, opponent_spec=OPPONENTS) as pool:
        result = pool.collect(expire, 10000, deadline=5.)
        state = pool.state_dict()
    assert result["deadline_hit"]
    assert len(batches) >= 4
    assert len(result["hands"]) == 4
    assert all(hand["nodes"] == 8 for hand in result["hands"])
    assert len(result["samples"]) == 16
    assert state["workers"][0]["hands_started"] == 4


@pytest.mark.parametrize("workers", [0, 1])
def test_bad_inference_fails_and_closes_workers(workers):
    pool = RolloutPool(GameSpec(), 33, workers=workers, envs_per_worker=2, opponent_spec=OPPONENTS)

    def invalid(requests):
        return [{"probabilities": np.ones(5) / 5, "value": np.nan} for _ in requests]

    with pytest.raises(ValueError):
        pool.collect(invalid, 10)
    assert pool._closed
    assert all(not process.is_alive() for process in pool._processes)
    with pytest.raises(RuntimeError, match="failed rollout"):
        pool.state_dict()


@pytest.mark.parametrize("workers", [0, 2])
def test_opponent_refresh_preserves_rng_counters_and_resumes_exactly(workers):
    config = dict(spec=GameSpec(), seed=421, workers=workers, envs_per_worker=3)
    refreshed = [{"policy_id": "latest-7", "kind": "neural", "weight": .5},
                 {"policy_id": "archive-2", "kind": "neural", "weight": .35},
                 {"policy_id": "frozen", "kind": "neural", "weight": .15}]
    with RolloutPool(**config, opponent_spec=OPPONENTS) as pool:
        pool.collect(inference, 32)
        before = pool.state_dict()
        pool.set_opponents(refreshed)
        state = pool.state_dict()
        for key in ("workers", "rollouts", "counters"):
            assert state[key] == before[key]
        assert state["configuration"]["opponents"] == refreshed
        expected = comparable(pool.collect(inference, 192))
        assert set(expected["per_opponent"]) == {entry["policy_id"] for entry in refreshed}
        expected_state = pool.state_dict()
    with RolloutPool(**config, opponent_spec=refreshed, state=state) as restored:
        assert comparable(restored.collect(inference, 192)) == expected
        assert restored.state_dict() == expected_state
    with pytest.raises(ValueError, match="settings differ"):
        RolloutPool(**config, opponent_spec=OPPONENTS, state=state)


@pytest.mark.parametrize("workers", [0, 1])
def test_refresh_during_inference_is_rejected_without_changing_active_hands(workers):
    refreshed = [{"policy_id": "new", "kind": "neural", "weight": 1.}]
    with RolloutPool(GameSpec(), 997, workers=workers, envs_per_worker=2, opponent_spec=OPPONENTS) as pool:
        rejected = []

        def callback(requests):
            with pytest.raises(RuntimeError, match="active"):
                pool.set_opponents(refreshed)
            rejected.append(True)
            return inference(requests)

        result = pool.collect(callback, 32)
        assert rejected
        assert set(result["per_opponent"]) == {"frozen"}
        assert pool.state_dict()["configuration"]["opponents"] == OPPONENTS
