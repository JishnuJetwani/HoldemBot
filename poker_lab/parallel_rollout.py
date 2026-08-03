"""Complete-hand CPU rollouts with batched inference in the parent process."""
from __future__ import annotations

import copy
import multiprocessing as mp
import time
import traceback

import numpy as np

from .game import GameSpec, HoldemState


DEFAULT_OPPONENTS = ({"policy_id": "selfplay-latest", "kind": "neural", "weight": 1.},)


def _counters():
    return {"hands": 0, "steps": 0, "nodes": 0, "street_samples": [0] * 4,
            "learner_action_counts": [0] * 5,
            "learner_action_counts_by_street": [[0] * 5 for _ in range(4)],
            "learner_seats": [0, 0]}


def _add_counters(target, source):
    for key, value in source.items():
        if isinstance(value, list):
            target[key] = (np.asarray(target[key]) + np.asarray(value)).tolist()
        else:
            target[key] += value


def _probabilities(result, observation):
    probabilities = np.asarray(result["probabilities"], dtype=np.float64)
    legal = np.asarray(observation.legal_mask, dtype=bool)
    if (probabilities.shape != (5,) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities[~legal] != 0).any()
            or not np.isclose(probabilities.sum(), 1., rtol=1e-6, atol=1e-7)):
        raise ValueError("Inference returned illegal or unnormalized probabilities")
    value = float(result.get("value", 0.))
    if not np.isfinite(value):
        raise ValueError("Inference returned a non-finite value")
    return probabilities / probabilities.sum(), value


def _opponents(entries):
    entries = copy.deepcopy(list(entries))
    if not entries:
        raise ValueError("At least one fixed opponent is required")
    ids = set()
    for entry in entries:
        policy = entry.get("policy_id")
        if not isinstance(policy, str) or not policy or policy == "learner" or policy in ids:
            raise ValueError("Opponent policy IDs must be unique and cannot be learner")
        ids.add(policy)
        if entry.get("kind") != "neural":
            raise ValueError("Opponent kind must be neural")
        weight = entry.get("weight", 1.)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("Opponent weights must be finite and positive")
        entry["weight"] = float(weight)
    return entries


class _Worker:
    def __init__(self, spec, worker_id, env_count, opponents, state):
        self.spec, self.worker_id = spec, worker_id
        self.envs = [None] * env_count
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.hands_started = state["hands_started"]
        self.pending = []
        self.set_opponents(opponents)

    def set_opponents(self, entries):
        state = self.state_dict()
        opponents = _opponents(entries)
        weights = np.asarray([entry["weight"] for entry in opponents], dtype=float)
        self.opponents, self.weights = opponents, weights / weights.sum()
        return state

    def state_dict(self):
        if any(env is not None for env in self.envs) or self.pending:
            raise RuntimeError("Worker state is available only between complete rollouts")
        return {"rng": copy.deepcopy(self.rng.bit_generator.state),
                "hands_started": self.hands_started}

    def _new_hand(self):
        opponent = self.opponents[int(self.rng.choice(len(self.opponents), p=self.weights))]
        game = HoldemState.new(int(self.rng.integers(2**32, 2**63)), self.spec)
        number = self.hands_started
        self.hands_started += 1
        return {"game": game, "seat": (number + self.worker_id) % 2,
                "opponent": opponent, "number": number, "observations": [],
                "actions": [], "log_probs": [], "values": [], "nodes": 0}

    def _apply(self, env, observation, result, counters):
        probabilities, value = _probabilities(result, observation)
        action = int(self.rng.choice(5, p=probabilities))
        if observation.player == env["seat"]:
            env["observations"].append(observation)
            env["actions"].append(action)
            env["log_probs"].append(float(np.log(probabilities[action])))
            env["values"].append(value)
            street = env["game"].street
            counters["steps"] += 1
            counters["street_samples"][street] += 1
            counters["learner_action_counts"][action] += 1
            counters["learner_action_counts_by_street"][street][action] += 1
        env["game"].apply_slot(action)
        counters["nodes"] += 1
        env["nodes"] += 1
        if env["nodes"] > 4096:
            raise RuntimeError("Hand exceeded the rollout action limit")

    def advance(self, responses, start_new):
        if len(responses) != len(self.pending):
            raise ValueError("Inference response count differs from the pending requests")
        counters, completed, requests = _counters(), [], []
        for (index, observation), result in zip(self.pending, responses):
            self._apply(self.envs[index], observation, result, counters)
        self.pending = []
        for index in range(len(self.envs)):
            while True:
                env = self.envs[index]
                if env is not None and env["game"].is_terminal():
                    returns = env["game"].returns()
                    if not np.isfinite(returns).all() or abs(sum(returns)) > 1e-8:
                        raise RuntimeError("Rollout hand violated chip conservation")
                    completed.append({key: env[key] for key in
                                      ("observations", "actions", "log_probs", "values", "nodes", "seat", "number")})
                    completed[-1].update(worker_id=self.worker_id,
                                         opponent_id=env["opponent"]["policy_id"],
                                         payoff_bb=float(returns[env["seat"]] / self.spec.big_blind))
                    counters["hands"] += 1
                    counters["learner_seats"][env["seat"]] += 1
                    self.envs[index] = env = None
                if env is None:
                    if not start_new:
                        break
                    self.envs[index] = env = self._new_hand()
                observation = env["game"].observe()
                learner = observation.player == env["seat"]
                policy = "learner" if learner else env["opponent"]["policy_id"]
                requests.append({"policy_id": policy, "observation": observation})
                self.pending.append((index, observation))
                break
        drained = not any(env is not None for env in self.envs)
        return {"requests": requests, "completed": completed, "counters": counters,
                "drained": drained, "state": self.state_dict() if drained else None}


def _worker_main(connection, spec, worker_id, env_count, opponents, state):
    try:
        worker = _Worker(spec, worker_id, env_count, opponents, state)
        while True:
            command = connection.recv()
            if command["command"] == "close":
                break
            if command["command"] == "set_opponents":
                connection.send({"result": worker.set_opponents(command["opponents"])})
                continue
            if command["command"] != "advance":
                raise ValueError("Unknown rollout worker command")
            connection.send({"result": worker.advance(command["responses"], command["start_new"])})
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send({"error": traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class RolloutPool:
    """Run several games with shared policy inference.

    After ``deadline`` (a ``time.monotonic()`` timestamp), finish active hands
    without starting new ones. Save only between hands. Zero workers runs
    in-process with the same RNG stream as one worker process.
    """

    def __init__(self, spec: GameSpec, seed: int, workers=0, envs_per_worker=16,
                 opponent_spec=None, state=None, gamma=1., gae_lambda=.95,
                 reward_scale=1., worker_timeout=120.):
        if type(workers) is not int or workers < 0:
            raise ValueError("workers must be a nonnegative integer")
        if type(envs_per_worker) is not int or envs_per_worker < 1:
            raise ValueError("envs_per_worker must be a positive integer")
        if not np.isfinite([gamma, gae_lambda, reward_scale, worker_timeout]).all():
            raise ValueError("Rollout settings must be finite")
        if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1 or reward_scale <= 0 or worker_timeout <= 0:
            raise ValueError("Invalid GAE, reward scale, or worker timeout")
        self.spec, self.seed, self.workers = spec, seed, workers
        self.worker_count, self.envs_per_worker = max(1, workers), envs_per_worker
        self.opponents = _opponents(DEFAULT_OPPONENTS if opponent_spec is None else opponent_spec)
        self.gamma, self.gae_lambda, self.reward_scale = gamma, gae_lambda, reward_scale
        self.worker_timeout = worker_timeout
        self.configuration = {"spec": spec.to_dict(), "seed": seed, "worker_count": self.worker_count,
                              "envs_per_worker": envs_per_worker, "opponents": self.opponents,
                              "gamma": gamma, "gae_lambda": gae_lambda, "reward_scale": reward_scale}
        seeds = np.random.SeedSequence(seed).spawn(self.worker_count)
        self._states = [{"rng": np.random.default_rng(child).bit_generator.state, "hands_started": 0}
                        for child in seeds]
        self._rollouts, self._totals = 0, _counters()
        self._connections, self._processes, self._serial = [], [], None
        self._closed, self._collecting, self._failed = False, False, False
        if state is not None:
            self._restore(state)
        try:
            self._start()
        except BaseException:
            self.close()
            raise

    def _start(self):
        if not self.workers:
            self._serial = _Worker(self.spec, 0, self.envs_per_worker, self.opponents, self._states[0])
            return
        context = mp.get_context("spawn")
        for index in range(self.worker_count):
            parent, child = context.Pipe()
            process = context.Process(target=_worker_main, args=(
                child, self.spec, index, self.envs_per_worker, self.opponents, self._states[index]), daemon=True)
            try:
                process.start()
            except BaseException:
                parent.close()
                child.close()
                raise
            child.close()
            self._connections.append(parent)
            self._processes.append(process)

    def _restore(self, state):
        if state.get("version") != 1 or state.get("configuration") != self.configuration:
            raise ValueError("Saved rollout topology, opponents, or settings differ")
        if len(state["workers"]) != self.worker_count:
            raise ValueError("Saved rollout worker count differs")
        self._states = copy.deepcopy(state["workers"])
        self._rollouts, self._totals = state["rollouts"], copy.deepcopy(state["counters"])

    def state_dict(self):
        if self._collecting or self._failed:
            raise RuntimeError("Cannot save an unfinished or failed rollout")
        return copy.deepcopy({"version": 1, "configuration": self.configuration,
                              "workers": self._states, "rollouts": self._rollouts,
                              "counters": self._totals})

    def load_state_dict(self, state):
        if self._collecting or self._closed:
            raise RuntimeError("Cannot restore an active or closed pool")
        self._restore(state)
        self.close()
        self._closed, self._failed = False, False
        self._connections, self._processes = [], []
        try:
            self._start()
        except BaseException:
            self.close()
            raise

    def _advance(self, responses, start_new):
        if self._serial is not None:
            return [self._serial.advance(responses[0], start_new)]
        for connection, result in zip(self._connections, responses):
            connection.send({"command": "advance", "responses": result, "start_new": start_new})
        return self._receive_workers()

    def _receive_workers(self):
        outputs = []
        for index, connection in enumerate(self._connections):
            if not connection.poll(self.worker_timeout):
                raise TimeoutError(f"Rollout worker {index} did not respond within {self.worker_timeout} seconds")
            try:
                output = connection.recv()
            except (EOFError, OSError) as error:
                raise RuntimeError(f"Rollout worker {index} exited before returning its batch") from error
            if "error" in output:
                raise RuntimeError(f"Rollout worker {index} failed:\n{output['error']}")
            outputs.append(output["result"])
        return outputs

    def set_opponents(self, entries):
        """Change opponents and sampling weights between rollouts."""
        if self._closed or self._collecting or self._failed:
            raise RuntimeError("Cannot refresh opponents in a closed, active, or failed pool")
        opponents = _opponents(entries)
        try:
            if self._serial is not None:
                states = [self._serial.set_opponents(opponents)]
            else:
                for connection in self._connections:
                    connection.send({"command": "set_opponents", "opponents": opponents})
                states = self._receive_workers()
            if states != self._states:
                raise RuntimeError("Opponent refresh changed worker RNG or hand counters")
            self.opponents = opponents
            self.configuration["opponents"] = copy.deepcopy(opponents)
        except BaseException:
            self._failed = True
            self.close()
            raise

    def collect(self, infer_callback, target_steps, deadline=None):
        if self._closed or self._collecting:
            raise RuntimeError("Cannot collect from a closed or active pool")
        if type(target_steps) is not int or target_steps < 1:
            raise ValueError("target_steps must be a positive integer")
        if deadline is not None and not np.isfinite(deadline):
            raise ValueError("deadline must be finite")
        self._collecting = True
        started = time.monotonic()
        responses = [[] for _ in range(self.worker_count)]
        counters, completed = _counters(), []
        inference_seconds, worker_seconds, inference_calls, inference_requests = 0., 0., 0, 0
        learner_responses, deadline_hit = 0, False
        try:
            while True:
                deadline_hit |= deadline is not None and time.monotonic() >= deadline
                start_new = counters["steps"] + learner_responses < target_steps and not deadline_hit
                phase_start = time.monotonic()
                outputs = self._advance(responses, start_new)
                worker_seconds += time.monotonic() - phase_start
                requests = []
                for output in outputs:
                    _add_counters(counters, output["counters"])
                    completed.extend(output["completed"])
                    requests.extend(output["requests"])
                if all(output["drained"] for output in outputs):
                    self._states = [output["state"] for output in outputs]
                    break
                if not requests:
                    raise RuntimeError("Active rollout hands produced no inference requests")
                phase_start = time.monotonic()
                predictions = list(infer_callback(requests))
                inference_seconds += time.monotonic() - phase_start
                inference_calls += 1
                inference_requests += len(requests)
                if len(predictions) != len(requests):
                    raise ValueError("Inference callback returned the wrong number of results")
                for request, prediction in zip(requests, predictions):
                    _probabilities(prediction, request["observation"])
                learner_responses = sum(request["policy_id"] == "learner" for request in requests)
                responses, offset = [], 0
                for output in outputs:
                    length = len(output["requests"])
                    responses.append(predictions[offset:offset + length])
                    offset += length
            samples, hands, per_opponent = self._finish(completed)
            if len(samples) != counters["steps"]:
                raise RuntimeError("Complete-hand sample count differs from learner decisions")
            _add_counters(self._totals, counters)
            self._rollouts += 1
            return {"samples": samples, "counters": counters, "hands": hands,
                    "per_opponent": per_opponent, "elapsed_seconds": time.monotonic() - started,
                    "deadline_hit": bool(deadline_hit),
                    "timing": {"inference_seconds": inference_seconds, "worker_seconds": worker_seconds,
                               "inference_calls": inference_calls, "inference_requests": inference_requests,
                               "mean_inference_batch": inference_requests / max(1, inference_calls)}}
        except BaseException:
            self._failed = True
            self.close()
            raise
        finally:
            self._collecting = False

    def _finish(self, completed):
        from .ppo import generalized_advantages
        samples, hands, per_opponent = [], [], {}
        for hand in completed:
            length = len(hand["actions"])
            offset = len(samples)
            if length:
                rewards = np.zeros(length, dtype=np.float32)
                rewards[-1] = hand["payoff_bb"] * self.reward_scale
                advantages, returns = generalized_advantages(
                    rewards, hand["values"], self.gamma, self.gae_lambda)
                samples.extend(zip(hand["observations"], hand["actions"], hand["log_probs"],
                                   advantages.tolist(), returns.tolist(), hand["values"]))
            hands.append({key: hand[key] for key in ("worker_id", "number", "seat", "opponent_id", "payoff_bb", "nodes")})
            hands[-1].update(steps=length, sample_start=offset)
            record = per_opponent.setdefault(hand["opponent_id"], {"hands": 0, "steps": 0, "return_bb_sum": 0.})
            record["hands"] += 1
            record["steps"] += length
            record["return_bb_sum"] += hand["payoff_bb"]
        for record in per_opponent.values():
            record["mean_return_bb"] = record["return_bb_sum"] / record["hands"]
        return samples, hands, per_opponent

    def close(self):
        if self._closed:
            return
        self._closed = True
        for connection in self._connections:
            try:
                connection.send({"command": "close"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=1.)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.)
        for connection in self._connections:
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()
