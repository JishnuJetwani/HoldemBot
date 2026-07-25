"""Clipped PPO against current and historical policy snapshots."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from .agents import PPOAgent, cpu_state
from .game import GameSpec, HoldemState
from .networks import PokerNetwork, batch_observations, masked_probs


DEFAULT_CONFIG = {
    "device": "cpu", "rollout_steps": 128, "batch_size": 64, "epochs": 4,
    "learning_rate": .0003, "gamma": 1., "gae_lambda": .95,
    "clip_epsilon": .2, "entropy_coefficient": .01, "value_coefficient": .5,
    "gradient_clip": .5, "pool_capacity": 8, "historical_probability": .5,
}


def generalized_advantages(rewards, values, gamma=1., gae_lambda=.95, bootstrap=0.):
    """GAE over consecutive decisions of one player, terminal bootstrap by default."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    advantage, next_value = 0., float(bootstrap)
    for index in reversed(range(len(rewards))):
        delta = float(rewards[index]) + gamma * next_value - float(values[index])
        advantage = delta + gamma * gae_lambda * advantage
        advantages[index] = advantage
        next_value = float(values[index])
    return advantages, advantages + values


def clipped_policy_loss(new_log_probs, old_log_probs, advantages, epsilon=.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    return -torch.minimum(ratio * advantages, ratio.clamp(1 - epsilon, 1 + epsilon) * advantages).mean()


class PPOTrainer:
    def __init__(self, seed, config=None, payload=None):
        self.config = dict(DEFAULT_CONFIG, **(config or {}))
        for key in ("rollout_steps", "batch_size", "epochs", "pool_capacity"):
            if int(self.config[key]) < 1:
                raise ValueError(f"{key} must be positive")
        for key in ("gamma", "gae_lambda", "historical_probability"):
            if not 0 <= self.config[key] <= 1:
                raise ValueError(f"{key} must lie in [0, 1]")
        self.spec = GameSpec(**payload["game_spec"]) if payload else GameSpec()
        self.rng = np.random.default_rng(seed)
        self.model = PokerNetwork().to(self.config["device"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        self.pool = [cpu_state(self.model)]
        self.rollout_model_state = cpu_state(self.model)
        self.samples = []
        self.active = None
        self.game = None
        self.opponent = None
        self.counters = {"iterations": 0, "steps": 0, "nodes": 0, "hands": 0,
                         "gradient_steps": 0, "historical_opponent_hands": 0,
                         "street_samples": [0, 0, 0, 0]}
        self.progress = {"phase": "collect", "epoch": 0, "offset": 0, "order": None}
        self.losses = []
        if payload:
            self._restore(payload)

    def _restore(self, payload):
        self.model.load_state_dict(payload["model_state"])
        state = payload["trainer_state"]
        self.optimizer.load_state_dict(state["optimizer"])
        self.pool, self.samples = state["pool"], state["samples"]
        self.rollout_model_state = state["rollout_model_state"]
        self.active, self.progress, self.losses = state["active"], state["progress"], state["losses"]
        self.counters = payload["counters"]
        self.rng.bit_generator.state = state["rng"]
        if self.active is not None:
            self.game = HoldemState.from_dict(state["game"])
            self.opponent = PPOAgent(self.active["opponent_state"], self.config["device"])
            self.opponent.rng.bit_generator.state = self.active["opponent_rng"]

    def _new_hand(self):
        historical = self.rng.random() < self.config["historical_probability"] and bool(self.pool)
        opponent_state = self.pool[int(self.rng.integers(len(self.pool)))] if historical else self.rollout_model_state
        self.game = HoldemState.new(int(self.rng.integers(2**32, 2**63)), self.spec)
        self.opponent = PPOAgent(opponent_state, self.config["device"], int(self.rng.integers(2**63)))
        self.active = {"seat": self.counters["hands"] % 2, "observations": [], "actions": [],
                       "values": [], "log_probs": [], "opponent_state": opponent_state,
                       "opponent_rng": copy.deepcopy(self.opponent.rng.bit_generator.state)}
        self.counters["historical_opponent_hands"] += int(historical)

    def _collect_step(self):
        if self.active is None:
            self._new_hand()
        if self.game.is_terminal():
            self._finish_hand()
            return
        seat = self.game.current_player()
        obs = self.game.observe(seat)
        self.counters["nodes"] += 1
        if seat != self.active["seat"]:
            action = self.opponent.act(obs)
        else:
            self.model.eval()
            with torch.no_grad():
                batch = batch_observations([obs], self.config["device"])
                logits, values = self.model(batch)
                probabilities = masked_probs(logits, batch["mask"])[0].cpu().numpy().astype(np.float64)
                probabilities /= probabilities.sum()
            action = int(self.rng.choice(5, p=probabilities))
            self.active["observations"].append(obs)
            self.active["actions"].append(action)
            self.active["values"].append(float(values[0].cpu()))
            self.active["log_probs"].append(float(np.log(probabilities[action])))
            self.counters["steps"] += 1
            board_count = sum(c < 52 for c in obs.cards[2:])
            self.counters["street_samples"][{0: 0, 3: 1, 4: 2, 5: 3}.get(board_count, 0)] += 1
        self.game.apply_slot(action)
        if self.game.is_terminal():
            self._finish_hand()

    def _finish_hand(self):
        length = len(self.active["actions"])
        if length:
            rewards = np.zeros(length, dtype=np.float32)
            rewards[-1] = self.game.returns()[self.active["seat"]] / self.spec.big_blind
            advantages, returns = generalized_advantages(rewards, self.active["values"], self.config["gamma"], self.config["gae_lambda"])
            self.samples.extend(zip(self.active["observations"], self.active["actions"], self.active["log_probs"],
                                    advantages.tolist(), returns.tolist()))
        self.counters["hands"] += 1
        self.active, self.game, self.opponent = None, None, None
        if len(self.samples) >= self.config["rollout_steps"]:
            self.progress.update(phase="fit", epoch=0, offset=0, order=self.rng.permutation(len(self.samples)).tolist())

    def _fit_batch(self):
        indices = self.progress["order"][self.progress["offset"]:self.progress["offset"] + self.config["batch_size"]]
        samples = [self.samples[i] for i in indices]
        device = self.config["device"]
        batch = batch_observations([s[0] for s in samples], device)
        actions = torch.as_tensor([s[1] for s in samples], dtype=torch.long, device=device)
        old_log_probs = torch.as_tensor([s[2] for s in samples], dtype=torch.float32, device=device)
        all_advantages = np.asarray([s[3] for s in self.samples], dtype=np.float32)
        normalized = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
        advantages = torch.as_tensor(normalized[indices], device=device)
        returns = torch.as_tensor([s[4] for s in samples], dtype=torch.float32, device=device)
        self.model.train()
        logits, values = self.model(batch)
        probabilities = masked_probs(logits, batch["mask"])
        logs = probabilities.clamp_min(1e-12).log()
        log_probs = logs.gather(-1, actions[:, None]).squeeze(-1)
        policy_loss = clipped_policy_loss(log_probs, old_log_probs, advantages, self.config["clip_epsilon"])
        value_loss = (values - returns).square().mean()
        entropy = -(probabilities * logs).sum(-1).mean()
        loss = policy_loss + self.config["value_coefficient"] * value_loss - self.config["entropy_coefficient"] * entropy
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite PPO loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip"])
        self.optimizer.step()
        self.counters["gradient_steps"] += 1
        self.losses = (self.losses + [{"total": float(loss.detach().cpu()), "policy": float(policy_loss.detach().cpu()),
                                     "value": float(value_loss.detach().cpu()), "entropy": float(entropy.detach().cpu())}])[-100:]
        self.progress["offset"] += len(indices)
        if self.progress["offset"] >= len(self.samples):
            self.progress["epoch"] += 1
            self.progress["offset"] = 0
            if self.progress["epoch"] >= self.config["epochs"]:
                self.counters["iterations"] += 1
                self.rollout_model_state = cpu_state(self.model)
                self.pool.append(self.rollout_model_state)
                self.pool = self.pool[-self.config["pool_capacity"]:]
                self.samples = []
                self.progress.update(phase="collect", epoch=0, offset=0, order=None)
            else:
                self.progress["order"] = self.rng.permutation(len(self.samples)).tolist()

    def run(self, deadline, progress_callback=None):
        while time.monotonic() < deadline:
            if progress_callback is not None:
                progress_callback()
            if self.progress["phase"] == "collect":
                self._collect_step()
            else:
                self._fit_batch()
        return "time_budget"

    def payload(self):
        active = copy.deepcopy(self.active)
        if active is not None:
            active["opponent_rng"] = copy.deepcopy(self.opponent.rng.bit_generator.state)
        return {"algorithm": "ppo", "game_spec": self.spec.to_dict(), "config": self.config,
                "counters": self.counters, "model_state": cpu_state(self.model),
                "trainer_state": {"optimizer": self.optimizer.state_dict(), "pool": self.pool,
                                  "rollout_model_state": self.rollout_model_state,
                                  "samples": self.samples, "active": active,
                                  "game": self.game.to_dict() if self.game else None,
                                  "progress": self.progress, "losses": self.losses,
                                  "rng": self.rng.bit_generator.state}}
