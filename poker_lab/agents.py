"""Deployed policies receive observations, not hidden game state."""
from __future__ import annotations

import copy

import numpy as np
import torch

from .game import Observation
from .networks import PokerNetwork, batch_observations, masked_probs


def cpu_state(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def make_network(state: dict | None, device: str = "cpu") -> PokerNetwork | None:
    if state is None:
        return None
    # Loading a policy must not change the training RNG.
    with torch.random.fork_rng():
        net = PokerNetwork().to(device)
    net.load_state_dict(state)
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return net


def network_probs(net, observation, device="cpu", regret=False):
    mask = np.asarray(observation.legal_mask, dtype=np.float64)
    if mask.sum() <= 0:
        raise ValueError("A policy requires at least one legal action")
    if net is None:
        return mask / mask.sum()
    with torch.no_grad():
        batch = batch_observations([observation], device=device)
        logits, _ = net(batch)
        probabilities = masked_probs(logits, batch["mask"], regret=regret)[0]
    result = probabilities.detach().cpu().numpy().astype(np.float64)
    result *= mask
    return result / result.sum()


class RandomAgent:
    name = "uniform random"

    def __init__(self, seed=0):
        self.reset(seed)

    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def probabilities(self, observation: Observation):
        return network_probs(None, observation)

    def act(self, observation: Observation):
        return int(self.rng.choice(5, p=self.probabilities(observation)))

    def clone(self):
        other = copy.copy(self)
        other.rng = np.random.default_rng()
        other.rng.bit_generator.state = copy.deepcopy(self.rng.bit_generator.state)
        return other


class HeuristicAgent(RandomAgent):
    """Simple rank, pair, and draw rules for local play."""
    name = "card-strength heuristic"

    def __init__(self, style="equity", seed=0):
        if style not in {"equity", "tight", "loose", "aggressive", "passive"}:
            raise ValueError(f"Unknown heuristic style: {style}")
        self.style = style
        super().__init__(seed)

    def probabilities(self, observation):
        cards = [int(c) for c in observation.cards if int(c) < 52]
        hole, board = cards[:2], cards[2:]
        ranks = [c // 4 for c in hole]
        score = (sum(ranks) / 24.0 if ranks else 0.5)
        if len(ranks) == 2 and ranks[0] == ranks[1]:
            score = 0.7 + 0.25 * ranks[0] / 12
        if board:
            all_ranks = [c // 4 for c in cards]
            matches = max((all_ranks.count(r) for r in ranks), default=1)
            score = max(score * 0.7, {1: 0.2, 2: 0.65, 3: 0.9, 4: 0.99}[min(matches, 4)])
            if max((sum(c % 4 == suit for c in cards) for suit in range(4)), default=0) >= 4:
                score = max(score, 0.55)
        mask = np.asarray(observation.legal_mask, dtype=np.float64)
        threshold = {"tight": .72, "loose": .48, "aggressive": .48, "passive": .9}.get(self.style, .62)
        weights = np.array([max(.03, .5 - score), .45, .03, .01, .005])
        if score > threshold:
            weights[2:] = [.35, .2, .015 if score < .9 else .1]
        if self.style == "passive":
            weights[2:] *= .1
        weights *= mask
        return weights / weights.sum()


class PPOAgent(RandomAgent):
    name = "PPO self-play"

    def __init__(self, model_state, device="cpu", seed=0):
        self.device = device
        self.model = make_network(model_state, device)
        super().__init__(seed)

    def probabilities(self, observation):
        return network_probs(self.model, observation, self.device)



def load_agent(path: str, device="cpu"):
    from .checkpoint import load_checkpoint

    payload = load_checkpoint(path)
    if payload["algorithm"] != "ppo":
        raise ValueError(f"Unsupported policy: {payload['algorithm']}")
    return PPOAgent(payload["model_state"], device=device)
