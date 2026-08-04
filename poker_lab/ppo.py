"""PPO losses and advantage estimates used by the batched self-play learner."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


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


def critic_loss(values, returns, old_values=None, kind="mse", huber_delta=1., clip_epsilon=None):
    """Optional pessimistic value clipping, in the configured reward units."""
    def errors(predictions):
        if kind == "mse":
            return (predictions - returns).square()
        if kind == "huber":
            return F.huber_loss(predictions, returns, reduction="none", delta=huber_delta)
        raise ValueError("value_loss must be 'mse' or 'huber'")

    losses = errors(values)
    if clip_epsilon is not None:
        if old_values is None:
            raise ValueError("Value clipping requires old rollout value predictions")
        clipped = old_values + (values - old_values).clamp(-clip_epsilon, clip_epsilon)
        losses = torch.maximum(losses, errors(clipped))
    return losses.mean()


def balanced_batch_end(sample_count, batch_size, offset):
    """Partition an epoch into batches differing by at most one sample."""
    for name, value in (("sample_count", sample_count), ("batch_size", batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(offset) is not int or not 0 <= offset < sample_count:
        raise ValueError("Invalid balanced minibatch offset")
    count = (sample_count + batch_size - 1) // batch_size
    size, extra = divmod(sample_count, count)
    large_end = extra * (size + 1)
    if offset < large_end:
        width, origin = size + 1, 0
    else:
        width, origin = size, large_end
    if (offset - origin) % width:
        raise ValueError("Saved offset is not a balanced minibatch boundary")
    return offset + width
