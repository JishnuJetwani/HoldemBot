"""Batched policy inference and PPO updates."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from .agents import cpu_state, load_agent
from .checkpoint import capture_rng, load_checkpoint, restore_rng
from .game import GameSpec, canonical_cards
from .hybrid_policy import HybridPokerNetwork, architecture_metadata
from .networks import PokerNetwork, batch_observations, masked_probs
from .ppo import balanced_batch_end, clipped_policy_loss, critic_loss
from .paths import resolve_path
from .semantic_policy import SemanticPokerNetwork, semantic_features


DEFAULT_CONFIG = {
    'rollout_steps': 8192, 'batch_size': 256, 'epochs': 4,
    'learning_rate': .0001, 'gamma': 1., 'gae_lambda': .95,
    'clip_epsilon': .2, 'entropy_coefficient': .01, 'value_coefficient': .5,
    'gradient_clip': .5, 'reward_scale': .005, 'target_kl': .01,
}
SELF_PLAY_DEFAULT_CONFIG = {
    'latest_weight': .5, 'history_weight': .35,
    'archive_interval': 10, 'history_capacity': 32,
}
SELF_PLAY_LATEST = 'selfplay-latest'
SELF_PLAY_INITIAL = 'selfplay-history-000000'
_NETWORKS = {
    'hybrid_poker_v1': HybridPokerNetwork,
    'semantic_poker_v1': SemanticPokerNetwork,
    'poker_network_v1': PokerNetwork,
}


def prepare_observations(observations, device='cpu'):
    """Compute card features once; keep sequence lengths on CPU."""
    batch = batch_observations(observations)
    mask = batch['mask']
    if mask.ndim != 2 or mask.shape[1] != 5 or not ((mask == 0) | (mask == 1)).all():
        raise ValueError('Expected binary five-slot legal masks')
    if not (mask.sum(-1) > 0).all():
        raise ValueError('Expected at least one legal action per observation')
    if not torch.isfinite(batch['history']).all():
        raise ValueError('History must contain finite values')
    batch['features'] = semantic_features(batch)
    batch['canonical_cards'] = torch.tensor(
        [canonical_cards(tuple(row)) for row in batch['cards'].tolist()], dtype=torch.long)
    return {key: value if key == 'lengths' else value.to(device)
            for key, value in batch.items()}


def prepared_forward(model, batch):
    packed = pack_padded_sequence(batch['history'], batch['lengths'], batch_first=True,
                                  enforce_sorted=False)
    _, hidden = model.history(packed)
    if isinstance(model, HybridPokerNetwork):
        features = (model.cards(batch['canonical_cards']).flatten(1),
                    model.semantic_projection(batch['features']), hidden[-1], batch['scalars'])
    elif isinstance(model, SemanticPokerNetwork):
        features = (model.semantic_projection(batch['features']), hidden[-1], batch['scalars'])
    elif type(model) is PokerNetwork:
        features = (model.cards(batch['cards']).flatten(1), hidden[-1], batch['scalars'])
    else:
        raise TypeError(f'Unsupported batched network: {type(model).__name__}')
    body = model.body(torch.cat(features, dim=-1))
    return model.policy(body), model.value(body).squeeze(-1)


def _select(batch, cpu_indices, device_indices):
    return {key: value[cpu_indices if key == 'lengths' else device_indices]
            for key, value in batch.items()}


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_to_cpu(item) for item in value)
    return copy.deepcopy(value)


class BatchedPPOLearner:
    algorithm = 'ppo_hybrid'
    implementation = 'batched_ppo_v1'

    def __init__(self, seed, config=None, device='cpu', payload=None, opponent_paths=None):
        self.seed = int(seed)
        self.device = torch.device(device)
        self.config = {**DEFAULT_CONFIG, **(payload['config'] if payload else {}), **(config or {})}
        unknown = self.config.keys() - DEFAULT_CONFIG.keys()
        if unknown:
            raise ValueError(f'Unknown batched PPO settings: {sorted(unknown)}')
        for key in ('rollout_steps', 'batch_size', 'epochs'):
            if type(self.config[key]) is not int or self.config[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key, value in self.config.items():
            if key == 'target_kl' and value is None:
                continue
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        for key in ('learning_rate', 'gradient_clip', 'reward_scale', 'clip_epsilon'):
            if self.config[key] <= 0:
                raise ValueError(f'{key} must be positive')
        for key in ('gamma', 'gae_lambda'):
            if self.config[key] > 1:
                raise ValueError(f'{key} must lie in [0, 1]')
        if self.config['target_kl'] is not None and self.config['target_kl'] <= 0:
            raise ValueError('target_kl must be positive or None')
        if payload:
            if payload.get('artifact_kind') == 'deployment':
                raise ValueError('A deployment cannot resume training; use from_deployment for a new experiment')
            if (payload['algorithm'] != self.algorithm or payload['architecture'] != architecture_metadata()
                    or payload['trainer_state']['implementation'] != self.implementation):
                raise ValueError('Resume requires a batched hybrid PPO training checkpoint')
            if self.seed != payload['seed'] or self.config != payload['config']:
                raise ValueError('Seed and training settings cannot change during resume')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.model = HybridPokerNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.rng = np.random.default_rng(self.seed)
        self.spec = GameSpec(**payload['game_spec']) if payload else GameSpec()
        self.counters = {'iterations': 0, 'steps': 0, 'gradient_steps': 0, 'kl_early_stops': 0}
        self.metrics = {}
        self.initialization = copy.deepcopy(payload.get('initialization')) if payload else None
        self.opponents = {}
        self.opponent_sources = {}
        self.self_play = None
        self.population_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0x53454C46]))
        if payload:
            saved = payload['trainer_state']
            self.model.load_state_dict(payload['model_state'], strict=True)
            self.model.history.flatten_parameters()
            self.optimizer.load_state_dict(saved['optimizer'])
            self.rng.bit_generator.state = copy.deepcopy(saved['rng'])
            self.counters = copy.deepcopy(payload['counters'])
            self.metrics = copy.deepcopy(saved['metrics'])
            self.opponent_sources = copy.deepcopy(saved['opponent_sources'])
            for name, item in saved['opponents'].items():
                with torch.random.fork_rng(devices=[]):
                    model = _NETWORKS[item['network']]().to(self.device)
                model.load_state_dict(item['model_state'], strict=True)
                model.history.flatten_parameters()
                self.opponents[name] = model.eval().requires_grad_(False)
            self.self_play = copy.deepcopy(saved.get('self_play'))
            if self.self_play is not None:
                self.population_rng.bit_generator.state = copy.deepcopy(saved['population_rng'])
                self._validate_self_play()
            if opponent_paths is not None:
                if set(opponent_paths) != set(self.opponent_sources):
                    raise ValueError('Frozen opponent IDs changed during resume')
                for name, path in opponent_paths.items():
                    if self._digest(path) != self.opponent_sources[name]['sha256']:
                        raise ValueError(f'Frozen opponent changed during resume: {name}')
            restore_rng(saved['global_rng'])
        else:
            for name, path in (opponent_paths or {}).items():
                self.add_frozen_opponent(name, path)

    def add_frozen_opponent(self, policy_id, path):
        """Load a frozen opponent between rollouts."""
        if (not isinstance(policy_id, str) or not policy_id or policy_id.strip() != policy_id
                or policy_id == 'learner' or policy_id.startswith('selfplay-')):
            raise ValueError('Use a nonempty opponent ID outside learner and selfplay-')
        if self.self_play is not None:
            self._validate_self_play()
        if (policy_id in self.opponents) != (policy_id in self.opponent_sources):
            raise ValueError('Opponent ID is already used without a frozen source')
        source = resolve_path(path)
        before = source.stat()
        identity = (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        digest = self._digest(source)

        def verify_unchanged():
            after = source.stat()
            if identity != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise RuntimeError('Opponent checkpoint changed while registering it')

        if policy_id in self.opponent_sources:
            if digest != self.opponent_sources[policy_id]['sha256']:
                raise ValueError(f'Frozen opponent changed: {policy_id}')
            verify_unchanged()
            return False
        rng_state = capture_rng()
        try:
            payload = load_checkpoint(source)
            if payload.get('game_spec') != self.spec.to_dict():
                raise ValueError('Frozen opponent game specification differs from the learner')
            if payload.get('algorithm') not in {'ppo', 'ppo_semantic', 'ppo_hybrid'}:
                raise ValueError('Frozen opponent must use a supported direct neural policy')
            agent = load_agent(str(source), device=str(self.device))
            if not hasattr(agent, 'model') or type(agent.model) not in _NETWORKS.values():
                raise TypeError(f'Unsupported frozen opponent: {policy_id}')
            model = agent.model.eval().requires_grad_(False)
            if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
                raise ValueError('Frozen opponent weights must be finite')
            model.history.flatten_parameters()
            if self._digest(source) != digest:
                raise RuntimeError('Opponent checkpoint changed while registering it')
            verify_unchanged()
            provenance = {'path': str(source), 'sha256': digest, 'size_bytes': before.st_size,
                          'algorithm': payload['algorithm'], 'game_spec': copy.deepcopy(payload['game_spec']),
                          'artifact_kind': payload.get('artifact_kind', 'training'),
                          'source_checkpoint': copy.deepcopy(payload.get('source_checkpoint'))}
        finally:
            restore_rng(rng_state)
        self.opponents = {**self.opponents, policy_id: model}
        self.opponent_sources = {**self.opponent_sources, policy_id: provenance}
        return True

    @classmethod
    def from_deployment(cls, path, *, seed, config=None, device='cpu', opponent_paths=None):
        """Start from saved weights with a fresh optimizer."""
        source = resolve_path(path)
        before = source.stat()
        payload = load_checkpoint(source)
        if payload.get('artifact_kind') != 'deployment':
            raise ValueError('Warm start requires a deployment artifact, not a training resume')
        if payload.get('algorithm') != cls.algorithm or payload.get('architecture') != architecture_metadata():
            raise ValueError('Warm start requires the exact hybrid PPO architecture')
        source_config = payload.get('config')
        if not isinstance(source_config, dict) or 'reward_scale' not in source_config:
            raise ValueError('Deployment config must declare its value reward_scale')
        learner = cls(seed, config=config, device=device, opponent_paths=opponent_paths)
        if payload.get('game_spec') != learner.spec.to_dict():
            raise ValueError('Warm start game specification differs from the new experiment')
        if (not isinstance(source_config['reward_scale'], (int, float))
                or isinstance(source_config['reward_scale'], bool)
                or not np.isfinite(source_config['reward_scale'])
                or source_config['reward_scale'] != learner.config['reward_scale']):
            raise ValueError('Warm start must preserve reward_scale for the copied value head')
        state = payload.get('model_state')
        expected = learner.model.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError('Deployment is missing hybrid network weights')
        for name, weight in state.items():
            if (not isinstance(weight, torch.Tensor) or weight.shape != expected[name].shape
                    or weight.dtype != expected[name].dtype or not torch.isfinite(weight).all()):
                raise ValueError(f'Incompatible deployment weight: {name}')
        digest = cls._digest(source)
        after = source.stat()
        if ((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise RuntimeError('Deployment changed while creating the warm start')
        learner.model.load_state_dict(state, strict=True)
        learner.model.history.flatten_parameters()
        learner.initialization = {
            'version': 1, 'kind': 'deployment_warm_start', 'new_seed': learner.seed,
            'source': {'id': source.stem, 'path': str(source), 'sha256': digest,
                       'size_bytes': before.st_size, 'algorithm': payload['algorithm'],
                       'architecture': copy.deepcopy(payload['architecture']),
                       'game_spec': copy.deepcopy(payload['game_spec']),
                       'config': copy.deepcopy(source_config), 'seed': payload.get('seed'),
                       'counters': copy.deepcopy(payload.get('counters', {})),
                       'source_checkpoint': copy.deepcopy(payload.get('source_checkpoint'))},
        }
        random.seed(learner.seed)
        np.random.seed(learner.seed % 2**32)
        torch.manual_seed(learner.seed)
        return learner

    @staticmethod
    def _digest(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    def configure_self_play(self, config=None):
        settings = {**SELF_PLAY_DEFAULT_CONFIG, **(config or {})}
        if settings.keys() != SELF_PLAY_DEFAULT_CONFIG.keys():
            raise ValueError('Unknown self-play settings')
        for key, minimum in (('archive_interval', 1), ('history_capacity', 2)):
            if type(settings[key]) is not int or settings[key] < minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}')
        for key in ('latest_weight', 'history_weight'):
            value = settings[key]
            if isinstance(value, bool) or not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'{key} must lie in (0, 1]')
        if settings['latest_weight'] + settings['history_weight'] > 1:
            raise ValueError('Self-play mixture weights cannot sum above one')
        if self.self_play is not None:
            if settings != self.self_play['config']:
                raise ValueError('Self-play settings cannot change during resume')
            self._validate_self_play()
            return
        if self.counters['iterations']:
            raise ValueError('Configure self-play before the first training update')
        latest, initial = self._frozen_copy(), self._frozen_copy()
        self.opponents.update({SELF_PLAY_LATEST: latest, SELF_PLAY_INITIAL: initial})
        self.self_play = {
            'version': 1, 'config': settings, 'successful_updates': 0,
            'last_observed_iteration': 0, 'latest_iteration': 0, 'archive_candidates': 0,
            'history': [{'policy_id': SELF_PLAY_INITIAL, 'iteration': 0}],
        }

    def _validate_self_play(self):
        population = self.self_play
        if population['version'] != 1:
            raise ValueError('Unsupported self-play population version')
        history = population['history']
        ids = [item['policy_id'] for item in history]
        expected = set(ids) | {SELF_PLAY_LATEST} | set(self.opponent_sources)
        if (not history or ids[0] != SELF_PLAY_INITIAL or history[0]['iteration'] != 0
                or len(set(ids)) != len(ids) or set(self.opponents) != expected
                or len(history) > population['config']['history_capacity']):
            raise ValueError('Invalid self-play population snapshots')
        if not 0 <= population['last_observed_iteration'] <= self.counters['iterations']:
            raise ValueError('Invalid self-play iteration counter')

    def _frozen_copy(self):
        model = copy.deepcopy(self.model).eval().requires_grad_(False)
        model.history.flatten_parameters()
        return model

    def refresh_self_play(self):
        """Refresh only after a complete update and before collecting new hands."""
        if self.self_play is None:
            raise RuntimeError('Configure self-play before refreshing it')
        population = copy.deepcopy(self.self_play)
        iteration = self.counters['iterations']
        previous = population['last_observed_iteration']
        if iteration == previous:
            return False
        if iteration != previous + 1:
            raise RuntimeError('Refresh self-play after each training update')
        if not self.metrics.get('optimizer_steps', 0):
            population['last_observed_iteration'] = iteration
            self.self_play = population
            return False
        opponents = self.opponents.copy()
        rng = np.random.default_rng(0)
        rng.bit_generator.state = copy.deepcopy(self.population_rng.bit_generator.state)
        opponents[SELF_PLAY_LATEST] = self._frozen_copy()
        population['latest_iteration'] = iteration
        population['successful_updates'] += 1
        settings = population['config']
        if population['successful_updates'] % settings['archive_interval'] == 0:
            population['archive_candidates'] += 1
            history = population['history']
            capacity = settings['history_capacity'] - 1
            seen = population['archive_candidates']
            slot = seen - 1 if seen <= capacity else int(rng.integers(seen))
            if slot < capacity:
                policy_id = f'selfplay-history-{iteration:06d}'
                item = {'policy_id': policy_id, 'iteration': iteration}
                frozen = self._frozen_copy()
                if slot + 1 < len(history):
                    del opponents[history[slot + 1]['policy_id']]
                    history[slot + 1] = item
                else:
                    history.append(item)
                opponents[policy_id] = frozen
        population['last_observed_iteration'] = iteration
        self.opponents, self.self_play = opponents, population
        self.population_rng.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        return True

    def self_play_opponents(self):
        if self.self_play is None:
            raise RuntimeError('Configure self-play before requesting opponents')
        population = self.self_play
        settings, history = population['config'], population['history']
        return [{'policy_id': SELF_PLAY_LATEST, 'kind': 'neural', 'weight': settings['latest_weight']}] + [
            {'policy_id': item['policy_id'], 'kind': 'neural',
             'weight': settings['history_weight'] / len(history)} for item in history]

    def infer(self, requests):
        responses = [None] * len(requests)
        groups = {}
        for index, request in enumerate(requests):
            policy_id = request['policy_id']
            if policy_id != 'learner' and policy_id not in self.opponents:
                raise KeyError(f'Unknown policy ID: {policy_id}')
            groups.setdefault(policy_id, []).append(index)
        with torch.inference_mode():
            for policy_id, indices in groups.items():
                model = self.model if policy_id == 'learner' else self.opponents[policy_id]
                model.eval()
                batch = prepare_observations([requests[i]['observation'] for i in indices], self.device)
                logits, values = prepared_forward(model, batch)
                probabilities = masked_probs(logits, batch['mask']).cpu().numpy().astype(np.float64)
                values = values.cpu().numpy()
                if not np.isfinite(probabilities).all() or not np.isfinite(values).all():
                    raise FloatingPointError('Non-finite batched inference output')
                probabilities /= probabilities.sum(-1, keepdims=True)
                for row, index in enumerate(indices):
                    responses[index] = {'probabilities': probabilities[row].copy(),
                                        'value': float(values[row]) if policy_id == 'learner' else 0.}
        return responses

    def update(self, samples):
        """Update from one rollout using its saved targets."""
        if not samples or any(len(sample) != 6 for sample in samples):
            raise ValueError('Expected nonempty six-field PPO samples')
        batch = prepare_observations([sample[0] for sample in samples], self.device)
        for observation, action, *_ in samples:
            if (isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer))
                    or not 0 <= action < 5 or not observation.legal_mask[action]):
                raise ValueError('Rollout actions must be legal integer slots')
        scalars = np.asarray([sample[2:] for sample in samples], dtype=np.float32)
        if not np.isfinite(scalars).all() or np.any(scalars[:, 0] > 1e-6):
            raise ValueError('Rollout scalars must be finite and log probabilities nonpositive')
        advantages = scalars[:, 1]
        normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        targets = torch.tensor(np.column_stack((scalars[:, 0], normalized, scalars[:, 2:])),
                               dtype=torch.float32, device=self.device)
        actions = torch.tensor([sample[1] for sample in samples], dtype=torch.long, device=self.device)
        records = []
        stop = False
        self.model.train()
        for epoch in range(self.config['epochs']):
            order = torch.tensor(self.rng.permutation(len(samples)), dtype=torch.long)
            device_order = order.to(self.device)
            offset = 0
            while offset < len(samples):
                end = balanced_batch_end(len(samples), self.config['batch_size'], offset)
                cpu_indices, indices = order[offset:end], device_order[offset:end]
                selected = _select(batch, cpu_indices, indices)
                logits, values = prepared_forward(self.model, selected)
                probabilities = masked_probs(logits, selected['mask'])
                logs = probabilities.clamp_min(1e-12).log()
                new_logs = logs.gather(-1, actions[indices, None]).squeeze(-1)
                old_logs, adv, returns, old_values = targets[indices].unbind(-1)
                policy = clipped_policy_loss(new_logs, old_logs, adv, self.config['clip_epsilon'])
                value = critic_loss(values, returns, old_values)
                entropy = -(probabilities * logs).sum(-1).mean()
                loss = policy + self.config['value_coefficient'] * value - self.config['entropy_coefficient'] * entropy
                log_ratio = new_logs.detach() - old_logs
                ratios = log_ratio.exp()
                kl = ((ratios - 1.) - log_ratio).mean()
                diagnostics = torch.stack((loss.detach(), policy.detach(), value.detach(), entropy.detach(),
                                           kl, ((ratios - 1.).abs() > self.config['clip_epsilon']).float().mean()))
                numbers = diagnostics.cpu().numpy()
                if not np.isfinite(numbers).all():
                    raise FloatingPointError('Non-finite PPO update')
                record = dict(zip(('total', 'policy', 'value', 'entropy', 'approximate_kl', 'clip_fraction'),
                                  map(float, numbers)))
                record.update(epoch=epoch, minibatch_size=end-offset, optimizer_step=False)
                if self.config['target_kl'] is not None and record['approximate_kl'] > self.config['target_kl']:
                    self.counters['kl_early_stops'] += 1
                    record['early_stop'] = 'target_kl'
                    records.append(record)
                    stop = True
                    break
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'],
                                                      error_if_nonfinite=True)
                self.optimizer.step()
                self.counters['gradient_steps'] += 1
                record.update(optimizer_step=True, gradient_norm_before_clip=float(norm.detach().cpu()))
                records.append(record)
                offset = end
            if stop:
                break
        self.model.eval()
        self.counters['iterations'] += 1
        self.counters['steps'] += len(samples)
        self.metrics = {'samples': len(samples), 'optimizer_steps': sum(r['optimizer_step'] for r in records),
                        'early_stop': stop, 'last_minibatch': records[-1],
                        'advantage_mean': float(advantages.mean()), 'advantage_std': float(advantages.std()),
                        'return_mean_scaled': float(scalars[:, 2].mean())}
        return copy.deepcopy(self.metrics)

    def payload(self):
        opponents = {}
        for name, model in self.opponents.items():
            network = next(key for key, cls in _NETWORKS.items() if type(model) is cls)
            opponents[name] = {'network': network, 'model_state': cpu_state(model)}
        return {'algorithm': self.algorithm, 'architecture': architecture_metadata(), 'seed': self.seed,
                'game_spec': self.spec.to_dict(), 'config': copy.deepcopy(self.config),
                'initialization': copy.deepcopy(self.initialization),
                'counters': copy.deepcopy(self.counters), 'model_state': cpu_state(self.model),
                'trainer_state': {'implementation': self.implementation,
                                  'optimizer': _to_cpu(self.optimizer.state_dict()),
                                  'rng': copy.deepcopy(self.rng.bit_generator.state),
                                  'global_rng': capture_rng(), 'metrics': copy.deepcopy(self.metrics),
                                  'self_play': copy.deepcopy(self.self_play),
                                  'population_rng': copy.deepcopy(self.population_rng.bit_generator.state),
                                  'opponents': opponents, 'opponent_sources': copy.deepcopy(self.opponent_sources)}}
