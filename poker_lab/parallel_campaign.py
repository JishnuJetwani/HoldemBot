"""Run PPO self-play and save each completed update."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

from .batched_ppo import DEFAULT_CONFIG

SELF_PLAY = {'latest_weight': .7, 'history_weight': .3,
             'archive_interval': 10, 'history_capacity': 32}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


def sources():
    """Hash the source files used by a run."""
    names = ('train.py', 'parallel_campaign.py', 'league_training.py',
             'batched_ppo.py', 'parallel_rollout.py', 'ppo.py', 'game.py', 'networks.py',
             'hybrid_policy.py', 'semantic_policy.py', 'agents.py', 'checkpoint.py', 'export.py')
    return {name: sha(Path(__file__).with_name(name)) for name in names}


@contextmanager
def run_lock(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.training.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('This output directory already has an active trainer') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def opponent_inputs(paths):
    """Load fixed opponents and record their file hashes."""
    from .checkpoint import load_checkpoint
    result = {}
    for index, supplied in enumerate(paths or ()):
        path = Path(supplied).expanduser().resolve(strict=True)
        load_checkpoint(path)
        result[f'fixed-{index:03d}'] = {'path': str(path), 'sha256': sha(path)}
    return result


def self_play_settings(config=None, *, opponent_weight=0.):
    settings = {**SELF_PLAY, **(config or {})}
    if isinstance(opponent_weight, bool) or not math.isfinite(opponent_weight) or not 0 <= opponent_weight < 1:
        raise ValueError('opponent_weight must lie in [0, 1)')
    total = settings['latest_weight'] + settings['history_weight']
    if not math.isfinite(total) or total <= 0:
        raise ValueError('Self-play weights must have a finite positive sum')
    for key in ('latest_weight', 'history_weight'):
        settings[key] = settings[key] / total * (1 - opponent_weight)
    return settings


def read_config(path):
    value = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(value, dict) or value.keys() - {'ppo', 'self_play'}:
        raise ValueError('Configuration must be an object with ppo and/or self_play objects')
    if any(not isinstance(settings, dict) for settings in value.values()):
        raise ValueError('ppo and self_play settings must be JSON objects')
    return value


def run_training(output_dir, *, seconds, seed=None, config=None, self_play_config=None,
                 workers=None, envs_per_worker=None, warm_start=None, resume=False,
                 opponent_paths=None, opponent_weight=None, device='cpu', max_updates=None,
                 commit=lambda: None):
    """Train for additional wall time or resume a saved run.

    Save each completed update to alternating checkpoint files. Resume restores
    the optimizer, collector, and RNG from the last saved update. ``max_updates``
    limits updates in this call, including on resume.
    """
    from .batched_ppo import BatchedPPOLearner
    from .checkpoint import load_checkpoint
    from .game import GameSpec
    from .league_training import _commit_generation, _export, _load_generation, _restore
    from .parallel_rollout import RolloutPool
    import numpy as np
    import random
    import torch

    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('seconds must be finite and positive')
    if max_updates is not None and (type(max_updates) is not int or max_updates < 0):
        raise ValueError('max_updates must be a nonnegative integer')
    if warm_start is not None and resume:
        raise ValueError('Choose either a new warm start or resume')
    folder = Path(output_dir).expanduser().resolve()
    with run_lock(folder):
        settings_path = folder / 'run.json'
        if resume:
            if not settings_path.exists():
                raise FileNotFoundError('No saved run in output_dir')
            settings = json.loads(settings_path.read_text())
            supplied = {'seed': seed, 'workers': workers, 'envs_per_worker': envs_per_worker,
                        'opponent_weight': opponent_weight}
            for key, value in supplied.items():
                if value is not None and value != settings[key]:
                    raise ValueError(f'Resume cannot change {key}')
            if config is not None and {**DEFAULT_CONFIG, **config} != settings['config']:
                raise ValueError('Resume cannot change PPO settings')
            if self_play_config is not None and self_play_settings(
                    self_play_config, opponent_weight=settings['opponent_weight']) != settings['self_play']:
                raise ValueError('Resume cannot change self-play settings')
            if opponent_paths is not None and opponent_inputs(opponent_paths) != settings['opponents']:
                raise ValueError('Resume cannot change frozen opponents')
            if settings['sources'] != sources():
                raise ValueError('Training source changed since this run was created')
        else:
            if settings_path.exists() or any(p.name != '.training.lock' for p in folder.iterdir()):
                raise FileExistsError('Use an empty output directory, or pass resume=True')
            opponents = opponent_inputs(opponent_paths)
            weight = (0.3 if opponents else 0.) if opponent_weight is None else opponent_weight
            if (not opponents and weight != 0) or (opponents and weight <= 0):
                raise ValueError('Frozen opponents require a positive opponent_weight; omit the weight without opponents')
            settings = {'version': 1, 'seed': 0 if seed is None else seed,
                        'config': {**DEFAULT_CONFIG, **(config or {})},
                        'self_play': self_play_settings(self_play_config, opponent_weight=weight),
                        'workers': 1 if workers is None else workers,
                        'envs_per_worker': 8 if envs_per_worker is None else envs_per_worker,
                        'opponents': opponents, 'opponent_weight': weight,
                        'initialization': {'path': str(Path(warm_start).expanduser().resolve(strict=True)),
                                           'sha256': sha(Path(warm_start).expanduser())} if warm_start else None,
                        'sources': sources()}
        torch.set_num_threads(1)
        random.seed(settings['seed'])
        np.random.seed(settings['seed'] % 2**32)
        torch.manual_seed(settings['seed'])
        payload, pointer, recovery = _restore(folder, commit) if resume else (None, None, None)
        if payload is not None and payload.get('run_sha256') != digest(settings):
            raise ValueError('Checkpoint does not belong to the saved run')
        paths = {key: item['path'] for key, item in settings['opponents'].items()}
        for key, item in settings['opponents'].items():
            if sha(item['path']) != item['sha256']:
                raise ValueError(f'Frozen opponent changed: {key}')
        arguments = dict(seed=settings['seed'], config=settings['config'], device=device, opponent_paths=paths)
        if payload is not None:
            learner = BatchedPPOLearner(payload=payload, **arguments)
        elif settings['initialization']:
            initial = settings['initialization']
            if sha(initial['path']) != initial['sha256']:
                raise ValueError('Warm-start model changed')
            learner = BatchedPPOLearner.from_deployment(initial['path'], **arguments)
        else:
            learner = BatchedPPOLearner(**arguments)
        learner.configure_self_play(settings['self_play'])
        fixed = [{'policy_id': key, 'kind': 'neural', 'weight': settings['opponent_weight'] / len(paths)}
                 for key in paths]
        pool = RolloutPool(GameSpec(), settings['seed'], workers=settings['workers'],
                           envs_per_worker=settings['envs_per_worker'],
                           opponent_spec=learner.self_play_opponents() + fixed,
                           state=payload.get('collector_state') if payload else None,
                           gamma=learner.config['gamma'], gae_lambda=learner.config['gae_lambda'],
                           reward_scale=learner.config['reward_scale'])
        started = time.monotonic()
        deadline = started + seconds
        previous_seconds = payload.get('training_seconds', 0.) if payload else 0.
        hands = payload['counters'].get('hands', 0) if payload else 0
        initial_iteration = learner.counters['iterations']
        status = 'deadline'

        def checkpoint(metrics=None):
            nonlocal pointer
            state = {**learner.payload(), 'artifact_kind': 'training', 'collector_state': pool.state_dict(),
                     'counters': {**learner.counters, 'hands': hands},
                     'training_seconds': previous_seconds + time.monotonic() - started,
                     'run_sha256': digest(settings), 'source_code_sha256': digest(settings['sources'])}
            pointer = _commit_generation(folder, state, pointer, commit)
            progress = {'status': 'training', 'counters': state['counters'],
                        'training_seconds': state['training_seconds'], 'metrics': metrics}
            write(folder / 'progress.json', progress)
            if metrics is not None:
                write(folder / 'metrics' / f"update-{learner.counters['iterations']:06d}.json", progress)
            return state

        try:
            if not settings_path.exists():
                write(settings_path, settings)
            if pointer is None:
                checkpoint()
            while time.monotonic() < deadline:
                if max_updates is not None and learner.counters['iterations'] - initial_iteration >= max_updates:
                    status = 'update_cap'
                    break
                rollout = pool.collect(learner.infer, learner.config['rollout_steps'], deadline=deadline)
                if not rollout['samples']:
                    break
                metrics = learner.update(rollout['samples'])
                learner.refresh_self_play()
                pool.set_opponents(learner.self_play_opponents() + fixed)
                hands += len(rollout['hands'])
                checkpoint({'update': metrics, 'opponents': rollout['per_opponent']})
                print(json.dumps({'counters': {**learner.counters, 'hands': hands}}), flush=True)
        except KeyboardInterrupt:
            status = 'interrupted'
        finally:
            pool.close()
        saved = _load_generation(folder, pointer['current'])
        # Export only the last committed update, including after an interruption.
        exported = _export(folder, 'selfplay', saved)
        result = {'status': status, 'counters': saved['counters'],
                  'training_seconds': saved['training_seconds'],
                  'resume_path': str(folder / pointer['current']['file']),
                  'resume_sha256': pointer['current']['sha256'], 'recovery': recovery, **exported}
        write(folder / 'result.json', result)
        write(folder / 'progress.json', result)
        commit()
        return result
