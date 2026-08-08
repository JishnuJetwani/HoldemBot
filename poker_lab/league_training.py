"""Attacker and defender training with a checkpoint after each update."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import time


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read(path):
    return json.loads(Path(path).read_text())


def _same(left, right):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _same(value, right[key]) for key, value in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right))
    return left == right


def _identity(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': _sha(path),
            'manifest_sha256': _sha(path.with_suffix('.json'))}


def _load_generation(folder, entry):
    from .checkpoint import load_checkpoint
    if entry['file'] not in ('checkpoint-a.pt', 'checkpoint-b.pt'):
        raise ValueError('Unknown checkpoint generation')
    path = folder / entry['file']
    if (_sha(path) != entry['sha256']
            or _sha(path.with_suffix('.json')) != entry['manifest_sha256']):
        raise ValueError('Checkpoint generation is incomplete or changed')
    payload = load_checkpoint(path)
    if payload['counters'] != entry['counters']:
        raise ValueError('Checkpoint pointer counters disagree')
    return payload


def _restore(folder, commit):
    path = folder / 'checkpoint.json'
    if not path.exists():
        return None, None, None
    pointer = _read(path)
    try:
        return _load_generation(folder, pointer['current']), pointer, None
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        previous = pointer.get('previous')
        if previous is None:
            raise ValueError('No valid committed checkpoint generation') from error
        try:
            payload = _load_generation(folder, previous)
        except (OSError, ValueError, KeyError, RuntimeError) as second:
            raise ValueError('Neither committed checkpoint generation is valid') from second
        recovery = {'reason': str(error), 'discarded': pointer['current'],
                    'restored': previous, 'at_unix': time.time()}
        history = _read(folder / 'recovery.json') if (folder / 'recovery.json').exists() else []
        _write(folder / 'recovery.json', history + [recovery])
        pointer = {'current': previous, 'previous': None}
        _write(path, pointer)
        commit()
        return payload, pointer, recovery


def _commit_generation(folder, payload, pointer, commit):
    from .checkpoint import save_checkpoint
    current = pointer['current'] if pointer else None
    name = 'checkpoint-b.pt' if current and current['file'] == 'checkpoint-a.pt' else 'checkpoint-a.pt'
    path = folder / name
    save_checkpoint(path, payload)
    entry = {'file': name, 'sha256': _sha(path),
             'manifest_sha256': _sha(path.with_suffix('.json')), 'counters': payload['counters']}
    _load_generation(folder, entry)
    updated = {'current': entry, 'previous': current}
    _write(folder / 'checkpoint.json', updated)
    commit()
    return updated


def _check_recorded_exports(folder):
    from .checkpoint import load_checkpoint
    result = folder / 'result.json'
    if not result.exists():
        return set()
    recorded = _read(result)
    protected = set()
    for prefix in ('export_source', 'deployment'):
        try:
            path = Path(recorded[prefix + '_path'])
            if not path.with_suffix('.json').exists() or _sha(path) != recorded[prefix + '_sha256']:
                raise ValueError('Recorded export hash differs')
            load_checkpoint(path)
            protected.add(path.resolve())
        except (OSError, ValueError, EOFError, KeyError, RuntimeError, TypeError, pickle.UnpicklingError) as error:
            raise ValueError('Committed phase export changed: ' + prefix) from error
    return protected


def _quarantine_export(folder, path, reason):
    index = 1
    while True:
        destination = folder / 'quarantine' / f'{path.parent.name}-{path.stem}-{index:03d}'
        try:
            destination.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            index += 1
    _write(destination / 'reason.json', {'source': str(path), 'reason': str(reason)})
    for item in (path, path.with_suffix('.json')):
        if item.exists():
            os.replace(item, destination / item.name)


def _export(folder, phase_id, payload):
    from .checkpoint import load_checkpoint, save_checkpoint
    from .export import export_checkpoint
    protected = _check_recorded_exports(folder)
    identifier = f"{phase_id}-update-{payload['counters']['iterations']:06d}"
    source = folder / 'exports' / (identifier + '.pt')
    target = folder / 'deployment' / (identifier + '.pt')
    valid_source = False
    if source.exists() or source.with_suffix('.json').exists():
        try:
            if not source.exists() or not source.with_suffix('.json').exists():
                raise ValueError('Incomplete export source pair')
            saved = load_checkpoint(source)
            for key in ('algorithm', 'architecture', 'config', 'seed', 'counters',
                        'model_state', 'trainer_state', 'collector_state'):
                if not _same(saved[key], payload[key]):
                    raise ValueError('Export source differs from committed training state')
            valid_source = True
        except (OSError, ValueError, EOFError, KeyError, RuntimeError, TypeError, pickle.UnpicklingError) as error:
            if source.resolve() in protected:
                raise ValueError('Committed phase export differs from training state') from error
            _quarantine_export(folder, source, error)
    if not valid_source:
        save_checkpoint(source, payload)
    valid_target = False
    if target.exists() or target.with_suffix('.json').exists():
        try:
            if not target.exists() or not target.with_suffix('.json').exists():
                raise ValueError('Incomplete deployment pair')
            exported = load_checkpoint(target)
            if (exported['source_checkpoint']['sha256'] != _sha(source)
                    or not _same(exported['model_state'], payload['model_state'])):
                raise ValueError('Deployment differs from committed export source')
            valid_target = True
        except (OSError, ValueError, EOFError, KeyError, RuntimeError, TypeError, pickle.UnpicklingError) as error:
            if target.resolve() in protected:
                raise ValueError('Committed phase deployment differs from training state') from error
            _quarantine_export(folder, target, error)
    if not valid_target:
        export_checkpoint(source, target)
    return {'deployment_path': str(target.resolve()), 'deployment_sha256': _sha(target),
            'export_source_path': str(source.resolve()), 'export_source_sha256': _sha(source)}


def _build(settings, payload, opponent_paths, initialization_path):
    import torch
    from .batched_ppo import BatchedPPOLearner
    from .game import GameSpec
    from .parallel_rollout import RolloutPool
    torch.set_num_threads(1)
    if payload:
        saved_sources = payload['trainer_state']['opponent_sources']
        if not saved_sources.keys() <= opponent_paths.keys():
            raise ValueError('Previously frozen opponents must remain in the archive')
        learner = BatchedPPOLearner(settings['seed'], config=settings['config'], payload=payload,
                                    opponent_paths={key: opponent_paths[key] for key in saved_sources})
        for key, path in opponent_paths.items():
            learner.add_frozen_opponent(key, path)
    else:
        learner = BatchedPPOLearner.from_deployment(initialization_path, seed=settings['seed'],
                                                   config=settings['config'], opponent_paths=opponent_paths)
    if learner.spec.to_dict() != settings['spec']:
        raise ValueError('League phase game specification changed')
    if settings['role'] == 'defender':
        learner.configure_self_play(settings['self_play_config'])
        population = learner.self_play_opponents() + settings['opponent_spec']
    else:
        if learner.self_play is not None:
            raise ValueError('Attacker cannot inherit self-play population')
        population = settings['opponent_spec']
    collector = payload.get('collector_state') if payload else None
    previous = collector['configuration']['opponents'] if collector else population
    pool = RolloutPool(GameSpec(**settings['spec']), settings['seed'], workers=settings['workers'],
                       envs_per_worker=settings['envs_per_worker'], opponent_spec=previous, state=collector,
                       gamma=settings['config']['gamma'], gae_lambda=settings['config']['gae_lambda'],
                       reward_scale=settings['config']['reward_scale'])
    try:
        if population != previous:
            pool.set_opponents(population)
    except BaseException:
        pool.close()
        raise
    return learner, pool


def train_phase(folder, *, phase_id, role, initialization_path, resume_from=None,
                opponent_paths, opponent_spec, self_play_config=None, seed, config, spec,
                workers, envs_per_worker, seconds, global_deadline_unix,
                commit=lambda: None, max_updates=None):
    """Train an attacker or continue the defender until the saved deadline."""
    from .checkpoint import load_checkpoint
    from .game import GameSpec
    if not phase_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in phase_id):
        raise ValueError('Use a lowercase phase ID')
    if role not in ('defender', 'attacker'):
        raise ValueError('Unknown league role')
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Phase duration must be finite and positive')
    if not math.isfinite(global_deadline_unix):
        raise ValueError('Global deadline must be finite')
    if max_updates is not None and (type(max_updates) is not int or max_updates < 0):
        raise ValueError('max_updates must be a nonnegative phase update cap')
    if (role == 'defender') != (self_play_config is not None):
        raise ValueError('Only defender phases require self-play settings')
    if role == 'attacker' and resume_from is not None:
        raise ValueError('New attackers start from deployment weights, not another training phase')
    spec = GameSpec(**spec).to_dict() if isinstance(spec, dict) else spec.to_dict()
    opponent_paths = {key: str(Path(value).resolve()) for key, value in opponent_paths.items()}
    entries = copy.deepcopy(list(opponent_spec))
    ids = [row['policy_id'] for row in entries]
    if (not entries or len(set(ids)) != len(ids) or not set(ids) <= opponent_paths.keys()
            or any(row.get('kind') != 'neural' or not math.isfinite(row.get('weight', 1.))
                   or row.get('weight', 1.) <= 0 for row in entries)):
        raise ValueError('Expected a positive mixture of known frozen neural opponents')
    weight = sum(row.get('weight', 1.) for row in entries)
    if self_play_config is not None:
        weight += self_play_config['latest_weight'] + self_play_config['history_weight']
    if not math.isclose(weight, 1., abs_tol=1e-9):
        raise ValueError('League opponent weights must sum to one')
    if role == 'attacker' and len(entries) != 1:
        raise ValueError('Each attacker targets exactly one frozen policy')
    settings = {'version': 1, 'phase_id': phase_id, 'role': role, 'seed': seed,
                'config': copy.deepcopy(config), 'spec': spec, 'workers': workers,
                'envs_per_worker': envs_per_worker, 'seconds': seconds,
                'global_deadline_unix': global_deadline_unix,
                'self_play_config': copy.deepcopy(self_play_config), 'opponent_spec': entries,
                'opponents': {key: _identity(path) for key, path in opponent_paths.items()},
                'initialization': _identity(initialization_path),
                'resume_from': _identity(resume_from) if resume_from else None}
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    _check_recorded_exports(folder)
    phase_path = folder / 'phase.json'
    if phase_path.exists():
        phase = _read(phase_path)
        if phase['settings'] != settings:
            raise ValueError('League phase identity or settings changed')
    else:
        now = time.time()
        phase = {'settings': settings, 'started_unix': now,
                 'deadline_unix': min(now + seconds, global_deadline_unix)}
        _write(phase_path, phase)
        commit()
    restored, pointer, recovery = _restore(folder, commit)
    inherited = load_checkpoint(resume_from) if restored is None and resume_from else None
    payload = restored if restored is not None else inherited
    if restored is not None and restored['league_phase']['identity'] != phase:
        raise ValueError('Checkpoint belongs to a different league phase')
    if inherited is not None and inherited.get('league_phase', {}).get('role') != 'defender':
        raise ValueError('Defender continuation requires a committed defender checkpoint')
    started = time.monotonic()
    local_deadline = started + max(0., phase['deadline_unix'] - time.time())
    learner, pool = _build(settings, payload, opponent_paths, initialization_path)
    cumulative = {**learner.counters, 'hands': payload['counters']['hands'] if payload else 0}
    prior_phase = restored['league_phase'] if restored else None
    initial_counters = copy.deepcopy(prior_phase['initial_counters'] if prior_phase else cumulative)
    baseline_seconds = prior_phase['baseline_training_seconds'] if prior_phase else (
        inherited.get('training_seconds', 0.) if inherited else 0.)
    prior_active = prior_phase['active_training_seconds'] if prior_phase else 0.
    status = 'global_deadline' if phase['deadline_unix'] == global_deadline_unix else 'deadline'

    def checkpoint(last_update=None):
        nonlocal pointer
        active = prior_active + time.monotonic() - started
        state = {**learner.payload(), 'artifact_kind': 'training',
                 'collector_state': pool.state_dict(), 'counters': copy.deepcopy(cumulative),
                 'training_seconds': baseline_seconds + active,
                 'league_phase': {'identity': phase, 'phase_id': phase_id, 'role': role,
                                  'initial_counters': initial_counters,
                                  'baseline_training_seconds': baseline_seconds,
                                  'active_training_seconds': active}}
        if last_update is not None:
            _write(folder / 'metrics' / f"update-{cumulative['iterations']:06d}.json", last_update)
        pointer = _commit_generation(folder, state, pointer, commit)
        _write(folder / 'progress.json', {'phase_id': phase_id, 'role': role, 'status': 'training',
                                          'counters': cumulative, 'deadline_unix': phase['deadline_unix'],
                                          'active_training_seconds': active})
        return state

    try:
        if restored is None:
            checkpoint()
        while time.monotonic() < local_deadline:
            if max_updates is not None and cumulative['iterations'] - initial_counters['iterations'] >= max_updates:
                status = 'update_cap'
                break
            rollout = pool.collect(learner.infer, config['rollout_steps'], deadline=local_deadline)
            if not rollout['samples']:
                break
            metrics = learner.update(rollout['samples'])
            if role == 'defender':
                learner.refresh_self_play()
                pool.set_opponents(learner.self_play_opponents() + entries)
            cumulative = {**learner.counters, 'hands': cumulative['hands'] + len(rollout['hands'])}
            checkpoint({'counters': cumulative, 'metrics': metrics,
                        'opponents': rollout['per_opponent'], 'rollout_counters': rollout['counters']})
        saved = _load_generation(folder, pointer['current'])
        exported = _export(folder, phase_id, saved)
        result = {'phase_id': phase_id, 'role': role, 'status': status, 'counters': saved['counters'],
                  'phase_counters': {key: saved['counters'][key] - initial_counters[key] for key in initial_counters},
                  'resume_path': str(folder / pointer['current']['file']),
                  'resume_sha256': pointer['current']['sha256'], **exported,
                  'active_training_seconds': saved['league_phase']['active_training_seconds'],
                  'elapsed_phase_seconds': max(0., time.time() - phase['started_unix']),
                  'deadline_unix': phase['deadline_unix'], 'recovery': recovery}
        _write(folder / 'result.json', result)
        _write(folder / 'progress.json', result)
        commit()
        return result
    except Exception as error:
        _write(folder / 'failure.json', {'phase_id': phase_id, 'error': repr(error),
                                       'last_committed': pointer['current'] if pointer else None})
        commit()
        raise
    finally:
        pool.close()
