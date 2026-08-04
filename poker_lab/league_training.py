"""Checkpoint recovery and deployment exports."""
from __future__ import annotations

import hashlib
import json
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
