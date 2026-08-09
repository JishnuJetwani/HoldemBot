"""Run self-play or league training on Modal.

Start or resume a cloud run:
    modal run scripts/modal_train.py --name selfplay --seconds 3600
    modal run scripts/modal_train.py --name league --mode league --rounds 3
    modal run scripts/modal_train.py --name selfplay --resume --seconds 3600

Runs are saved in the holdembot-training volume. To download one:
    modal volume get holdembot-training RUN_NAME ./downloaded-run
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
app = modal.App('holdembot-training')
volume = modal.Volume.from_name('holdembot-training', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install_from_pyproject(str(ROOT / 'pyproject.toml'))
         .env({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
         .add_local_dir(ROOT / 'poker_lab', '/root/poker_lab', ignore=['**/__pycache__/**']))


@app.function(image=image, volumes={'/runs': volume}, cpu=4, memory=8192,
              timeout=86400, retries=0, max_containers=1)
def train_remote(name, mode, options, inputs):
    from poker_lab.parallel_campaign import run_training
    from poker_lab.league_campaign import run_league

    volume.reload()
    for filename, data in inputs.items():
        destination = Path('/runs/inputs') / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError('Uploaded model input changed')
        destination.write_bytes(data)
    volume.commit()
    runner = run_league if mode == 'league' else run_training
    return runner(Path('/runs') / name, **options, commit=volume.commit)


@app.local_entrypoint()
def main(name: str, mode: str = 'selfplay', seconds: float = 3600,
         config: str = '', warm_start: str = '', opponents: str = '', resume: bool = False,
         seed: int | None = None, workers: int | None = None, envs_per_worker: int | None = None,
         rounds: int | None = None, defender_seconds: float | None = None,
         attacker_seconds: float | None = None, max_updates: int | None = None):
    from poker_lab.parallel_campaign import read_config

    if not name or any(char not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for char in name) or name == 'inputs':
        raise ValueError('Use a lowercase run name containing letters, digits, hyphens or underscores')
    if mode not in ('selfplay', 'league'):
        raise ValueError('mode must be selfplay or league')
    if resume and warm_start:
        raise ValueError('A warm start creates a new run; omit it when resuming')
    settings = read_config(config or None)
    inputs = {}

    def upload(path):
        source = Path(path).expanduser().resolve(strict=True)
        data = source.read_bytes()
        filename = hashlib.sha256(data).hexdigest() + '.pt'
        inputs[filename] = data
        manifest = source.with_suffix('.json')
        if manifest.exists():
            inputs[filename.removesuffix('.pt') + '.json'] = manifest.read_bytes()
        return '/runs/inputs/' + filename

    options = {'config': settings.get('ppo'), 'self_play_config': settings.get('self_play'),
               'seed': seed, 'workers': workers, 'envs_per_worker': envs_per_worker,
               'resume': resume, 'max_updates': max_updates,
               'opponent_paths': [upload(path.strip()) for path in opponents.split(',')] if opponents else None}
    initial = upload(warm_start) if warm_start else None
    if mode == 'selfplay':
        options.update(seconds=seconds, warm_start=initial)
    else:
        options.update(initialization=initial, rounds=rounds,
                       defender_seconds=defender_seconds, attacker_seconds=attacker_seconds)
    result = train_remote.remote(name, mode, options, inputs)
    print(json.dumps({'status': result['status'], 'deployment_path': result['deployment_path'],
                      'volume': 'holdembot-training', 'run': name}, indent=2))
