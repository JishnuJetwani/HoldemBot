"""Alternate attacker training with defender self-play."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from .batched_ppo import DEFAULT_CONFIG
from .parallel_campaign import (digest, opponent_inputs, read_config, run_lock,
                                self_play_settings, sha, sources, write)


def archive_weights(archive, *, generation, half_life=2., uniform_share=.2):
    """Favor recent attackers while giving every saved attacker some weight."""
    if (not archive or not math.isfinite(half_life) or half_life <= 0
            or not 0 <= uniform_share <= 1):
        raise ValueError('Invalid archive weighting configuration')
    if any(entry['generation'] > generation for entry in archive):
        raise ValueError('Archive entry is from a future generation')
    recent = [2 ** (-(generation - entry['generation']) / half_life) for entry in archive]
    total = sum(recent)
    return {entry['id']: (1 - uniform_share) * value / total + uniform_share / len(archive)
            for entry, value in zip(archive, recent)}


def run_league(output_dir, *, initialization=None, rounds=None, defender_seconds=None,
               attacker_seconds=None, config=None, self_play_config=None, seed=None,
               workers=None, envs_per_worker=None, opponent_paths=None,
               opponent_weight=None, resume=False, max_updates=None, commit=lambda: None):
    """Train a new attacker, then continue the defender, on CPU.

    Resume keeps each phase's deadline and skips completed phases.
    ``max_updates`` limits updates per phase.
    """
    from .batched_ppo import BatchedPPOLearner
    from .checkpoint import load_checkpoint, save_checkpoint
    from .export import export_checkpoint
    from .game import GameSpec
    from .league_training import train_phase
    import torch

    folder = Path(output_dir).expanduser().resolve()
    with run_lock(folder):
        state_path = folder / 'league.json'
        requested = {'rounds': rounds, 'defender_seconds': defender_seconds,
                     'attacker_seconds': attacker_seconds, 'seed': seed, 'workers': workers,
                     'envs_per_worker': envs_per_worker, 'opponent_weight': opponent_weight,
                     'max_updates': max_updates}
        if resume:
            if not state_path.exists():
                raise FileNotFoundError('No saved league in output_dir')
            state = json.loads(state_path.read_text())
            settings = state['settings']
            for key, value in requested.items():
                if value is not None and value != settings[key]:
                    raise ValueError(f'Resume cannot change {key}')
            if initialization is not None and str(Path(initialization).expanduser().resolve()) != settings['initialization']:
                raise ValueError('Resume cannot change initialization')
            if config is not None and {**DEFAULT_CONFIG, **config} != settings['config']:
                raise ValueError('Resume cannot change PPO settings')
            if self_play_config is not None and self_play_settings(
                    self_play_config, opponent_weight=settings['opponent_weight']) != settings['self_play']:
                raise ValueError('Resume cannot change self-play settings')
            if opponent_paths is not None and opponent_inputs(opponent_paths) != settings['opponents']:
                raise ValueError('Resume cannot change frozen opponents')
            if settings['sources'] != sources():
                raise ValueError('Training source changed since this league was created')
        else:
            if state_path.exists() or any(p.name != '.training.lock' for p in folder.iterdir()):
                raise FileExistsError('Use an empty output directory, or pass resume=True')
            defaults = {'rounds': 3, 'defender_seconds': 1800., 'attacker_seconds': 900.,
                        'seed': 0, 'workers': 1, 'envs_per_worker': 8,
                        'opponent_weight': .3, 'max_updates': None}
            settings = {key: defaults[key] if value is None else value for key, value in requested.items()}
            if type(settings['rounds']) is not int or settings['rounds'] < 1:
                raise ValueError('rounds must be a positive integer')
            for key in ('defender_seconds', 'attacker_seconds'):
                value = settings[key]
                if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'{key} must be finite and positive')
            if not 0 < settings['opponent_weight'] < 1:
                raise ValueError('opponent_weight must lie in (0, 1)')
            if max_updates is not None and (type(max_updates) is not int or max_updates < 0):
                raise ValueError('max_updates must be a nonnegative integer')
            settings.update(config={**DEFAULT_CONFIG, **(config or {})},
                            self_play=self_play_settings(self_play_config, opponent_weight=settings['opponent_weight']),
                            opponents=opponent_inputs(opponent_paths), sources=sources(),
                            initialization=str(Path(initialization).expanduser().resolve(strict=True)) if initialization else None)
            state = {'version': 1, 'settings': settings, 'phases': {}, 'phase_deadlines': {},
                     'archive': [{'id': key, **item, 'generation': 0}
                                 for key, item in settings['opponents'].items()],
                     'defender': None, 'initial': None, 'status': 'training'}
        torch.set_num_threads(1)
        if state['initial'] is None:
            if settings['initialization']:
                learner = BatchedPPOLearner.from_deployment(settings['initialization'], seed=settings['seed'],
                                                           config=settings['config'])
            else:
                learner = BatchedPPOLearner(settings['seed'], config=settings['config'])
            source = folder / 'initial' / 'training.pt'
            target = folder / 'initial' / 'deployment.pt'
            save_checkpoint(source, {**learner.payload(), 'artifact_kind': 'training',
                                     'source_code_sha256': digest(settings['sources'])})
            export_checkpoint(source, target)
            state['initial'] = {'path': str(target), 'sha256': sha(target)}
            write(state_path, state)
            commit()
        for entry in [state['initial'], *state['archive']]:
            if sha(entry['path']) != entry['sha256']:
                raise ValueError('A frozen league model changed: ' + entry['path'])
            load_checkpoint(entry['path'])
        for phase in state['phases'].values():
            for prefix in ('resume', 'deployment', 'export_source'):
                if sha(phase[prefix + '_path']) != phase[prefix + '_sha256']:
                    raise ValueError('A completed phase artifact changed')
                load_checkpoint(phase[prefix + '_path'])
        for generation in range(1, settings['rounds'] + 1):
            for role in ('attacker', 'defender'):
                phase_id = f'{role}-{generation}'
                if phase_id in state['phases']:
                    continue
                duration = settings[role + '_seconds']
                if phase_id not in state['phase_deadlines']:
                    state['phase_deadlines'][phase_id] = time.time() + duration
                    write(state_path, state)
                    commit()
                target = state['defender']['deployment_path'] if state['defender'] else state['initial']['path']
                if role == 'attacker':
                    paths = {'target': target}
                    entries = [{'policy_id': 'target', 'kind': 'neural', 'weight': 1.}]
                    phase_seed = settings['seed'] + generation
                    inherited = None
                    self_play = None
                else:
                    paths = {entry['id']: entry['path'] for entry in state['archive']}
                    weights = archive_weights(state['archive'], generation=generation)
                    entries = [{'policy_id': key, 'kind': 'neural',
                                'weight': weight * settings['opponent_weight']}
                               for key, weight in weights.items()]
                    phase_seed = settings['seed']
                    inherited = state['defender']['resume_path'] if state['defender'] else None
                    self_play = settings['self_play']
                result = train_phase(folder / phase_id, phase_id=phase_id, role=role,
                                     initialization_path=target if role == 'attacker' else state['initial']['path'],
                                     resume_from=inherited, opponent_paths=paths, opponent_spec=entries,
                                     self_play_config=self_play, seed=phase_seed, config=settings['config'],
                                     spec=GameSpec().to_dict(), workers=settings['workers'],
                                     envs_per_worker=settings['envs_per_worker'], seconds=duration,
                                     global_deadline_unix=state['phase_deadlines'][phase_id],
                                     max_updates=settings['max_updates'], commit=commit)
                state['phases'][phase_id] = result
                if role == 'attacker':
                    state['archive'].append({'id': phase_id, 'generation': generation,
                                             'path': result['deployment_path'], 'sha256': result['deployment_sha256']})
                else:
                    state['defender'] = result
                write(state_path, state)
                commit()
                print(json.dumps({'phase': phase_id, 'status': result['status'], 'counters': result['counters']}), flush=True)
        state['status'] = 'complete'
        state['deployment_path'] = state['defender']['deployment_path']
        write(state_path, state)
        write(folder / 'result.json', state)
        commit()
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--initialization', help='Hybrid PPO weights to start the defender')
    parser.add_argument('--resume', action='store_true', help='Continue with the saved settings and phase deadlines')
    parser.add_argument('--config', help='JSON object with ppo and self_play settings')
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--defender-seconds', type=float)
    parser.add_argument('--attacker-seconds', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--envs-per-worker', type=int)
    parser.add_argument('--opponent', action='append', help='Saved neural opponent to add to the attacker pool')
    parser.add_argument('--opponent-weight', type=float, help='Share of hands against attackers (default: 0.3)')
    parser.add_argument('--max-updates', type=int, help='Optional update limit per phase')
    args = parser.parse_args()
    config = read_config(args.config)
    result = run_league(args.output_dir, initialization=args.initialization, resume=args.resume,
                        config=config.get('ppo'), self_play_config=config.get('self_play'),
                        rounds=args.rounds, defender_seconds=args.defender_seconds,
                        attacker_seconds=args.attacker_seconds, seed=args.seed, workers=args.workers,
                        envs_per_worker=args.envs_per_worker, opponent_paths=args.opponent,
                        opponent_weight=args.opponent_weight, max_updates=args.max_updates)
    print(json.dumps({'status': result['status'], 'deployment_path': result['deployment_path'],
                      'defender_counters': result['defender']['counters']}, indent=2))


if __name__ == '__main__':
    main()
