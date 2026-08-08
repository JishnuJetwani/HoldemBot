"""League continuation, fixed opponents, deadlines, and torn-write recovery."""
import copy
import json
from pathlib import Path
import time

import pytest
import torch

from poker_lab import league_training as league
from poker_lab.batched_ppo import BatchedPPOLearner
from poker_lab.checkpoint import load_checkpoint, save_checkpoint
from poker_lab.export import export_checkpoint
from poker_lab.game import GameSpec


@pytest.fixture
def inputs(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    for index, name in enumerate(('base', 'attack')):
        source = tmp_path / (name + '-training.pt')
        save_checkpoint(source, BatchedPPOLearner(1830 + index).payload())
        export_checkpoint(source, tmp_path / (name + '.pt'))
    config = {**BatchedPPOLearner(10).config, 'rollout_steps': 8, 'batch_size': 4,
              'epochs': 1, 'target_kl': None}
    yield {'phase_id': 'main-1', 'role': 'defender', 'initialization_path': tmp_path / 'base.pt',
           'opponent_paths': {'base': tmp_path / 'base.pt'},
           'opponent_spec': [{'policy_id': 'base', 'kind': 'neural', 'weight': .3}],
           'self_play_config': {'latest_weight': .5, 'history_weight': .2,
                                'archive_interval': 1, 'history_capacity': 4},
           'seed': 9933, 'config': config, 'spec': GameSpec(), 'workers': 0,
           'envs_per_worker': 2, 'seconds': 10, 'global_deadline_unix': 1_900_000_500.}, tmp_path
    torch.set_num_threads(previous)


class Clock:
    def __init__(self, elapsed=0.):
        self.base = time.monotonic() + 1000
        self.elapsed = elapsed

    def time(self):
        return 1_900_000_000. + self.elapsed

    def monotonic(self):
        return self.base + self.elapsed


def execute(settings, folder, monkeypatch, *, elapsed=0., inspect=None,
            fail_update=None, max_updates=None):
    original = league._build
    clock = Clock(elapsed)
    monkeypatch.setattr(league, 'time', clock)

    def build(*args, **kwargs):
        learner, pool = original(*args, **kwargs)
        if inspect:
            inspect(learner, pool)
        update = learner.update
        calls = 0

        def measured(samples):
            nonlocal calls
            result = update(samples)
            clock.elapsed += 3.
            calls += 1
            if calls == fail_update:
                raise RuntimeError('Injected optimizer failure')
            return result

        learner.update = measured
        return learner, pool

    monkeypatch.setattr(league, '_build', build)
    try:
        return league.train_phase(folder, **settings, max_updates=max_updates)
    finally:
        monkeypatch.setattr(league, '_build', original)


def test_same_phase_resume_preserves_next_updates(inputs, monkeypatch):
    settings, root = inputs
    first = execute(settings, root / 'resumed', monkeypatch, max_updates=1)
    assert first['phase_counters']['iterations'] == 1
    resumed = execute(settings, root / 'resumed', monkeypatch, elapsed=3, max_updates=3)
    whole = execute(settings, root / 'whole', monkeypatch, max_updates=3)
    assert resumed['phase_counters']['iterations'] == 3
    assert resumed['active_training_seconds'] == pytest.approx(9, rel=0, abs=1e-8)
    assert whole['active_training_seconds'] == pytest.approx(9, rel=0, abs=1e-8)
    a, b = load_checkpoint(resumed['resume_path']), load_checkpoint(whole['resume_path'])
    for key in ('model_state', 'trainer_state', 'collector_state', 'counters'):
        assert league._same(a[key], b[key])
    assert len(list((root / 'resumed').glob('checkpoint-?.pt'))) == 2
    assert league._sha(resumed['resume_path']) == resumed['resume_sha256']
    assert league._sha(resumed['deployment_path']) == resumed['deployment_sha256']


def test_defender_transition_retains_optimizer_history_and_collector(inputs, monkeypatch):
    settings, root = inputs
    previous = execute(settings, root / 'first', monkeypatch, max_updates=2)
    payload = load_checkpoint(previous['resume_path'])
    following = copy.deepcopy(settings)
    following.update(phase_id='main-2', resume_from=previous['resume_path'])
    following['opponent_paths']['attack'] = root / 'attack.pt'
    following['opponent_spec'] = [{'policy_id': 'base', 'kind': 'neural', 'weight': .1},
                                  {'policy_id': 'attack', 'kind': 'neural', 'weight': .2}]

    def inspect(learner, pool):
        assert league._same(learner.optimizer.state_dict(), payload['trainer_state']['optimizer'])
        assert league._same(learner.model.state_dict(), payload['model_state'])
        assert learner.self_play == payload['trainer_state']['self_play']
        assert learner.counters == payload['counters']
        assert league._same(pool.state_dict()['workers'], payload['collector_state']['workers'])
        assert pool.state_dict()['counters'] == payload['collector_state']['counters']
        assert pool.opponents == learner.self_play_opponents() + following['opponent_spec']
        assert set(learner.opponent_sources) == {'base', 'attack'}

    result = execute(following, root / 'second', monkeypatch, elapsed=6, max_updates=1, inspect=inspect)
    assert result['counters']['iterations'] == 3
    assert result['phase_counters']['iterations'] == 1
    assert result['active_training_seconds'] == pytest.approx(3)
    continued = load_checkpoint(result['resume_path'])
    assert continued['trainer_state']['self_play']['latest_iteration'] == 3
    assert continued['training_seconds'] == pytest.approx(9)


def test_attacker_is_fresh_and_opponent_never_changes(inputs, monkeypatch):
    settings, root = inputs
    settings.update(phase_id='attack-1', role='attacker', self_play_config=None,
                    opponent_paths={'target': root / 'attack.pt'},
                    opponent_spec=[{'policy_id': 'target', 'kind': 'neural', 'weight': 1.}])
    target = load_checkpoint(root / 'attack.pt')['model_state']
    base = load_checkpoint(root / 'base.pt')['model_state']

    def inspect(learner, pool):
        assert learner.self_play is None
        assert learner.optimizer.state_dict()['state'] == {}
        assert learner.counters['iterations'] == 0
        assert league._same(learner.model.state_dict(), base)
        assert league._same(learner.opponents['target'].state_dict(), target)

    result = execute(settings, root / 'attacker', monkeypatch, max_updates=2, inspect=inspect)
    payload = load_checkpoint(result['resume_path'])
    assert payload['trainer_state']['self_play'] is None
    assert list(payload['trainer_state']['opponents']) == ['target']
    assert league._same(payload['trainer_state']['opponents']['target']['model_state'], target)


def test_failure_after_optimizer_mutation_keeps_committed_generation(inputs, monkeypatch):
    settings, root = inputs
    with pytest.raises(RuntimeError, match='optimizer failure'):
        execute(settings, root / 'failure', monkeypatch, fail_update=2)
    pointer = json.loads((root / 'failure/checkpoint.json').read_text())
    saved = league._load_generation(root / 'failure', pointer['current'])
    assert saved['counters']['iterations'] == 1
    resumed = execute(settings, root / 'failure', monkeypatch, elapsed=3, max_updates=3)
    whole = execute(settings, root / 'whole', monkeypatch, max_updates=3)
    actual, expected = load_checkpoint(resumed['resume_path']), load_checkpoint(whole['resume_path'])
    for key in ('model_state', 'trainer_state', 'collector_state', 'counters'):
        assert league._same(actual[key], expected[key])


def test_corrupt_current_generation_falls_back_explicitly(inputs, monkeypatch):
    settings, root = inputs
    first = execute(settings, root / 'corrupt', monkeypatch, max_updates=2)
    Path(first['resume_path']).write_bytes(b'corrupt')
    result = execute(settings, root / 'corrupt', monkeypatch, elapsed=20)
    assert result['counters']['iterations'] == 1
    assert result['recovery']['discarded']['counters']['iterations'] == 2
    assert result['recovery']['restored']['counters']['iterations'] == 1
    assert json.loads((root / 'corrupt/recovery.json').read_text())[-1] == result['recovery']
