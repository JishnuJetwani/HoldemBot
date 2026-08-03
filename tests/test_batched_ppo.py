"""Batched inference identity and PPO continuation checks."""
import random

import numpy as np
import pytest
import torch

from poker_lab.agents import cpu_state, load_agent
from poker_lab.batched_ppo import (BatchedPPOLearner, SELF_PLAY_INITIAL, SELF_PLAY_LATEST,
                                   prepare_observations, prepared_forward)
from poker_lab.checkpoint import load_checkpoint, save_checkpoint
from poker_lab.export import export_checkpoint
from poker_lab.game import HoldemState
from poker_lab.hybrid_policy import HybridPokerNetwork
from poker_lab.networks import PokerNetwork, batch_observations
from poker_lab.semantic_policy import SemanticPokerNetwork


@pytest.fixture(autouse=True)
def torch_setup():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observations():
    game = HoldemState.new(8273)
    result = [game.observe()]
    while not game.is_terminal():
        game.apply_slot(1)
        if not game.is_terminal():
            result.append(game.observe())
    return result


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def rollout(learner):
    rows = observations() * 3
    responses = learner.infer([{'policy_id': 'learner', 'observation': obs} for obs in rows])
    result = []
    for index, (obs, response) in enumerate(zip(rows, responses)):
        action = next(i for i, probability in enumerate(response['probabilities']) if probability > 0)
        advantage = .7 if index % 2 else -.4
        result.append((obs, action, float(np.log(response['probabilities'][action])),
                       advantage, advantage + response['value'], response['value']))
    return result


@pytest.mark.parametrize('network', [HybridPokerNetwork, SemanticPokerNetwork, PokerNetwork])
def test_prepared_forward_matches_original(network):
    rows = observations()
    model = network().eval()
    original = model(batch_observations(rows))
    actual = prepared_forward(model, prepare_observations(rows))
    for wanted, got in zip(original, actual):
        torch.testing.assert_close(wanted, got, rtol=0, atol=0)


def test_collection_log_probabilities_match_training_batch():
    from poker_lab.networks import masked_probs
    learner = BatchedPPOLearner(6559)
    samples = rollout(learner)
    order = list(reversed(range(len(samples))))
    rows = [samples[index] for index in order]
    batch = prepare_observations([row[0] for row in rows])
    logits, _ = prepared_forward(learner.model, batch)
    probabilities = masked_probs(logits, batch['mask'])
    actions = torch.tensor([row[1] for row in rows])
    logs = probabilities.log().gather(-1, actions[:, None]).squeeze(-1)
    np.testing.assert_allclose(logs.detach().numpy(), [row[2] for row in rows], atol=2e-7)


def test_checkpoint_deployment_and_optimizer_resume_are_exact(tmp_path):
    config = {'rollout_steps': 16, 'batch_size': 7, 'epochs': 2, 'target_kl': None}
    learner = BatchedPPOLearner(7284, config=config)
    learner.update(rollout(learner))
    payload = learner.payload()
    path = tmp_path / 'learner.pt'
    save_checkpoint(path, payload)
    resumed = BatchedPPOLearner(7284, payload=load_checkpoint(path))
    assert_same(learner.model.state_dict(), resumed.model.state_dict())
    assert_same(learner.optimizer.state_dict(), resumed.optimizer.state_dict())
    deployed = load_agent(path)
    for row in observations():
        expected = learner.infer([{'policy_id': 'learner', 'observation': row}])[0]['probabilities']
        np.testing.assert_array_equal(expected, deployed.probabilities(row))
    samples = rollout(learner)
    assert_same(learner.update(samples), resumed.update(samples))
    assert_same(learner.model.state_dict(), resumed.model.state_dict())
    assert_same(learner.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_same(learner.counters, resumed.counters)
    assert_same(learner.rng.bit_generator.state, resumed.rng.bit_generator.state)
    for key, value in payload['model_state'].items():
        assert torch.equal(value, load_checkpoint(path)['model_state'][key])


def test_kl_gate_runs_before_optimizer_step():
    learner = BatchedPPOLearner(1417)
    samples = [(obs, action, old_log-4, adv, ret, value)
               for obs, action, old_log, adv, ret, value in rollout(learner)]
    before = cpu_state(learner.model)
    result = learner.update(samples)
    assert result['early_stop'] and result['optimizer_steps'] == 0
    assert_same(before, learner.model.state_dict())


def trained_deployment(tmp_path):
    base = BatchedPPOLearner(7221, config={'epochs': 1, 'target_kl': None})
    base.configure_self_play({'archive_interval': 1})
    base.update(rollout(base))
    base.refresh_self_play()
    training, deployment = tmp_path / 'base-training.pt', tmp_path / 'base-deployment.pt'
    save_checkpoint(training, base.payload())
    export_checkpoint(training, deployment)
    return base, deployment


def test_deployment_warm_start_copies_all_weights_and_resets_training_state(tmp_path):
    base, deployment = trained_deployment(tmp_path)
    with pytest.raises(ValueError, match='cannot resume'):
        BatchedPPOLearner(base.seed, payload=load_checkpoint(deployment))
    seed = 8759
    fork = BatchedPPOLearner.from_deployment(
        deployment, seed=seed, config={'learning_rate': .00005, 'epochs': 2},
        opponent_paths={'base': deployment})
    assert_same(base.model.state_dict(), fork.model.state_dict())
    assert not fork.optimizer.state
    assert fork.config['learning_rate'] == .00005
    assert fork.config['epochs'] == 2
    assert fork.optimizer.param_groups[0]['lr'] == .00005
    assert all(value == 0 for value in fork.counters.values())
    assert fork.metrics == {} and fork.self_play is None
    assert set(fork.opponents) == {'base'}
    assert all(parameter.requires_grad for parameter in fork.model.parameters())
    assert_same(fork.rng.bit_generator.state, np.random.default_rng(seed).bit_generator.state)
    expected_population = np.random.default_rng(np.random.SeedSequence([seed, 0x53454C46]))
    assert_same(fork.population_rng.bit_generator.state, expected_population.bit_generator.state)
    assert random.getstate() == random.Random(seed).getstate()
    assert_same(np.random.get_state(), np.random.RandomState(seed).get_state())
    assert torch.equal(torch.get_rng_state(), torch.Generator().manual_seed(seed).get_state())
    for own, source, frozen in zip(fork.model.parameters(), base.model.parameters(),
                                   fork.opponents['base'].parameters()):
        assert own.data_ptr() != source.data_ptr()
        assert own.data_ptr() != frozen.data_ptr()
    provenance = fork.initialization
    assert provenance['kind'] == 'deployment_warm_start'
    assert provenance['new_seed'] == seed
    assert provenance['source']['sha256'] == fork._digest(deployment)
    assert provenance['source']['seed'] == base.seed
    assert provenance['source']['counters'] == base.counters
    assert provenance['source']['source_checkpoint'] == load_checkpoint(deployment)['source_checkpoint']


def test_self_play_changes_only_on_explicit_successful_refresh():
    learner = BatchedPPOLearner(2764, config={'epochs': 1, 'target_kl': None})
    learner.configure_self_play({'archive_interval': 1})
    initial = cpu_state(learner.opponents[SELF_PLAY_INITIAL])
    for frozen in learner.opponents.values():
        assert not any(parameter.requires_grad for parameter in frozen.parameters())
        for parameter, own in zip(frozen.parameters(), learner.model.parameters()):
            assert parameter.data_ptr() != own.data_ptr()
    learner.update(rollout(learner))
    assert_same(initial, learner.opponents[SELF_PLAY_LATEST].state_dict())
    assert learner.refresh_self_play()
    assert learner.self_play['latest_iteration'] == 1
    assert_same(learner.model.state_dict(), learner.opponents[SELF_PLAY_LATEST].state_dict())
    assert_same(initial, learner.opponents[SELF_PLAY_INITIAL].state_dict())
    assert len(learner.self_play_opponents()) == 3
    assert sum(entry['weight'] for entry in learner.self_play_opponents()) == pytest.approx(.85)
    previous = learner.payload()['trainer_state']['self_play']
    assert not learner.refresh_self_play()
    assert learner.self_play == previous


def test_self_play_reservoir_is_bounded_reproducible_and_preserves_initial():
    learner = BatchedPPOLearner(7223, config={'epochs': 1, 'target_kl': None})
    learner.configure_self_play({'archive_interval': 1, 'history_capacity': 3})
    expected_rng = np.random.default_rng(np.random.SeedSequence([7223, 0x53454C46]))
    expected = [0]
    initial = cpu_state(learner.opponents[SELF_PLAY_INITIAL])
    for iteration in range(1, 9):
        learner.update(rollout(learner))
        learner.refresh_self_play()
        slot = iteration - 1 if iteration <= 2 else int(expected_rng.integers(iteration))
        if slot < 2:
            if slot + 1 < len(expected):
                expected[slot + 1] = iteration
            else:
                expected.append(iteration)
        assert [item['iteration'] for item in learner.self_play['history']] == expected
        assert len(learner.opponents) <= 4
        assert_same(initial, learner.opponents[SELF_PLAY_INITIAL].state_dict())
    assert learner.self_play['archive_candidates'] == 8
    assert_same(learner.population_rng.bit_generator.state, expected_rng.bit_generator.state)


@pytest.mark.parametrize('workers', [0, 2])
def test_actual_rollout_update_and_population_resume(workers):
    from poker_lab.parallel_rollout import RolloutPool
    learner = BatchedPPOLearner(5778, config={'epochs': 1, 'target_kl': None})
    learner.configure_self_play()
    config = dict(spec=learner.spec, seed=234, workers=workers, envs_per_worker=2,
                  opponent_spec=learner.self_play_opponents(), reward_scale=learner.config['reward_scale'])
    with RolloutPool(**config) as pool:
        result = pool.collect(learner.infer, 32)
        learner.update(result['samples'])
        learner.refresh_self_play()
        payload, pool_state = learner.payload(), pool.state_dict()
        expected = pool.collect(learner.infer, 48)
        expected_metrics = learner.update(expected['samples'])
        learner.refresh_self_play()
        expected_state = pool.state_dict()
    resumed = BatchedPPOLearner(5778, payload=payload)
    resumed.configure_self_play()
    with RolloutPool(**config, state=pool_state) as pool:
        actual = pool.collect(resumed.infer, 48)
        for key in ('samples', 'counters', 'hands', 'per_opponent'):
            assert_same(actual[key], expected[key])
        assert_same(resumed.update(actual['samples']), expected_metrics)
        resumed.refresh_self_play()
        assert_same(pool.state_dict(), expected_state)
    assert_same(learner.model.state_dict(), resumed.model.state_dict())
    assert_same(learner.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_same(learner.self_play, resumed.self_play)
    for name in learner.opponents:
        assert_same(learner.opponents[name].state_dict(), resumed.opponents[name].state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device unavailable')
def test_cuda_inference_update_and_resume():
    cpu = BatchedPPOLearner(7183, config={'batch_size': 4, 'epochs': 1, 'target_kl': None})
    gpu = BatchedPPOLearner(7183, config=cpu.config, device='cuda')
    requests = [{'policy_id': 'learner', 'observation': row} for row in observations()]
    for expected, actual in zip(cpu.infer(requests), gpu.infer(requests)):
        np.testing.assert_allclose(expected['probabilities'], actual['probabilities'], atol=2e-6)
    samples = rollout(gpu)
    assert gpu.update(samples)['optimizer_steps'] > 0
    resumed = BatchedPPOLearner(7183, payload=gpu.payload(), device='cuda')
    for expected, actual in zip(gpu.infer(requests), resumed.infer(requests)):
        np.testing.assert_array_equal(expected['probabilities'], actual['probabilities'])
    assert all(value.device.type == 'cuda' for state in resumed.optimizer.state.values()
               for key, value in state.items() if isinstance(value, torch.Tensor) and key != 'step')
