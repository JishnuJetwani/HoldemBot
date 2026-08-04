"""Train hybrid PPO against self-play and optional fixed opponents."""
from __future__ import annotations

import argparse
import json

from .parallel_campaign import read_config, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='Run directory containing checkpoints and metrics')
    parser.add_argument('--seconds', type=float, default=3600, help='Additional seconds to train')
    parser.add_argument('--config', help='JSON object with ppo and self_play settings')
    parser.add_argument('--seed', type=int, default=None, help='New-run seed (default: 0)')
    parser.add_argument('--workers', type=int, default=None, help='Rollout processes (default: 1; 0 runs in-process)')
    parser.add_argument('--envs-per-worker', type=int, default=None, help='Concurrent hands per worker (default: 8)')
    parser.add_argument('--device', default='cpu', help='Device for inference and training')
    start = parser.add_mutually_exclusive_group()
    start.add_argument('--warm-start', help='Deployment weights for a new run with a fresh optimizer')
    start.add_argument('--resume', action='store_true', help='Continue the saved run in output-dir')
    parser.add_argument('--opponent', action='append', default=None, help='Frozen neural checkpoint; repeat for a mixture')
    parser.add_argument('--opponent-weight', type=float, default=None, help='Total frozen-opponent share (default: 0.3 when present)')
    parser.add_argument('--max-updates', type=int, help='Maximum updates during this run')
    args = parser.parse_args()
    config = read_config(args.config)
    result = run_training(args.output_dir, seconds=args.seconds, config=config.get('ppo'),
                          self_play_config=config.get('self_play'), seed=args.seed,
                          workers=args.workers, envs_per_worker=args.envs_per_worker,
                          device=args.device, warm_start=args.warm_start, resume=args.resume,
                          opponent_paths=args.opponent, opponent_weight=args.opponent_weight,
                          max_updates=args.max_updates)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
