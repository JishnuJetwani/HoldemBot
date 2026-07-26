"""Compare two frozen checkpoints on duplicate deals with swapped seats."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path):
    import hashlib
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    import torch
    from poker_lab.agents import load_agent
    from poker_lab.checkpoint import load_checkpoint
    from poker_lab.evaluation import evaluate_matchup
    from poker_lab.game import GameSpec

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--opponent', required=True)
    parser.add_argument('--pairs', type=int, default=1000, help='Independent duplicate-deal pairs (two hands each)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True, help='JSON result including per-pair returns')
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error('--pairs must be positive')
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error('--output already exists; choose a new result file')
    torch.set_num_threads(1)
    paths = [Path(value).expanduser().resolve(strict=True) for value in (args.checkpoint, args.opponent)]
    if output in paths or output in [path.with_suffix('.json') for path in paths]:
        parser.error('--output must be separate from both model files and their manifests')
    identities = [{'path': str(path), 'sha256': sha(path)} for path in paths]
    payloads = [load_checkpoint(path) for path in paths]
    if payloads[0]['game_spec'] != payloads[1]['game_spec']:
        parser.error('Checkpoint game specifications differ')
    result = evaluate_matchup(load_agent(paths[0]), load_agent(paths[1]),
                              num_pairs=args.pairs, seed=args.seed,
                              spec=GameSpec(**payloads[0]['game_spec']),
                              name_a=paths[0].stem, name_b=paths[1].stem)
    for entry in identities:
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('Checkpoint changed while evaluating')
    result['checkpoints'] = identities
    write(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'raw_pairs'}, indent=2))


if __name__ == '__main__':
    main()
