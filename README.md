# HoldemBot

A PPO bot for heads-up no-limit Hold'em with 200bb stacks, 50/100 blinds, and
no rake. The network combines hand features, card embeddings, and a GRU
over betting history.

Requires Python 3.11+ on macOS or Linux.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m poker_lab.train --output-dir artifacts/runs/selfplay \
  --config configs/ppo.json --seconds 3600 --workers 4 --envs-per-worker 16
python -m poker_lab.train --output-dir artifacts/runs/selfplay --resume --seconds 3600
```

Training runs several games in parallel against recent and older copies of the
bot. Opponents update between batches. Add fixed opponents with `--opponent`,
or use `--warm-start` to start a new run from saved weights.

Each update saves the optimizer, game workers, opponent pool, and random state.
Two alternating checkpoint files allow recovery if a write is interrupted.

## Evaluate

```sh
python scripts/evaluate.py --checkpoint path/to/policy.pt \
  --opponent path/to/opponent.pt --pairs 10000 --seed 42 --output artifacts/evaluation.json
python -m poker_lab.slumbot_benchmark --help
pytest -q
```

Each deal is played from both seats. Results include returns for each pair,
bb/100, and a 95% confidence interval. Tests compare rules with PokerKit and
check card visibility, suit symmetry, legal actions, and training recovery.

## Adversarial training

```sh
python -m poker_lab.league_campaign --output-dir artifacts/runs/league \
  --initialization path/to/policy.pt --config configs/league.json
```

Each round trains a new attacker against the bot, then trains the bot against
self-play opponents and saved attackers. Newer attackers get more weight, but
all remain in the pool. The bot keeps its optimizer and history between phases.
Resume keeps the original phase deadlines.

## Modal

Install `pip install -e '.[cloud]'`, then run
`modal run scripts/modal_train.py --help` for cloud options. Modal uses the
same training code and saves runs in a volume. Warm starts upload saved weights.
