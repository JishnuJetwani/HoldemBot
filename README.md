# HoldemBot

A heads-up no-limit Hold'em bot trained with PPO on 5.13 million simulated hands.
Training used self-play and opponents trained to exploit the bot.

Games use 200bb stacks, 50/100 blinds, and no rake. The bot sees its own cards,
the board, stacks, and betting history. It chooses fold, check/call, half-pot,
pot, or all-in. The engine also accepts other legal raise amounts.

## Play

Requires Python 3.11+, Node.js 20.19+ or 22.12+, and macOS or Linux.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m uvicorn poker_lab.server:app --host 127.0.0.1 --port 8001
```

In another terminal:

```sh
cd frontend
npm ci
npm run dev
```

Open http://localhost:5173 to play, replay hands, and view the bot's action
probabilities. Sessions are saved; stacks reset after each hand.

## Train

Training runs several games in parallel against recent and older copies of the
bot. Opponents update between batches. Checkpoints save the optimizer, opponent
pool, and random state so runs can resume.

```sh
python -m poker_lab.train --output-dir artifacts/runs/selfplay \
  --config configs/ppo.json --seconds 3600 --workers 4 --envs-per-worker 16

python -m poker_lab.train --output-dir artifacts/runs/finetune \
  --warm-start artifacts/deployment/holdem-ppo.pt --seconds 3600

python -m poker_lab.train --output-dir artifacts/runs/selfplay --resume --seconds 3600
```

Adversarial training first trains an attacker against the bot, then trains the
bot against self-play opponents and saved attackers. Newer attackers get more
weight.

```sh
python -m poker_lab.league_campaign --output-dir artifacts/runs/league \
  --initialization artifacts/deployment/holdem-ppo.pt --config configs/league.json
```

For Modal, install `pip install -e '.[cloud]'` and run
`modal run scripts/modal_train.py --help`. New runs use your config; the model
notes below record the included bot's training setup.

## Evaluate

```sh
python scripts/evaluate.py --checkpoint artifacts/deployment/holdem-ppo.pt \
  --opponent path/to/opponent.pt --pairs 10000 --seed 42 \
  --output artifacts/evaluation.json
```

Each deal is played from both seats. Results include the returns for each pair,
bb/100, and a 95% confidence interval. For Slumbot, see
`python -m poker_lab.slumbot_benchmark --help`.

See the [model notes](docs/model.md) for the network, training settings, and
results. The bot remains exploitable; these results do not establish competitive
poker strength.

## Checks

```sh
pytest -q
npm --prefix frontend run build
```

Tests cover rules against PokerKit, card visibility, legal actions, training
recovery, evaluation, and saved sessions. With both servers running,
`npm --prefix frontend run test:browser` checks play and replay.
