# HoldemBot

A heads-up no-limit Hold'em bot trained with PPO on 5.13 million simulated hands.
Training used self-play and opponents trained to exploit the bot.

Games use 200bb stacks, 50/100 blinds, and no rake. The bot sees its own cards,
the board, stacks, and betting history. It chooses fold, check/call, half-pot,
pot, or all-in. The engine also accepts other legal raise amounts.

## Run locally

Use Python 3.11 and Node.js 22.12+ on macOS or Linux. Clone the repository,
install the locked Python dependencies, and start the API:

```sh
git clone https://github.com/JishnuJetwani/HoldemBot.git
cd HoldemBot
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m uvicorn poker_lab.server:app --host 127.0.0.1 --port 8001
```

In a second terminal, start the frontend from the repository root:

```sh
cd frontend
npm ci
npm run dev
```

With both servers running, open [localhost:5173](http://localhost:5173) on your
computer to play and replay hands. Sessions are stored in `artifacts/demo.sqlite3`;
stacks reset after each hand.

## Train

Training runs several games in parallel against recent and older copies of the
bot. Opponents update between batches. Checkpoints save the optimizer, opponent
pool, and random state so runs can resume. Run these commands from the repository
root with the Python environment activated:

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

For Modal, install the cloud extra and authenticate once:

```sh
python -m pip install -e '.[cloud]'
modal setup
modal run scripts/modal_train.py --help
```

New runs use your config; the model notes record the included bot's training setup.

## Evaluate

Supply a second trained checkpoint as `--opponent`. The original evaluation
opponents are not bundled with this repository.

```sh
python scripts/evaluate.py --checkpoint artifacts/deployment/holdem-ppo.pt \
  --opponent path/to/opponent.pt --pairs 10000 --seed 42 \
  --output artifacts/evaluation.json
```

Each deal is played from both seats. Results include the returns for each pair,
bb/100, and a 95% confidence interval. For Slumbot, see
`python -m poker_lab.slumbot_benchmark --help`.

See the [model notes](docs/model.md) for the network, training settings, and
results. The bot remains vulnerable to newly trained attackers.

## Checks

```sh
pytest -q
npm --prefix frontend run build
```

Tests cover rules against PokerKit, card visibility, legal actions, training
recovery, evaluation, and saved sessions. The browser check requires both servers
and an installed Chrome or Chromium:

```sh
npm --prefix frontend run test:browser
```

Set `CHROME_EXECUTABLE` if the browser is installed outside the usual locations.
