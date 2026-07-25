# HoldemBot

Heads-up no-limit Hold'em with 200bb stacks, 50/100 blinds, and no rake.
OpenSpiel handles the rules. The action menu offers fold, check/call, half-pot,
pot, and all-in. The engine also accepts other legal raise amounts.

Requires Python 3.11+ on macOS or Linux.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Tests compare payouts, betting order, and raises with PokerKit. Observations
contain the player's cards and public information.

## Self-play

The PPO network combines card embeddings with a GRU over betting history.
It trains from both seats against fixed copies of recent and older models.
The reward is profit at the end of each hand; GAE credits earlier decisions.

```python
import time
import torch
from poker_lab.ppo import PPOTrainer

torch.set_num_threads(1)
trainer = PPOTrainer(seed=0)
trainer.run(time.monotonic() + 60)
print(trainer.counters)
```
