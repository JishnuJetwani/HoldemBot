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
