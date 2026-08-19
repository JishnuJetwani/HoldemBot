# Included policy

`artifacts/deployment/holdem-ppo.pt` contains the bot's weights after three rounds
of adversarial training. The matching JSON file records its hash, settings,
dependency versions, and training counts.

## Network and observations

The network has 227,030 parameters. It combines 16-dimensional card embeddings
and 52 hand and board features with a 128-unit GRU over betting history. Two
256-unit dense layers feed the action and value heads.

Card positions distinguish hole cards, flop, turn, and river. Equivalent suit
patterns share a representation. Features include hand rank, draws, blockers,
board texture, stacks, pot, call amount, and legal actions.

The bot sees no opponent cards or future cards. It gets no opponent range or
calculated win odds. Action probabilities describe how often it chooses each move.

## Training

| Phase | Completed hands | Opponents |
| --- | ---: | --- |
| Initial PPO | 4,231,766 | 50% latest self-play, 35% historical self-play, 10% frozen neural policies, 5% heuristics |
| Defender post-training | 901,961 | 50% latest self-play, 20% historical self-play, 30% trained exploiters |
| Total policy training | 5,133,727 | Excludes attacker training and evaluation |

Post-training ran for three rounds, adding one attacker per round and keeping
earlier attackers. Each hand picks one opponent. Copies of the bot refresh
between PPO updates; a fixed-size pool keeps older versions.

A hand's reward is its net profit in big blinds multiplied by 0.005. PPO uses gamma 1,
GAE lambda 0.95, clipping 0.2, entropy coefficient 0.01, and a KL limit of 0.01.
Stacks reset each hand. Use the deployment file for play or to start a new run
from its weights. Resuming an existing run requires its full training checkpoint.

## Evaluation

Each matchup used 5,000 deals played from both seats, for 10,000 hands. The table
shows the included bot's returns in bb/100. Confidence intervals reflect deal and
action randomness for one training seed.

| Opponent | Hands | bb/100 | 95% interval |
| --- | ---: | ---: | --- |
| Initial PPO policy | 10,000 | -39.13 | [-75.68, -2.57] |
| Training exploiter 1 | 10,000 | +237.92 | [+171.71, +304.12] |
| Training exploiter 2 | 10,000 | +530.76 | [+448.13, +613.38] |
| Training exploiter 3 | 10,000 | +131.37 | [+96.97, +165.76] |
| Fresh attacker trained against initial PPO | 10,000 | +139.09 | [+73.71, +204.46] |
| Fresh attacker trained against final PPO | 10,000 | -448.89 | [-523.81, -373.97] |

Two control matchups tested the fresh attackers against initial PPO, bringing
the total to 80,000 hands. The fresh attackers were not used in training or model
selection. All eight results are in [evaluation.json](evaluation.json).

The bot beat its three training attackers but lost to a new one. These results
do not establish competitive poker strength or measure whole-game exploitability.
We did not compare this with continued self-play using the same training budget.
