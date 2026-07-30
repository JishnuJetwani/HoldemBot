import copy
import hashlib
import json

import numpy as np

from poker_lab.slumbot import ENDPOINTS, SLUMBOT_SPEC
from poker_lab.slumbot_benchmark import _validate_response, run_benchmark


class Clock:
    def __init__(self):
        self.now = 0.0
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


class Agent:
    def __init__(self, slot=4):
        self.slot = slot
        self.seeds = []
    def reset(self, seed):
        self.seeds.append(seed)
    def probabilities(self, observation):
        p = np.zeros(5)
        p[self.slot] = 1
        return p
    def act(self, observation):
        return self.slot


def checkpoint(tmp_path):
    path = tmp_path / "frozen.pt"
    path.write_bytes(b"mock-frozen-policy")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps({"sha256": sha, "game_spec": SLUMBOT_SPEC.to_dict(),
                                                    "algorithm": "ppo", "artifact_kind": "deployment"}))
    return path, sha


def response(**updates):
    return {"client_pos": 0, "hole_cards": ["Ac", "9d"], "board": [], "action": "b333",
            "token": "secret-token-A", **updates}


def play(tmp_path, responses, *, agent=None, transport_hook=None):
    path, sha = checkpoint(tmp_path)
    output = tmp_path / "session.json"
    clock, calls = Clock(), []
    pending = iter(responses)
    def transport(endpoint, payload):
        calls.append((endpoint, copy.deepcopy(payload), clock()))
        if transport_hook:
            transport_hook(output, calls)
        reply = next(pending)
        if isinstance(reply, BaseException):
            raise reply
        return copy.deepcopy(reply)
    agent = agent or Agent()
    report = run_benchmark(path, expected_sha256=sha, num_hands=2, output=output,
                           transport=transport, agent_loader=lambda _: agent,
                           clock=clock, sleep=clock.sleep, min_interval_seconds=.25)
    return report, calls, output, agent


def test_zero_terminal_and_server_payout_are_authoritative(tmp_path):
    board = ["2c", "3d", "8h", "Ts", "Jh"]
    replies = [response(action="b20000"), response(action="b20000c///", board=board, winnings=0),
               response(client_pos=1, action=""), response(client_pos=1, action="cb20000"),
               response(client_pos=1, action="cb20000c///", board=board, winnings=20000)]
    report, calls, _, _ = play(tmp_path, replies, agent=Agent(slot=1))
    assert report["status"] == "complete"
    assert [h["winnings_chips"] for h in report["raw_hands"]] == [0, 20000]
    assert len(calls) == 5
    assert report["statistics"]["bb_per_100"] == 10000
    # The unknown placeholder cards would award a different result. The server
    # knows the actual opponent cards, so its valid numeric payout is the score.
    terminal = _validate_response(replies[-1])
    assert terminal.returns()[1] != 20000


def test_ambiguous_action_failure_is_not_replayed_or_scored(tmp_path):
    report, calls, output, _ = play(tmp_path, [response(), TimeoutError("secret-token-A")])
    assert len(calls) == 2 and calls[-1][0] == ENDPOINTS["act"]
    assert report["status"] == "incomplete"
    assert report["statistics"]["completed_hands"] == 0
    assert report["statistics"]["bb_per_100"] is None
    assert report["incomplete_hand"]["stage"] == "act"
    assert "secret-token" not in output.read_text()
