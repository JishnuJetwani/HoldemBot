import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from poker_lab.server import create_app


BASE = "/api/v1/holdem/sessions"


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Fixed deals and independent fixed policy seed make replay assertions stable.
    monkeypatch.setattr("poker_lab.server.secrets.randbelow", lambda _: 123)
    return TestClient(create_app(tmp_path / "sessions.sqlite3", tmp_path / "artifacts"))


def new(client, **kwargs):
    response = client.post(BASE, json={"checkpoint_id": "diagnostic:random", **kwargs})
    assert response.status_code == 201, response.text
    return response.json()


def act(client, session, kind, amount=None):
    body = {"revision": session["revision"], "kind": kind}
    if amount is not None:
        body["amount"] = amount
    return client.post(f"{BASE}/{session['session_id']}/act", json=body)


def assert_public(value):
    if isinstance(value, dict):
        assert not {"deck", "seed", "bot_seed", "rng_state"}.intersection(value)
        if "opponent_cards" in value:
            if not value["terminal"] or value["history"][-1]["kind"] == "fold":
                assert value["opponent_cards"] == []
            assert len(value["board"]) <= [0, 3, 4, 5][value["street"]]
        for child in value.values():
            assert_public(child)
    elif isinstance(value, list):
        for child in value:
            assert_public(child)


def test_hidden_state_stays_server_side_and_arbitrary_raise(client):
    session = new(client)
    assert len(session["state"]["hole_cards"]) == 2
    assert session["state"]["board"] == []
    assert_public(session)
    # 333 is intentionally outside the five-slot neural action menu.
    assert 333 not in [a["amount"] for a in session["state"]["legal_actions"]]
    response = act(client, session, "raise_to", 333)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["events"][0]["kind"] == "raise_to"
    assert result["events"][0]["amount"] == 333
    assert_public(result)
    for event in result["events"]:
        if event["probabilities"] is not None:
            probabilities = np.array(event["probabilities"])
            assert probabilities.sum() == pytest.approx(1)
            legal = [a["slot"] for a in event["action_menu"]]
            assert event["slot"] in legal
            assert all(p == 0 for i, p in enumerate(probabilities) if i not in legal)
    assert_public(client.get(f"{BASE}/{session['session_id']}/hands").json())


def test_invalid_and_stale_actions_do_not_commit(client):
    session = new(client)
    assert act(client, session, "raise_to", 101).status_code == 422
    assert act(client, session, "call", 100).status_code == 422
    assert act(client, session, "raise_to", 222.5).status_code == 422
    assert client.get(f"{BASE}/{session['session_id']}").json() == session
    success = act(client, session, "call")
    assert success.status_code == 200
    assert act(client, session, "call").status_code == 409


def test_terminal_and_new_hand_keep_history_private(client):
    session = new(client)
    path = f"{BASE}/{session['session_id']}"
    assert client.post(path + "/new-hand", json={"revision": 0}).status_code == 409
    terminal = act(client, session, "fold").json()
    assert terminal["state"]["terminal"]
    assert terminal["state"]["stacks"] == [20050, 19950]
    assert terminal["state"]["opponent_cards"] == []
    assert act(client, terminal, "call").status_code == 409
    response = client.post(path + "/new-hand", json={"revision": terminal["revision"]})
    assert response.status_code == 200
    assert response.json()["hand_number"] == 2
    history = client.get(path + "/hands").json()
    assert len(history["hands"]) == 2
    assert history["hands"][0]["terminal"]
    assert_public(history)


def test_session_and_agent_rng_survive_process_restart(tmp_path, monkeypatch):
    monkeypatch.setattr("poker_lab.server.secrets.randbelow", lambda _: 54321)
    first = TestClient(create_app(tmp_path / "sessions.sqlite3", tmp_path / "artifacts"))
    session = new(first)
    session = act(first, session, "call").json()
    second = TestClient(create_app(tmp_path / "sessions.sqlite3", tmp_path / "artifacts"))
    assert second.get(f"{BASE}/{session['session_id']}").json() == session
    if not session["state"]["terminal"]:
        response = act(second, session, "call")
        assert response.status_code == 200, response.text
        assert response.json()["revision"] == session["revision"] + 1
        assert_public(response.json())
    saved = second.get(BASE).json()["sessions"]
    assert saved[0]["checkpoint_id"] == "diagnostic:random"


def test_checkpoint_listing_reads_manifests_without_loading_weights(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "policy.pt").write_text("Not a torch checkpoint")
    (artifacts / "policy.json").write_text(json.dumps({"format_version": 1, "algorithm": "ppo", "id": "policy", "sha256": "example", "game_spec": {"stack": 2000, "small_blind": 50, "big_blind": 100}}))
    monkeypatch.setattr("torch.load", lambda *a, **kw: pytest.fail("Registry must not load weights"))
    client = TestClient(create_app(tmp_path / "sessions.sqlite3", artifacts))
    response = client.get("/api/v1/checkpoints")
    assert response.status_code == 200
    assert response.json()["checkpoints"][0]["id"] == "policy"
    assert response.json()["diagnostics"][1]["label"] == "Card-strength heuristic"


def test_selected_policy_loads_once_and_uses_its_game_spec(tmp_path, monkeypatch):
    from poker_lab.agents import RandomAgent
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    manifest_path = artifacts / "small-stack.json"
    (artifacts / "small-stack.pt").write_text("Weights are replaced by the test loader")
    manifest = {"format_version": 1, "algorithm": "ppo", "id": "small-stack", "sha256": "original",
                "game_spec": {"stack": 2000, "small_blind": 50, "big_blind": 100}}
    manifest_path.write_text(json.dumps(manifest))
    loads = []

    def fake_load(path, device):
        loads.append(path)
        return RandomAgent()

    monkeypatch.setattr("poker_lab.agents.load_agent", fake_load)
    client = TestClient(create_app(tmp_path / "db.sqlite3", artifacts))
    first = new(client, checkpoint_id="small-stack")
    second = new(client, checkpoint_id="small-stack")
    assert first["state"]["game_spec"]["stack"] == 2000
    assert second["checkpoint"]["sha256"] == "original"
    assert len(loads) == 1
    manifest_path.write_text(json.dumps({**manifest, "sha256": "changed"}))
    assert act(client, first, "call").status_code == 409


def test_unknown_session_and_checkpoint_are_404(client):
    assert client.get(BASE + "/missing").status_code == 404
    assert client.post(BASE, json={"checkpoint_id": "not-a-checkpoint"}).status_code == 404
