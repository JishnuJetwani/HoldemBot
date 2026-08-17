"""Hold'em play API with private game state stored in SQLite."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import sqlite3
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from .game import Action, GameSpec, HoldemState

ROOT = Path(__file__).resolve().parents[1]
class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint_id: str = "diagnostic:random"
    human_seat: int = Field(default=1, ge=0, le=1)


class ActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=0)
    kind: str
    amount: int | None = Field(default=None, ge=0, strict=True)


class NextHandRequest(BaseModel):
    revision: int = Field(ge=0)


def _now():
    return datetime.now(timezone.utc).isoformat()


def create_app(db_path=None, artifacts_dir=None):
    artifacts = Path(artifacts_dir or os.environ.get("POKER_ARTIFACTS_DIR", ROOT / "artifacts"))
    database = Path(db_path or os.environ.get("POKER_DEMO_DB", artifacts / "demo.sqlite3"))
    database.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.RLock()
    deployed_cache = {}
    app = FastAPI(title="HoldemBot", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
    )

    def connect():
        return sqlite3.connect(database)

    with connect() as con:
        con.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, updated TEXT NOT NULL, body TEXT NOT NULL)")

    def persist(session):
        session["updated"] = _now()
        with connect() as con:
            con.execute("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)",
                        (session["id"], session["updated"], json.dumps(session)))

    def fetch(session_id):
        with connect() as con:
            row = con.execute("SELECT body FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Session not found")
        return json.loads(row[0])

    def checkpoints():
        # List models from their manifests without loading weights.
        from .checkpoint import list_checkpoints
        return list_checkpoints(str(artifacts))

    def public_checkpoint(value):
        fields = ("id", "name", "algorithm", "sha256", "game_spec", "artifact_kind",
                  "counters", "training_lineage", "size_bytes", "label", "status")
        return {key: value[key] for key in fields if key in value}

    def recommendation(entries):
        chosen = next((item for item in entries if item["id"] == "diagnostic:random"), None)
        return {**public_checkpoint(chosen), "label": "Holdem PPO"} if chosen else None

    def resolve_checkpoint(checkpoint_id):
        if checkpoint_id in ("diagnostic:random", "diagnostic:equity"):
            return {"id": checkpoint_id, "algorithm": "diagnostic", "status": "untrained",
                    "label": "Uniform random" if checkpoint_id.endswith("random") else "Card-strength heuristic",
                    "game_spec": GameSpec().to_dict()}
        for item in checkpoints():
            if item.get("id") == checkpoint_id:
                return item
        raise HTTPException(404, "Checkpoint not found; refresh the checkpoint list")

    def make_agent(session, hand):
        from .agents import RandomAgent, HeuristicAgent, load_agent
        selected = session["checkpoint"]
        if selected["id"] == "diagnostic:random":
            agent = RandomAgent()
        elif selected["id"] == "diagnostic:equity":
            agent = HeuristicAgent(style="equity")
        else:
            current = resolve_checkpoint(selected["id"])
            if current.get("sha256") != selected.get("sha256"):
                raise HTTPException(409, "This checkpoint changed. Start a new session with the current checkpoint.")
            try:
                key = (current["path"], current["sha256"])
                if key not in deployed_cache:
                    # Clones share frozen weights but keep separate random streams.
                    deployed_cache.clear()
                    deployed_cache[key] = load_agent(current["path"], device="cpu")
                agent = deployed_cache[key].clone()
            except Exception as exc:
                raise HTTPException(503, "The selected checkpoint could not be loaded") from exc
        agent.reset(hand["bot_seed"])
        return agent

    def replay_for_agent(session, hand):
        """Replay saved actions to restore the policy RNG."""
        agent = make_agent(session, hand)
        state = HoldemState.new(hand["seed"], GameSpec(**session["game_spec"]))
        for event in hand["events"]:
            if event["actor"] != session["human_seat"]:
                slot = agent.act(state.observe())
                if int(slot) != event["slot"]:
                    raise HTTPException(409, "Saved agent replay differs from this runtime; start a new session")
            state.apply_action(Action(event["kind"], event.get("amount")))
        return state, agent

    def append_event(session, hand, state, action, actor, slot=None, probabilities=None):
        menu = state.public_view(session["human_seat"]).get("legal_actions", [])
        event = {"actor": actor, "kind": action.kind, "amount": action.amount, "slot": slot,
                 "street": state.public_view(session["human_seat"])["street"],
                 "probabilities": probabilities, "action_menu": menu if probabilities is not None else None}
        state.apply_action(action)
        event["check"] = state.history[-1]["check"]
        event["state"] = state.public_view(session["human_seat"])
        hand["events"].append(event)
        hand["state"] = state.to_dict()

    def run_bot(session, hand, state, agent):
        import numpy as np
        for _ in range(64):
            if state.is_terminal() or state.current_player() == session["human_seat"]:
                return
            obs = state.observe()
            probs = np.asarray(agent.probabilities(obs), dtype=float)
            mask = np.asarray(state.legal_mask(), dtype=bool)
            if (probs.shape != (5,) or not np.all(np.isfinite(probs)) or np.any(probs < 0)
                    or np.any(probs[~mask] > 1e-8) or not np.isclose(probs.sum(), 1)):
                raise HTTPException(503, "Agent produced an invalid action distribution")
            slot = int(agent.act(obs))
            if slot not in range(5) or not mask[slot]:
                raise HTTPException(503, "Agent produced an illegal action")
            action = state.action_menu()[slot]
            append_event(session, hand, state, action, state.current_player(), slot, probs.tolist())
        raise HTTPException(503, "Hand exceeded the automatic action limit")

    def add_hand(session):
        hand = {"id": str(uuid.uuid4()), "seed": secrets.randbelow(2**31),
                "bot_seed": secrets.randbelow(2**31), "events": []}
        state = HoldemState.new(hand["seed"], GameSpec(**session["game_spec"]))
        hand["initial"] = state.public_view(session["human_seat"])
        hand["state"] = state.to_dict()
        agent = make_agent(session, hand)
        run_bot(session, hand, state, agent)
        session["hands"].append(hand)

    def public_session(session):
        hand = session["hands"][-1]
        state = HoldemState.from_dict(hand["state"])
        return {"session_id": session["id"], "revision": session["revision"], "hand_id": hand["id"],
                "hand_number": len(session["hands"]), "human_seat": session["human_seat"],
                "checkpoint": public_checkpoint(session["checkpoint"]), "state": state.public_view(session["human_seat"]),
                "events": hand["events"], "updated": session["updated"],
                "probability_note": "Bot probabilities are conditional on the policy selected for this hand; they are not confidence or win probability."}

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "storage": "sqlite", "game": "heads-up-no-limit-holdem"}

    @app.get("/api/v1/checkpoints")
    def checkpoint_registry():
        entries = checkpoints()
        return {"checkpoints": [public_checkpoint(item) for item in entries],
                "diagnostics": [resolve_checkpoint("diagnostic:random"), resolve_checkpoint("diagnostic:equity")],
                "recommended_checkpoint": recommendation(entries)}

    @app.get("/api/v1/holdem/sessions")
    def session_list():
        with connect() as con:
            rows = con.execute("SELECT body FROM sessions ORDER BY updated DESC LIMIT 30").fetchall()
        items = []
        for row in rows:
            s = json.loads(row[0])
            items.append({"session_id": s["id"], "updated": s["updated"], "human_seat": s["human_seat"],
                          "checkpoint_id": s["checkpoint"]["id"], "hands": len(s["hands"])})
        return {"sessions": items}

    @app.post("/api/v1/holdem/sessions", status_code=201)
    def new_session(body: SessionRequest):
        with lock:
            chosen = resolve_checkpoint(body.checkpoint_id)
            session = {"id": str(uuid.uuid4()), "revision": 0, "human_seat": body.human_seat,
                       "checkpoint": chosen, "game_spec": chosen.get("game_spec", GameSpec().to_dict()), "hands": []}
            add_hand(session)
            persist(session)
            return public_session(session)

    @app.get("/api/v1/holdem/sessions/{session_id}")
    def get_session(session_id: str):
        return public_session(fetch(session_id))

    @app.post("/api/v1/holdem/sessions/{session_id}/act")
    def act(session_id: str, body: ActionRequest):
        with lock:
            session = fetch(session_id)
            if body.revision != session["revision"]:
                raise HTTPException(409, "This hand changed; reload the current session")
            hand = session["hands"][-1]
            state = HoldemState.from_dict(hand["state"])
            if state.is_terminal() or state.current_player() != session["human_seat"]:
                raise HTTPException(409, "It is not your turn")
            if body.kind not in ("fold", "call", "raise_to") or (body.kind != "raise_to" and body.amount is not None):
                raise HTTPException(422, "Use fold, call, or raise_to with a total street contribution")
            state, agent = replay_for_agent(session, hand)
            try:
                append_event(session, hand, state, Action(body.kind, body.amount), session["human_seat"])
            except (ValueError, TypeError, AssertionError) as exc:
                raise HTTPException(422, str(exc)) from exc
            run_bot(session, hand, state, agent)
            session["revision"] += 1
            persist(session)
            return public_session(session)

    @app.post("/api/v1/holdem/sessions/{session_id}/new-hand")
    def new_hand(session_id: str, body: NextHandRequest):
        with lock:
            session = fetch(session_id)
            if body.revision != session["revision"]:
                raise HTTPException(409, "This hand changed; reload the current session")
            if not HoldemState.from_dict(session["hands"][-1]["state"]).is_terminal():
                raise HTTPException(409, "Finish the current hand first")
            add_hand(session)
            session["revision"] += 1
            persist(session)
            return public_session(session)

    @app.get("/api/v1/holdem/sessions/{session_id}/hands")
    def hand_history(session_id: str):
        session = fetch(session_id)
        return {"hands": [{"hand_id": h["id"], "number": i + 1, "initial": h["initial"],
                           "events": h["events"], "terminal": HoldemState.from_dict(h["state"]).is_terminal()}
                          for i, h in enumerate(session["hands"])]}

    return app


app = create_app()
