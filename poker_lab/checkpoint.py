"""Save and load trusted local checkpoints."""
from __future__ import annotations

import hashlib
import fcntl
import importlib.metadata
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 1


def capture_rng():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def dependency_versions():
    names = ('torch', 'numpy', 'open_spiel', 'pokerkit')
    return {"python": platform.python_version(), **{n: importlib.metadata.version(n) for n in names}}


def save_checkpoint(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {**payload, "format_version": FORMAT_VERSION, "dependencies": dependency_versions()}
    if "rng_state" not in content and content.get("artifact_kind") != "deployment":
        content["rng_state"] = capture_rng()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(content, temporary)
    artifact_root = next((p for p in path.parents if p.name == "artifacts"), None)
    # Lock checkpoint writes. Other writers may rename files during the size scan.
    with (artifact_root or path.parent).joinpath('.checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if artifact_root is not None:
            existing = 0
            for item in artifact_root.rglob('*'):
                try:
                    if item.is_file() and item not in (path, temporary):
                        existing += item.stat().st_size
                except FileNotFoundError:
                    pass
            if existing + temporary.stat().st_size > 10 * 1024**3:
                temporary.unlink()
                raise RuntimeError("10GiB artifact cap exceeded; previous checkpoint retained")
        with temporary.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(temporary, path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {"id": path.stem, "algorithm": content["algorithm"], "path": path.name,
                "format_version": FORMAT_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
                "size_bytes": path.stat().st_size, "sha256": digest,
                "game_spec": content["game_spec"], "config": content.get("config", {}),
                "counters": content.get("counters", {}), "dependencies": content["dependencies"],
                "artifact_kind": content.get("artifact_kind", "training"),
                "source_checkpoint": content.get("source_checkpoint"),
                "source_training_seconds": content.get("source_training_seconds"),
                "seed": content.get("seed"), "training_seconds": content.get("training_seconds"),
                "cpu_seconds": content.get("cpu_seconds"),
                "source_code_sha256": content.get('source_code_sha256'),
                "evidence_level": content.get('evidence_level', "unrated"), "action_abstraction": ["fold", "check/call", "half-pot", "pot", "all-in"]}
    manifest_path = path.with_suffix(".json")
    temporary_json = manifest_path.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary_json, manifest_path)
    return manifest


def load_checkpoint(path):
    from .paths import resolve_path
    path = resolve_path(path)
    manifest_path = path.with_suffix('.json')
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != manifest.get('sha256'):
            raise ValueError('Checkpoint hash differs from its manifest')
    content = torch.load(path, map_location="cpu", weights_only=False)
    if content.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format")
    return content


def list_checkpoints(root="artifacts"):
    entries = []
    for path in sorted(Path(root).rglob("*.json")):
        try:
            weight_path = path.with_suffix(".pt")
            if not weight_path.exists():
                continue
            item = json.loads(path.read_text())
            if isinstance(item, dict) and item.get("format_version") == FORMAT_VERSION and "sha256" in item and "algorithm" in item:
                item["name"] = item["id"]
                item["id"] = str(path.relative_to(Path(root)).with_suffix(""))
                item["path"] = str(weight_path.resolve())
                entries.append(item)
        except (ValueError, OSError):
            continue
    return entries
