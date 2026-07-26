"""Resumable PPO training with bounded wall time."""
from __future__ import annotations

import argparse
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoint import capture_rng, load_checkpoint, restore_rng, save_checkpoint
from .ppo import PPOTrainer


class ResourceLimit(Exception):
    """A configured local resource bound was reached."""


class ProgressMonitor:
    def __init__(self, trainer, output, start, requested_seconds):
        self.trainer, self.start, self.requested_seconds = trainer, start, requested_seconds
        self.path = Path(output).resolve().with_suffix(".progress.json")
        self.artifact_root = next((p for p in self.path.parents if p.name == "artifacts"), self.path.parent)
        self.last_write = self.last_check = float("-inf")
        self.rss_bytes, self.artifact_bytes = 0, 0

    def __call__(self, force=False, status="running"):
        now = time.monotonic()
        if force or now - self.last_check >= 1.:
            self.last_check = now
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.rss_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
            self.artifact_bytes = 0
            for path in self.artifact_root.rglob("*") if self.artifact_root.exists() else []:
                try:
                    if path.is_file():
                        self.artifact_bytes += path.stat().st_size
                except FileNotFoundError:
                    pass  # An atomic writer may have renamed a temporary file.
            if status == "running" and self.rss_bytes > 16 * 1024**3:
                raise ResourceLimit("rss_limit_16_gib")
            if status == "running" and self.artifact_bytes > 10 * 1024**3:
                raise ResourceLimit("artifact_limit_10_gib")
        if force or now - self.last_write >= 15.:
            self.last_write = now
            report = {"algorithm": "ppo",
                      "status": status, "elapsed_seconds": now - self.start,
                      "requested_seconds": self.requested_seconds,
                      "counters": self.trainer.counters, "progress": self.trainer.progress,
                      "latest_loss": self.trainer.losses[-1] if self.trainer.losses else None,
                      "peak_rss_bytes": self.rss_bytes, "artifact_bytes": self.artifact_bytes}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            os.replace(temporary, self.path)


def run_training(seconds, seed, output, resume=None, **overrides):
    whole_start = time.monotonic()
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    torch.set_num_threads(int(overrides.pop("threads", 1)))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    payload = load_checkpoint(resume) if resume else None
    if payload and payload["algorithm"] != "ppo":
        raise ValueError("Checkpoint algorithm does not match the requested trainer")
    config = dict(payload["config"] if payload else {}, **overrides)
    trainer = PPOTrainer(seed, config=config, payload=payload)
    if payload:
        restore_rng(payload["rng_state"])
    start = time.monotonic()
    # Reserve time for an atomic checkpoint. A single optimizer kernel and disk
    # flush cannot be interrupted mid-operation, so report actual elapsed too.
    save_reserve = min(5., float(seconds) * .1)
    deadline = whole_start + float(seconds) - save_reserve
    monitor = ProgressMonitor(trainer, output, whole_start, seconds)
    reason = "time_budget"
    try:
        reason = trainer.run(deadline, progress_callback=monitor)
    except KeyboardInterrupt:
        reason = "interrupted"
    except ResourceLimit as error:
        reason = str(error)
    elapsed = time.monotonic() - start
    result = trainer.payload()
    result["rng_state"] = capture_rng()
    result["seed"] = payload.get("seed", seed) if payload else seed
    result["training_seconds"] = float(payload.get("training_seconds", 0) if payload else 0) + elapsed
    result["stop_reason"] = reason
    manifest = save_checkpoint(output, result)
    try:
        monitor(force=True, status=reason)
    except ResourceLimit:
        # A guard may have caused this checkpoint; don't discard that outcome.
        pass
    return {"algorithm": "ppo", "elapsed": time.monotonic() - whole_start,
            "training_elapsed": elapsed, "requested_seconds": seconds,
            "total_training_seconds": result["training_seconds"], "counters": trainer.counters,
            "completed_iterations": trainer.counters["iterations"], "partial": trainer.progress,
            "checkpoint": manifest, "stop_reason": reason}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    overrides = {"threads": args.threads}
    if args.device is not None:
        overrides["device"] = args.device
    summary = run_training(args.seconds, args.seed, args.output, args.resume, **overrides)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
