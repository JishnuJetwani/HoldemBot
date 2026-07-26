"""Export or repackage policy weights from a trusted local checkpoint.

Example:
    python -m poker_lab.export artifacts/runs/training/checkpoint.pt artifacts/deployment/policy.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .checkpoint import load_checkpoint, save_checkpoint


def _fingerprint(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def export_checkpoint(input_path, output_path):
    """Copy policy weights without optimizer or replay state.

    Leave the source untouched and reject changes made during the export.
    """
    source = Path(input_path).resolve(strict=True)
    target = Path(output_path).resolve()
    if target.suffix != ".pt":
        raise ValueError("Deployment artifact output must use the .pt extension")
    if source == target or (target.exists() and os.path.samefile(source, target)):
        raise ValueError("Export must not overwrite its source checkpoint")
    before = _fingerprint(source)
    payload = load_checkpoint(source)
    if payload.get("artifact_kind", "training") not in {"training", "deployment"}:
        raise ValueError("Unsupported checkpoint artifact kind")
    algorithm = payload["algorithm"]
    if algorithm not in {"ppo"}:
        raise ValueError(f"Cannot export unsupported algorithm: {algorithm}")
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if before != _fingerprint(source):
        raise RuntimeError("Source checkpoint changed during export; retry after its writer finishes")

    exported = {
        "algorithm": algorithm,
        "artifact_kind": "deployment",
        "game_spec": payload["game_spec"],
        "config": payload.get("config", {}),
        "counters": payload.get("counters", {}),
        "seed": payload.get("seed"), "cpu_seconds": payload.get("cpu_seconds"),
        "source_code_sha256": payload.get('source_code_sha256'),
        "policy_kind": "ppo_last_iterate",
        "source_checkpoint": (payload.get("source_checkpoint")
                              if payload.get("artifact_kind") == "deployment" and payload.get("source_checkpoint")
                              else {"sha256": digest, "size_bytes": before[1]}),
        "source_dependencies": (payload.get("source_dependencies", payload.get("dependencies", {}))
                                if payload.get("artifact_kind") == "deployment"
                                else payload.get("dependencies", {})),
        "source_training_seconds": (payload.get("source_training_seconds")
                                   if payload.get("artifact_kind") == "deployment"
                                   else payload.get("training_seconds")),
        "evidence_level": payload.get("evidence_level", "unrated"),
    }
    if "training_lineage" in payload:
        exported["training_lineage"] = payload["training_lineage"]
    if "model_state" not in payload:
        raise ValueError("Training checkpoint is missing model weights")
    exported["model_state"] = payload["model_state"]
    if "architecture" in payload:
        exported["architecture"] = payload["architecture"]
    return save_checkpoint(target, exported)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Trusted local training checkpoint or deployment artifact")
    parser.add_argument("output", help="Distinct .pt deployment artifact path")
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
