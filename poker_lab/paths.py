"""Resolve explicit checkpoint paths."""
from pathlib import Path


def resolve_path(filename):
    return Path(filename).expanduser().resolve()
