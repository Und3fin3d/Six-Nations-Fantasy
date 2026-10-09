"""Write-once outputs and hashes for the P3 shadow."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..data import ROOT

DATA = ROOT / "data"
OUT = DATA / "unified" / "v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, content: str | bytes) -> None:
    """Create an immutable artifact; refuse silent retrospective regeneration."""
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "x"
    with path.open(mode) as handle:
        handle.write(content)
