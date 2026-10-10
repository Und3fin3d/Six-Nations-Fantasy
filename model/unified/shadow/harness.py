from __future__ import annotations

import hashlib
import os

OUT = "data/unified/v3"


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: str, content: str | bytes) -> None:
    if os.path.exists(path):
        raise FileExistsError(f"immutable artifact already exists: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "x"
    with open(path, mode) as handle:
        handle.write(content)
