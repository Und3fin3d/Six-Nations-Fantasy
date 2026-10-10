from __future__ import annotations

from pipeline.sixn.compare_api_official import norm_key
from pipeline.sources.rugbypass_batch import NAME_TO_SLUG

SLUG_TO_NAME = {v: k for k, v in NAME_TO_SLUG.items()}


def slug_key(slug: str) -> str:
    name = SLUG_TO_NAME.get(slug)
    if name:
        return norm_key(name)
    parts = slug.split("-")
    return f"{parts[0][0]}|{''.join(parts[1:])}" if len(parts) > 1 else slug
