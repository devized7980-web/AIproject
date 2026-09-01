"""Canonical, collision-safe identity helpers for source videos."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

RAW_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}


def output_stem(filename: str | Path) -> str:
    """Return the exact stem used by the processing pipeline for output names."""
    name = Path(filename).name
    return name.rsplit(".", 1)[0] if "." in name else name


def canonical_video_id(filename: str | Path) -> str:
    """Return a stable readable ID with a collision-proof filename digest."""
    name = Path(filename).name
    base = name
    while Path(base).suffix.lower() in RAW_EXTENSIONS:
        base = Path(base).stem
    slug = re.sub(r"[^a-z0-9]+", "_", base.casefold()).strip("_") or "video"
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}"
