"""Small, dependency-free loader for local development environment files."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> None:
    """Load simple ``KEY=VALUE`` entries without overriding the process environment."""

    source = Path(path)
    if not source.is_file():
        return
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key.replace("_", "").isalnum():
            os.environ.setdefault(key, value)
