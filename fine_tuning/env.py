from __future__ import annotations

import os
from pathlib import Path

from .constants import PROJECT_ROOT


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_project_env(*, override: bool = False) -> Path | None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None

    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(
                f"Malformed .env line {line_number} in {env_path}: missing '='."
            )

        key, value = line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError(
                f"Malformed .env line {line_number} in {env_path}: empty key."
            )

        if normalized_key in os.environ and not override:
            continue

        os.environ[normalized_key] = _strip_wrapping_quotes(value.strip())

    return env_path
