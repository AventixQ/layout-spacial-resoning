"""Environment and runtime configuration helpers."""

import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs into os.environ if they are not set already."""
    env_path = Path(path)
    if not env_path.exists():
        return

    with env_path.open(encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = _strip_quotes(value.strip())
            if key and key not in os.environ:
                os.environ[key] = value


def env_str(name: str, default: str | None = None) -> str | None:
    load_env()
    return os.environ.get(name, default)


def env_float(name: str, default: float) -> float:
    value = env_str(name)
    if value is None or value == "":
        return default
    return float(value)


def env_int(name: str, default: int) -> int:
    value = env_str(name)
    if value is None or value == "":
        return default
    return int(value)


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
