"""Top-level linc.yaml schema.

The yaml file looks like:

    data_dir: ~/.linc
    poll_interval_ms: 100   # outbox dispatcher tick (optional)
    adapters:
      slack:
        bot_token: xoxb-...
        app_token: xapp-...

`adapters` is an open dict: each key must be a registered platform name and
its value is fed verbatim to that adapter's `Config.model_validate(...)`.
Cred validation is therefore the adapter's responsibility, not ours.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .errors import ConfigError


class LincConfig(BaseModel):
    """Process-wide configuration loaded from linc.yaml."""

    data_dir: Path = Field(default_factory=lambda: Path.home() / ".linc")
    poll_interval_ms: int = 100
    adapters: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LincConfig":
        p = Path(path).expanduser()
        if not p.exists():
            raise ConfigError(f"linc.yaml not found at {p}")
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"failed to parse {p}: {e}") from e
        if not isinstance(raw, dict):
            raise ConfigError(f"{p}: top-level must be a mapping, got {type(raw).__name__}")
        # Expand ~ in data_dir if present
        if "data_dir" in raw and isinstance(raw["data_dir"], str):
            raw["data_dir"] = os.path.expanduser(raw["data_dir"])
        try:
            return cls.model_validate(raw)
        except Exception as e:
            raise ConfigError(f"{p}: invalid configuration: {e}") from e
