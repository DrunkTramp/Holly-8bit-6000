"""Persistent UI settings: what the settings panel last held, restored at the next launch.

One small JSON file beside the calibration config (same per-machine, gitignored logic). The
panel saves it whenever the backends load and again when the window closes, so the next run
opens with the voice/brain choices exactly where the user left them.

Precedence everywhere is explicit CLI flag > saved settings > built-in default. The file can
hold an API key in plain text -- which is why it sits beside a gitignored sibling and not
inside tracked files.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Read at launch, written by the settings panel. Override with --settings.
DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "hollyhost_settings.json"


def load_settings(path: str | Path | None = None) -> dict:
    """Read the settings file, or {} when absent/unreadable (a normal run then uses defaults)."""
    path = Path(path) if path else DEFAULT_SETTINGS_PATH
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(settings: dict, path: str | Path | None = None) -> Path:
    path = Path(path) if path else DEFAULT_SETTINGS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
    return path
