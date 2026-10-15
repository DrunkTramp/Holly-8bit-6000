"""Viseme timeline format.

One record per analysis frame: the dominant viseme, its weight, and the runner-up
that the renderer will cross-fade against. That is the whole contract between the
analysis side and the drawing side.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .vise import VISEMES, VISEME_INDEX

FORMAT = "holly-viseme-timeline/1"


def build_timeline(
    weights: np.ndarray,
    times: np.ndarray,
    fps: float,
    weight_precision: int = 3,
) -> dict:
    """Turn a (n_frames, 15) weight matrix into a timeline document."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("expected a (n_frames, n_visemes) weight matrix")
    if weights.shape[0] != times.shape[0]:
        raise ValueError(f"frame count mismatch: {weights.shape[0]} weights vs {times.shape[0]} times")

    frames = []
    for i in range(weights.shape[0]):
        row = weights[i]
        order = np.argsort(row)[::-1]
        top = int(order[0])
        runner = int(order[1])

        alt = {}
        for j in order[1:]:
            if row[j] >= 0.02:
                alt[VISEMES[int(j)]] = round(float(row[j]), weight_precision)

        frames.append(
            {
                "t": round(float(times[i]), 4),
                "viseme": VISEMES[top],
                "weight": round(float(row[top]), weight_precision),
                "alt": alt,
            }
        )

    return {
        "format": FORMAT,
        "fps": round(float(fps), 6),
        "visemes": list(VISEMES),
        "frame_count": weights.shape[0],
        "duration_s": round(float(times[-1]) if times.size else 0.0, 4),
        "frames": frames,
    }


def write_timeline(timeline: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timeline, indent=2) + "\n")


def read_timeline(path: str | Path) -> dict:
    timeline = json.loads(Path(path).read_text())
    if timeline.get("format") != FORMAT:
        raise ValueError(f"unsupported timeline format: {timeline.get('format')!r}")
    return timeline


def timeline_weights(timeline: dict) -> np.ndarray:
    """Reconstruct the dense (n_frames, 15) weight matrix from a timeline document."""
    frames = timeline["frames"]
    out = np.zeros((len(frames), len(VISEMES)), dtype=np.float64)
    for i, frame in enumerate(frames):
        idx = VISEME_INDEX.get(frame["viseme"])
        if idx is None:
            raise ValueError(f"unknown viseme in timeline: {frame['viseme']!r}")
        out[i, idx] = float(frame["weight"])
        for name, value in frame.get("alt", {}).items():
            idx = VISEME_INDEX.get(name)
            if idx is None:
                raise ValueError(f"unknown viseme in timeline: {name!r}")
            out[i, idx] = float(value)
    return out


def sample_frames(timeline: dict, count: int) -> list[dict]:
    """Evenly spaced frames across the timeline, for contact sheets."""
    frames = timeline["frames"]
    if not frames or count >= len(frames):
        return list(frames)
    step = (len(frames) - 1) / (count - 1) if count > 1 else 0
    return [frames[min(int(round(i * step)), len(frames) - 1)] for i in range(count)]
