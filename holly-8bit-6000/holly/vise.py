"""Shared viseme constants.

The 15 Oculus visemes. Case is significant and matches the authored layer
names in viseme.psd after the `.png` suffix is stripped.
"""

from __future__ import annotations

VISEMES: tuple[str, ...] = (
    "sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "ih", "oh", "ou",
)

BASE_VISEME = "sil"

VISEME_INDEX: dict[str, int] = {name: i for i, name in enumerate(VISEMES)}


def viseme_weights_to_array(weights: dict[str, float]) -> list[float]:
    """Dense weight vector in canonical order. Unknown names are dropped."""
    out = [0.0] * len(VISEMES)
    for name, value in weights.items():
        idx = VISEME_INDEX.get(name)
        if idx is not None:
            out[idx] = float(value)
    return out
