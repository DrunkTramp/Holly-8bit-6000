"""Per-viseme attack/release envelopes.

Frame-wise classification is noisy: a 10 ms decision flipping between two visemes
is exactly the "popping between images" the plan's rendering rule is trying to
avoid. One-pole envelopes with a fast attack and a slow release make mouth
openings snap shut visually while closures ease out, which matches how speech
actually looks.
"""

from __future__ import annotations

import numpy as np

from .vise import BASE_VISEME, VISEME_INDEX

ATTACK_MS = 30.0
RELEASE_MS = 120.0


def attack_release(
    weights: np.ndarray,
    fps: float,
    attack_ms: float = ATTACK_MS,
    release_ms: float = RELEASE_MS,
) -> np.ndarray:
    """Apply per-viseme one-pole envelopes to a (n_frames, n_visemes) weight matrix."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("expected a (n_frames, n_visemes) weight matrix")

    dt = 1.0 / fps
    attack = 1.0 - np.exp(-dt / (attack_ms / 1000.0))
    release = 1.0 - np.exp(-dt / (release_ms / 1000.0))

    out = np.zeros_like(weights)
    current = np.zeros(weights.shape[1], dtype=np.float64)

    for i in range(weights.shape[0]):
        target = weights[i]
        coef = np.where(target > current, attack, release)
        current = current + (target - current) * coef
        out[i] = current

    return out


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Scale each frame so viseme weights sum to 1, falling back to the base viseme."""
    weights = np.asarray(weights, dtype=np.float64)
    totals = weights.sum(axis=1)
    out = np.array(weights, copy=True)

    live = totals > 1e-6
    out[live] /= totals[live, None]

    base = VISEME_INDEX[BASE_VISEME]
    dead = ~live
    if np.any(dead):
        out[dead] = 0.0
        out[dead, base] = 1.0

    return out
