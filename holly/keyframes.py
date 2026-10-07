"""Analysis-rate weights -> cartoon keyframes.

The continuous pipeline (classify -> attack/release -> one record per analysis
frame) aims for frame-accurate A/V alignment. Measured on `test_audio.flac` that
yields ~10 top-viseme switches per second with 16 of 79 runs lasting a single
16 ms frame, which reads as flickering rather than talking.

This module trades timing accuracy for legibility, which is how cartoon lip sync
actually works: the mouth is a *pose*, held long enough to read, changed only a
few times a second, with a short cross-fade so the change is not a hard cut.

    analysis weights -> pool_keys() -> [Keyframe] -> expand_keys() -> render weights

`expand_keys` emits the same dense (n_frames, 15) weight matrix the rest of the
pipeline already consumes, so `holly/timeline.py` and `holly/render.py` are
unchanged: keyframing is a re-timing transform inserted before the timeline, not
a new output format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import groupby, pairwise

import numpy as np

from .vise import BASE_VISEME, VISEMES, VISEME_INDEX

# Pose rate. The ceiling on how often the mouth may change. Measured on
# test_audio.flac, the achieved switch rate is ~57% of this because adjacent slots
# often pick the same pose: 12/s here yields ~7 changes/s. Analysis fps is
# independent -- pooling averages however many analysis frames fall in each slot.
# 12/s is the rate chosen by eye on the debug video; 8/s (~4.5 changes/s) is the
# calmer alternative if a real classifier turns out to be more volatile.
KEY_HZ = 12.0

# Cross-fade length between consecutive poses. Deliberately short: at 30 output
# fps a 50 ms fade spans ~1.5 frames, so the change is visible as a change.
FADE_S = 0.05

# Output frame rate for the keyframed timeline. Keeping the timeline at the
# render rate makes tools/render_filmstrip.py's resampling 1:1 instead of nearest.
RENDER_FPS = 30.0

# Flicker suppression between mouth visemes: a new pose must beat the current
# pose's pooled weight by this much to take over. Not applied to or from the base
# viseme -- opening and closing the mouth should be eager, and a margin there
# would delay mouth opening by a whole key slot.
STICKY_MARGIN = 0.05

FADE_SHAPES = ("center", "lead", "lag")

POOL_MODES = ("mean", "max")


@dataclass(frozen=True)
class Keyframe:
    """One held pose. `t` is the exact start of its slot on the key grid."""

    t: float
    viseme: str
    weight: float  # winner's share of the pooled weight mass for that slot


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def pool_weights(
    weights: np.ndarray,
    analysis_fps: float,
    key_hz: float = KEY_HZ,
    pool: str = POOL_MODES[0],
) -> list[np.ndarray]:
    """Average (or peak) the analysis weights into one vector per key slot.

    Slot boundaries are exact multiples of 1/key_hz in *time*, so the pose rate
    is exactly `key_hz` regardless of the analysis frame rate. Pooling is what
    does the smoothing here: at 12 poses/s and 62.5 analysis fps each slot averages
    ~5 per-frame decisions, so a single noisy frame cannot move the mouth.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("expected a (n_frames, n_visemes) weight matrix")
    if analysis_fps <= 0.0 or key_hz <= 0.0:
        raise ValueError(f"invalid rates: analysis_fps={analysis_fps} key_hz={key_hz}")
    if pool not in POOL_MODES:
        raise ValueError(f"unknown pool mode {pool!r}, expected one of {POOL_MODES}")

    n = weights.shape[0]
    duration = n / analysis_fps
    n_keys = max(1, math.ceil(duration * key_hz - 1e-9))

    pooled = []
    for j in range(n_keys):
        lo = min(max(round(j * analysis_fps / key_hz), 0), n - 1)
        hi = min(max(round((j + 1) * analysis_fps / key_hz), lo + 1), n)
        block = weights[lo:hi]
        row = block.max(axis=0) if pool == "max" else block.mean(axis=0)
        pooled.append(row)
    return pooled


def pool_keys(
    weights: np.ndarray,
    analysis_fps: float,
    key_hz: float = KEY_HZ,
    pool: str = POOL_MODES[0],
    sticky: float = STICKY_MARGIN,
) -> list[Keyframe]:
    """Reduce analysis weights to a held-pose timeline, one Keyframe per key slot."""
    base = VISEME_INDEX[BASE_VISEME]
    keys: list[Keyframe] = []
    prev_idx = -1

    for j, row in enumerate(pool_weights(weights, analysis_fps, key_hz, pool)):
        total = float(row.sum())
        if total <= 1e-12:
            idx, conf = base, 0.0
        else:
            row = row / total
            idx = int(np.argmax(row))
            conf = float(row[idx])

            # Hysteresis only between mouth shapes. sil <-> mouth transitions stay
            # eager so the mouth opens on the slot where speech actually starts.
            mouth_to_mouth = idx != prev_idx and idx != base and prev_idx != base
            if sticky > 0.0 and prev_idx >= 0 and mouth_to_mouth and conf < float(row[prev_idx]) + sticky:
                idx = prev_idx
                conf = float(row[idx])

        keys.append(Keyframe(t=j / key_hz, viseme=VISEMES[idx], weight=conf))
        prev_idx = idx

    return keys


def fade_frames(fade_s: float, out_fps: float) -> int:
    """Whole output frames for the cross-fade.

    A fade shorter than one output frame is not rendered at all, and a fade
    expressed in seconds can miss its midpoint entirely: with a 125 ms key grid
    and a 33.3 ms frame grid, a 50 ms fade sampled at frame boundaries only ever
    produced 0.074 and 0.259 blends, never 0.5. Quantising the fade to frames and
    driving the ramp from frame index makes the blend deterministic.
    """
    return max(1, round(fade_s * out_fps))


def expand_keys(
    keys: list[Keyframe],
    key_hz: float = KEY_HZ,
    out_fps: float = RENDER_FPS,
    fade_s: float = FADE_S,
    shape: str = FADE_SHAPES[0],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Turn keyframes into dense weights sampled at `out_fps`.

    Returns (weights, times, out_fps) ready for `build_timeline`. Exactly two
    visemes are non-zero per frame: the outgoing pose at (1 - u) and the incoming
    pose at u, so the existing top-2 cross-fade renderer draws it unchanged.

    The fade runs over an integer number of output frames, and u is driven by
    frame index rather than by time, so the midpoint is always actually sampled.

    `shape` positions the fade relative to the pose boundary:
      center -- half before the boundary, half after
      lead   -- the new pose is fully reached *before* the boundary (anticipation)
      lag    -- the old pose holds to the boundary, then the fade runs
    """
    if not keys:
        raise ValueError("no keyframes to expand")
    if shape not in FADE_SHAPES:
        raise ValueError(f"unknown fade shape {shape!r}, expected one of {FADE_SHAPES}")
    if key_hz <= 0.0 or out_fps <= 0.0:
        raise ValueError(f"invalid rates: key_hz={key_hz} out_fps={out_fps}")

    period = 1.0 / key_hz
    last_t = keys[-1].t + period
    n_out = max(1, math.ceil(last_t * out_fps - 1e-9))

    # First output frame index of every key slot. Slots are not equal length when
    # out_fps is not a multiple of key_hz, so the grid is derived by rounding each
    # exact boundary time instead of fixed steps.
    bounds = [min(round(j * out_fps / key_hz), n_out) for j in range(len(keys) + 1)]
    for j in range(1, len(bounds)):
        bounds[j] = max(bounds[j], bounds[j - 1] + 1)
    n_out = max(n_out, bounds[-1])

    slot_len = [bounds[j + 1] - bounds[j] for j in range(len(keys))]
    min_slot = min(slot_len)

    # A fade cannot be longer than the two slots it straddles, or it never reaches
    # a held pose and the animation becomes a continuous dissolve.
    frames = min(fade_frames(fade_s, out_fps), 2 * min_slot)
    before = min({"center": frames // 2, "lead": frames, "lag": 0}[shape], min_slot)

    index = VISEME_INDEX
    out = np.zeros((n_out, len(VISEMES)), dtype=np.float64)

    for k in range(n_out):
        j = max((i for i in range(len(keys)) if bounds[i] <= k), default=len(keys) - 1)
        cur = keys[j]

        u = 0.0
        prev = None
        if j > 0:
            p = k - (bounds[j] - before)
            if 0 <= p < frames:
                u, prev = _smoothstep((p + 1) / frames), keys[j - 1]
        if prev is None and j + 1 < len(keys):
            p = k - (bounds[j + 1] - before)
            if 0 <= p < frames:
                u, prev = _smoothstep((p + 1) / frames), cur
                cur = keys[j + 1]

        if prev is None or u <= 0.0 or u >= 1.0 or prev.viseme == cur.viseme:
            out[k, index[cur.viseme]] = 1.0
        else:
            out[k, index[prev.viseme]] = 1.0 - u
            out[k, index[cur.viseme]] = u

    times = np.arange(n_out, dtype=np.float64) / out_fps
    return out, times, out_fps


def keyframe_weights(
    weights: np.ndarray,
    analysis_fps: float,
    key_hz: float = KEY_HZ,
    out_fps: float = RENDER_FPS,
    fade_s: float = FADE_S,
    shape: str = FADE_SHAPES[0],
    pool: str = POOL_MODES[0],
    sticky: float = STICKY_MARGIN,
) -> tuple[list[Keyframe], np.ndarray, np.ndarray, float]:
    """pool_keys + expand_keys in one call. Returns (keys, weights, times, out_fps)."""
    keys = pool_keys(weights, analysis_fps, key_hz, pool, sticky)
    dense, times, fps = expand_keys(keys, key_hz, out_fps, fade_s, shape)
    return keys, dense, times, fps


def draw_stats(weights: np.ndarray, fps: float) -> dict:
    """Cost profile of a dense weight matrix under on-demand drawing.

    The target is live display, not video, so 30 fps is a logical clock rather than a
    refresh rate: a frame only costs anything if it differs from the one before it, and
    a frame at pure base pose costs nothing at all. These are the numbers that decide
    whether the render budget is viable on low-end hardware.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("expected a (n_frames, n_visemes) weight matrix")
    if fps <= 0.0:
        raise ValueError(f"invalid fps: {fps}")

    n = weights.shape[0]
    base = VISEME_INDEX[BASE_VISEME]
    needs = np.empty(n, dtype=bool)
    needs[0] = True
    needs[1:] = np.any(weights[1:] != weights[:-1], axis=1)

    closed = weights[:, base] >= 1.0 - 1e-9
    speaking = ~closed
    bursts = [len(list(group)) for key, group in groupby(needs) if key]

    duration = n / fps
    speech_s = speaking.sum() / fps
    return {
        "frames": n,
        "fps": fps,
        "duration_s": duration,
        "redraws": int(needs.sum()),
        "redraw_share": float(needs.mean()),
        "skippable": int((~needs).sum()),
        "idle_frames": int(closed.sum()),
        "idle_share": float(closed.mean()),
        "idle_periods": sum(1 for key, group in groupby(closed) if key),
        "mean_idle_s": closed.sum() / max(sum(1 for key, _ in groupby(closed) if key), 1) / fps,
        "draws_per_s_speaking": float(needs[speaking].sum() / speech_s) if speech_s > 0 else 0.0,
        "draws_per_s_overall": float(needs.sum() / duration) if duration > 0 else 0.0,
        "longest_burst_frames": max(bursts, default=0),
        "longest_burst_s": max(bursts, default=0) / fps,
    }


def key_stats(keys: list[Keyframe], key_hz: float, duration: float) -> dict:
    """Measure what the keyframing actually produced, against the intended rate."""
    if not keys:
        return {}

    names = [k.viseme for k in keys]
    changes = sum(1 for a, b in pairwise(names) if a != b)
    runs = [len(list(group)) for _, group in groupby(names)]
    period = 1.0 / key_hz

    shares: dict[str, float] = {}
    for k in keys:
        shares[k.viseme] = shares.get(k.viseme, 0.0) + period

    return {
        "keys": len(keys),
        "changes": changes,
        "changes_per_s": changes / duration if duration > 0 else 0.0,
        "ceiling_per_s": key_hz,
        "runs": len(runs),
        "mean_hold_s": sum(runs) / len(runs) * period,
        "shortest_hold_s": min(runs) * period,
        "single_slot_runs": sum(1 for r in runs if r == 1),
        "distinct": len(set(names)),
        "time_share": shares,
    }
