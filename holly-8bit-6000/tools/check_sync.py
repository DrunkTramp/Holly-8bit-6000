#!/usr/bin/env python3
"""Measure A/V offset objectively: known acoustic events vs when the mouth moves.

Phase A's exit criteria are "sync verified by eye, idle is zero-draw, no final-pose
latch, mouth-rect-only blits" -- three of those four are already pinned by tests, and
"by eye" is the one that isn't. This closes that gap with a stimulus whose event times
are known by construction: a train of short broadband clicks.

For each click it finds the first frame where the mouth leaves the base pose and reports
the offset in milliseconds, against the timing grid the pipeline actually imposes:

    slot        1/key_hz          the pose grid; nothing can move between slots
    fade        frames/out_fps    `center` straddles the boundary, `lead` lands early,
                                  `lag` lands late
    vote        vote_window frames of majority voting before pooling
    gate        vad_ms of hysteresis before the classifier engages

So an offset of one slot is not a bug, it is the cartoon timing model working: poses are
held and quantised. What matters is that the offsets are *consistent* (low spread) and
*centred* on the grid, because a systematic half-slot skew is what makes a mouth look
detached from a voice.

    .venv/bin/python tools/check_sync.py
    .venv/bin/python tools/check_sync.py --fade-shape lead --key-hz 10
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from holly.audio import TARGET_RATE
from holly.keyframes import FADE_S, FADE_SHAPES, KEY_HZ, RENDER_FPS
from holly.runtime import HollyFace
from holly.vise import VISEME_INDEX

BASE = VISEME_INDEX["sil"]


def click_train(
    n: int = 12,
    spacing: float = 0.5,
    burst_ms: float = 120.0,
    amplitude: float = 0.5,
    tail: float = 0.3,
    seed: int = 7,
    stimulus: str = "vowel",
) -> tuple[np.ndarray, np.ndarray]:
    """Bursts at exact times, on a floor quiet enough for the auto gate.

    Returns (samples, event_times). The events are the *start* of each burst, which is
    where a lip-sync observer would expect the mouth to react.

    `vowel` is a harmonic tone, which the HeadAudio prototypes actually recognise; `click`
    is white noise, which is an unambiguous "something happened" but is not a speech-like
    spectrum, so the real classifier reacts to far fewer of them. Timing geometry is best
    measured with `--classifier openness-ramp` (deterministic in RMS, reacts to every
    burst); viseme quality needs real speech, which is filmstrip territory, not this tool.
    """
    rng = np.random.default_rng(seed)
    total = round((n * spacing + tail) * TARGET_RATE)
    samples = (rng.standard_normal(total) * 1e-4).astype(np.float32)
    burst = round(burst_ms / 1000.0 * TARGET_RATE)
    times = []
    for i in range(n):
        start = round(i * spacing * TARGET_RATE)
        stop = min(start + burst, total)
        length = stop - start
        if stimulus == "click":
            body = rng.standard_normal(length) * amplitude
        else:
            t = np.arange(length) / TARGET_RATE
            f0 = 110.0 * (1.0 + (i % 3) * 0.25)
            # Harmonics with a crude formant tilt: speech-like enough for the prototypes.
            body = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3, 4, 5))
            body = (body / np.abs(body).max() * amplitude).astype(np.float32)
        samples[start:stop] += body.astype(np.float32)
        times.append(start / TARGET_RATE)
    return samples, np.array(times, dtype=np.float64)


def offsets(face: HollyFace, samples: np.ndarray, events: np.ndarray,
            window: float = 0.35) -> tuple[list[float], list[int]]:
    """Signed ms from each event to the *first* frame whose mouth is open.

    Searching forward (from one slot before the event, which is where a `center` fade is
    legitimately early) rather than for the strongest reaction, because "when did the mouth
    start moving" is the sync question; "where did it move most" is a viseme question.
    """
    utterance = face.speak(samples)
    fps = utterance.render_fps
    rows = utterance.weights
    slot = 1.0 / face.key_hz
    got, frames = [], []
    for t in events:
        lo = max(0, round((t - slot) * fps))
        hi = min(rows.shape[0], round((t + window) * fps) + 1)
        block = rows[lo:hi, BASE]
        open_at = np.flatnonzero(block < 1.0 - 1e-9)
        if open_at.size == 0:
            continue
        k = lo + int(open_at[0])
        got.append((k / fps - t) * 1000.0)
        frames.append(k)
    return got, frames


def report(args) -> int:
    samples, events = click_train(
        n=args.clicks, spacing=args.spacing, burst_ms=args.burst_ms,
        amplitude=args.amplitude, seed=args.seed, stimulus=args.stimulus,
    )
    slot_ms = 1000.0 / args.key_hz
    fade_frames = max(1, round(args.fade_s * args.render_fps))

    face = HollyFace(
        classifier=args.classifier,
        key_hz=args.key_hz,
        render_fps=args.render_fps,
        fade_s=args.fade_s,
        fade_shape=args.fade_shape,
        sticky=args.sticky,
        gain=not args.no_gain,
    )
    got, _frames = offsets(face, samples, events)

    print(f"stimulus: {args.clicks} {args.stimulus} bursts at {args.spacing*1000:.0f} ms spacing, "
          f"{args.burst_ms:.0f} ms long, {samples.size / TARGET_RATE:.2f} s total "
          f"({args.classifier})")
    print(f"grid: slot {slot_ms:.1f} ms ({args.key_hz:g}/s), fade {fade_frames} frames "
          f"({fade_frames / args.render_fps * 1000:.0f} ms, {args.fade_shape}), "
          f"analysis {TARGET_RATE / 256:.1f} fps, render {args.render_fps:g} fps")
    print(f"detected {len(got)} of {len(events)} clicks")
    if not got:
        print("no mouth movement detected -- the stimulus or the gate is wrong", file=sys.stderr)
        return 1

    med = statistics.median(got)
    mean = statistics.fmean(got)
    print("\noffset (mouth-opens minus click), ms")
    print(f"  median {med:+7.1f}   mean {mean:+7.1f}   min {min(got):+7.1f}   max {max(got):+7.1f}")
    print(f"  spread (max-min) {max(got) - min(got):7.1f}   stdev {statistics.pstdev(got):7.1f}")
    print(f"  per click: {' '.join(f'{o:+.0f}' for o in got)}")

    # The judgement that matters for a cartoon mouth: consistent, and not skewed by more
    # than half a slot. A half-slot systematic shift is ~42 ms at 12/s -- audible as
    # "the mouth is ahead of the voice".
    skew = abs(med)
    verdict = "ok" if skew <= slot_ms / 2 else "SKEWED"
    spread = max(got) - min(got)
    jitter = "ok" if spread <= slot_ms * 1.5 else "JITTERY"
    print(f"\n  skew {skew:.1f} ms vs half-slot {slot_ms/2:.1f} ms -> {verdict}")
    print(f"  spread {spread:.1f} ms vs 1.5 slots {slot_ms*1.5:.1f} ms -> {jitter}")
    if args.compare:
        print("\nfade-shape sweep (the runtime path, not the offline tool):")
        for shape in FADE_SHAPES:
            face2 = HollyFace(classifier=args.classifier, key_hz=args.key_hz,
                              render_fps=args.render_fps, fade_s=args.fade_s, fade_shape=shape,
                              sticky=args.sticky, gain=not args.no_gain)
            got2, _ = offsets(face2, samples.copy(), events)
            if got2:
                print(f"  {shape:<7} median {statistics.median(got2):+7.1f} ms  "
                      f"spread {max(got2)-min(got2):6.1f} ms  n={len(got2)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure pose-vs-event offset on a click train.")
    parser.add_argument("--clicks", type=int, default=12)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--burst-ms", type=float, default=120.0)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stimulus", choices=("vowel", "click"), default="vowel")
    parser.add_argument("--classifier", choices=("headaudio", "openness-ramp"), default="headaudio")
    parser.add_argument("--key-hz", type=float, default=KEY_HZ)
    parser.add_argument("--render-fps", type=float, default=RENDER_FPS)
    parser.add_argument("--fade-s", type=float, default=FADE_S)
    parser.add_argument("--fade-shape", choices=FADE_SHAPES, default="center")
    parser.add_argument("--sticky", type=float, default=0.05)
    parser.add_argument("--no-gain", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Also sweep lead/center/lag.")
    return report(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
