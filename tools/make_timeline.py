#!/usr/bin/env python3
"""Phase 1: audio file -> viseme timeline JSON.

Offline proof of the analysis side of the plan: decode, frame, feature, classify,
smooth, emit {t, viseme, weight} records. The classifier defaults to the Phase 2
HeadAudio Mahalanobis port; `--classifier openness-ramp` selects the Phase 1
placeholder so the two can be A/B'd on the same audio.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from holly.audio import HOP_SIZE, TARGET_RATE, decode, frame, frame_times
from holly.classify import (
    HeadAudioClassifier,
    OpennessRamp,
    derive_le_thresholds,
    derive_vad_thresholds,
)
from holly.features import extract_features
from holly.keyframes import (
    FADE_S,
    FADE_SHAPES,
    KEY_HZ,
    POOL_MODES,
    RENDER_FPS,
    STICKY_MARGIN,
    Keyframe,
    draw_stats,
    fade_frames,
    key_stats,
    keyframe_weights,
)
from holly.smooth import ATTACK_MS, RELEASE_MS, attack_release, normalize_weights
from holly.timeline import build_timeline, write_timeline
from holly.vise import BASE_VISEME, VISEMES

CLASSIFIERS = ("openness-ramp", "headaudio")
DEFAULT_MODEL = Path("model/model-en-mixed.bin")


def build_classifier(args, analysis_fps: float, floor: float, ceiling: float, features):
    """Both classifiers satisfy the same weight-matrix contract; they just differ in what they need."""
    if args.classifier == "headaudio":
        vad = not args.no_vad
        vad_active_db = -40.0
        vad_inactive_db = -50.0
        if vad and args.vad == "auto":
            # HeadAudio's shipped -40/-50 dBFS gate assumes a noise floor under
            # -50 dBFS; recordings that sit above it (this TTS clip: quiet frames
            # at le -4.9..-4.4) never close the gate and the mouth hangs open.
            # Derive the same hysteresis from the file's own log-energy instead.
            floor_le, ceiling_le = derive_le_thresholds(features.log_energy)
            vad_inactive_db, vad_active_db = floor_le * 10.0, ceiling_le * 10.0
            print(f"vad gate: le {floor_le:.2f} -> {ceiling_le:.2f} (auto, log-energy percentiles)")
        return HeadAudioClassifier(
            args.model,
            sil_sensitivity=args.sil_sensitivity,
            vote_window=0 if args.no_vote else 6,
            vad=vad,
            vad_active_db=vad_active_db,
            vad_inactive_db=vad_inactive_db,
            analysis_fps=analysis_fps,
            gain=not args.no_gain,
        )
    return OpennessRamp(silence_floor=floor, speech_ceiling=ceiling)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Turn an audio file into a viseme timeline JSON.")
    parser.add_argument("--audio", type=Path, default=Path("test_audio.flac"))
    parser.add_argument("--out", type=Path, default=Path("build/debug/timeline.json"))
    parser.add_argument("--classifier", choices=CLASSIFIERS, default="headaudio")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="HeadAudio binary model (only used by the headaudio classifier).")
    parser.add_argument("--sil-sensitivity", type=float, default=1.2,
                        help="HeadAudio silSensitivity: >1 favours silence prototypes.")
    parser.add_argument("--no-vad", action="store_true",
                        help="Skip HeadAudio's log-energy gate and classify every frame.")
    parser.add_argument("--no-vote", action="store_true",
                        help="Skip HeadAudio's 6-frame majority vote (pooling still smooths).")
    parser.add_argument("--no-gain", action="store_true",
                        help="Raw one-hot poses instead of headaudio.mjs visemeMaxs display gain "
                             "(mouth poses capped at 0.65/0.75 alpha, remainder held by sil).")
    parser.add_argument("--attack-ms", type=float, default=ATTACK_MS)
    parser.add_argument("--release-ms", type=float, default=RELEASE_MS)
    parser.add_argument("--no-smooth", action="store_true", help="Emit raw per-frame decisions.")
    parser.add_argument("--vad", choices=("auto", "absolute"), default="auto",
                        help="Derive the silence gate from this file's own energy, or use the fixed thresholds "
                             "(rms percentiles for the stub, -40/-50 dBFS log-energy for headaudio).")
    parser.add_argument("--vad-floor", type=float, default=None, help="Absolute RMS gate start (default 0.005).")
    parser.add_argument("--vad-ceiling", type=float, default=None, help="Absolute RMS gate end (default 0.020).")
    parser.add_argument("--rate", type=int, default=TARGET_RATE)
    parser.add_argument("--hop", type=int, default=HOP_SIZE)

    parser.add_argument("--key-hz", type=float, default=KEY_HZ,
                        help="Pose rate: how many mouth poses per second. Ceiling on switch rate.")
    parser.add_argument("--render-fps", type=float, default=RENDER_FPS,
                        help="Frame rate the keyframed timeline is written at.")
    parser.add_argument("--fade-s", type=float, default=FADE_S, help="Cross-fade length between poses.")
    parser.add_argument("--fade-shape", choices=FADE_SHAPES, default="center",
                        help="center: fade straddles the pose boundary. lead: new pose lands ON the boundary. "
                             "lag: old pose holds to the boundary, then fades.")
    parser.add_argument("--pool", choices=POOL_MODES, default="mean",
                        help="How a pose slot picks a viseme: mean of the analysis frames, or their peak.")
    parser.add_argument("--sticky", type=float, default=STICKY_MARGIN,
                        help="How much a new mouth pose must beat the current one by to take over. "
                             "Not applied to sil transitions. 0 disables.")
    parser.add_argument("--continuous", action="store_true",
                        help="Old behaviour: one timeline record per analysis frame, attack/release envelopes.")
    parser.add_argument("--smooth-before-keys", action="store_true",
                        help="Apply attack/release envelopes before pooling into pose slots.")
    args = parser.parse_args(argv)

    samples = decode(args.audio, target_rate=args.rate)
    duration = samples.size / args.rate
    frames = frame(samples, hop=args.hop)
    fps = args.rate / args.hop
    times = frame_times(frames.shape[0], hop=args.hop, rate=args.rate)

    print(f"decoded {args.audio}: {samples.size} samples, {duration:.2f} s at {args.rate} Hz")
    print(f"framed: {frames.shape[0]} frames of 512 at hop {args.hop} -> {fps:.1f} analysis fps")
    if abs(fps - 100.0) > 0.5:
        print(f"  note: the plan's '~100 viseme frames/s' needs hop {args.rate // 100} at {args.rate} Hz")

    t0 = time.perf_counter()
    features = extract_features(frames, sample_rate=args.rate, hop=args.hop)
    feature_ms = (time.perf_counter() - t0) * 1000

    if args.vad == "auto":
        floor, ceiling = derive_vad_thresholds(features.rms)
        if args.vad_floor is not None:
            floor = args.vad_floor
        if args.vad_ceiling is not None:
            ceiling = args.vad_ceiling
    else:
        floor = args.vad_floor if args.vad_floor is not None else 0.005
        ceiling = args.vad_ceiling if args.vad_ceiling is not None else 0.020

    classifier = build_classifier(args, fps, floor, ceiling, features)
    if isinstance(classifier, OpennessRamp):
        print(f"vad gate: rms {classifier.silence_floor:.5f} -> {classifier.speech_ceiling:.5f} ({args.vad})")

    t1 = time.perf_counter()
    weights = classifier(features)
    classify_ms = (time.perf_counter() - t1) * 1000

    weights = normalize_weights(weights)

    continuous = args.continuous
    keys: list[Keyframe] = []

    # Keyframing does its own smoothing by pooling, so the attack/release envelope
    # is only applied in continuous mode unless --smooth-before-keys asks for it.
    smooth = (not args.no_smooth) if continuous else args.smooth_before_keys
    if smooth:
        weights = attack_release(weights, fps, attack_ms=args.attack_ms, release_ms=args.release_ms)
        weights = normalize_weights(weights)

    if continuous:
        out_times, out_fps = times, fps
    else:
        keys, weights, out_times, out_fps = keyframe_weights(
            weights,
            analysis_fps=fps,
            key_hz=args.key_hz,
            out_fps=args.render_fps,
            fade_s=args.fade_s,
            shape=args.fade_shape,
            pool=args.pool,
            sticky=args.sticky,
        )

    per_frame_ms = (feature_ms + classify_ms) / frames.shape[0]

    timeline = build_timeline(weights, out_times, out_fps)
    write_timeline(timeline, args.out)

    top = [f["viseme"] for f in timeline["frames"]]
    changes = sum(1 for a, b in pairwise(top) if a != b)
    mean_weight = float(np.mean([f["weight"] for f in timeline["frames"]]))
    duration = float(timeline["duration_s"]) or duration

    print()
    print(f"classifier: {classifier.name}")
    print(f"analysis cost: {per_frame_ms:.3f} ms/frame "
          f"(features {feature_ms / frames.shape[0]:.3f} + classify {classify_ms / frames.shape[0]:.3f})")
    print(f"  -> {per_frame_ms * fps / 10:.1f}% of one core sustaining {fps:.0f} analysis fps")

    if continuous:
        silence = top.count("sil")
        print(f"mode: continuous ({fps:.1f} records/s, attack {args.attack_ms} ms / release {args.release_ms} ms"
              f"{', smoothing off' if args.no_smooth else ''})")
        print()
        print(f"records: {len(top)}  silence: {silence / len(top):.1%}  mean top weight: {mean_weight:.3f}")
        print(f"top-viseme switches: {changes} total, {changes / duration:.1f}/s")
    else:
        stats = key_stats(keys, args.key_hz, duration)
        period = 1.0 / args.key_hz
        frames = fade_frames(args.fade_s, out_fps)
        print(f"mode: keyframed ({args.key_hz:g} poses/s, {frames}-frame {args.fade_shape} cross-fade "
              f"= {frames / out_fps * 1000:.0f} ms effective at {out_fps:g} fps)")
        print(f"  slot = {period * 1000:.1f} ms ({period * out_fps:.2f} output frames), pooling "
              f"{fps / args.key_hz:.1f} analysis frames ({args.pool}) per pose, sticky margin {args.sticky:g}")
        if frames != round(args.fade_s * out_fps):
            print(f"  note: --fade-s {args.fade_s:g} s snapped to {frames} whole frames; "
                  f"a sub-frame fade is not rendered")
        if frames > args.render_fps / args.key_hz:
            print(f"  warning: the fade is longer than a pose slot ({args.render_fps / args.key_hz:.2f} frames), "
                  f"so poses are never held pure -- lower --fade-s or --render-fps")
        print(f"  smoothing before pooling: {'on' if args.smooth_before_keys else 'off'}")
        print()
        print(f"poses: {stats['keys']} slots, {stats['distinct']} distinct, mean top weight: {mean_weight:.3f}")
        print(f"pose changes: {stats['changes']} total, {stats['changes_per_s']:.1f}/s "
              f"(ceiling {args.key_hz:g}/s, {stats['changes_per_s'] / args.key_hz:.0%} of it)")
        print(f"hold length: mean {stats['mean_hold_s'] * 1000:.0f} ms, "
              f"shortest {stats['shortest_hold_s'] * 1000:.0f} ms, "
              f"{stats['single_slot_runs']} of {stats['runs']} runs change on every slot")
        distinct_rows = len({tuple(np.round(row, 3)) for row in weights})
        print(f"drawn states: {distinct_rows} distinct weight rows across {len(weights)} output frames "
              f"({distinct_rows / len(weights):.1%}) -- the mouth holds poses")

        ds = draw_stats(weights, out_fps)
        print()
        print("on-demand draw cost (30 fps is a logical clock, not a refresh rate)")
        print(f"  redraws needed: {ds['redraws']} of {ds['frames']} frames ({ds['redraw_share']:.1%}); "
              f"{ds['skippable']} identical to their predecessor and free")
        print(f"  idle at {BASE_VISEME}: {ds['idle_frames']} frames ({ds['idle_share']:.1%}) "
              f"in {ds['idle_periods']} periods, mean {ds['mean_idle_s'] * 1000:.0f} ms -- zero draw")
        print(f"  while speaking: {ds['draws_per_s_speaking']:.1f} draws/s "
              f"({ds['draws_per_s_overall']:.1f}/s averaged over the clip)")
        print(f"  worst burst: {ds['longest_burst_frames']} consecutive draws "
              f"in {ds['longest_burst_s'] * 1000:.0f} ms")
        print()
        print("viseme time share")
        for viseme, share in sorted(stats["time_share"].items(), key=lambda kv: -kv[1]):
            bar = "#" * round(40 * share / duration)
            print(f"  {viseme:<5} {share / duration:>6.1%} {bar}")
        print()
        print(f"pose sequence ({len(keys)}): {' '.join(k.viseme for k in keys)}")

    print()
    if continuous:
        print("viseme histogram")
        for viseme, count in Counter(top).most_common():
            bar = "#" * int(round(40 * count / len(top)))
            print(f"  {viseme:<5} {count:>4} {count / len(top):>6.1%} {bar}")
    print()
    print(f"wrote {args.out}")
    print(f"weights sum to 1 per frame: max deviation {float(np.abs(weights.sum(axis=1) - 1.0).max()):.2e}")

    unused = [v for v in VISEMES if v not in set(top)]
    if unused:
        print(f"note: never selected by this classifier: {' '.join(unused)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
