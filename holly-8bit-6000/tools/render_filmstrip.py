#!/usr/bin/env python3
"""Phase 1: viseme timeline -> debug filmstrip and a muxed debug video.

Two artefacts, for two different questions:

  strip.png  every sampled frame side by side, so you can scan the whole utterance
             for stuck or flickering visemes at a glance
  video      the mouth animated at a real frame rate with the source audio and a
             playhead over the waveform, so you can watch mouth and sound together
             and judge the A/V offset directly
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

from holly.audio import decode
from holly.render import Renderer
from holly.timeline import read_timeline, sample_frames, timeline_weights

BAND_HEIGHT = 96


def waveform_band(samples: np.ndarray, rate: int, width: int, height: int) -> np.ndarray:
    """Min/max envelope of the whole utterance as a grayscale band."""
    columns = max(width, 1)
    per_column = max(1, samples.size // columns)
    usable = (samples.size // per_column) * per_column
    reshaped = samples[:usable].reshape(-1, per_column)
    peak = np.max(np.abs(reshaped), axis=1)

    if peak.size < columns:
        peak = np.pad(peak, (0, columns - peak.size))
    peak = peak[:columns]

    band = np.full((height, columns), 28, dtype=np.uint8)
    half = height // 2
    reach = np.clip(peak * (half - 2), 1, half - 2).astype(int)
    rows, cols = np.mgrid[0:height, 0:columns]
    mask = np.abs(rows - half) <= reach[None, :]
    band[mask] = 150
    band[half - 1 : half + 1, :] = 90
    return band


def render_frames(
    renderer: Renderer,
    weights: np.ndarray,
    timeline: dict,
    out_fps: float,
    band: np.ndarray | None,
    duration: float,
    show_labels: bool,
) -> np.ndarray:
    """Render one RGB frame for each out_fps tick, resampling the timeline."""
    source_fps = timeline["fps"]
    n_out = max(1, int(round(duration * out_fps)))
    out = []

    for k in range(n_out):
        t = k / out_fps
        src = min(int(round(t * source_fps)), weights.shape[0] - 1)
        # snapshot(), not draw_row()'s return value: draw_row hands back a persistent
        # buffer it overwrites in place, so anything retained across draws must be copied.
        rgb = renderer.draw_row(weights[src]).copy()
        frame = Image.fromarray(rgb, "RGB")

        if band is not None:
            frame = frame.convert("RGB")
            frame.paste(Image.fromarray(band, "L").convert("RGB"), (0, frame.height - BAND_HEIGHT))

        if show_labels:
            row = weights[src]
            order = np.argsort(row)[::-1]
            names = timeline["visemes"]
            top, runner = names[int(order[0])], names[int(order[1])]
            draw = ImageDraw.Draw(frame)
            label = f"t={t:5.2f}s  {top} {row[int(order[0])]:.2f}   alt {runner} {row[int(order[1])]:.2f}"
            draw.rectangle((0, 0, frame.width, 22), fill=(0, 0, 0))
            draw.text((6, 5), label, fill=(255, 230, 0))

            if band is not None and duration > 0:
                x = int(round(t / duration * frame.width))
                top_of_band = frame.height - BAND_HEIGHT
                draw.line((x, top_of_band, x, frame.height - 1), fill=(255, 60, 60), width=2)

        out.append(np.array(frame, dtype=np.uint8))

    return np.stack(out)


def contact_sheet(frames: list[np.ndarray], labels: list[str], tile_width: int, out: Path) -> None:
    tiles = []
    for frame, label in zip(frames, labels):
        image = Image.fromarray(frame, "RGB")
        scale = tile_width / image.width
        image = image.resize((tile_width, max(1, int(round(image.height * scale)))))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, tile_width, 18), fill=(0, 0, 0))
        draw.text((4, 3), label, fill=(255, 230, 0))
        tiles.append(image)

    width = max(t.width for t in tiles)
    height = max(t.height for t in tiles)
    per_row = max(1, 1440 // width)
    rows = (len(tiles) + per_row - 1) // per_row

    sheet = Image.new("RGB", (width * per_row, height * rows), (18, 18, 18))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % per_row) * width, (i // per_row) * height))

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def encode_video(frames: np.ndarray, audio: Path, out: Path, fps: float, keep_frames: Path | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(keep_frames) if keep_frames else Path(tempfile.mkdtemp(prefix="holly_frames_"))
    tmp.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        Image.fromarray(frame, "RGB").save(tmp / f"{i:05d}.png")

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-framerate", f"{fps}",
        "-i", str(tmp / "%05d.png"),
        "-i", str(audio),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.decode('utf-8', 'replace').strip()}")

    if keep_frames is None:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a viseme timeline into a contact sheet and a muxed debug video.")
    parser.add_argument("--timeline", type=Path, default=Path("build/debug/timeline.json"))
    parser.add_argument("--buffers", type=Path, default=Path("build/visemes/visemes.npz"))
    parser.add_argument("--audio", type=Path, default=None, help="Source audio, for the muxed video and waveform band.")
    parser.add_argument("--strip", type=Path, default=Path("build/debug/strip.png"))
    parser.add_argument("--video", type=Path, default=None, help="Write a muxed debug video here (needs --audio).")
    parser.add_argument("--tiles", type=int, default=40, help="Frames in the contact sheet.")
    parser.add_argument("--tile-width", type=int, default=288)
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate.")
    parser.add_argument("--no-band", action="store_true", help="Skip the waveform band in the video.")
    args = parser.parse_args(argv)

    timeline = read_timeline(args.timeline)
    weights = timeline_weights(timeline)
    duration = float(timeline["duration_s"])

    renderer = Renderer(args.buffers)
    print(f"loaded {args.buffers}: canvas {renderer.size[0]}x{renderer.size[1]}, "
          f"mouth rect {renderer.patch_shape[1]}x{renderer.patch_shape[0]} at {renderer.origin}")
    print(f"loaded {args.timeline}: {weights.shape[0]} frames, {duration:.2f} s at {timeline['fps']} fps")

    sampled = sample_frames(timeline, args.tiles)
    strip_frames, labels = [], []
    for record in sampled:
        row = np.zeros(len(timeline["visemes"]), dtype=np.float64)
        index = {v: i for i, v in enumerate(timeline["visemes"])}
        row[index[record["viseme"]]] = record["weight"]
        for name, value in record.get("alt", {}).items():
            row[index[name]] = value
        strip_frames.append(renderer.draw_row(row).copy())
        alt = max(record.get("alt", {}).items(), key=lambda kv: kv[1])[0] if record.get("alt") else "-"
        labels.append(f"{record['t']:5.2f}s {record['viseme']} {record['weight']:.2f} / {alt}")

    contact_sheet(strip_frames, labels, args.tile_width, args.strip)
    print(f"wrote {args.strip} ({len(strip_frames)} tiles)")

    if args.video:
        if args.audio is None:
            print("error: --video needs --audio so the mouth and the sound share a timeline", file=sys.stderr)
            return 2

        samples = decode(args.audio)
        band = None if args.no_band else waveform_band(samples, 16000, renderer.size[0], BAND_HEIGHT)
        frames = render_frames(renderer, weights, timeline, args.fps, band, duration, show_labels=True)
        encode_video(frames, args.audio, args.video, args.fps, keep_frames=None)
        print(f"wrote {args.video} ({frames.shape[0]} frames at {args.fps} fps, muxed with {args.audio})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
