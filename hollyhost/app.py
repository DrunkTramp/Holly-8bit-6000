"""Phase A0: the corner face, driven by the audio device's position.

    python -m hollyhost.app --audio ../Holly/test_audio.flac --audio ../Holly/test_audio.flac

The face is a **corner element** (576x432 host bake), not a full-window canvas. Only the
mouth rect is ever uploaded, and only when `HollyFace.frame_at` says the pose changed.
`--dry` runs the identical loop against FakeDevice, so the clock path is exercisable on a
box with no audio hardware.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .audio import FakeDevice, SAMPLE_RATE, SounddeviceOutput
from .session import build_session, from_file


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Holly host Phase A0: audio files in, corner face out.")
    p.add_argument("--audio", type=Path, action="append", required=True,
                   help="WAV/FLAC/MP3 to speak. Repeat to queue. Phase A1 replaces this with TTS.")
    p.add_argument("--buffers", type=Path, default=None, help="Viseme buffers (default: holly.DEFAULT_BUFFERS).")
    p.add_argument("--gap", type=float, default=0.25, help="Silence between utterances, seconds.")
    p.add_argument("--blocksize", type=int, default=512)
    p.add_argument("--device", default=None, help="sounddevice output device index or name.")
    p.add_argument("--poll-hz", type=float, default=60.0)
    p.add_argument("--corner", choices=("bottom-left", "bottom-right", "top-left", "top-right"),
                   default="bottom-left")
    p.add_argument("--window", type=str, default="960x540", help="Host UI canvas the face sits in.")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--dry", action="store_true", help="FakeDevice: no audio, no window. Clock proof only.")
    return p.parse_args(argv)


def run(argv=None) -> int:
    args = parse_args(argv)
    from holly.render import Renderer
    from holly.runtime import HollyFace

    buffers_path = args.buffers
    if buffers_path is None:
        from holly import DEFAULT_BUFFERS as buffers_path

    if not Path(buffers_path).exists():
        print(f"missing buffers: {buffers_path}\n  (cd ../Holly && python tools/export_visemes.py "
              f"--scale 0.5 --resample nearest --out build/visemes_pixel)", file=sys.stderr)
        return 2

    print("decoding + analysing (all of it before playback)")
    samples = []
    for path in args.audio:
        b = from_file(path)
        b.name = str(path)
        samples.append(b)
        print(f"  {path}: {b.size / SAMPLE_RATE:.2f} s")

    face = HollyFace(renderer=Renderer(buffers_path))
    face, sources, playback = build_session(samples, face=face, gap=args.gap)
    total = playback.size / SAMPLE_RATE
    print(f"queued {len(sources)} utterances, {total:.2f} s of clock, "
          f"mouth rect {face.renderer.mouth_rect}")

    device = FakeDevice(playback) if args.dry else SounddeviceOutput(
        playback, rate=SAMPLE_RATE, blocksize=args.blocksize,
        device=args.device if args.device is None else int(args.device) if str(args.device).isdigit() else args.device,
    )

    if args.dry:
        return dry_run(face, device, total, args)
    return ui_run(face, device, total, args)


def _tick(face, device) -> bool:
    """One poll: ask for the pose at the device position, blit only if it changed."""
    _frame, _rect, changed = face.frame_at(device.t)
    return changed


def dry_run(face, device, total: float, args) -> int:
    """Same clock and same dirty check, no pixels. Prints the numbers Phase A0 must show."""
    poll = 1.0 / max(1.0, args.poll_hz)
    draws = 0
    polls = 0
    while not device.finished:
        device.advance(round(SAMPLE_RATE * poll))
        polls += 1
        draws += _tick(face, device)
    # past the end: the no-latch restore, then silence
    for _ in range(int(poll and 30)):
        device.advance(round(SAMPLE_RATE * poll))
        polls += 1
        draws += _tick(face, device)
    print(f"\n{polls} polls, {draws} draws ({draws/max(polls,1):.1%}), "
          f"final pose {face.viseme_at(device.t)!r}")
    assert face.viseme_at(device.t) == "sil", "the mouth did not return to rest"
    return 0


def ui_run(face, device, total: float, args) -> int:
    import pygame
    import pygame.surfarray

    x, y, w, h = face.renderer.mouth_rect
    fw, fh = face.renderer.size
    win_w, win_h = (int(v) for v in args.window.lower().split("x"))
    ox, oy = corner_offset(args.corner, win_w, win_h, fw, fh)

    pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, args.blocksize)
    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Holly host - Phase A0")
    mouth = pygame.Rect(ox + x, oy + y, w, h)

    def upload(region: np.ndarray, px_w: int, px_h: int, dest) -> None:
        surf = pygame.surfarray.make_surface(np.transpose(region, (1, 0, 2)))
        screen.blit(surf, dest)

    # Static background: the base face in its corner. Drawn once; the rest of the window is
    # the host UI that Phase C/D will fill (chat log, input line).
    screen.fill((16, 16, 18))
    upload(face.renderer.frame, fw, fh, (ox, oy))
    pygame.display.flip()

    device.start()
    poll = 1.0 / max(1.0, args.poll_hz)
    started = time.perf_counter()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        if args.seconds is not None and time.perf_counter() - started > args.seconds:
            break
        time.sleep(poll)

        if _tick(face, device):
            frame = face.renderer.frame
            upload(frame[y : y + h, x : x + w], w, h, mouth)
            pygame.display.update(mouth)
        if device.finished:
            # keep polling so the return-to-rest draw happens, then hold the idle face
            _tick(face, device)
            break

    device.stop()
    ds = face.stats
    print(f"polls {ds['polls']}  draws {ds['draws']}  skips {ds['skips']} "
          f"({face.draw_share():.1%})  underrun frames {getattr(device, 'underrun_frames', 0)}")
    pygame.quit()
    return 0


def corner_offset(corner: str, win_w: int, win_h: int, face_w: int, face_h: int) -> tuple[int, int]:
    pad = 8
    right, bottom = win_w - face_w - pad, win_h - face_h - pad
    return {
        "bottom-left": (pad, max(0, bottom)),
        "bottom-right": (max(0, right), max(0, bottom)),
        "top-left": (pad, pad),
        "top-right": (max(0, right), pad),
    }[corner]


if __name__ == "__main__":
    raise SystemExit(run())
