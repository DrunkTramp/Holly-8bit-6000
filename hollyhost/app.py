"""The Holly host: a corner face driven by the audio device's position.

Two modes, one clock rule (`t` is frames the output callback has consumed, never
`time.time()`):

    # A0 -- batch: queue files, play once, prove the clock.
    python -m hollyhost.app --audio ../Holly/test_audio.flac --audio ../Holly/test_audio.flac

    # A1 -- interactive: type, Holly speaks it with a synced mouth (no --audio).
    python -m hollyhost.app --tts null
    python -m hollyhost.app --tts piper --voice voices/en-gb.mp3

The face is a **corner element** (the 576x432 host bake). Only the mouth rect is ever
uploaded, and only when `HollyFace.frame_at` says the pose changed -- the dirty check, the
zero-draw idle and the return-to-rest all live in the component, not here. `--dry` runs the
identical loop against a hand-driven clock, so the whole thing is exercisable on a box with
no audio hardware (this dev box has no libportaudio).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .audio import FakeDevice, LiveAudio, SAMPLE_RATE, SounddeviceOutput
from .session import LiveSession, build_session, from_file


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Holly host: keyboard in, lip-synced corner face out.")
    p.add_argument("--audio", type=Path, action="append", default=None,
                   help="WAV/FLAC/MP3 to speak (batch mode). Omit for the interactive text line.")
    p.add_argument("--buffers", type=Path, default=None, help="Viseme buffers (default: holly.DEFAULT_BUFFERS).")
    p.add_argument("--gap", type=float, default=0.25, help="Silence between utterances, seconds.")
    p.add_argument("--blocksize", type=int, default=512)
    p.add_argument("--device", default=None, help="sounddevice output device index or name.")
    p.add_argument("--poll-hz", type=float, default=60.0)
    p.add_argument("--corner", choices=("bottom-left", "bottom-right", "top-left", "top-right"),
                   default="bottom-left")
    p.add_argument("--window", type=str, default="960x540", help="Host UI canvas the face sits in.")
    p.add_argument("--seconds", type=float, default=None, help="Batch mode: stop after this much wall time.")

    # A1: text -> speech.
    p.add_argument("--tts", choices=("null", "piper", "script"), default="null",
                   help="Speech backend for the interactive line (null runs with no downloads).")
    p.add_argument("--voice", default=None, help="Piper .onnx voice (with --tts piper).")
    p.add_argument("--tts-rate", type=int, default=None, help="Voice output rate; defaults per backend.")
    p.add_argument("--dry", action="store_true", help="Hand-driven clock: no audio, no window. Loop proof only.")
    return p.parse_args(argv)


def _resolve_buffers(arg):
    from holly.runtime import DEFAULT_BUFFERS

    path = Path(arg) if arg else DEFAULT_BUFFERS
    if not path.exists():
        print(f"missing buffers: {path}\n  (cd ../Holly && python tools/export_visemes.py "
              f"--scale 0.5 --resample nearest --out build/visemes_pixel)", file=sys.stderr)
        raise SystemExit(2)
    return path


def _device_arg(value):
    if value is None:
        return None
    return int(value) if str(value).isdigit() else value


def run(argv=None) -> int:
    args = parse_args(argv)
    buffers_path = _resolve_buffers(args.buffers)

    if args.audio:
        return run_batch(args, buffers_path)
    return run_interactive(args, buffers_path)


# --------------------------------------------------------------------------- A0: batch


def run_batch(args, buffers_path) -> int:
    from holly.render import Renderer
    from holly.runtime import HollyFace

    print("decoding + analysing (all of it before playback)")
    samples = []
    names = []
    for path in args.audio:
        b = from_file(path)
        samples.append(b)
        names.append(str(path))
        print(f"  {path}: {b.size / SAMPLE_RATE:.2f} s")

    face = HollyFace(renderer=Renderer(buffers_path))
    face, sources, playback = build_session(samples, face=face, gap=args.gap, names=names)
    total = playback.size / SAMPLE_RATE
    print(f"queued {len(sources)} utterances, {total:.2f} s of clock, "
          f"mouth rect {face.renderer.mouth_rect}")

    device = FakeDevice(playback) if args.dry else SounddeviceOutput(
        playback, rate=SAMPLE_RATE, blocksize=args.blocksize, device=_device_arg(args.device))

    if args.dry:
        return _batch_dry(face, device, args)
    return _batch_ui(face, device, total, args)


def _tick(face, device) -> bool:
    """One poll: ask for the pose at the device position, blit only if it changed."""
    _frame, _rect, changed = face.frame_at(device.t)
    return changed


def _batch_dry(face, device, args) -> int:
    """Same clock and same dirty check, no pixels. Prints the numbers Phase A0 must show."""
    poll = 1.0 / max(1.0, args.poll_hz)
    draws = polls = 0
    while not device.finished:
        device.advance(round(SAMPLE_RATE * poll))
        polls += 1
        draws += _tick(face, device)
    for _ in range(30):  # past the end: the no-latch restore, then silence
        device.advance(round(SAMPLE_RATE * poll))
        polls += 1
        draws += _tick(face, device)
    print(f"\n{polls} polls, {draws} draws ({draws/max(polls,1):.1%}), "
          f"final pose {face.viseme_at(device.t)!r}")
    assert face.viseme_at(device.t) == "sil", "the mouth did not return to rest"
    return 0


def _batch_ui(face, device, total: float, args) -> int:
    import pygame
    import pygame.surfarray

    x, y, w, h = face.renderer.mouth_rect
    fw, fh = face.renderer.size
    win_w, win_h = _window_size(args.window)
    ox, oy = corner_offset(args.corner, win_w, win_h, fw, fh)

    pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, args.blocksize)
    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Holly host - Phase A0")
    mouth = pygame.Rect(ox + x, oy + y, w, h)

    def upload(region: np.ndarray, dest) -> None:
        surf = pygame.surfarray.make_surface(np.transpose(region, (1, 0, 2)))
        screen.blit(surf, dest)

    screen.fill((16, 16, 18))
    upload(face.renderer.frame, (ox, oy))
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
            upload(frame[y : y + h, x : x + w], mouth)
            pygame.display.update(mouth)
        if device.finished:
            _tick(face, device)  # let the return-to-rest draw happen
            break

    device.stop()
    ds = face.stats
    print(f"polls {ds['polls']}  draws {ds['draws']}  skips {ds['skips']} "
          f"({face.draw_share():.1%})  underrun frames {getattr(device, 'underrun_frames', 0)}")
    pygame.quit()
    return 0


# --------------------------------------------------------------------------- A1: interactive


def run_interactive(args, buffers_path) -> int:
    from holly.render import Renderer
    from holly.runtime import HollyFace
    from .tts import make_tts

    tts = make_tts(args.tts, voice=args.voice, rate=args.tts_rate)
    rate = args.tts_rate or getattr(tts, "rate", SAMPLE_RATE)
    audio = LiveAudio(rate, blocksize=args.blocksize)
    face = HollyFace(renderer=Renderer(buffers_path))
    session = LiveSession(face, audio, gap=args.gap, tts=tts)
    print(f"interactive: tts={tts.name} @ {rate} Hz, mouth rect {face.renderer.mouth_rect}, "
          f"Esc quits, Enter speaks")

    if args.dry:
        return _interactive_dry(session, args)
    return _interactive_ui(session, args)


def _interactive_dry(session, args) -> int:
    """Drive the live clock by hand and feed a scripted line, no window, no audio.

    A faithful device rehearsal: the clock advances by exactly the frames a healthy output
    callback would have consumed since the last poll (elapsed * rate), so the mouth animates
    on real audio position -- just silently. This is what the host test asserts deterministically.
    """
    session.say("Hello. This is Holly speaking in sync.")
    poll = 1.0 / max(1.0, args.poll_hz)
    started = time.perf_counter()
    last = started
    while not session.audio.idle() or time.perf_counter() - started < 0.5:
        now = time.perf_counter()
        session.audio.advance(max(1, round((now - last) * session.audio.rate)))
        last = now
        session.poll()
        time.sleep(poll)
        if time.perf_counter() - started > (args.seconds or 12.0):
            break
    ds = session.face.stats
    print(f"\n{ds['polls']} polls, {ds['draws']} draws ({session.face.draw_share():.1%}), "
          f"final pose {session.face.viseme_at(session.audio.t)!r}")
    assert session.face.viseme_at(session.audio.t) == "sil", "the mouth did not return to rest"
    return 0


def _interactive_ui(session, args) -> int:
    import pygame
    import pygame.surfarray

    x, y, w, h = session.face.renderer.mouth_rect
    fw, fh = session.face.renderer.size
    win_w, win_h = _window_size(args.window)
    ox, oy = corner_offset(args.corner, win_w, win_h, fw, fh)
    line_h = 34

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Holly host - Phase A1 (type, Enter to speak)")
    font = pygame.font.Font(None, 26)
    mouth = pygame.Rect(ox + x, oy + y, w, h)
    input_rect = pygame.Rect(8, win_h - line_h, win_w - 16, line_h - 6)

    def upload(region: np.ndarray, dest) -> None:
        surf = pygame.surfarray.make_surface(np.transpose(region, (1, 0, 2)))
        screen.blit(surf, dest)

    screen.fill((16, 16, 18))
    upload(session.face.renderer.frame, (ox, oy))
    pygame.display.flip()

    try:
        session.audio.start()
    except (ImportError, RuntimeError, OSError) as exc:  # no libportaudio / no device on this box
        print(f"audio output unavailable: {exc}\n  (try --dry, or install libportaudio2)", file=sys.stderr)
        pygame.quit()
        return 1

    text = ""
    dirty_line = True
    poll = 1.0 / max(1.0, args.poll_hz)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RETURN:
                    if text.strip():
                        session.say(text)
                        text = ""
                        dirty_line = True
                elif event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                    dirty_line = True
                elif event.unicode and event.unicode.isprintable():
                    text += event.unicode
                    dirty_line = True
        time.sleep(poll)

        # The face: blit the mouth rect only when the pose changed.
        if session.poll()[2]:
            frame = session.face.renderer.frame
            upload(frame[y : y + h, x : x + w], mouth)
            pygame.display.update(mouth)

        # The input line: host UI, redrawn only when the text changed.
        if dirty_line:
            pygame.draw.rect(screen, (30, 30, 34), input_rect)
            pygame.draw.rect(screen, (90, 90, 100), input_rect, 1)
            shown = "> " + text + ("_" if (time.time() * 2) % 2 < 1 else "")
            screen.blit(font.render(shown, True, (220, 220, 220)), (input_rect.x + 6, input_rect.y + 5))
            pygame.display.update(input_rect)
            dirty_line = False

    session.audio.stop()
    pygame.quit()
    return 0


# --------------------------------------------------------------------------- shared helpers


def _window_size(spec: str) -> tuple[int, int]:
    w, h = spec.lower().split("x")
    return int(w), int(h)


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
