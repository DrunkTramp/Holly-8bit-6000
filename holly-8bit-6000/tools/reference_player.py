#!/usr/bin/env python3
"""Phase 3a: standalone reference player -- Holly speaks an audio file in sync.

This is not the host application (that does not exist yet). It exists to *prove* the
four properties the runtime model rests on, and to double as the eyeball harness for
every future tuning pass:

    1. the animation clock is the audio output position, never `time.time()`
    2. dirty-check: a poll whose pose equals the last drawn one costs no draw
    3. idle is a true zero-draw state on the base layer
    4. no final-pose latch: when the utterance ends the mouth returns to rest

Everything it does, a real host does identically -- except that a real host feeds
`speak()` a TTS buffer instead of a file and owns its own audio device. The core stays
free of pygame: this file is the only place pygame is touched.

    .venv/bin/python tools/reference_player.py --audio test_audio.flac
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holly.audio import TARGET_RATE, decode
from holly.keyframes import FADE_S, FADE_SHAPES, KEY_HZ, POOL_MODES, RENDER_FPS, STICKY_MARGIN
from holly.render import Renderer
from holly.runtime import CLASSIFIERS, DEFAULT_BUFFERS, DEFAULT_MODEL, HollyFace
from holly.vise import BASE_VISEME

POLL_HZ = 60.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play an audio file with Holly lip-syncing to it -- the Phase 3a reference player."
    )
    parser.add_argument("--audio", type=Path, action="append", required=True,
                        help="Audio file to speak. Repeat to queue several back-to-back.")
    parser.add_argument("--buffers", type=Path, default=DEFAULT_BUFFERS,
                        help="Premultiplied viseme buffers (tools/export_visemes.py).")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--classifier", choices=CLASSIFIERS, default="headaudio")
    parser.add_argument("--gap", type=float, default=0.25,
                        help="Seconds of silence between queued utterances (exercises the idle path).")

    # Tuning knobs, mirroring tools/make_timeline.py so this doubles as the harness.
    parser.add_argument("--key-hz", type=float, default=KEY_HZ)
    parser.add_argument("--render-fps", type=float, default=RENDER_FPS)
    parser.add_argument("--fade-s", type=float, default=FADE_S)
    parser.add_argument("--fade-shape", choices=FADE_SHAPES, default="center")
    parser.add_argument("--pool", choices=POOL_MODES, default="mean")
    parser.add_argument("--sticky", type=float, default=STICKY_MARGIN)
    parser.add_argument("--no-gain", action="store_true")

    parser.add_argument("--scale", type=int, default=1, help="Integer window zoom (nearest-neighbour).")
    parser.add_argument("--poll-hz", type=float, default=POLL_HZ,
                        help="How often the loop asks for a pose. The clock is the audio position, "
                             "so polling faster than the render rate only ever produces skips.")
    parser.add_argument("--loop", action="store_true", help="Replay when the queue drains.")
    parser.add_argument("--headless", action="store_true",
                        help="Run the same code path on SDL dummy drivers (smoke test, no window).")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Stop after this much wall time (default: run to the end of the queue).")
    return parser.parse_args(argv)


def build_chunks(utterances: list[np.ndarray], gap: float) -> list[np.ndarray]:
    """Each utterance, separated by `gap` seconds of silence.

    The gaps are what make the audio length equal the face clock: `speak(at=...)`
    places utterance N+1 at utterance N's end plus the gap, and the host plays exactly
    that much silence in between.
    """
    chunks = []
    for i, samples in enumerate(utterances):
        chunks.append(np.clip(samples, -1.0, 1.0))
        if i < len(utterances) - 1 and gap > 0:
            chunks.append(np.zeros(round(TARGET_RATE * max(0.0, gap)), dtype=np.float32))
    return chunks


def wav_bytes(chunks: list[np.ndarray], rate: int = TARGET_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        for samples in chunks:
            w.writeframes((samples.astype(np.float32) * 32767.0).astype("<i2").tobytes())
    return buf.getvalue()


def load_music(pygame, wav: bytes):
    """`mixer.music` gives a real playback position; `mixer.Sound` does not.

    Newer SDL_mixer reads a file-like, older builds only a path, so fall back to a
    temp file rather than assuming which the host box has.
    """
    try:
        pygame.mixer.music.load(io.BytesIO(wav))
        return None
    except pygame.error:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav)
            path = Path(tmp.name)
        pygame.mixer.music.load(path)
        return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    try:
        import pygame
        import pygame.surfarray
    except ImportError as exc:  # pragma: no cover - optional runtime dep
        print(f"the reference player needs pygame (the core does not): {exc}", file=sys.stderr)
        print("    .venv/bin/pip install pygame", file=sys.stderr)
        return 2

    if not args.buffers.exists():
        print(f"missing baked buffers: {args.buffers}\n"
              f"    .venv/bin/python tools/export_visemes.py", file=sys.stderr)
        return 2

    print("decoding")
    buffers = []
    for path in args.audio:
        samples = decode(path)
        buffers.append(samples)
        print(f"  {path}: {samples.size} samples, {samples.size / TARGET_RATE:.2f} s")

    renderer = Renderer(args.buffers)
    face = HollyFace(
        renderer=renderer,
        model_path=args.model,
        classifier=args.classifier,
        key_hz=args.key_hz,
        render_fps=args.render_fps,
        fade_s=args.fade_s,
        fade_shape=args.fade_shape,
        pool=args.pool,
        sticky=args.sticky,
        gain=not args.no_gain,
    )

    def queue() -> float:
        """Analyse everything up front, laid back-to-back on the face clock."""
        clock = 0.0
        for name, samples in zip([str(p) for p in args.audio], buffers):
            u = face.speak(samples, TARGET_RATE, at=clock if clock > 0 else None)
            print(f"  {Path(name).name}: {u.frames} frames, {u.audio_duration:.2f} s, "
                  f"{u.analysis_ms:.0f} ms analysis, {len(u.keys)} poses")
            clock = u.end + args.gap
        return face.queued_duration

    print("analysing (ahead of playback, as the component API requires)")
    total = queue()
    wav = wav_bytes(build_chunks(buffers, args.gap))

    pygame.mixer.pre_init(TARGET_RATE, -16, 1, 512)
    pygame.init()
    if not pygame.mixer.get_init():
        print("no audio device: the player cannot prove sync without a playback clock",
              file=sys.stderr)
        pygame.quit()
        return 1
    tmp_path = load_music(pygame, wav)
    pygame.mixer.music.set_volume(1.0 if not args.headless else 0.0)

    scale = max(1, int(args.scale))
    x, y, w, h = renderer.mouth_rect
    size = (renderer.size[0] * scale, renderer.size[1] * scale)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Holly-8bit-6000 reference player")
    mouth = pygame.Rect(x * scale, y * scale, w * scale, h * scale)

    def surface_of(region: np.ndarray, px_w: int, px_h: int) -> pygame.Surface:
        # pygame's surfarray is width-major; the renderer's frame is row-major.
        surface = pygame.surfarray.make_surface(np.transpose(region, (1, 0, 2)))
        return surface if scale == 1 else pygame.transform.scale(surface, (px_w * scale, px_h * scale))

    def paint_full(frame: np.ndarray) -> None:
        screen.blit(surface_of(frame, *renderer.size), (0, 0))
        pygame.display.flip()

    def paint_mouth(frame: np.ndarray) -> None:
        """The mouth rect and nothing else -- ~9% of the pixels."""
        screen.blit(surface_of(frame[y : y + h, x : x + w], w, h), mouth)
        pygame.display.update(mouth)

    running = True
    started = False
    wall_start = time.perf_counter()
    poll = 1.0 / max(1.0, args.poll_hz)
    deadline = wall_start + args.seconds if args.seconds else None
    paints = 0

    print(f"\nplaying {total:.2f} s of face clock at {size[0]}x{size[1]} "
          f"({scale}x), polling {args.poll_hz:g} Hz -- Esc quits")
    paint_full(renderer.frame)
    pygame.mixer.music.play()

    while running:
        now = time.perf_counter()
        if deadline is not None and now >= deadline:
            break
        time.sleep(poll)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        busy = pygame.mixer.music.get_busy()
        if busy:
            started = True
            t = max(0, pygame.mixer.music.get_pos()) / 1000.0
        elif not started:
            t = 0.0  # the mixer has not spun up yet; do not jump the clock
        else:
            t = total + 1.0  # past the end: this is the no-latch path

        frame, _rect, changed = face.frame_at(t)
        if changed:
            # The first paint after the queue drains restores the base mouth; it is
            # still only the mouth rect, because that is the only thing that moved.
            paint_mouth(frame)
            paints += 1

        if not busy and started:
            if not args.loop:
                break
            face.reset()
            total = queue()
            started = False
            pygame.mixer.music.play()

    pygame.mixer.music.stop()
    pygame.quit()
    if tmp_path is not None:
        tmp_path.unlink(missing_ok=True)

    ds = face.stats
    print(f"\npolls {ds['polls']}  draws {ds['draws']}  skips {ds['skips']} "
          f"-- {face.draw_share():.1%} of polls cost a draw, {paints} blits")
    print(f"analysis: {ds['analyses']} utterances, {ds['analysis_ms']:.0f} ms total "
          f"({ds['analysis_ms'] / max(total, 1e-9):.3f} ms per second of speech)")
    print(f"queue drained, last pose {face.viseme_at(total + 1.0)!r} == {BASE_VISEME!r}: "
          f"the mouth returned to rest and stopped drawing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
