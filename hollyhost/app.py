"""The Holly host: an animated face driven by the audio device's position.

Two modes, one clock rule (`t` is frames the output callback has consumed, never
`time.time()`):

    # A0 -- batch: queue files, play once, prove the clock.
    python -m hollyhost.app --audio holly-8bit-6000/test_audio.flac --audio holly-8bit-6000/test_audio.flac

    # A1 -- interactive: type, Holly speaks it with a synced mouth (no --audio).
    python -m hollyhost.app --tts null
    python -m hollyhost.app --tts piper --voice voices/en-gb.mp3

The face is drawn on a Tkinter canvas (the 576x432 host bake, scaled into the right-hand
pane). Only the mouth rect is ever redrawn, and only when `HollyFace.frame_at` says the pose
changed -- the dirty check, the zero-draw idle and the return-to-rest all live in the
component, not here. `--dry` runs the identical loop against a hand-driven clock, so the
whole thing is exercisable on a box with no audio hardware and no display (this dev box has
no libportaudio). The Tk UI itself lives in `tkui.py`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .audio import FakeDevice, LiveAudio, SAMPLE_RATE, SounddeviceOutput
from .session import LiveSession, build_session, from_file


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Holly host: keyboard in, lip-synced animated face out.")
    p.add_argument("--audio", type=Path, action="append", default=None,
                   help="WAV/FLAC/MP3 to speak (batch mode). Omit for the interactive text line.")
    p.add_argument("--buffers", type=Path, default=None, help="Viseme buffers (default: holly.DEFAULT_BUFFERS).")
    p.add_argument("--gap", type=float, default=0.25, help="Silence between utterances, seconds.")
    p.add_argument("--blocksize", type=int, default=512)
    p.add_argument("--device", default=None, help="sounddevice output device index or name.")
    p.add_argument("--poll-hz", type=float, default=60.0)
    p.add_argument("--corner", choices=("bottom-left", "bottom-right", "top-left", "top-right"),
                   default="bottom-left",
                   help="Deprecated: the Tk layout pins the face to the top-right pane.")
    p.add_argument("--window", type=str, default="960x540", help="Host UI window size (WxH).")
    p.add_argument("--seconds", type=float, default=None, help="Batch mode: stop after this much wall time.")

    # A1: text -> speech.
    p.add_argument("--tts", choices=("null", "piper", "omnivoice", "script"), default=None,
                   help="Speech backend for the interactive line (default: saved settings, "
                        "else null -- which runs with no downloads).")
    p.add_argument("--voice", default=None,
                   help="Piper .onnx voice (with --tts piper); omnivoice reference audio "
                        "(with --tts omnivoice, default voices/holly_base.flac).")
    p.add_argument("--tts-rate", type=int, default=None, help="Voice output rate; defaults per backend.")
    p.add_argument("--tts-device", default=None,
                   help="Pin the omnivoice model to a torch device: cuda, cuda:0, cpu "
                        "(default: cuda when a GPU is present).")
    p.add_argument("--tts-steps", type=int, default=None,
                   help="omnivoice diffusion steps (default: the model's 32). Fewer steps cut "
                        "the per-sentence floor at some quality cost -- the dial to turn if "
                        "brain_check.py reports a STALL on a short first sentence.")
    p.add_argument("--dry", action="store_true", help="Hand-driven clock: no audio, no window. Loop proof only.")

    # C: the brain. `none` keeps A1: Holly speaks exactly what you type.
    p.add_argument("--brain", choices=("none", "script", "llama", "http"), default=None,
                   help="Reply generator (default: saved settings, else none, which speaks "
                        "what you type). llama needs --model.")
    p.add_argument("--model", type=Path, default=None, help="GGUF model for --brain llama.")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible endpoint for --brain http (llama-server, Ollama). "
                        "Default: saved settings, else http://127.0.0.1:8080.")
    p.add_argument("--http-model", default=None, help="Model name for --brain http (llama.cpp ignores it).")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                   help="Bearer token for --brain http (default: $OPENAI_API_KEY, if set).")
    p.add_argument("--ctx", type=int, default=4096,
                   help="Context window tokens. The file persona is ~2k tokens, so this needs "
                        "4096+; keep it as small as the persona allows (llama tier).")
    p.add_argument("--gpu-layers", type=int, default=None,
                   help="GGUF layers to offload (-1 = all). Default comes from the calibration "
                        "config when present, else -1.")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Reply length cap (conversational, not a sync lever). Default comes from "
                        "the calibration config when present, else 200.")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--threads", type=int, default=None, help="CPU inference threads (default: all).")
    p.add_argument("--persona", default=None,
                   help="System prompt text, or @file. Default: holly_personality_prompt.md "
                        "(edit that file to tune the character).")
    p.add_argument("--no-normalize", action="store_true",
                   help="Skip Phase D TTS text preprocessing (expand numbers/abbreviations/"
                        "symbols to spoken words before synthesis). Off only if your voice's own "
                        "normalizer already reads them correctly and you want the raw text spoken.")
    p.add_argument("--max-chars", type=int, default=None,
                   help="Sentence chunk cap: the streaming splitter's breath length. Default comes "
                        "from the calibration config when present, else 220.")
    p.add_argument("--captions", action="store_true", help="Show Holly's reply text in the window.")

    # First-run calibration: measure this box and record its tier (the production form of
    # tools/brain_check.py). Runs with whatever --brain/--tts you pass; CI uses script+null.
    p.add_argument("--calibrate", action="store_true",
                   help="Measure first-audio latency + queue margin on this box, print the table, "
                        "and write the chosen defaults (--max-chars etc.) to the config file. "
                        "Runs the same pipeline headless; no window, no audio device.")
    p.add_argument("--config", type=Path, default=None,
                   help="Calibration config file (default: hollyhost_config.json in the repo root). "
                        "Read for defaults on a normal run; written by --calibrate.")
    p.add_argument("--settings", type=Path, default=None,
                   help="UI settings file (default: hollyhost_settings.json in the repo root). "
                        "Read at launch to pre-fill the settings panel; written by the panel "
                        "when the backends load and when the window closes.")
    p.add_argument("--no-write", action="store_true",
                   help="With --calibrate: print and derive, but do not write the config file.")

    # E: polish.
    p.add_argument("--barge-at", type=float, default=None,
                   help="With the interactive --dry run: cut Holly off at this clock position "
                        "(seconds), exercising the barge-in path headlessly (Tab does it live).")
    p.add_argument("--idle-motion", action=argparse.BooleanOptionalAction, default=None,
                   help="Throttled 'breathing' on the idle mouth (at most 2 draws/s). A host-side "
                        "overlay that restores the pure idle pose whenever speech is queued; with "
                        "this off the component's zero-draw idle is untouched. "
                        "Default: saved settings, else off.")
    p.add_argument("--idle-weight", type=float, default=None,
                   help="Peak breath weight with --idle-motion (default: saved settings, else "
                        "0.25; speech poses sit at 0.65, and measured on the bake, 0.12 is "
                        "effectively invisible).")
    p.add_argument("--no-gain", action="store_true",
                   help="Turn off the classifier's per-viseme display gain (HeadAudio visemeMaxs "
                        "caps: poses at 0.65, PP/FF at 0.75) and get raw one-hot poses -- the A/B "
                        "dial for eyeballing the look.")
    p.add_argument("--key-hz", type=float, default=None,
                   help="Pose rate (default: the component's 12/s). `--key-hz 10` keeps the 2-frame "
                        "fade from collapsing short holds to one 33 ms frame -- settle by eye.")
    p.add_argument("--fade-shape", choices=("center", "lag", "lead"), default=None,
                   help="Fade placement around a pose change (default: the component's center).")
    return p.parse_args(argv)


def _resolve_buffers(arg):
    from holly.runtime import DEFAULT_BUFFERS

    path = Path(arg) if arg else DEFAULT_BUFFERS
    if not path.exists():
        print(f"missing buffers: {path}\n  (cd holly-8bit-6000 && ../.venv/bin/python tools/export_visemes.py "
              f"--scale 0.5 --resample nearest --out build/visemes_pixel)", file=sys.stderr)
        raise SystemExit(2)
    return path


def _device_arg(value):
    if value is None:
        return None
    return int(value) if str(value).isdigit() else value


def _face_kwargs(args) -> dict:
    """Phase E look knobs for `HollyFace`. Unset flags pass nothing: the component's shipped
    defaults are the shipped look, and these are only ever A/B dials for the human's eye."""
    kwargs = {}
    if args.no_gain:
        kwargs["gain"] = False
    if args.key_hz is not None:
        kwargs["key_hz"] = args.key_hz
    if args.fade_shape is not None:
        kwargs["fade_shape"] = args.fade_shape
    return kwargs


def run(argv=None) -> int:
    args = parse_args(argv)
    _apply_ui_settings(args)
    _apply_config_defaults(args)
    if args.calibrate:
        return run_calibrate(args)
    buffers_path = _resolve_buffers(args.buffers)

    if args.audio:
        return run_batch(args, buffers_path)
    return run_interactive(args, buffers_path)


# --------------------------------------------------------------------------- config + calibrate


def _config_path(args):
    from .calibrate import DEFAULT_CONFIG_PATH

    return Path(args.config) if args.config else DEFAULT_CONFIG_PATH


def _apply_ui_settings(args) -> None:
    """Fill unset backend/UI flags from the persistent settings file (saved by the panel).

    Precedence is explicit flag > saved settings > built-in default, and it runs before
    `_apply_config_defaults` so a model chosen in the panel outranks the calibration
    config's suggestion. May contain an API key in plain text -- the file is gitignored.
    """
    from .settings import load_settings

    s = load_settings(args.settings)
    if args.tts is None:
        args.tts = s.get("tts") or "null"
    if args.voice is None and s.get("voice"):
        args.voice = s["voice"]
    if args.brain is None:
        args.brain = s.get("brain") or "none"
    if args.model is None and s.get("llama_model"):
        args.model = Path(s["llama_model"])
    if args.base_url is None:
        host = str(s.get("host") or "").strip()
        port = str(s.get("port") or "").strip()
        if host:
            scheme, sep, rest = host.partition("://")
            if not sep:
                scheme, rest = "http", host
            default_port = "443" if scheme == "https" else "8080"
            from .brain import normalize_base_url
            args.base_url = normalize_base_url(
                f"{scheme}://{rest.rstrip('/')}:{port or default_port}")
    if args.http_model is None and s.get("http_model"):
        args.http_model = s["http_model"]
    if args.api_key is None and s.get("api_key"):
        args.api_key = s["api_key"]
    if args.idle_motion is None:
        args.idle_motion = bool(s.get("idle_motion", False))
    if args.idle_weight is None:
        args.idle_weight = float(s.get("idle_weight", 0.25))


def _apply_config_defaults(args) -> None:
    """Fill any unset tier flag from the calibration config, else the hard defaults.

    Precedence is explicit flag > config > built-in default, and the built-in defaults match the
    pre-calibration values, so a box that has never calibrated behaves exactly as before.
    """
    from .calibrate import HARD_DEFAULTS, load_config

    cfg = load_config(_config_path(args))
    if args.max_chars is None:
        args.max_chars = int(cfg.get("max_chars", HARD_DEFAULTS["max_chars"]))
    if args.max_tokens is None:
        args.max_tokens = int(cfg.get("max_tokens", HARD_DEFAULTS["max_tokens"]))
    if args.gpu_layers is None:
        args.gpu_layers = int(cfg.get("gpu_layers", HARD_DEFAULTS["gpu_layers"]))
    if args.model is None and cfg.get("model"):
        args.model = Path(cfg["model"])


def _build_brain_and_tts(args):
    """Build the voice (warm) and the brain, before any window opens.

    Shared by the interactive run and `--calibrate`: the slow parts happen up front, and a failure
    is a clean message + a non-zero code rather than a dead window or a traceback. Returns
    `(brain, tts, error_code)` with `error_code == 0` on success.
    """
    from .brain import make_brain
    from .tts import make_tts

    try:
        tts = make_tts(args.tts, voice=args.voice, rate=args.tts_rate, device=args.tts_device,
                       num_step=args.tts_steps)
    except (ImportError, FileNotFoundError, ValueError) as exc:
        print(f"tts {args.tts!r} unavailable: {exc}", file=sys.stderr)
        return None, None, 2

    # omnivoice loads lazily, so without this the first sentence would absorb ~5-11 s of one-off
    # cost (see OmniVoiceTTS.warmup). Same rule as the brain below: the slow part happens before
    # the window opens, and a failure is a message rather than a dead window.
    warmup = getattr(tts, "warmup", None)
    if warmup is not None:
        t0 = time.perf_counter()
        try:
            warmup()
        except (ImportError, FileNotFoundError, ValueError) as exc:
            print(f"tts {args.tts!r} unavailable: {exc}", file=sys.stderr)
            return None, None, 2
        print(f"tts {args.tts} warm in {time.perf_counter() - t0:.1f} s", file=sys.stderr)

    # The model loads before the window opens: it is the slow part, and a failure here should be a
    # message, not a dead window.
    try:
        brain = make_brain(args.brain, model=args.model, base_url=args.base_url,
                           http_model=args.http_model, ctx=args.ctx, gpu_layers=args.gpu_layers,
                           max_tokens=args.max_tokens, temperature=args.temp, threads=args.threads,
                           api_key=args.api_key)
    except ImportError as exc:
        print(f"brain {args.brain!r} needs its dependency: {exc}\n"
              f"  (.venv/bin/pip install -e '.[ui,dev,piper,llm]')", file=sys.stderr)
        return None, None, 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"brain {args.brain!r} unavailable: {exc}", file=sys.stderr)
        return None, None, 2
    return brain, tts, 0


def run_calibrate(args) -> int:
    """First-run calibration: measure this box, print the table, write the chosen defaults.

    The production form of `tools/brain_check.py` -- same measurement (`hollyhost.calibrate`),
    reachable by the end user. Runs headless (no window, no audio device); needs a brain, so
    `--brain none` is rejected. Model-free in CI with `--brain script --tts null`.
    """
    from holly.runtime import HollyFace

    from .calibrate import DEFAULT_PROMPT, measure, recommend, report, write_config

    brain, tts, err = _build_brain_and_tts(args)
    if err:
        return err
    if brain is None:
        print("--calibrate needs a brain (--brain script|llama|http); --brain none has nothing to "
              "measure.", file=sys.stderr)
        return 2

    face = HollyFace()  # analysis only: needs the classifier model, not the viseme bake
    cfg_path = _config_path(args)
    print(f"calibrate: brain={brain.name} tts={tts.name} @ {getattr(tts, 'rate', '?')} Hz  "
          f"gap {args.gap} s  max-chars {args.max_chars}  config -> {cfg_path}\nprompt: {DEFAULT_PROMPT}\n")
    rows = measure(brain, tts, face, DEFAULT_PROMPT, reps=2, gap=args.gap, max_chars=args.max_chars)
    report(rows, args.gap)

    cfg = recommend(rows, max_chars=args.max_chars, max_tokens=args.max_tokens,
                    gpu_layers=args.gpu_layers, model=args.model, tts=tts.name, brain=brain.name)
    print("\nrecommended defaults for this box:")
    print(f"  --max-chars  {cfg['max_chars']}")
    print(f"  --max-tokens {cfg['max_tokens']}")
    print(f"  --gpu-layers {cfg['gpu_layers']}")
    if cfg["model"]:
        print(f"  --model      {cfg['model']}")
    for note in cfg["notes"]:
        print(f"  - {note}")

    if args.no_write:
        print("\n(--no-write: config not written)")
    else:
        written = write_config(cfg, cfg_path)
        print(f"\nwrote {written}. A normal run now reads these as defaults (override any with an "
              f"explicit flag).")
    return 0


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

    face = HollyFace(renderer=Renderer(buffers_path), **_face_kwargs(args))
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
    from . import tkui

    return tkui.run_batch(face, device, args)


# --------------------------------------------------------------------------- A1: interactive


def run_interactive(args, buffers_path) -> int:
    from holly.render import Renderer
    from holly.runtime import HollyFace

    if args.dry:
        # The rehearsal keeps the eager path: the tests pin the whole headless pipeline, and
        # there is no window to keep responsive.
        brain, tts, err = _build_brain_and_tts(args)
        if err:
            return err
        rate = args.tts_rate or getattr(tts, "rate", SAMPLE_RATE)

        audio = LiveAudio(rate, blocksize=args.blocksize)
        face = HollyFace(renderer=Renderer(buffers_path), **_face_kwargs(args))
        session = LiveSession(face, audio, gap=args.gap, tts=tts, normalize=not args.no_normalize)

        from .brain import SentencePump
        pump = None
        if brain is not None:
            pump = SentencePump(brain, tts, persona=args.persona, max_chars=args.max_chars,
                                normalize=not args.no_normalize)
        print(f"interactive: tts={tts.name} @ {rate} Hz, brain={brain.name if brain else 'none'}, "
              f"mouth rect {face.renderer.mouth_rect}, Esc quits, Enter speaks, Tab/Space stops Holly")
        return _interactive_dry(session, args, pump)

    # Live: the window opens with NO model loaded. The settings panel carries the backend
    # choices (pre-filled from these flags), and the UI builds brain + TTS on a worker thread
    # at the first message -- the face and the chat stay responsive through the load. The
    # device runs at SAMPLE_RATE; `session.say` resamples whatever voice loads into it.
    audio = LiveAudio(args.tts_rate or SAMPLE_RATE, blocksize=args.blocksize)
    face = HollyFace(renderer=Renderer(buffers_path), **_face_kwargs(args))
    session = LiveSession(face, audio, gap=args.gap, tts=None, normalize=not args.no_normalize)
    print(f"interactive: mouth rect {face.renderer.mouth_rect}; voice + brain load on the "
          f"first message (settings panel, bottom right). Esc quits, Enter speaks, "
          f"Tab/Space stops Holly")
    return _interactive_ui(session, args, None)


def _interactive_dry(session, args, pump=None) -> int:
    """Drive the live clock by hand and feed a scripted line, no window, no audio.

    A faithful device rehearsal: the clock advances by exactly the frames a healthy output
    callback would have consumed since the last poll (elapsed * rate), so the mouth animates
    on real audio position -- just silently. This is what the host test asserts deterministically.

    With a brain, the scripted line becomes a *prompt* and everything Holly says arrives from
    the chat thread, sentence by sentence -- the Phase C path, exercised with no model.
    """
    if pump is not None:
        pump.submit("Hello Holly, what are you?")
    else:
        session.say("Hello. This is Holly speaking in sync.")
    poll = 1.0 / max(1.0, args.poll_hz)
    started = time.perf_counter()
    last = started
    budget = args.seconds or 12.0
    timed_out = False
    barged = args.barge_at is None
    while not session.audio.idle() or time.perf_counter() - started < 0.5:
        now = time.perf_counter()
        session.audio.advance(max(1, round((now - last) * session.audio.rate)))
        last = now
        for said, samples, sample_rate in (pump.sentences() if pump else ()):  # UI thread only
            session.say(said, samples=samples, rate=sample_rate)
            print(f"  [{session.audio.t:6.2f} s] {said}")
        if not barged and session.audio.t >= args.barge_at:
            # Phase E barge-in rehearsal: cut the clock, drop the queue, back to idle.
            barged = True
            if pump is not None:
                pump.interrupt()
            session.interrupt()
            print(f"  [{session.audio.t:6.2f} s] BARGE-IN: playback cut, queue dropped")
        session.poll()
        time.sleep(poll)
        if pump is not None and pump.error:
            print(f"brain: {pump.error}", file=sys.stderr)
            break
        if pump is not None and not pump.busy and pump.pending == 0 and session.audio.idle():
            break
        if time.perf_counter() - started > budget:
            timed_out = True
            break
    if pump is not None:
        pump.stop()
    ds = session.face.stats
    print(f"\n{ds['polls']} polls, {ds['draws']} draws ({session.face.draw_share():.1%}), "
          f"final pose {session.face.viseme_at(session.audio.t)!r}")
    if timed_out:
        # Cutting a turn short is not a sync failure -- the mouth is legitimately mid-vowel, so
        # the return-to-rest assertion below does not apply. Say so loudly instead of asserting.
        left = max(0.0, session.audio.queued_seconds - session.audio.t)
        extra = f", {pump.pending} synthesised but not yet queued" if pump else ""
        print(f"TRUNCATED after {budget:.1f} s: {left:.1f} s still playing, "
              f"{session.face.pending} utterance(s) queued{extra}. "
              f"Raise --seconds to rehearse a whole turn.", file=sys.stderr)
        return 0
    if args.barge_at is not None and not barged:
        print(f"note: --barge-at {args.barge_at:.2f} s was never reached (the turn ended first)",
              file=sys.stderr)
    assert session.face.viseme_at(session.audio.t) == "sil", "the mouth did not return to rest"
    return 0


def _interactive_ui(session, args, pump=None) -> int:
    from . import tkui

    return tkui.run_interactive(session, args, pump)


# --------------------------------------------------------------------------- shared helpers


def _window_size(spec: str) -> tuple[int, int]:
    w, h = spec.lower().split("x")
    return int(w), int(h)


if __name__ == "__main__":
    raise SystemExit(run())
