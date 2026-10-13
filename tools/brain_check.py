#!/usr/bin/env python3
"""Phase C latency budget: how long before Holly's mouth starts moving, and does he stall?

Perceived latency after the keypress is **not** the whole reply: the streaming splitter sends
the first sentence to TTS as soon as its boundary is confirmed, and the host queues audio
before it plays. What the user actually waits for is

    first token -> first sentence -> its TTS -> its viseme analysis -> audio starts

and the failure mode this tool exists to catch is the one that follows: if sentence N+1 is not
queued by the time the sentences ahead of it have finished playing, Holly goes **silent in the
middle of his answer**. That is what decides `--max-tokens` and the model tier for a given box
(the plan's Phase C exit criteria), and it is a property of the hardware, not of the code.

The measurement and the table live in `hollyhost.calibrate` (shared with the product's
`--calibrate` first-run step); this is the developer-facing CLI around them.

No window, no audio device: the clock is not needed to measure the budget.

    .venv/bin/python tools/brain_check.py --brain script --tts null         # pipeline only
    .venv/bin/python tools/brain_check.py --brain llama --model model.gguf  # real brain
    .venv/bin/python tools/brain_check.py --brain http --base-url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holly.runtime import HollyFace

from hollyhost.brain import make_brain
from hollyhost.calibrate import DEFAULT_PROMPT, measure, report
from hollyhost.tts import make_tts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Measure Phase C first-audio latency and queue stalls.")
    p.add_argument("--brain", default="script", choices=("script", "llama", "http"))
    p.add_argument("--model", type=Path, default=None, help="GGUF for --brain llama.")
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--http-model", default=None)
    p.add_argument("--tts", default="piper", choices=("null", "piper", "omnivoice"))
    p.add_argument("--voice", default=None,
                   help="Piper .onnx (default: first in voices/); omnivoice reference audio.")
    p.add_argument("--tts-device", default=None,
                   help="Pin the omnivoice model: cuda, cuda:0, cpu (default: cuda when present).")
    p.add_argument("--tts-steps", type=int, default=None,
                   help="omnivoice diffusion steps (default: the model's 32).")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--gap", type=float, default=0.25, help="The host's inter-utterance gap.")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--gpu-layers", type=int, default=-1)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--max-chars", type=int, default=220,
                   help="The splitter's clause cap (same knob as the app's --max-chars). Lower "
                        "it to cut a long first sentence short: when a model rambles without a "
                        "full stop, that cap is what sets 1audio ms.")
    args = p.parse_args(argv)

    loaded = time.perf_counter()
    brain = make_brain(args.brain, model=args.model, base_url=args.base_url,
                       http_model=args.http_model, ctx=args.ctx, gpu_layers=args.gpu_layers,
                       max_tokens=args.max_tokens, temperature=args.temp)
    load_s = time.perf_counter() - loaded
    loaded = time.perf_counter()
    tts = make_tts(args.tts, voice=args.voice, device=args.tts_device, num_step=args.tts_steps)
    # Warm up exactly as the app does, so `1audio ms` measures a conversation and not a
    # cold-start artefact: omnivoice loads lazily and would otherwise spend ~5-11 s of
    # one-off model load inside the first sentence.
    warmup = getattr(tts, "warmup", None)
    if warmup is not None:
        warmup()
    tts_load_s = time.perf_counter() - loaded
    face = HollyFace()  # shipping defaults: the analysis cost the UI thread actually pays

    print(f"brain={getattr(brain, 'name', '?')}  tts={tts.name} @ {getattr(tts, 'rate', '?')} Hz  "
          f"brain load {load_s:.1f} s  tts load {tts_load_s:.1f} s  gap {args.gap} s\nprompt: {args.prompt}\n")
    rows = measure(brain, tts, face, args.prompt, args.reps, args.gap, args.max_chars)
    report(rows, args.gap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
