#!/usr/bin/env python3
"""Cross-voice classifier spot-check (Phase A1 / B): does each TTS voice drive the mouth?

The HeadAudio prototypes were trained on human speech; TTS is cleaner but its prosody
differs, so a voice can give poor visemes even when the pipeline is correct. This runs the
**shipping** configuration (real classifier, auto `le` gate + display gain, 12/s @ 30 fps)
over a few sentences per voice and prints the numbers that separate a good voice from a
degenerate one — no audio device, no window, just `HollyFace.analyse()`.

If a voice looks wrong, the fix is the classifier's gate/gain knobs (see `HollyFace`), never
the renderer. The by-eye judgement is `../Holly/tools/reference_player.py --audio <file>`.

    .venv/bin/python tools/voice_check.py                       # every voice in voices/
    .venv/bin/python tools/voice_check.py --voices voices/en_GB-alan-medium.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holly.runtime import HollyFace
from holly.vise import BASE_VISEME, VISEMES, VISEME_INDEX

from hollyhost.tts import PiperVoiceTTS, prepare

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SENTENCES = [
    "Hello! I am Holly, the computer aboard the mining ship Starbug.",
    "I'm feeling 52 times the intelligence of a human being today.",
    "Could you pass the salt? And mind the cat.",
]


def analyse_voice(voice: Path, sentences: list[str], face: HollyFace) -> list[dict]:
    tts = PiperVoiceTTS(voice)
    out = []
    for text in sentences:
        samples, rate = tts.synthesize(text)
        prepared = prepare(text, samples, rate)  # analysis @ 16k, playback @ native
        weights, keys, ms, fps = face.analyse(prepared.analysis, 16000)
        dur = prepared.duration
        argmax = weights.argmax(axis=1)
        changes = int(np.any(weights[1:] != weights[:-1], axis=1).sum()) if weights.shape[0] > 1 else 0
        distinct = sorted({VISEMES[i] for i in set(argmax.tolist())})
        out.append({
            "text": text,
            "seconds": dur,
            "frames": int(weights.shape[0]),
            "poses": len(keys),
            "changes": changes,
            "changes_per_s": changes / max(dur, 1e-9),
            "sil_frac": float(np.mean(argmax == VISEME_INDEX[BASE_VISEME])),
            "mean_top": float(weights.max(axis=1).mean()),
            "analysis_ms": ms,
            "distinct": distinct,
        })
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cross-voice viseme classifier spot-check.")
    p.add_argument("--voices", type=Path, action="append", default=None,
                   help="Piper .onnx voices (default: every *.onnx in voices/).")
    p.add_argument("--sentence", action="append", dest="sentences", default=None)
    args = p.parse_args(argv)

    voices = args.voices or sorted((ROOT / "voices").glob("*.onnx"))
    if not voices:
        print("no voices found; fetch one:\n"
              "  .venv/bin/python -m piper.download_voices en_GB-alan-medium --download-dir voices",
              file=sys.stderr)
        return 2

    sentences = args.sentences or DEFAULT_SENTENCES
    face = HollyFace()  # shipping defaults: headaudio + auto gate + gain, 12/s @ 30 fps

    print(f"classifier={face.classifier}  gate={face.vad}  gain={face.gain}  "
          f"{face.key_hz:g}/s @ {face.render_fps:g} fps\n")
    header = f"{'voice':22} {'sec':>5} {'frames':>6} {'poses':>5} {'chg/s':>6} {'sil%':>6} {'top':>5} {'ms':>5}  visemes"
    for voice in voices:
        rows = analyse_voice(voice, sentences, face)
        print(f"== {voice.name} ==")
        print(header)
        for r in rows:
            print(f"{'':22} {r['seconds']:5.2f} {r['frames']:6d} {r['poses']:5d} "
                  f"{r['changes_per_s']:6.1f} {100*r['sil_frac']:6.1f} {r['mean_top']:5.2f} "
                  f"{r['analysis_ms']:5.0f}  {len(r['distinct'])}/15: {','.join(r['distinct'])}")
        # A degenerate voice: mouth barely moves, or is stuck open / stuck shut.
        cps = np.mean([r["changes_per_s"] for r in rows])
        sil = np.mean([r["sil_frac"] for r in rows])
        verdict = "OK"
        if cps < 2.0:
            verdict = "SUSPECT (mouth barely moves)"
        elif sil > 0.9:
            verdict = "SUSPECT (stays shut)"
        elif sil < 0.02:
            verdict = "SUSPECT (never closes)"
        print(f"   -> {cps:.1f} changes/s, {100*sil:.0f}% idle  [{verdict}]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
