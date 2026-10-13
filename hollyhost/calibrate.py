"""Phase C's measurement, reused as the product's first-run `--calibrate`.

`tools/brain_check.py` already measures the one number that decides a machine's configuration:
how long until Holly's mouth starts moving, and whether the sentence queue keeps ahead of the
voice or goes **silent mid-reply**. That table *is* the machine's settings -- `--max-chars`,
`--max-tokens`, `--gpu-layers` and the model tier all follow from it -- so in the shipped product
the same measurement is reachable by the end user as `python -m hollyhost.app --calibrate`, which
runs it with a real brain + voice, prints the table, derives defaults, and writes them to a config
file the normal run then reads back.

This module owns the measurement so the developer tool and the product flow share one
implementation. It is deliberately **model-free in CI**: `measure()` runs the same `SentencePump`
and `HollyFace.analyse()` the UI pays, but with `--brain script --tts null` it needs no GGUF and no
GPU -- which is what the tests rely on. Only the *real* calibration (a real tier) needs the user to
run it on their own hardware.

The derivation in `recommend()` is honest about what the numbers can and cannot decide: the queue
margin sets `--max-chars` (the first-sentence length is the latency lever); `--max-tokens` is a
conversational-length choice, not a sync one, so it is echoed rather than tuned; `--gpu-layers` and
the model are recorded as the tier that was measured, not invented.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

#: Where `--calibrate` writes and a normal run reads, unless `--config` says otherwise. A
#: per-machine file (the whole point is that it records *this* box's tier), so it is gitignored.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "hollyhost_config.json"

DEFAULT_PROMPT = "Holly, how many times the intelligence of a human being are you today?"
CHARS_PER_TOKEN = 4.0  # rough, for a tokens/s column comparable across backends

#: Fallbacks when no config exists -- identical to the pre-calibration CLI defaults, so adding
#: the config step changes nothing until a calibration has actually been run.
HARD_DEFAULTS = {"max_chars": 220, "max_tokens": 200, "gpu_layers": -1}


# --------------------------------------------------------------------------- measurement


def starvation(arrivals: list[dict], gap: float) -> tuple[float, int]:
    """The tightest margin in the queue, and the sentence it happens before.

    Signed on purpose: **positive == Holly goes silent** for that long before sentence `before`;
    negative == the queue was that far ahead at the tightest moment. The negative reading is what
    makes tiers comparable -- a tier with a 0.2 s margin and one with a 6 s margin both "pass", but
    only one of them has room for a bigger model.
    """
    worst, before = 0.0, 0
    for k in range(len(arrivals) - 1):
        # Everything queued so far plays back-to-back from the first arrival, `gap` apart.
        plays_until = arrivals[0]["ready"] + sum(a["audio"] for a in arrivals[: k + 1]) + k * gap
        starved = arrivals[k + 1]["ready"] - plays_until
        if starved > worst:
            worst, before = starved, k + 2
    if worst <= 0.0 and len(arrivals) > 1:
        worst, before = min(
            (arrivals[k + 1]["ready"]
             - (arrivals[0]["ready"] + sum(a["audio"] for a in arrivals[: k + 1]) + k * gap), k + 2)
            for k in range(len(arrivals) - 1))
    return worst, before


def measure(brain, tts, face, prompt: str, reps: int, gap: float,
            max_chars: int = 220) -> list[dict]:
    """Run `reps` turns and return one row per rep: latencies, per-sentence arrivals, margin.

    The clock is not needed to measure the budget, so this runs with no window and no audio
    device -- the same `SentencePump` the UI runs, plus the same `HollyFace.analyse()` the UI
    thread pays.
    """
    rows = []
    for rep in range(reps):
        pump = _make_pump(brain, tts, max_chars)
        started = time.perf_counter()
        pump.submit(prompt)
        arrivals: list[dict] = []
        while time.perf_counter() - started < 300.0:
            for text, samples, rate in pump.sentences():
                ready = time.perf_counter() - started
                prepared = _prepare(text, samples, rate)      # 16k analysis + native playback
                _weights, _keys, analysis_ms, _fps = face.analyse(prepared.analysis, 16000)
                arrivals.append({"text": text, "ready": ready,
                                 "audio": prepared.duration, "analysis": analysis_ms})
            if not pump.busy and pump.pending == 0:
                break
            time.sleep(0.005)
        if pump.error:
            print(f"  brain error: {pump.error}", file=sys.stderr)
        stats = pump.stats
        generated = stats.get("finished", started) - started
        margin, before = starvation(arrivals, gap)
        rows.append({
            "rep": rep,
            "first_token_ms": 1000 * (stats.get("first_delta", started) - started),
            "first_sentence_ms": 1000 * (stats.get("first_sentence", started) - started),
            "first_audio_ms": 1000 * arrivals[0]["ready"] if arrivals else None,
            "reply": pump.reply_text,
            "generated": generated,
            "arrivals": arrivals,
            "margin": margin,
            "starved_before": before,
        })
    return rows


def _make_pump(brain, tts, max_chars: int):
    from .brain import SentencePump
    return SentencePump(brain, tts, max_chars=max_chars)


def _prepare(text, samples, rate):
    from .tts import prepare
    return prepare(text, samples, rate)


# --------------------------------------------------------------------------- reporting


def verdict_for(row: dict, gap: float) -> str:
    arrivals = row["arrivals"]
    if len(arrivals) < 2:
        # One utterance means nothing was ever queued behind anything: the margin the verdict
        # reports would be invented. Say so instead of printing a passing grade.
        return "n/a single sentence -- no overlap to measure (try --max-chars 120)"
    if row["margin"] > 0:
        return f"STALL {row['margin']:.2f} s before sentence {row['starved_before']}"
    spoken = sum(a["audio"] for a in arrivals)
    return f"OK queue stays ahead (tightest margin {-row['margin']:.1f} s, {spoken:.1f} s spoken)"


def report(rows: list[dict], gap: float) -> None:
    """Print the latency table + verdicts (the same view for the dev tool and `--calibrate`)."""
    print(f"{'rep':>3} {'1tok ms':>8} {'1sent ms':>9} {'1audio ms':>10} {'sents':>6} "
          f"{'1audio s':>9} {'tts ms':>7} {'anal ms':>8} {'chars':>6} {'tok/s':>6}  verdict")
    for row in rows:
        arrivals = row["arrivals"]
        first = arrivals[0] if arrivals else None
        reply = row["reply"]
        tokens = len(reply) / CHARS_PER_TOKEN
        tok_s = tokens / row["generated"] if row["generated"] > 0 else 0.0
        tts_ms = 1000 * (first["ready"] - row["first_sentence_ms"] / 1000) if first else 0.0
        print(f"{row['rep']:>3} {row['first_token_ms']:8.0f} {row['first_sentence_ms']:9.0f} "
              f"{(1000 * first['ready']) if first else -1:10.0f} {len(arrivals):>6} "
              f"{(first['audio'] if first else 0):9.2f} {tts_ms:7.0f} "
              f"{(first['analysis'] if first else 0):8.0f} {len(reply):>6} {tok_s:6.1f}  "
              f"{verdict_for(row, gap)}")
        for i, a in enumerate(arrivals, start=1):
            print(f"      #{i} at {a['ready']:6.2f} s  {a['audio']:4.2f} s audio  {a['text'][:70]}")
    print("\n'1audio ms' is what the user waits for. A STALL means Holly goes silent mid-reply:")
    print("cap --max-tokens at what the queue can keep ahead of, or move to a faster hardware tier.")
    print("Run this on the target machine to choose ITS defaults (model, --gpu-layers,")
    print("--max-tokens, --max-chars): the queue margin is a property of the hardware, not the code.")


# --------------------------------------------------------------------------- derivation


def summarize(rows: list[dict]) -> dict:
    """Worst-case numbers across reps: what a config decision should key off, not the averages."""
    margins = [r["margin"] for r in rows]
    firsts = [r["first_audio_ms"] for r in rows if r["first_audio_ms"] is not None]
    multi = [r for r in rows if len(r["arrivals"]) >= 2]
    return {
        "reps": len(rows),
        "measurable": len(multi),           # reps with an overlap to judge the margin on
        "worst_first_audio_ms": round(max(firsts), 1) if firsts else None,
        "tightest_margin_s": round(max(margins), 2) if margins else 0.0,  # positive == a stall
        "stall": any(r["margin"] > 0 for r in rows),
    }


def recommend(rows: list[dict], *, max_chars: int, max_tokens: int, gpu_layers: int,
              model: str | None, tts: str, brain: str) -> dict:
    """Turn measurements into a config. Honest about what the numbers decide (see module doc)."""
    s = summarize(rows)
    notes: list[str] = []
    margin = s["tightest_margin_s"]

    if s["measurable"] == 0:
        rec_max_chars = max_chars
        notes.append("no rep produced two queued sentences, so the queue margin was not measured; "
                     "--max-chars kept as run. Use a longer reply or --max-chars 120.")
    elif s["stall"]:
        rec_max_chars = 120 if max_chars > 120 else max(80, max_chars // 2)
        notes.append(f"queue STALLed (tightest margin {margin:.2f} s): shorten the first sentence "
                     f"to --max-chars {rec_max_chars} and re-measure; a smaller/faster model or a "
                     f"cheaper voice are the other levers -- never the queue or the renderer.")
    elif margin > -1.0:
        rec_max_chars = min(max_chars, 120)
        notes.append(f"queue margin is thin ({-margin:.1f} s): --max-chars {rec_max_chars} buys "
                     f"headroom for a longer first sentence.")
    else:
        rec_max_chars = max_chars
        notes.append(f"queue stays ahead (tightest margin {-margin:.1f} s): --max-chars {max_chars} "
                     f"is fine on this box.")

    notes.append("--max-tokens sets conversational length, not sync; left at the value you ran with.")

    return {
        "max_chars": rec_max_chars,
        "max_tokens": max_tokens,
        "gpu_layers": gpu_layers,
        "model": str(model) if model else None,
        "tts": tts,
        "brain": brain,
        "measured": s,
        "notes": notes,
    }


# --------------------------------------------------------------------------- config file


def load_config(path: str | Path | None = None) -> dict:
    """Read a calibration config, or {} when absent/unreadable (a normal run then uses defaults)."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_config(cfg: dict, path: str | Path | None = None) -> Path:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    return path
