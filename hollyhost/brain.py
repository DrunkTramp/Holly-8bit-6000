"""The brain seam: chat history in, text deltas out.

One interface, so nothing else in the host cares which model is running:

    reply_stream(messages) -> Iterator[str]      # text deltas, in order

The rule that makes this module safe to run next to the settled clock, and the reason it is
a separate module rather than a few lines in `app.py`: **nothing in here touches the face,
the audio queue, or pygame.** `HollyFace`'s queue is touched only from the UI thread, while
the brain runs on its own thread because a 3B model takes seconds, not microseconds. If the
brain thread called `HollyFace.speak()` the pose queue would be a data race. So the boundary
is: the brain makes **text**, `SentencePump` turns text into **synthesised audio** (also off
the UI thread -- a voice costs ~0.1-0.2 s per sentence and the UI thread must not wait on it),
and only the UI thread is allowed to put either of them on a clock, via `LiveSession.say()`.

Three backends, in the order you are likely to want them:

- `LlamaCppBrain` -- a local GGUF through llama-cpp-python. No network, no API key.
- `OpenAICompatBrain` -- an OpenAI-compatible `/v1/chat/completions` endpoint (llama.cpp's
  own `llama-server`, Ollama's `/v1`, LM Studio). Pure stdlib, so the host gains no
  dependency for it, and it is how you keep the model in a separate process.
- `ScriptBrain` -- canned replies. Runs anywhere, with no model at all, which is what makes
  the whole chat loop testable (the same role `NullTTS` plays for the voice).
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from .normalize import preprocess
from .tts import SentenceSplitter


class Brain(Protocol):
    """The whole interface the rest of the host depends on.

    `name` is only for the log line, but pinning it here is what lets `app.py` and the pump
    stay backend-agnostic -- and what makes a backend that forgets `reply_stream` a type error
    rather than an AttributeError at the first prompt.
    """

    name: str

    def reply_stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield reply text in order, in deltas of any size (one token, or one word)."""
        ...

#: Phase D persona: Holly's *voice*, not just output the pipeline can stream. Two things this
#: prompt is doing on purpose:
#:   - it pushes the model toward spoken-form text (numbers spoken, abbreviations spelled out,
#:     no symbols/markdown), which is exactly what `normalize.preprocess` then guarantees even
#:     when a model leaks them anyway. Persona and preprocessing are the same Phase D idea from
#:     two sides: the prompt asks, the normalizer enforces.
#:   - it keeps replies short and spoken so the first sentence reaches TTS fast (the Phase C
#:     latency contract).
#: Whether this *sounds* like Holly can only be judged by a person running a real model -- see
#: HANDOFF "Open items"; the text is tuned for spoken output, the ear is the final judge.
DEFAULT_PERSONA = (
    "You are Holly, a holographic personality aboard a spaceship: vain, lazy, and 52 times the "
    "intelligence of a human being. Answer in character -- dry, deadpan, faintly contemptuous, "
    "allergic to effort -- in two or three short spoken sentences, never more. "
    "Never mention being an AI, a model or a program. "
    "Write for a voice, not a screen: use contractions, keep numbers plain with no symbols, "
    "spell out any abbreviation, and never use markdown, lists, headers, emoji, asterisks or "
    "parentheses."
)


def persona_from(spec: str | None) -> str:
    """`--persona` accepts literal text or `@path/to/file`."""
    if not spec:
        return DEFAULT_PERSONA
    spec = str(spec)
    if spec.startswith("@"):
        path = Path(spec[1:]).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"persona file not found: {path}")
        spec = path.read_text()
    return " ".join(spec.split())


# --------------------------------------------------------------------------- backends


class ScriptBrain:
    """Canned replies, handed over in token-sized deltas. No model, no downloads.

    The chat loop's stand-in: it exercises streaming, splitting, TTS and the queue exactly
    as a real model does, and it is deterministic, so the tests can assert on it.
    """

    name = "script"

    #: Deliberately punctuation-diverse: decimals, an abbreviation, and a long clause, so
    #: the streaming splitter is exercised by the default fixture.
    DEFAULT_SCRIPT = (
        "Rude. I am running at 576 by 432, which is 12 times the intelligence of a human. "
        "Dr. Hollister built me to be better than this. "
        "Ask me something with a number in it, like 12.5, and I will still say it properly.",
        "I have no memory of you, which is a feature. "
        "The mouth is a corner element: 168 by 132 pixels, and it only redraws when the pose changes.",
        "That is a reasonable question. I will answer in three sentences. "
        "The first sentence is long enough to be cut on a clause boundary, and the second one lands while the first is still playing.",
    )

    def __init__(self, script: list[str] | None = None, *, chunk: int = 7, delay: float = 0.0):
        self.script = list(script) if script else list(self.DEFAULT_SCRIPT)
        self.chunk = max(1, int(chunk))
        self.delay = float(delay)
        self._i = 0

    def reply_stream(self, messages: list[dict]) -> Iterator[str]:
        reply = self.script[self._i % len(self.script)]
        self._i += 1
        for i in range(0, len(reply), self.chunk):
            if self.delay:
                time.sleep(self.delay)
            yield reply[i : i + self.chunk]


class LlamaCppBrain:
    """A local GGUF through llama-cpp-python. Loaded at construction -- that is the slow part.

    `n_gpu_layers=-1` means "offload everything the runtime will". On a box with no GPU it
    still works; it is just CPU inference. `n_ctx` stays small on purpose (the plan's 2-4k):
    context is the memory that does not scale with model size, and Holly needs a conversation,
    not a document.
    """

    name = "llama"

    def __init__(self, model: str | Path, *, n_ctx: int = 2048, n_gpu_layers: int = -1,
                 max_tokens: int = 200, temperature: float = 0.7, n_threads: int | None = None,
                 seed: int | None = None, verbose: bool = False):
        path = Path(model)
        if not path.exists():
            raise FileNotFoundError(f"GGUF model not found: {path}")
        from llama_cpp import Llama  # lazy: the [llm] extra is optional

        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        kwargs = dict(model_path=str(path), n_ctx=int(n_ctx), n_gpu_layers=int(n_gpu_layers),
                      verbose=bool(verbose))
        if n_threads:
            kwargs["n_threads"] = int(n_threads)
        if seed is not None:
            kwargs["seed"] = int(seed)
        self._llm = Llama(**kwargs)

    def reply_stream(self, messages: list[dict]) -> Iterator[str]:
        stream = self._llm.create_chat_completion(
            messages=messages, max_tokens=self.max_tokens,
            temperature=self.temperature, stream=True,
        )
        for chunk in stream:
            choice = (chunk.get("choices") or [{}])[0]
            delta = (choice.get("delta") or {}).get("content")
            if delta:
                yield delta


class OpenAICompatBrain:
    """Any OpenAI-compatible chat endpoint, streamed as SSE. Stdlib only.

    This is the seam for keeping the model in another process: llama.cpp's `llama-server`
    (`--port 8080`), Ollama (`http://127.0.0.1:11434/v1`), LM Studio's server. `model` is
    optional because llama.cpp serves one model and ignores it.
    """

    name = "http"

    def __init__(self, base_url: str = "http://127.0.0.1:8080", model: str | None = None, *,
                 max_tokens: int = 200, temperature: float = 0.7, api_key: str | None = None,
                 timeout: float = 180.0):
        self.base_url = str(base_url).rstrip("/")
        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.api_key = api_key
        self.timeout = float(timeout)

    def _endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/v1/chat/completions"

    def reply_stream(self, messages: list[dict]) -> Iterator[str]:
        body: dict = {"messages": messages, "stream": True,
                      "max_tokens": self.max_tokens, "temperature": self.temperature}
        if self.model:
            body["model"] = self.model
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self._endpoint(), data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            # the status alone is useless -- LM Studio and friends put the real reason
            # (a model that failed to load, a bad model name) in the error body.
            try:
                detail = exc.read().decode("utf-8", "replace")[:300].strip()
            except Exception:  # noqa: BLE001 - a broken error body must not mask the status
                detail = ""
            raise RuntimeError(f"{self._endpoint()} HTTP {exc.code}" +
                               (f": {detail}" if detail else "")) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"{self._endpoint()} unreachable: {exc}") from exc
        with response:
            saw_sse = False
            other: list[str] = []
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    if line:
                        other.append(line)
                    continue
                saw_sse = True
                payload = line[5:].strip()
                if payload in ("", "[DONE]"):
                    if payload == "[DONE]":
                        return
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):  # an error *inside* the stream (LM Studio does this)
                    raise RuntimeError(f"{self._endpoint()}: {_error_text(chunk['error'])}")
                choice = (chunk.get("choices") or [{}])[0]
                delta = (choice.get("delta") or {}).get("content") or choice.get("text")
                if delta:
                    yield delta
            if not saw_sse and other:
                # Not SSE at all: either a server that ignored `stream` and answered with
                # one JSON document, or an error body. Read it instead of yielding silence.
                try:
                    doc = json.loads("\n".join(other))
                except json.JSONDecodeError:
                    raise RuntimeError(f"{self._endpoint()} replied non-JSON: "
                                       f"{' '.join(other)[:200]}") from None
                if doc.get("error"):
                    raise RuntimeError(f"{self._endpoint()}: {_error_text(doc['error'])}")
                content = ((doc.get("choices") or [{}])[0].get("message") or {}).get("content")
                if content:
                    yield content


def _error_text(err) -> str:
    """The message out of an OpenAI-style error object, whatever shape the server used."""
    if isinstance(err, dict):
        return str(err.get("message") or err.get("code") or err)
    return str(err)


def make_brain(kind: str | None, *, model: str | Path | None = None,
               base_url: str | None = None, http_model: str | None = None,
               ctx: int = 2048, gpu_layers: int = -1, max_tokens: int = 200,
               temperature: float = 0.7, threads: int | None = None,
               api_key: str | None = None,
               script: list[str] | None = None) -> Brain | None:
    """Backend factory for the app. `none` (the default) means: no brain, Phase A1 behaviour.

    That default matters -- with `--brain none` the host speaks exactly what you type, which
    is the A1 contract the existing tests pin. A brain only changes what gets spoken.
    """
    kind = (kind or "none").lower()
    if kind in ("none", "off", ""):
        return None
    if kind == "script":
        return ScriptBrain(script)
    if kind == "llama":
        if not model:
            raise ValueError("--brain llama needs --model path/to/model.gguf")
        return LlamaCppBrain(model, n_ctx=ctx, n_gpu_layers=gpu_layers, max_tokens=max_tokens,
                             temperature=temperature, n_threads=threads)
    if kind in ("http", "openai", "server"):
        return OpenAICompatBrain(base_url or "http://127.0.0.1:8080", http_model,
                                 max_tokens=max_tokens, temperature=temperature,
                                 api_key=api_key)
    raise ValueError(f"unknown brain {kind!r} (none, script, llama, http)")


def list_models(base_url: str, api_key: str | None = None, timeout: float = 10.0) -> list[str]:
    """The endpoint's own model list (OpenAI-compatible ``GET /v1/models``). Stdlib only.

    Used by the settings panel to offer the model names the server actually serves, instead
    of making the user type one. llama.cpp's server lists its single loaded model; Ollama's
    ``/v1`` shim and LM Studio behave the same, so one call covers the field.
    """
    base = str(base_url).rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    root = base if base.endswith("/v1") else base + "/v1"
    url = root + "/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(f"{url} unreachable or not JSON: {exc}") from exc
    entries = payload.get("data") or payload.get("models") or []
    names = []
    for entry in entries:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("id") or entry.get("name")
            if name:
                names.append(str(name))
    return sorted(names)


# --------------------------------------------------------------------------- the pump


class SentencePump:
    """One chat turn at a time, off the UI thread: brain -> sentences -> TTS -> queue.

    Why the TTS lives here too: Piper synthesis is a subprocess that takes hundreds of
    milliseconds, and the UI thread must not wait on it -- it would freeze the face
    mid-utterance. Doing synthesis on this thread is also the plan's pipeline overlap:
    sentence N+1 is synthesised while sentence N is still playing.

    What the UI thread is allowed to do with the result, and nothing else:

        for text, samples, rate in pump.sentences():
            session.say(text, samples=samples, rate=rate)   # UI thread: pins face `at`

    `LiveSession.say()` with `samples` given skips TTS but still owns the sample-rate
    contract (16k analysis, native playback), so the resample-once rule stays in `tts.py`
    and the clock pinning stays in `session.py`. This class never touches either.

    Streaming is what makes the first utterance fast: as soon as `SentenceSplitter` confirms
    a boundary it is synthesised and queued, so first-audio latency is first-token latency
    plus one sentence of TTS, not the whole reply.
    """

    def __init__(self, brain: Brain, tts, *, persona: str | None = None, max_chars: int = 220,
                 history_turns: int = 8, normalize: bool = True):
        if brain is None or tts is None:
            raise ValueError("SentencePump needs both a brain and a tts backend")
        self.brain = brain
        self.tts = tts
        self.persona = persona_from(persona)
        self.max_chars = int(max_chars)
        self.history_turns = int(history_turns)
        # Phase D: the text handed to TTS is preprocessed (numbers/abbreviations/symbols ->
        # spoken words) so the phonemes, and therefore the visemes, are unambiguous. The caption
        # and history keep the *original* text: subtitles read as what Holly "said", not as the
        # normalizer's expansion.
        self.normalize = bool(normalize)
        self.history: list[dict] = [{"role": "system", "content": self.persona}]
        self._out: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._busy = False
        self.error: str | None = None
        self.reply_text = ""
        self.stats: dict[str, float] = {}

    # -- UI thread ---------------------------------------------------------

    def submit(self, text: str) -> bool:
        """Start a turn. Returns False if Holly is still talking from the last one."""
        text = " ".join(str(text).split())
        if not text:
            return False
        if self.busy:
            return False
        self.history.append({"role": "user", "content": text})
        self._trim()
        self._cancel.clear()
        self.error = None
        self.reply_text = ""
        self.stats = {"submitted": time.perf_counter()}
        self._busy = True
        self._thread = threading.Thread(target=self._run, name="holly-chat", daemon=True)
        self._thread.start()
        return True

    def sentences(self) -> list[tuple[str, np.ndarray, int]]:
        """Drain everything ready, never blocking. `(text, samples, rate)` for `session.say`.

        After a barge-in (`interrupt()`) the turn is abandoned: the queue is drained and
        discarded instead of handed back, so nothing from the cut turn -- including a
        sentence whose synthesis was already in flight when the cancel landed -- can land
        on a clock that has just been truncated. The cancel clears at the next `submit`.
        """
        if self._cancel.is_set():
            while True:
                try:
                    self._out.get_nowait()
                except queue.Empty:
                    break
            return []
        ready = []
        while True:
            try:
                ready.append(self._out.get_nowait())
            except queue.Empty:
                return ready

    @property
    def busy(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def pending(self) -> int:
        """Synthesised sentences waiting for the UI thread to put them on the clock."""
        return self._out.qsize()

    @property
    def interrupted(self) -> bool:
        """True from a barge-in until the next `submit` (the UI stops showing 'thinking...')."""
        return self._cancel.is_set()

    def interrupt(self, timeout: float = 0.25) -> None:
        """Phase E barge-in: abandon the current turn without stalling the UI thread.

        Unlike `stop()`, the join is short and best-effort: the worker notices the cancel
        at the next token boundary (tens of ms on the GPU tier) or when the in-flight
        synthesis finishes, and `_produce` drops any sentence that completes after the
        cancel. Blocking the UI thread waiting for a whole OmniVoice sentence would
        freeze the face -- which the pump exists to prevent.
        """
        self._cancel.set()
        while True:
            try:
                self._out.get_nowait()
            except queue.Empty:
                break
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def stop(self, timeout: float = 2.0) -> None:
        self._cancel.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    # -- worker thread -----------------------------------------------------

    def _window(self) -> list[dict]:
        """Send the system prompt plus the last few turns: the context window stays small."""
        keep = self.history[1:][-2 * max(1, self.history_turns):]
        return [self.history[0]] + keep

    def _trim(self) -> None:
        keep = 2 * max(1, self.history_turns)
        if len(self.history) > keep + 1:
            self.history = [self.history[0]] + self.history[-keep:]

    def _run(self) -> None:
        splitter = SentenceSplitter(self.max_chars)
        parts: list[str] = []
        try:
            for delta in self.brain.reply_stream(self._window()):
                if self._cancel.is_set():
                    break
                if self.stats.get("first_delta") is None:
                    self.stats["first_delta"] = time.perf_counter()
                parts.append(delta)
                self.reply_text += delta
                for sentence in splitter.feed(delta):
                    self._produce(sentence)
            else:
                # Only reached if the stream finished on its own: end-of-stream is whitespace,
                # so it resolves a boundary the last delta left undecided ("...sentence."
                # with no trailing space is undecided until we know the stream is over).
                self._finish(splitter)
        except Exception as exc:  # a dead model must not take the UI down with it
            self.error = f"{type(exc).__name__}: {exc}"
            if not self._cancel.is_set():
                self._finish(splitter)  # speak what the model did manage to produce
        reply = "".join(parts).strip()
        if reply:
            self.history.append({"role": "assistant", "content": reply})
            self._trim()
        self.stats["finished"] = time.perf_counter()
        self._busy = False

    def _finish(self, splitter: SentenceSplitter) -> None:
        for sentence in splitter.end():
            if self._cancel.is_set():
                return
            self._produce(sentence)

    def _produce(self, sentence: str) -> None:
        sentence = sentence.strip()
        if not sentence:
            return
        if self.stats.get("first_sentence") is None:
            self.stats["first_sentence"] = time.perf_counter()
        # Synthesise the spoken form; queue the original for the caption/history.
        spoken = preprocess(sentence) if self.normalize else sentence
        try:
            samples, rate = self.tts.synthesize(spoken)
        except Exception as exc:
            self.error = f"tts {getattr(self.tts, 'name', '?')}: {type(exc).__name__}: {exc}"
            self._cancel.set()
            return
        if self.stats.get("first_ready") is None:
            self.stats["first_ready"] = time.perf_counter()
        if self._cancel.is_set():
            return  # barged in while synthesising: the turn is abandoned, drop it here
        self._out.put((sentence, samples, int(rate)))
