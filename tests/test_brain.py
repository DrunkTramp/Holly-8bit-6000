"""Phase C: the streaming splitter, the brain seam, and the chat thread's boundary.

The splitter tests are the important ones. `split_sentences` (batch) and `SentenceSplitter`
(streaming) share one boundary predicate, so the property that matters is: **chunking the
same text must not change the sentences it produces.** A delta stream that splits a decimal
across chunks ("12." + "5") or a word across chunks ("won" + "derful") is the normal case for
a token stream, and a splitter that gets it wrong sends a fragment to TTS -- which synthesises
badly and gives the viseme classifier nothing to chew on.

The pump tests pin the threading contract the HANDOFF demands: the chat thread may run the
brain and the TTS, and must never touch the face or the audio clock.
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hollyhost.brain import (DEFAULT_PERSONA, OpenAICompatBrain, ScriptBrain, SentencePump,
                             make_brain, persona_from)
from hollyhost.tts import SentenceSplitter, split_sentences

CORPUS = [
    "One. Two! Three?",
    "He weighed 12.5 kg. Rude, isn't it?",
    "Dr. Who said no. Honestly.",
    "Well... that is one sentence. Next one.",
    "Short.\nThen a newline.",
    "No punctuation at all, just a clause, and then it carries on",
    "Hello there. How are you? I am fine! Really.",
    "J. R. R. wrote books. That is true.",
    "12.5 and 3.0 stay put. Mrs. Holt agreed.",
    "Multiple\nlines\nhere.",
    "Trailing space.   ",
    "A  double   spaced   line.",
    "   leading and one sentence.",
]


def chunkings(text: str):
    """Ways a token stream might actually deliver this text, including the nasty ones."""
    yield "whole", [text]
    for size in (1, 2, 3, 4, 7, 13):
        yield f"every-{size}", [text[i:i + size] for i in range(0, len(text), size)]
    words = text.split(" ")
    yield "word-wise", [w + (" " if i < len(words) - 1 else "") for i, w in enumerate(words)]
    # Every 4 chars deliberately straddles both word seams and sentence seams at once.
    yield "straddling", [text[i:i + 4] for i in range(0, len(text), 4)]


def stream(pieces: list[str], **kwargs) -> list[str]:
    splitter = SentenceSplitter(**kwargs)
    out: list[str] = []
    for piece in pieces:
        out.extend(splitter.feed(piece))
    out.extend(splitter.end())
    return out


class TestStreamingSplitter(unittest.TestCase):
    def test_chunking_does_not_change_the_sentences(self):
        """The property Phase C rests on: streaming == batch, for any delta boundaries."""
        for text in CORPUS:
            for label, pieces in chunkings(text):
                with self.subTest(text=text[:24], chunking=label):
                    self.assertEqual(stream(pieces), split_sentences(text))

    def test_a_decimal_split_across_chunks_does_not_split_the_sentence(self):
        self.assertEqual(stream(["12.", "5 kg."]), ["12.5 kg."])
        self.assertEqual(stream(["It is 12.", "5 litres."]), ["It is 12.5 litres."])

    def test_an_abbreviation_at_a_chunk_seam_is_not_a_boundary(self):
        self.assertEqual(stream(["Dr.", " Who."]), ["Dr. Who."])
        self.assertEqual(stream(["Stop!", " Again."]), ["Stop!", "Again."])  # a real boundary

    def test_end_of_stream_resolves_a_held_boundary(self):
        self.assertEqual(stream(["Done."]), ["Done."])
        self.assertEqual(stream(["Done", "."]), ["Done."])

    def test_a_newline_ends_the_sentence_candidate(self):
        self.assertEqual(stream(["one", "\n", "two"]), ["one", "two"])
        self.assertEqual(stream(["one\n"]), ["one"])

    def test_whitespace_is_collapsed_across_the_chunk_seam(self):
        self.assertEqual(stream(["a ", " b."]), ["a b."])
        self.assertEqual(stream(["a", " ", "b."]), ["a b."])
        self.assertEqual(stream(["won", "derful."]), ["wonderful."])
        self.assertEqual(stream(["a\r", "\nb."]), ["a", "b."])

    def test_long_clause_cut_matches_the_batch_form(self):
        text = "word " * 200
        for pieces in ([text], [text[i:i + 5] for i in range(0, len(text), 5)]):
            self.assertEqual(stream(pieces, max_chars=120), split_sentences(text, max_chars=120))

    def test_clause_cut_on_commas_matches_the_batch_form(self):
        """The comma-cut path must agree too, not just the space-cut fallback."""
        text = ("It's a matter of personal preference, but I have the capacity to process an "
                "enormous amount of information at once, which is quite impressive.")
        for _label, pieces in chunkings(text):
            self.assertEqual(stream(pieces, max_chars=120), split_sentences(text, max_chars=120))

    def test_junk_is_never_its_own_utterance(self):
        """A one-character fragment goes nowhere near TTS: it is merged or dropped.

        `_boundary` already refuses a single-letter stem ("J.", "!"), so a 1-char candidate
        can only arrive from a flush -- a line or a stream that is just punctuation.
        """
        self.assertEqual(stream(["Fine.", " !"]), ["Fine."])       # dropped, something was said
        self.assertEqual(stream(["Fine.\n", "!"]), ["Fine."])
        self.assertEqual(stream(["!"]), ["!"])                     # all there is: say it
        self.assertEqual(stream(["Fine.", "!"]), ["Fine.!"])      # no space seen: one utterance

    def test_pass_junk_keeps_the_batch_contract(self):
        self.assertEqual(stream(["a\nb"], junk="pass"), ["a", "b"])

    def test_empty_and_whitespace(self):
        self.assertEqual(stream([]), [])
        self.assertEqual(stream(["", "", ""]), [])
        self.assertEqual(stream(["   "]), [])

    def test_bad_junk_policy_rejected(self):
        with self.assertRaises(ValueError):
            SentenceSplitter(junk="shred")


class _FakeTTS:
    """Records the thread it ran on: the pump must synthesise off the UI thread."""

    name = "fake"

    def __init__(self, rate: int = 22050):
        self.rate = rate
        self.calls: list[tuple[str, str]] = []

    def synthesize(self, text: str):
        self.calls.append((text, threading.current_thread().name))
        n = max(160, round(0.05 * self.rate))
        return (0.1 * np.sin(np.arange(n) / 40.0)).astype(np.float32), self.rate


class _SlowBrain:
    name = "slow"

    def __init__(self, reply: str, chunk: int = 6, delay: float = 0.02):
        self.reply, self.chunk, self.delay = reply, chunk, delay

    def reply_stream(self, messages):
        for i in range(0, len(self.reply), self.chunk):
            time.sleep(self.delay)
            yield self.reply[i:i + self.chunk]


class _BrokenBrain:
    name = "broken"

    def reply_stream(self, messages):
        yield "Starts fine."
        raise RuntimeError("model exploded")


class TestScriptBrain(unittest.TestCase):
    def test_deterministic_and_cycling(self):
        a, b = ScriptBrain(), ScriptBrain()
        first = "".join(a.reply_stream([]))
        self.assertEqual(first, "".join(b.reply_stream([])))
        self.assertEqual("".join(a.reply_stream([])), ScriptBrain.DEFAULT_SCRIPT[1])

    def test_deltas_rejoin_the_reply(self):
        brain = ScriptBrain(["One. Two! Three?"], chunk=4)
        self.assertEqual("".join(brain.reply_stream([])), "One. Two! Three?")

    def test_factory(self):
        self.assertIsNone(make_brain(None))
        self.assertIsNone(make_brain("none"))
        self.assertIsInstance(make_brain("script"), ScriptBrain)
        with self.assertRaises(ValueError):
            make_brain("gpt")
        with self.assertRaises(ValueError):
            make_brain("llama")  # no --model
        with self.assertRaises(FileNotFoundError):
            make_brain("llama", model="does-not-exist.gguf")

    def test_persona(self):
        self.assertEqual(persona_from(None), DEFAULT_PERSONA)
        self.assertEqual(persona_from("one\n  line"), "one line")
        with self.assertRaises(FileNotFoundError):
            persona_from("@nope.txt")

    def test_default_persona_lives_in_the_md_file(self):
        """The character comes from holly_personality_prompt.md, not from a string in code."""
        from hollyhost.brain import DEFAULT_PERSONA_FILE, SPOKEN_FORM_RULES

        if not DEFAULT_PERSONA_FILE.exists():
            self.skipTest("persona file not in this checkout")
        body = " ".join(DEFAULT_PERSONA_FILE.read_text().split())
        self.assertTrue(DEFAULT_PERSONA.startswith(body[:200]),
                        "the default persona should open with the file's text")
        self.assertIn(SPOKEN_FORM_RULES, DEFAULT_PERSONA,
                      "the spoken-form rules must ride along with the file persona")


class TestOpenAICompatBrainErrors(unittest.TestCase):
    """The endpoint seam: a server-side failure (LM Studio's model refusing to load, a bad
    model name) must surface as an error the chat log can show -- never a silent empty reply."""

    @staticmethod
    def _serve(body_lines: list[bytes], status: int = 200):
        import http.server

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self, *a):
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for ln in body_lines:
                    self.wfile.write(ln)

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def _drain(self, srv):
        brain = OpenAICompatBrain(f"http://127.0.0.1:{srv.server_address[1]}/v1", "m")
        return list(brain.reply_stream([{"role": "user", "content": "Boop"}]))

    def test_error_chunk_inside_sse_raises(self):
        srv = self._serve([b'data: {"error": {"message": "V Cache Quantization requires '
                                b'flash attention to be enabled"}}\n\n', b"data: [DONE]\n\n"])
        try:
            with self.assertRaises(RuntimeError) as cm:
                self._drain(srv)
            self.assertIn("flash attention", str(cm.exception))
        finally:
            srv.shutdown()

    def test_plain_json_error_body_raises(self):
        srv = self._serve([b'{"error": {"message": "model failed to load"}}'])
        try:
            with self.assertRaises(RuntimeError) as cm:
                self._drain(srv)
            self.assertIn("model failed to load", str(cm.exception))
        finally:
            srv.shutdown()

    def test_plain_json_completion_is_read_when_stream_is_ignored(self):
        srv = self._serve([b'{"choices": [{"message": {"content": "Boop back."}}]}'])
        try:
            self.assertEqual("".join(self._drain(srv)), "Boop back.")
        finally:
            srv.shutdown()

    def test_http_error_status_includes_the_body(self):
        srv = self._serve([b'{"error": {"message": "no such model"}}'], status=500)
        try:
            with self.assertRaises(RuntimeError) as cm:
                self._drain(srv)
            self.assertIn("HTTP 500", str(cm.exception))
            self.assertIn("no such model", str(cm.exception))
        finally:
            srv.shutdown()


class TestSentencePump(unittest.TestCase):
    """The chat thread: brain -> splitter -> TTS -> queue, and nothing else."""

    def drain(self, pump, limit=6.0):
        """Collect sentences the way the UI thread would, without touching a face."""
        said, started = [], time.perf_counter()
        while time.perf_counter() - started < limit:
            said.extend(pump.sentences())
            if not pump.busy and pump.pending == 0:
                break
            time.sleep(0.01)
        return said

    def test_a_turn_produces_the_batch_sentences_in_order(self):
        reply = "First sentence here. Second one too! Third? Fourth."
        pump = SentencePump(_SlowBrain(reply), _FakeTTS())
        self.assertTrue(pump.submit("hello"))
        got = self.drain(pump)
        self.assertEqual([t for t, _s, _r in got], split_sentences(reply))
        self.assertEqual(pump.error, None)
        self.assertEqual(pump.reply_text, reply)

    def test_the_first_sentence_is_ready_before_the_reply_is_generated(self):
        """Pipeline overlap: first-audio latency is one sentence, not the whole answer."""
        reply = "One full sentence arrives early. " + "and this keeps going on and on. " * 6
        brain = _SlowBrain(reply, chunk=6, delay=0.02)
        pump = SentencePump(brain, _FakeTTS())
        pump.submit("go")
        started = time.perf_counter()
        while not pump.sentences():
            self.assertLess(time.perf_counter() - started, 4.0, "no sentence ever arrived")
            time.sleep(0.005)
        self.assertTrue(pump.busy, "the whole reply was already generated -- not streaming")

    def test_synthesis_happens_off_the_ui_thread(self):
        """The HANDOFF's rule, pinned: the chat thread may synthesise, never the face."""
        tts = _FakeTTS()
        pump = SentencePump(_SlowBrain("One. Two."), tts)
        pump.submit("go")
        self.drain(pump)
        self.assertTrue(tts.calls, "nothing was synthesised")
        for _text, thread in tts.calls:
            self.assertNotEqual(thread, threading.current_thread().name,
                                "TTS ran on the UI thread -- it would freeze the face")

    def test_history_grows_and_stays_trimmed(self):
        pump = SentencePump(ScriptBrain(["A. B."]), _FakeTTS(), history_turns=2)
        for i in range(5):
            pump.submit(f"question {i}")
            self.drain(pump)
        self.assertEqual(pump.history[0]["role"], "system")
        self.assertEqual(pump.history[-1], {"role": "assistant", "content": "A. B."})
        # system + 2 turns kept = at most 5 messages, and the newest turn is present.
        self.assertLessEqual(len(pump.history), 5)
        self.assertIn("question 4", [m["content"] for m in pump.history])

    def test_one_turn_at_a_time(self):
        pump = SentencePump(_SlowBrain("Slow. reply."), _FakeTTS())
        self.assertTrue(pump.submit("first"))
        self.assertFalse(pump.submit("second"), "a second turn was allowed while busy")
        self.drain(pump)
        self.assertTrue(pump.submit("third"))
        self.drain(pump)
        users = [m["content"] for m in pump.history if m["role"] == "user"]
        self.assertEqual(users, ["first", "third"])

    def test_a_brain_exception_becomes_an_error_not_a_crash(self):
        pump = SentencePump(_BrokenBrain(), _FakeTTS())
        pump.submit("go")
        got = self.drain(pump)
        self.assertEqual([t for t, _s, _r in got], ["Starts fine."])
        self.assertIn("model exploded", pump.error or "")

    def test_a_tts_failure_is_reported_on_the_turn(self):
        class BrokenTTS:
            name = "broken"

            def synthesize(self, text):
                raise RuntimeError("no voice on disk")

        pump = SentencePump(ScriptBrain(["One. Two."], chunk=200), BrokenTTS())
        pump.submit("go")
        self.assertEqual(self.drain(pump), [])
        self.assertIn("no voice on disk", pump.error or "")

    def test_stop_cancels_a_running_turn(self):
        pump = SentencePump(_SlowBrain("word " * 400, chunk=4, delay=0.005), _FakeTTS())
        pump.submit("go")
        time.sleep(0.2)
        pump.stop()
        self.assertFalse(pump.busy)
        time.sleep(0.2)
        self.assertEqual(pump.pending, 0, "the pump kept producing after stop()")

    def test_empty_prompt_is_not_a_turn(self):
        pump = SentencePump(ScriptBrain(), _FakeTTS())
        self.assertFalse(pump.submit("   "))
        self.assertFalse(pump.busy)

    def test_pump_needs_both_halves(self):
        with self.assertRaises(ValueError):
            SentencePump(ScriptBrain(), None)

    def test_tts_gets_preprocessed_text_but_the_caption_stays_original(self):
        """Phase D: the voice hears spoken words; the caption reads what Holly 'said'."""
        reply = "I run at 576x432, that is 12/s, Dr. Hollister built me."
        tts = _FakeTTS()
        pump = SentencePump(_SlowBrain(reply, chunk=200), tts)
        pump.submit("go")
        got = self.drain(pump)
        # The caption/queue text is the original sentence, unchanged.
        self.assertEqual([t for t, _s, _r in got], [reply])
        # What actually reached TTS is the spoken form.
        spoken = " ".join(text for text, _thread in tts.calls)
        self.assertIn("576 by 432", spoken)
        self.assertIn("12 per second", spoken)
        self.assertIn("Doctor Hollister", spoken)

    def test_normalization_can_be_turned_off(self):
        """`--no-normalize`: the raw text is spoken verbatim."""
        reply = "I run at 576x432, that is 12/s."
        tts = _FakeTTS()
        pump = SentencePump(_SlowBrain(reply, chunk=200), tts, normalize=False)
        pump.submit("go")
        self.drain(pump)
        spoken = " ".join(text for text, _thread in tts.calls)
        self.assertIn("576x432", spoken)
        self.assertIn("12/s", spoken)


if __name__ == "__main__":
    unittest.main()
