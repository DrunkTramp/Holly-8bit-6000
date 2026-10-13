"""Phase D: TTS text preprocessing.

Visemes follow phonemes, and the phonemes follow the *text* the voice is handed. A model that
leaks `576x432`, `12/s`, `12.5`, `etc.` or a stray `**bold**` into its reply gives the TTS a
string it either misreads or spells letter-by-letter, and the mouth inherits whatever the voice
made of it. This module rewrites those into explicit spoken words **after** the sentence splitter
has done its job (so it can never disturb a decimal or a `Dr.` boundary the splitter was careful
to keep) and **before** synthesis, so every backend -- Piper, OmniVoice, a future fine-tune --
receives text whose phonemes are unambiguous.

It is deliberately conservative. It expands what a phonemizer tends to misread (decimals, unit
slashes, symbols, a handful of abbreviations, markdown a model leaked despite the persona) and
leaves everything else alone. Bare integers stay digits: Piper/espeak already read `576` as
"five hundred seventy six", and rewriting large numbers to words is a liability, not a help.

The rule that makes this safe next to the settled splitter: run it on a **whole sentence**, never
on a partial token stream. The splitter owns boundaries; this owns phonemes.
"""

from __future__ import annotations

import re

#: Curly punctuation and unicode symbols a phonemizer handles inconsistently.
_UNICODE = {
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2026": "...",  # horizontal ellipsis
    "\u00a0": " ",   # non-breaking space
    "\u00d7": "x",   # multiplication sign -> the digit-x-digit rule below turns it into "by"
}

#: Whole-word abbreviations a phonemizer reads as letters or gets wrong. Matched on word
#: boundaries. Titles beyond `Dr.`/`St.` are left to the phonemizer, which handles them.
_ABBREV = {
    "e.g.": "for example",
    "i.e.": "that is",
    "etc.": "et cetera",
    "vs.": "versus",
    "approx.": "approximately",
    "Dr.": "Doctor",
    "St.": "Saint",
}

#: Standalone unit tokens (matched case-insensitively, never glued to letters). None of these
#: collide with English words, so a word-boundary match is safe.
_UNITS = {
    "fps": "frames per second",
    "mph": "miles per hour",
    "kph": "kilometres per hour",
    "km/h": "kilometres per hour",
    "m/s": "metres per second",
    "km": "kilometres",
    "cm": "centimetres",
    "mm": "millimetres",
    "kg": "kilograms",
    "hz": "hertz",
    "khz": "kilohertz",
    "mhz": "megahertz",
    "ghz": "gigahertz",
}

# Markdown a model leaks despite the persona's "no markdown" -- removed so the voice never says
# "asterisk" or "hash". Line-start markers (headings, bullets, numbered lists) are stripped as
# markers, not as content.
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.MULTILINE)
_BULLET = re.compile(r"^[ \t]*[-*+][ \t]+", re.MULTILINE)
_NUMLIST = re.compile(r"^[ \t]*\d+[.)][ \t]+", re.MULTILINE)

# Symbols between/after numbers, and the decimal point. The digit-x-digit rule is what turns a
# dimension like `576x432` into "576 by 432" (the `x` would otherwise be read as the letter).
_MULT_X = re.compile(r"(?<=\d)[xX](?=\d)")
_MULT_STAR = re.compile(r"(?<=\d)\*(?=\d)")
_PCT = re.compile(r"(?<=\d)%")
_PER_SLASH_S = re.compile(r"(?<=\d)/s(?![A-Za-z])")  # a rate like 12/s -> "12 per second"
_PLUS = re.compile(r"(?<=\d)[ \t]*\+[ \t]*(?=\d)")
_DEGREES = re.compile("\u00b0[CF]?")  # °C / °F / lone degree sign
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")

# After all the insertions, tidy spacing so the voice never opens on a stray pause.
_SPACE_RUN = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.;:!?])")


def _degree(match: re.Match) -> str:
    symbol = match.group(0)
    if symbol.endswith("C"):
        return " degrees Celsius"
    if symbol.endswith("F"):
        return " degrees Fahrenheit"
    return " degrees"


def _replace_word(text: str, token: str, spoken: str) -> str:
    """Replace a whole-word token (e.g. an abbreviation) with its spoken form."""
    pattern = re.compile(r"(?<!\w)" + re.escape(token) + r"(?!\w)")
    return pattern.sub(" " + spoken + " ", text)


def _replace_unit(text: str, unit: str, spoken: str) -> str:
    """Replace a standalone unit token, but never one glued inside a word (`skg`, `whiz`)."""
    pattern = re.compile(r"(?<![A-Za-z])" + re.escape(unit) + r"(?![A-Za-z])", re.IGNORECASE)
    return pattern.sub(" " + spoken + " ", text)


def preprocess(text: str) -> str:
    """Rewrite one finished sentence into spoken form for TTS.

    Deterministic, never raises, and conservative: it only changes what a phonemizer is likely
    to mispronounce. Running it twice is a no-op on already-spoken text (idempotent for the
    cases it handles).
    """
    if not text:
        return ""
    s = str(text)

    # 1. Curly punctuation and unicode symbols -> plain ASCII the phonemizer handles.
    for bad, good in _UNICODE.items():
        s = s.replace(bad, good)

    # 2. Markdown a model leaked: strip the markers, keep the words.
    s = _HEADING.sub("", s)
    s = _BULLET.sub("", s)
    s = _NUMLIST.sub("", s)
    s = _MULT_STAR.sub(" by ", s)
    s = s.replace("**", "").replace("__", "").replace("`", "").replace("*", "")

    # 3. Whole-word abbreviations and standalone units.
    for token, spoken in _ABBREV.items():
        s = _replace_word(s, token, spoken)
    for unit, spoken in _UNITS.items():
        s = _replace_unit(s, unit, spoken)

    # 4. Symbols and the decimal point, mostly in numeric contexts.
    s = _MULT_X.sub(" by ", s)
    s = _PCT.sub(" percent", s)
    s = _PER_SLASH_S.sub(" per second", s)
    s = s.replace("&", " and ")
    s = _DEGREES.sub(_degree, s)
    s = _PLUS.sub(" plus ", s)
    s = _DECIMAL.sub(" point ", s)

    # 5. Tidy spacing: collapse runs, pull punctuation back onto the word that owns it.
    s = _SPACE_BEFORE_PUNCT.sub(r"\1", s)
    s = _SPACE_RUN.sub(" ", s)
    return s.strip()
