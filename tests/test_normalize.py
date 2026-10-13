"""Phase D: TTS text preprocessing.

The premise this module exists on: visemes follow phonemes, phonemes follow the *text* the voice
is handed, so a reply that leaks `576x432`, `12/s`, `12.5`, `etc.` or a stray `**bold**` gives
the TTS a string it misreads -- and the mouth inherits the mistake. These tests pin the two
contract points:

  1. what a phonemizer is likely to misread IS expanded to spoken words, and
  2. everything else is left alone (bare integers, real words that merely *contain* a unit).

They also pin the safety property that matters next to the settled splitter: `preprocess` runs on
a finished sentence, never re-splits it, and is idempotent on text it has already handled.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hollyhost.normalize import preprocess


class TestNumbers(unittest.TestCase):
    def test_a_decimal_point_becomes_spoken(self):
        self.assertEqual(preprocess("It is 12.5 litres."), "It is 12 point 5 litres.")

    def test_bare_integers_are_left_as_digits(self):
        """espeak/Piper already read 576 correctly; rewriting big numbers to words is a liability."""
        self.assertEqual(preprocess("576 by 432 pixels."), "576 by 432 pixels.")
        self.assertEqual(preprocess("I run at 52 times."), "I run at 52 times.")

    def test_a_dimension_x_becomes_by(self):
        """The letter-x in 576x432 would otherwise be read as the letter, not 'by'."""
        self.assertEqual(preprocess("576x432"), "576 by 432")
        self.assertEqual(preprocess("the 640x480 buffer"), "the 640 by 480 buffer")

    def test_a_star_between_digits_is_a_multiply(self):
        self.assertEqual(preprocess("2*3"), "2 by 3")

    def test_plus_between_numbers_is_spoken(self):
        self.assertEqual(preprocess("2 + 3 is five"), "2 plus 3 is five")

    def test_percent_is_spoken(self):
        self.assertEqual(preprocess("12% idle"), "12 percent idle")

    def test_a_version_number_reads_as_points(self):
        self.assertEqual(preprocess("1.2.3"), "1 point 2 point 3")


class TestUnitsAndAbbrev(unittest.TestCase):
    def test_unit_slashes_become_spoken_units(self):
        self.assertEqual(preprocess("12/s"), "12 per second")
        self.assertEqual(preprocess("100 m/s"), "100 metres per second")
        self.assertEqual(preprocess("90 km/h"), "90 kilometres per hour")

    def test_standalone_units_are_spelled_out(self):
        self.assertEqual(preprocess("30 fps"), "30 frames per second")
        self.assertEqual(preprocess("70 mph"), "70 miles per hour")
        self.assertEqual(preprocess("12 kg"), "12 kilograms")
        self.assertEqual(preprocess("a 3 km run"), "a 3 kilometres run")

    def test_a_unit_glued_inside_a_word_is_untouched(self):
        """`kg`, `mm`, `hz` must not fire inside real words."""
        self.assertEqual(preprocess("a lovely summer accent"), "a lovely summer accent")
        self.assertEqual(preprocess("the whiz kid"), "the whiz kid")

    def test_common_abbreviations_are_expanded(self):
        self.assertEqual(preprocess("tools e.g. hammers"), "tools for example hammers")
        self.assertEqual(preprocess("and so on etc."), "and so on et cetera")
        self.assertEqual(preprocess("Dr. Hollister built me."), "Doctor Hollister built me.")

    def test_abbreviations_are_not_matched_mid_word(self):
        self.assertEqual(preprocess("the Dracula film"), "the Dracula film")


class TestSymbolsAndMarkdown(unittest.TestCase):
    def test_ampersand_becomes_and(self):
        self.assertEqual(preprocess("R&D"), "R and D")

    def test_degrees_are_spoken(self):
        self.assertEqual(preprocess("20°C outside"), "20 degrees Celsius outside")
        self.assertEqual(preprocess("98.6°F"), "98 point 6 degrees Fahrenheit")
        self.assertEqual(preprocess("a 45° angle"), "a 45 degrees angle")

    def test_markdown_leaks_are_stripped_not_spoken(self):
        self.assertEqual(preprocess("**bold** and *ital*"), "bold and ital")
        self.assertEqual(preprocess("run `make test` now"), "run make test now")
        self.assertEqual(preprocess("# A heading"), "A heading")
        self.assertEqual(preprocess("- a bullet"), "a bullet")

    def test_curly_punctuation_is_plain(self):
        self.assertEqual(preprocess("that\u2019s Holly"), "that's Holly")
        self.assertEqual(preprocess("two\u2013three words"), "two-three words")


class TestSafety(unittest.TestCase):
    def test_plain_spoken_text_is_unchanged(self):
        text = "Hello there, how are you? I am fine."
        self.assertEqual(preprocess(text), text)

    def test_empty_and_none_are_empty(self):
        self.assertEqual(preprocess(""), "")
        self.assertEqual(preprocess("   "), "")

    def test_it_is_idempotent_on_text_it_already_handled(self):
        once = preprocess("It is 12.5 kg, 576x432, etc.")
        self.assertEqual(preprocess(once), once)

    def test_it_never_adds_a_sentence_boundary(self):
        """Preprocessing runs after splitting, so it must not re-split the utterance."""
        out = preprocess("One. Two! Three?")
        self.assertEqual(out.count("."), 1)
        self.assertEqual(out.count("!"), 1)
        self.assertEqual(out.count("?"), 1)


if __name__ == "__main__":
    unittest.main()
