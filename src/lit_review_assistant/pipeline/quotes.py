"""Locate a quoted passage inside source text, tolerating the ways PDF text differs from what a model copies.

LLMs can't count characters, so instead of asking for offsets we ask for a verbatim quote and find it
here. A copied quote differs from the PDF text in predictable ways -- line breaks become spaces,
ligatures ("ﬁ") become plain letters, a word hyphenated across a line break is rejoined, curly quotes
become straight -- so matching falls back from exact, to normalized, to a strict fuzzy alignment.
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher

_PUNCTUATION = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "­": ""})

# Fuzzy matching: ignore aligned runs shorter than this, and require this share of the quote to align.
FUZZY_MIN_BLOCK = 8
FUZZY_MIN_COVERAGE = 0.8


def _is_space(char: str) -> bool:
    # Control characters (stray PDF math/symbol-font bytes) count as whitespace, matching how
    # strip_control_chars shows them to the model.
    return char.isspace() or unicodedata.category(char) == "Cc"


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize text for matching, returning it with a map from each normalized char to its source index."""
    chars: list[str] = []
    index_map: list[int] = []
    index = 0
    while index < len(text):
        if _is_space(text[index]):
            run_start = index
            while index < len(text) and _is_space(text[index]):
                index += 1
            # Collapse each whitespace run to one space -- except a line-break hyphenation
            # ("end-\nto-end"), which keeps the hyphen and drops the break.
            hyphenated_break = bool(chars) and chars[-1] == "-" and "\n" in text[run_start:index]
            if chars and index < len(text) and not hyphenated_break:
                chars.append(" ")
                index_map.append(run_start)
            continue
        for char in unicodedata.normalize("NFKC", text[index].translate(_PUNCTUATION)).casefold():
            chars.append(char)
            index_map.append(index)
        index += 1
    return "".join(chars), index_map


def strip_control_chars(text: str) -> str:
    """Replace control characters (other than newline and tab) with spaces, preserving length."""
    return "".join(" " if unicodedata.category(char) == "Cc" and char not in "\n\t" else char for char in text)


def find_quote_span(text: str, quote: str) -> tuple[int, int] | None:
    """Return [start, end) offsets of `quote` within `text`, or None if it can't be found reliably."""
    quote = quote.strip()
    if not quote:
        return None
    exact = text.find(quote)
    if exact >= 0:
        return exact, exact + len(quote)

    haystack, index_map = _normalize_with_map(text)
    needle, _ = _normalize_with_map(quote)
    if not needle:
        return None
    start = haystack.find(needle)
    if start >= 0:
        return index_map[start], index_map[start + len(needle) - 1] + 1

    matcher = SequenceMatcher(None, needle, haystack, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size >= FUZZY_MIN_BLOCK]
    if not blocks or sum(block.size for block in blocks) < FUZZY_MIN_COVERAGE * len(needle):
        return None
    first, last = blocks[0], blocks[-1]
    # Reject alignments scattered across far more text than the quote itself.
    if (last.b + last.size - first.b) > 1.5 * len(needle):
        return None
    return index_map[first.b], index_map[last.b + last.size - 1] + 1
