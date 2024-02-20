from __future__ import annotations

import re

try:
    from pythainlp.tokenize import word_tokenize

    TH_AVAILABLE = True
except ImportError:
    TH_AVAILABLE = False
    word_tokenize = None


class ThaiTokenizer:
    def __init__(self, ignore_case: bool):
        self.ignore_case = ignore_case

    def __call__(self, text):
        new_text = []
        morphs = word_tokenize(text)
        pronunciations = []

        # some "words" have spaces in them which messes with processing reduplication
        actual_normalized = []
        for normalized in morphs:
            actual_normalized.extend(normalized.split())

        for normalized in actual_normalized:
            if normalized in {"ฯ"}:
                continue
            p = normalized
            if normalized == "ๆ" and new_text:  # Reduplication
                pronunciations[-1] += pronunciations[-1]
                new_text[-1] += normalized
                continue
            elif "ๆ" in normalized:
                p = re.sub(r"^(.+)ๆ", r"\1\1", normalized)
            join = False
            m = re.search(r"[]})>][<({[]", normalized)
            if new_text and m:
                new_text[-1] += normalized[: m.start() + 1]
                normalized = normalized[m.end() - 1 :]
            elif new_text and re.match(r"^[<({\[].*", new_text[-1]):
                join = True
            elif new_text and re.match(r".*[-_~]$", new_text[-1]):
                join = True
            elif new_text and re.match(r".*[>)}\]]$", normalized):
                join = True
            elif new_text and re.match(r"^[-_~].*", normalized):
                join = True
            if new_text and any(new_text[-1].endswith(x) for x in {">", ")", "}", "]"}):
                join = False
            if join:
                new_text[-1] += normalized
                pronunciations[-1] += p
                continue
            if re.match(r"^\W+$", normalized):
                continue
            new_text.append(normalized)
            pronunciations.append(p)
        assert len(new_text) == len(pronunciations)
        new_text = " ".join(new_text)
        pronunciations = " ".join(pronunciations)
        if self.ignore_case:
            new_text = new_text.lower()
            pronunciations = pronunciations.lower()
        return new_text, pronunciations


def th_spacy(ignore_case: bool = True):
    if not TH_AVAILABLE:
        raise ImportError("Please install Thai tokenization support via `pip install pythainlp`")
    return ThaiTokenizer(ignore_case)
