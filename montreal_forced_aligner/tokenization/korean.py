from __future__ import annotations

import re

try:
    import jamo
    from mecab import MeCab

    KO_AVAILABLE = True
except ImportError:
    KO_AVAILABLE = False
    MeCab = None
    jamo = None


class KoreanTokenizer:
    def __init__(self, ignore_case: bool = True):
        self.ignore_case = ignore_case
        self.tokenizer = MeCab()

    def __call__(self, text):
        new_text = []
        morphs = self.tokenizer.parse(text)
        pronunciations = []
        for morph in morphs:
            normalized = morph.surface
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
                pronunciations[-1] += jamo.h2j(normalized)
                continue
            if morph.pos in {"SF", "SY", "SC"} and normalized not in {"<", "(", "{", "["}:
                continue
            new_text.append(normalized)
            pronunciations.append(jamo.h2j(normalized))
        new_text = " ".join(new_text)
        pronunciations = " ".join(pronunciations)
        if self.ignore_case:
            new_text = new_text.lower()
            pronunciations = pronunciations.lower()
        return new_text, pronunciations


def ko_spacy(ignore_case: bool = True):
    if not KO_AVAILABLE:
        raise ImportError("Please install Korean support via `pip install python-mecab-ko jamo`")
    return KoreanTokenizer(ignore_case)
