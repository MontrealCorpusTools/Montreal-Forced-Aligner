from __future__ import annotations

import re

try:
    import hanziconv
    import spacy_pkuseg
    from dragonmapper.hanzi import to_pinyin

    ZH_AVAILABLE = True
except ImportError:
    ZH_AVAILABLE = False
    spacy_pkuseg = None
    hanziconv = None
    to_pinyin = None


class ChineseTokenizer:
    def __init__(self, ignore_case):
        self.tokenizer = spacy_pkuseg.pkuseg(postag=True)
        self.ignore_case = ignore_case

    def __call__(self, text):
        for t in [
            "·",
            ",",
            "!",
            '"',
            "～",
            "?",
            "•",
            "‧",
        ]:  # Remove punctuation that pkuseg doesn't recognize
            text = text.replace(t, " ")
        new_text = []

        # pkuseg was trained on simplified characters
        simplified = hanziconv.HanziConv.toSimplified(text)
        is_traditional = simplified != text

        morphs = self.tokenizer.cut(simplified)
        pronunciations = []
        for normalized, pos in morphs:
            join = False
            if pos in {"w"} and normalized not in {"<", "(", "{", "["}:
                continue
            m = re.search(r"[]})>][<({[]", normalized)
            p = to_pinyin(normalized)
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
            if pos == "m":  # numerals
                for c in normalized:
                    new_text.append(c)
                    pronunciations.append(to_pinyin(c))
                continue
            new_text.append(normalized)
            pronunciations.append(p)
        assert len(new_text) == len(pronunciations)
        new_text = " ".join(new_text)
        pronunciations = " ".join(pronunciations)
        if is_traditional:
            new_text = hanziconv.HanziConv.toTraditional(new_text)
        if self.ignore_case:
            new_text = new_text.lower()
            pronunciations = pronunciations.lower()
        return new_text, pronunciations


def zh_spacy(ignore_case: bool = True):
    if not ZH_AVAILABLE:
        raise ImportError(
            "Please install Chinese tokenization support via `pip install spacy-pkuseg dragonmapper hanziconv`"
        )
    return ChineseTokenizer(ignore_case)
