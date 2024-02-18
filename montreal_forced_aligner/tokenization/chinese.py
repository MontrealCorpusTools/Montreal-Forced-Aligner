from __future__ import annotations

import re

try:
    import hanziconv
    import spacy_pkuseg

    ZH_AVAILABLE = True
except ImportError:
    ZH_AVAILABLE = False
    spacy_pkuseg = None
    hanziconv = None


class ChineseTokenizer:
    def __init__(self, ignore_case):
        self.tokenizer = spacy_pkuseg.pkuseg(postag=True)
        self.ignore_case = ignore_case

    def __call__(self, text):
        new_text = []

        # pkuseg was trained on simplified characters
        simplified = hanziconv.HanziConv.toSimplified(text)
        is_traditional = simplified != text

        morphs = self.tokenizer.cut(simplified)
        for normalized, pos in morphs:
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
                continue
            if pos in {"w"} and normalized not in {"<", "(", "{", "["}:
                continue
            if "·" in normalized:
                normalized = " ".join(normalized.split("·"))
            new_text.append(normalized)
        new_text = " ".join(new_text)
        if is_traditional:
            new_text = hanziconv.HanziConv.toTraditional(new_text)
        if self.ignore_case:
            new_text = new_text.lower()
        return new_text


def zh_spacy(ignore_case: bool = True):
    if not ZH_AVAILABLE:
        raise ImportError(
            "Please install Chinese tokenization support via `pip install spacy-pkuseg hanziconv`"
        )
    return ChineseTokenizer(ignore_case)
