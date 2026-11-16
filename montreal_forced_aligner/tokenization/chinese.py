#!/usr/bin/env Python
# coding=utf-8
from __future__ import annotations

import re

from montreal_forced_aligner.compat import ZH_AVAILABLE
from montreal_forced_aligner.tokenization.classes import BaseTokenizer, TokenizedText


class ChineseTokenizer(BaseTokenizer):
    def __init__(self, ignore_case: bool = True):
        import hanziconv
        import spacy_pkuseg
        from dragonmapper.hanzi import to_pinyin

        super().__init__(spacy_pkuseg.pkuseg(postag=True), ignore_case)
        self.to_simplified = hanziconv.HanziConv.toSimplified
        self.to_traditional = hanziconv.HanziConv.toTraditional
        self.to_pinyin = to_pinyin

    def __call__(self, text: str) -> TokenizedText:
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
        simplified = self.to_simplified(text)
        is_traditional = simplified != text

        morphs = self.tokenizer.cut(simplified)
        pronunciations = []
        for normalized, pos in morphs:
            join = False
            if pos in {"w"} and normalized not in {"<", "(", "{", "["}:
                continue
            m = re.search(r"[]})>][<({[]", normalized)
            p = self.to_pinyin(normalized)
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
                    pronunciations.append(self.to_pinyin(c))
                continue
            new_text.append(normalized)
            pronunciations.append(p)
        assert len(new_text) == len(pronunciations)
        new_text = " ".join(new_text)
        pronunciations = " ".join(pronunciations)
        oov_indices = [
            i for i, x in enumerate(pronunciations) if re.search(r"[^A-Za-z0-9]", x) is not None
        ]
        oovs = [x for i, x in enumerate(new_text) if i in oov_indices]
        if is_traditional:
            new_text = self.to_traditional(new_text)
        if self.ignore_case:
            new_text = new_text.lower()
            pronunciations = pronunciations.lower()
        return TokenizedText(new_text, pronunciations, oovs)


def zh_spacy(ignore_case: bool = True):
    if not ZH_AVAILABLE:
        raise ImportError(
            "Please install Chinese tokenization support via `pip install spacy-pkuseg dragonmapper hanziconv`"
        )
    return ChineseTokenizer(ignore_case)
