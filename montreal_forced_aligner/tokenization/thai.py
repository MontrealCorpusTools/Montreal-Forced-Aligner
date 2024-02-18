from __future__ import annotations

import re


class ThaiTokenizer:
    def __init__(self, ignore_case: bool):
        import attacut

        self.tokenizer = attacut.Tokenizer(model="attacut-sc")
        self.ignore_case = ignore_case

    def __call__(self, text):
        new_text = []
        morphs = self.tokenizer.tokenize(text)
        for normalized in morphs:
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
            # if normalized in {"w"} and normalized not in {"<", "(", "{", "["}:
            #    continue
            if re.match(r"^\W+$", normalized):
                continue
            new_text.append(normalized)
        new_text = " ".join(new_text)
        if self.ignore_case:
            new_text = new_text.lower()
        return new_text


def th_spacy(ignore_case: bool = True):
    try:
        import attacut  # noqa
    except ImportError:
        raise ImportError("Please install Thai tokenization support via `pip install attacut`")
    return ThaiTokenizer(ignore_case)
