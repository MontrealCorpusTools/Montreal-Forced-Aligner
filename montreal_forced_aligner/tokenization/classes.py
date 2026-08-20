import dataclasses


@dataclasses.dataclass
class TokenizedText:
    normalized_text: str
    pronunciation_text: str
    oovs: list[str] = None


class BaseTokenizer:
    def __init__(self, tokenizer, ignore_case: bool = True):
        self.tokenizer = tokenizer
        self.ignore_case = ignore_case

    def __call__(self, text: str) -> TokenizedText:
        if self.tokenizer is not None:
            text = " ".join([x.text for x in self.tokenizer(text)])
        if self.ignore_case:
            text = text.lower()
        return TokenizedText(text, text, [])
