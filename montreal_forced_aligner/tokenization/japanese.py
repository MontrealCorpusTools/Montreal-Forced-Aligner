from __future__ import annotations

try:
    import spacy
    from spacy.lang.ja import Japanese

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
try:
    import sudachipy

    JA_AVAILABLE = True
except ImportError:
    JA_AVAILABLE = False


class BracketedReTokenize:
    def __init__(self, vocab):
        pass

    def __call__(self, doc):
        spans = []
        initial_span = None
        for w in doc:
            if w.text in {"<", "(", "{", "["}:
                initial_span = w.left_edge.i
            elif w.text in {">", ")", "}", "]"}:
                if initial_span is not None:
                    spans.append(doc[initial_span : w.right_edge.i + 1])
                    initial_span = None
        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span, attrs={"POS": "X"})
        return doc


class JapaneseReTokenize:
    def __init__(self, vocab):
        pass

    def __call__(self, doc):
        spans = []
        for j, w in enumerate(doc):
            if w.text in {"えっと", "そのー", "あのー", "えっ", "まあ", "このー", "えー"}:
                spans.append((doc[w.left_edge.i : w.right_edge.i + 1], "INTJ"))
            elif (
                w.text in {"あっ", "まっ"}
                and j < len(doc) - 1
                and doc[j + 1].pos not in "AUX"
                and doc[j + 1].text != "て"
            ):
                spans.append((doc[w.left_edge.i : w.right_edge.i + 1], "INTJ"))

        with doc.retokenize() as retokenizer:
            for span, pos in spans:
                retokenizer.merge(span, attrs={"POS": pos})
        return doc


def ja_spacy(accurate=True):
    if not JA_AVAILABLE:
        raise ImportError("Please install Japanese support via `conda install spacy[ja]`")
    nlp = Japanese.from_config({"nlp": {"tokenizer": {"split_mode": "C"}}})

    @spacy.Language.factory("bracketed_re_tokenize")
    def bracketed_re_tokenize(_nlp, name):
        return BracketedReTokenize(_nlp.vocab)

    @spacy.Language.factory("ja_re_tokenize")
    def ja_re_tokenize(_nlp, name):
        return JapaneseReTokenize(_nlp.vocab)

    nlp.tokenizer.tokenizer = sudachipy.Dictionary(dict="full" if accurate else "core").create()
    nlp.add_pipe("bracketed_re_tokenize")
    nlp.add_pipe("ja_re_tokenize")
    return nlp
