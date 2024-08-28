from __future__ import annotations

import os
import re
import subprocess
import typing

try:
    import spacy
    from spacy.symbols import NORM, ORTH
    from spacy.tokens import Doc, Token

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

GENERIC_PREFIXES = {"non", "electro", "multi", "cross", "pseudo", "techno", "robo", "thermo"}


class EnglishReTokenize:
    """
    Retokenizer for fixing English splitting
    """

    def __init__(self, vocab):
        pass

    def __call__(self, doc):
        spans = []
        for j, w in enumerate(doc):
            if j > 0 and w.text == "'" and doc[j - 1].text.endswith("in"):
                spans.append((doc[j - 1 : j + 1], {"NORM": doc[j - 1].text + "g"}))
            elif j > 0 and w.text == "-" and doc[j - 1].text in GENERIC_PREFIXES:
                spans.append((doc[j - 1 : j + 1], {}))
        with doc.retokenize() as retokenizer:
            for span, attrs in spans:
                retokenizer.merge(span, attrs=attrs)
        return doc


class EnglishSplitPrefixes:
    """
    Retokenizer for splitting prefixes
    """

    def __init__(self, vocab: spacy.Vocab):
        self.vocab = vocab

    def __call__(self, doc):
        spans = []
        for w in doc:
            verb_prefixes = ["re"]
            adjective_prefixes = ["in", "un", "non"]
            if w.pos_ == "VERB":
                for vp in verb_prefixes:
                    if w.text.startswith(vp) and self.vocab[w.lemma_].is_oov:
                        lemma = re.sub(rf"^{vp}", "", w.lemma_)
                        if not self.vocab[lemma].is_oov:
                            orth = re.sub(rf"^{vp}", "", w.text)
                            norm = re.sub(rf"^{vp}", "", w.norm_)
                            lemma_form = f"{vp}-"
                            spans.append(
                                (
                                    w,
                                    [vp, orth],
                                    [(w, 1), w.head],
                                    {
                                        "POS": ["VERB", "VERB"],
                                        "NORM": [lemma_form, norm],
                                        "LEMMA": [lemma_form, lemma],
                                        "MORPH": [str(w.morph), str(w.morph)],
                                    },
                                )
                            )
            elif w.pos_ == "ADJ":
                for ap in adjective_prefixes:
                    if w.text.startswith(ap) and self.vocab[w.lemma_].is_oov:
                        lemma = re.sub(rf"^{ap}", "", w.lemma_)
                        if not self.vocab[lemma].is_oov:
                            orth = re.sub(rf"^{ap}", "", w.text)
                            norm = re.sub(rf"^{ap}", "", w.norm_)

                            lemma_form = f"{ap}-"
                            spans.append(
                                (
                                    w,
                                    [ap, orth],
                                    [(w, 1), w.head],
                                    {
                                        "POS": ["ADJ", "ADJ"],
                                        "NORM": [lemma_form, norm],
                                        "LEMMA": [lemma_form, lemma],
                                        "MORPH": [str(w.morph), str(w.morph)],
                                    },
                                )
                            )
            for ap in GENERIC_PREFIXES:
                if w.text.startswith(ap) and self.vocab[w.lemma_].is_oov:
                    lemma = re.sub(rf"^{ap}", "", w.lemma_)
                    if not self.vocab[lemma].is_oov:
                        orth = re.sub(rf"^{ap}", "", w.text)
                        norm = re.sub(rf"^{ap}", "", w.norm_)

                        lemma_form = f"{ap}-"
                        spans.append(
                            (
                                w,
                                [ap, orth],
                                [(w, 1), w.head],
                                {
                                    "POS": [w.pos_, w.pos_],
                                    "NORM": [lemma_form, norm],
                                    "LEMMA": [lemma_form, lemma],
                                    "MORPH": [str(w.morph), str(w.morph)],
                                },
                            )
                        )
        with doc.retokenize() as retokenizer:
            for details in spans:
                if len(details) == 4:
                    span, orths, heads, attrs = details
                    retokenizer.split(span, orths, heads, attrs=attrs)
                else:
                    span, attrs = details
                    retokenizer.merge(span, attrs=attrs)

        return doc


class EnglishSplitSuffixes:
    """
    Retokenizer for splitting suffixes
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def find_base_form(
        self, w: Token, suffix: str
    ) -> typing.Tuple[typing.Optional[str], typing.Optional[str], typing.Optional[str]]:
        base_lemma = re.sub(rf"{suffix}$", "", w.lemma_)
        base_norm = re.sub(rf"{suffix}$", "", w.norm_)
        if not base_lemma:
            return None, None, None
        base_text = re.sub(rf"{suffix}$", "", w.text)
        if not self.vocab[base_lemma].is_oov:
            if base_lemma.endswith("e") and not base_norm.endswith("e"):
                base_norm += "e"
            return base_lemma, base_norm, base_text

        if not self.vocab[base_lemma + "e"].is_oov:
            return base_lemma + "e", base_norm + "e", base_text
        if base_lemma.endswith("i") and not self.vocab[base_lemma[:-1] + "y"].is_oov:
            return base_lemma[:-1] + "y", base_norm[:-1] + "y", base_text

        if re.search(r"(\w)\1$", base_lemma) and not self.vocab[base_lemma[:-1]].is_oov:
            return base_lemma[:-1], base_norm[:-1], base_text
        return None, None, None

    def handle_ing(self, w: Token):
        base_lemma, base_norm, base_text = self.find_base_form(w, "ing")
        if base_lemma is None:
            return None
        return (
            w,
            [base_text, "ing"],
            [w.head, (w, 0)],
            {
                "POS": ["VERB", "VERB"],
                "NORM": [base_norm, "-ing"],
                "LEMMA": [base_lemma, "-ing"],
                "MORPH": [str(w.morph), str(w.morph)],
            },
        )

    def handle_ness(self, w: Token):
        base_lemma, base_norm, base_text = self.find_base_form(w, "ness")
        if base_lemma is None:
            return None
        if base_text.endswith("li"):
            base_text = base_text[:-1] + "y"
        return (
            w,
            [base_text, "ness"],
            [w.head, (w, 0)],
            {
                "POS": ["ADJ", "NOUN"],
                "NORM": [base_norm, "-ness"],
                "LEMMA": [base_lemma, "-ness"],
                "MORPH": ["Degree=Pos", "Number=Sing"],
            },
        )

    def handle_less(self, w: Token):
        base_lemma, base_norm, base_text = self.find_base_form(w, "less")
        if base_lemma is None:
            return None
        return (
            w,
            [base_text, "ing"],
            [w.head, (w, 0)],
            {
                "POS": ["NOUN", "ADJ"],
                "NORM": [base_norm, "-less"],
                "LEMMA": [base_lemma, "-less"],
                "MORPH": ["Number=Sing", "Degree=Pos"],
            },
        )

    def handle_able(self, w: Token):
        if w.text.endswith("able"):
            suffix = "able"
        else:
            suffix = "ible"
        base_lemma, base_norm, base_text = self.find_base_form(w, suffix)
        if base_lemma is None:
            return None
        return (
            w,
            [base_text, suffix],
            [w.head, (w, 0)],
            {
                "POS": ["VERB", "ADJ"],
                "NORM": [base_norm, "-able"],
                "LEMMA": [base_lemma, "-able"],
                "MORPH": ["VerbForm=Inf", "Degree=Pos"],
            },
        )

    def handle_ability(self, w: Token):
        if w.text.endswith("ability"):
            suffix = "ability"
        else:
            suffix = "ibility"
        base_lemma, base_norm, base_text = self.find_base_form(w, suffix)
        if base_lemma is None:
            return None
        return (
            w,
            [base_text, suffix],
            [w.head, (w, 0)],
            {
                "POS": ["VERB", "NOUN"],
                "NORM": [base_norm, "-ability"],
                "LEMMA": [base_lemma, "-ability"],
                "MORPH": ["VerbForm=Inf", "Number=Sing"],
            },
        )

    def handle_ably(self, w: Token):
        if w.text.endswith("ably"):
            suffix = "ably"
        else:
            suffix = "ibly"
        base_lemma, base_norm, base_text = self.find_base_form(w, suffix)
        if base_lemma is None:
            return None
        return (
            w,
            [base_text, suffix],
            [w.head, (w, 0)],
            {
                "POS": ["ADJ", "ADV"],
                "NORM": [base_norm, "-ly"],
                "LEMMA": [base_lemma, "-ly"],
                "MORPH": ["Degree=Pos", ""],
            },
        )

    def handle_plural(self, w: Token):
        if w.text == "'s":
            return None
        if w.text.endswith("ies"):
            orth = re.sub(r"es$", "", w.text)
            norm = re.sub(r"ies$", "y", w.text)
            s_form = "es"
        elif w.text.endswith("es"):
            if w.lemma_.endswith("e"):
                orth = re.sub(r"s$", "", w.text)
                s_form = "s"
                norm = re.sub(r"s$", "", w.norm_)
            else:
                orth = re.sub(r"es$", "", w.text)
                s_form = "es"
                norm = re.sub(r"es$", "", w.norm_)
        else:
            orth = re.sub(r"s$", "", w.text)
            norm = re.sub(r"s$", "", w.norm_)
            s_form = "s"
        if self.vocab[norm].is_oov:
            return None
        return (
            w,
            [orth, s_form],
            [w.head, (w, 0)],
            {
                "POS": ["NOUN", "NOUN"],
                "NORM": [norm, "-s"],
                "LEMMA": [norm, "-s"],
                "MORPH": ["Number=Sing", "Number=Plur"],
            },
        )

    def handle_3p_pres(self, w: Token):
        if w.text == "'s":
            return None
        base_lemma = None
        if w.text.endswith("es"):
            suffix = "es"
            base_lemma, base_norm, base_text = self.find_base_form(w, suffix)

        if base_lemma is None:
            suffix = "s"
            base_lemma, base_norm, base_text = self.find_base_form(w, suffix)
            if base_lemma is None:
                return None
        elif base_norm.endswith("i"):
            base_norm = base_norm[:-1] + "y"
        return (
            w,
            [base_text, suffix],
            [w.head, (w, 0)],
            {
                "POS": ["VERB", "VERB"],
                "NORM": [base_norm, "-s"],
                "LEMMA": [base_lemma, "-s"],
                "MORPH": ["VerbForm=Inf", "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin"],
            },
        )

    def handle_ly(self, w: Token):
        suffix = "ly"
        base_lemma, base_norm, base_text = self.find_base_form(w, suffix)
        if base_lemma is None:
            return None
        if base_norm.endswith("i"):
            base_norm = base_norm[:-1] + "y"
        return (
            w,
            [base_text, suffix],
            [w.head, (w, 0)],
            {
                "POS": ["ADJ", "ADV"],
                "NORM": [base_norm, "-ly"],
                "LEMMA": [base_lemma, "-ly"],
                "MORPH": ["Degree=Pos", ""],
            },
        )

    def handle_ed(self, w: Token):
        base_lemma, base_norm, base_text = self.find_base_form(w, "ed")
        if base_lemma is None:
            return None
        if base_norm.endswith("i"):
            base_norm = base_norm[:-1] + "y"
        return (
            w,
            [base_text, "ed"],
            [w.head, (w, 0)],
            {
                "POS": ["VERB", "VERB"],
                "NORM": [base_norm, "-ed"],
                "LEMMA": [base_lemma, "-ed"],
                "MORPH": ["VerbForm=Inf", str(w.morph)],
            },
        )

    def __call__(self, doc: Doc):
        while True:
            for j, w in enumerate(doc):
                try:
                    if w.lemma_.startswith("-") or w.lemma_.endswith("-"):
                        continue
                except KeyError:
                    continue
                span = None
                if "Prog" in w.morph.get("Aspect") and w.text.endswith("ing"):
                    span = self.handle_ing(w)
                elif (
                    w.pos_ == "ADJ"
                    and (w.text.endswith("able") or w.text.endswith("ible"))
                    and self.vocab[w.lemma_].is_oov
                ):
                    span = self.handle_able(w)

                elif (w.text.endswith("ability") or w.text.endswith("ibility")) and self.vocab[
                    w.lemma_
                ].is_oov:
                    span = self.handle_ability(w)
                    break
                elif (w.text.endswith("ably") or w.text.endswith("ibly")) and (
                    w.pos_ == "ADV" or self.vocab[w.lemma_].is_oov
                ):
                    span = self.handle_ably(w)
                    break
                elif w.pos_ == "NOUN" and "Plur" in w.morph.get("Number") and w.text.endswith("s"):
                    span = self.handle_plural(w)
                elif (
                    w.pos_ == "VERB"
                    and "Sing" in w.morph.get("Number")
                    and "3" in w.morph.get("Person")
                    and "Pres" in w.morph.get("Tense")
                    and w.text.endswith("s")
                ):
                    span = self.handle_3p_pres(w)
                elif w.pos_ == "VERB" and "Past" in w.morph.get("Tense") and w.text.endswith("ed"):
                    span = self.handle_ed(w)
                elif w.text in {"n't"} and w.norm_ != "-n't":
                    span = (
                        doc[j : j + 1],
                        {
                            "NORM": "-n't",
                        },
                    )
                elif w.pos_ in {"ADJ", "ADV"} and w.text.endswith("ly"):
                    span = self.handle_ly(w)
                if span is not None:
                    break
            else:
                break
            if span is not None:
                with doc.retokenize() as retokenizer:
                    if len(span) == 4:
                        span, orths, heads, attrs = span
                        retokenizer.split(span, orths, heads, attrs=attrs)
                    else:
                        span, attrs = span
                        retokenizer.merge(span, attrs=attrs)
            else:
                break
        return doc


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


def en_spacy(ignore_case: bool = True):
    name = "en_core_web_sm"
    try:
        en_nlp = spacy.load(name)
    except OSError:
        subprocess.call(["python", "-m", "spacy", "download", name], env=os.environ)
        en_nlp = spacy.load(name)

    @spacy.Language.factory("en_re_tokenize")
    def en_re_tokenize(nlp, name):
        return EnglishReTokenize(nlp.vocab)

    @spacy.Language.factory("en_split_suffixes")
    def en_split_suffixes(nlp, name):
        return EnglishSplitSuffixes(nlp.vocab)

    @spacy.Language.factory("en_split_prefixes")
    def en_split_prefixes(nlp, name):
        return EnglishSplitPrefixes(nlp.vocab)

    @spacy.Language.factory("en_bracketed_re_tokenize")
    def en_bracketed_re_tokenize(nlp, name):
        return BracketedReTokenize(nlp.vocab)

    initial_brackets = r"\(\[\{<"
    final_brackets = r"\)\]\}>"

    en_nlp.tokenizer.token_match = re.compile(
        rf"[{initial_brackets}][-\w_']+[?!,][{final_brackets}]"
    ).match
    en_nlp.tokenizer.add_special_case(
        "wanna", [{ORTH: "wan", NORM: "want"}, {ORTH: "na", NORM: "to"}]
    )
    en_nlp.tokenizer.add_special_case(
        "dunno", [{ORTH: "dun", NORM: "don't"}, {ORTH: "no", NORM: "know"}]
    )
    en_nlp.tokenizer.add_special_case(
        "woulda", [{ORTH: "would", NORM: "would"}, {ORTH: "a", NORM: "have"}]
    )
    en_nlp.tokenizer.add_special_case(
        "sorta", [{ORTH: "sort", NORM: "sort"}, {ORTH: "a", NORM: "of"}]
    )
    en_nlp.tokenizer.add_special_case(
        "kinda", [{ORTH: "kind", NORM: "kind"}, {ORTH: "a", NORM: "of"}]
    )
    en_nlp.tokenizer.add_special_case(
        "coulda", [{ORTH: "could", NORM: "could"}, {ORTH: "a", NORM: "have"}]
    )
    en_nlp.tokenizer.add_special_case(
        "shoulda", [{ORTH: "should", NORM: "should"}, {ORTH: "a", NORM: "have"}]
    )
    en_nlp.tokenizer.add_special_case(
        "finna", [{ORTH: "fin", NORM: "fixing"}, {ORTH: "na", NORM: "to"}]
    )
    en_nlp.tokenizer.add_special_case(
        "yknow", [{ORTH: "y", NORM: "you"}, {ORTH: "know", NORM: "know"}]
    )
    en_nlp.tokenizer.add_special_case(
        "y'know", [{ORTH: "y'", NORM: "you"}, {ORTH: "know", NORM: "know"}]
    )
    en_nlp.add_pipe("en_re_tokenize", before="tagger")
    en_nlp.add_pipe("en_bracketed_re_tokenize", before="tagger")
    en_nlp.add_pipe("en_split_prefixes")
    en_nlp.add_pipe("en_split_suffixes")
    return en_nlp
