from __future__ import annotations

import re
import typing

import pywrapfst

from montreal_forced_aligner.data import BRACKETED_WORD, CUTOFF_WORD, LAUGHTER_WORD, OOV_WORD
from montreal_forced_aligner.helper import make_re_character_set_safe

__all__ = ["SanitizeFunction", "SplitWordsFunction", "SimpleTokenizer"]


class SanitizeFunction:
    """
    Class for functions that sanitize text and strip punctuation

    Parameters
    ----------
    punctuation: list[str]
        List of characters to treat as punctuation
    clitic_markers: list[str]
        Characters that mark clitics
    compound_markers: list[str]
        Characters that mark compound words
    brackets: list[tuple[str, str]]
        List of bracket sets to not strip from the ends of words
    ignore_case: bool
        Flag for whether all items should be converted to lower case, defaults to True
    quote_markers: list[str], optional
        Quotation markers to use when parsing text
    quote_markers: list[str], optional
        Quotation markers to use when parsing text
    word_break_markers: list[str], optional
        Word break markers to use when parsing text
    """

    def __init__(
        self,
        word_table: pywrapfst.SymbolTable,
        clitic_marker: str,
        clitic_cleanup_regex: typing.Optional[re.Pattern],
        clitic_quote_regex: typing.Optional[re.Pattern],
        punctuation_regex: typing.Optional[re.Pattern],
        word_break_regex: typing.Optional[re.Pattern],
        bracket_regex: typing.Optional[re.Pattern],
        bracket_sanitize_regex: typing.Optional[re.Pattern],
        cutoff_regex: typing.Optional[re.Pattern],
        ignore_case: bool = True,
    ):
        self.word_table = word_table
        self.clitic_marker = clitic_marker
        self.clitic_cleanup_regex = clitic_cleanup_regex
        self.clitic_quote_regex = clitic_quote_regex
        self.punctuation_regex = punctuation_regex
        self.word_break_regex = word_break_regex
        self.cutoff_regex = cutoff_regex
        self.bracket_regex = bracket_regex
        self.bracket_sanitize_regex = bracket_sanitize_regex

        self.ignore_case = ignore_case

    def __call__(self, text) -> typing.Generator[str]:
        """
        Sanitize text according to punctuation, quotes, and word break characters

        Parameters
        ----------
        text: str
            Text to sanitize

        Returns
        -------
        Generator[str]
            Sanitized form
        """
        if self.ignore_case:
            text = text.lower()
            text = text.replace("i̇", "i")  # Turkish normalization
        if self.bracket_regex:
            for word_object in self.bracket_regex.finditer(text):
                word = word_object.group(0)
                if self.cutoff_regex is not None and self.cutoff_regex.match(word):
                    continue
                if self.word_table and self.word_table.member(word):
                    continue
                new_word = self.bracket_sanitize_regex.sub("_", word)
                text = text.replace(word, new_word)

        if self.clitic_cleanup_regex:
            text = self.clitic_cleanup_regex.sub(self.clitic_marker, text)

        if self.clitic_quote_regex is not None and self.clitic_marker in text:
            text = self.clitic_quote_regex.sub(r"\g<word>", text)

        words = self.word_break_regex.split(text)

        for w in words:
            if not w:
                continue
            if self.punctuation_regex is not None and self.punctuation_regex.match(w):
                continue
            if w:
                yield w


class SplitWordsFunction:
    """
    Class for functions that splits words that have compound and clitic markers

    Parameters
    ----------
    word_table: :class:`pywrapfst.SymbolTable`
        Symbol table to look words up
    clitic_marker: str
        Character that marks clitics
    initial_clitic_regex: :class:`re.Pattern`
        Regex for splitting off initial clitics
    final_clitic_regex: :class:`re.Pattern`
        Regex for splitting off final clitics
    compound_regex: :class:`re.Pattern`
        Regex for splitting compound words
    non_speech_regexes: dict[str, :class:`re.Pattern`]
        Regex for detecting and sanitizing non-speech words
    oov_word : str
        What to label words not in the dictionary, defaults to None
    """

    def __init__(
        self,
        word_table: pywrapfst.SymbolTable,
        clitic_marker: str,
        initial_clitic_regex: typing.Optional[re.Pattern],
        final_clitic_regex: typing.Optional[re.Pattern],
        compound_regex: typing.Optional[re.Pattern],
        cutoff_regex: typing.Optional[re.Pattern],
        non_speech_regexes: typing.Dict[str, re.Pattern],
        oov_word: typing.Optional[str] = None,
        grapheme_set: typing.Optional[typing.Collection[str]] = None,
        always_split_compounds: bool = False,
    ):
        self.word_table = word_table
        self.clitic_marker = clitic_marker
        self.compound_regex = compound_regex
        self.cutoff_regex = cutoff_regex
        self.oov_word = oov_word
        self.specials_set = {self.oov_word, "<s>", "</s>"}
        if not grapheme_set:
            grapheme_set = None
        self.grapheme_set = grapheme_set
        self.compound_pattern = None
        self.clitic_pattern = None
        self.non_speech_regexes = non_speech_regexes
        self.initial_clitic_regex = initial_clitic_regex
        self.final_clitic_regex = final_clitic_regex
        self.has_initial = False
        self.has_final = False
        if self.initial_clitic_regex is not None:
            self.has_initial = True
        if self.final_clitic_regex is not None:
            self.has_final = True
        self.always_split_compounds = always_split_compounds

    def to_str(self, normalized_text: str) -> str:
        """
        Convert normalized text to an integer ID

        Parameters
        ----------
        normalized_text:
            Word to convert

        Returns
        -------
        str
            Normalized string
        """
        if normalized_text in self.specials_set:
            return self.oov_word
        if self.word_table and self.word_table.member(normalized_text):
            return normalized_text
        if self.cutoff_regex is not None and self.cutoff_regex.match(normalized_text):
            return normalized_text
        for word, regex in self.non_speech_regexes.items():
            if regex.match(normalized_text):
                return word
        return normalized_text

    def split_clitics(
        self,
        item: str,
    ) -> typing.List[str]:
        """
        Split a word into subwords based on dictionary information

        Parameters
        ----------
        item: str
            Word to split

        Returns
        -------
        list[str]
            List of subwords
        """
        split = []
        benefit = False
        if self.compound_regex is not None:
            s = [x for x in self.compound_regex.split(item) if x]
            if self.always_split_compounds and len(s) > 1:
                benefit = True
        else:
            s = [item]
        if self.word_table is None:
            return [item]
        clean_initial_quote_regex = re.compile("^'")
        clean_final_quote_regex = re.compile("'$")
        for seg in s:
            if not seg:
                continue
            if not self.clitic_marker or self.clitic_marker not in seg:
                split.append(seg)
                if not benefit and self.word_table.member(seg):
                    benefit = True
                continue
            elif seg.startswith(self.clitic_marker):
                if self.word_table.member(seg[1:]):
                    split.append(seg[1:])
                    benefit = True
                    continue
            elif seg.endswith(self.clitic_marker):
                if self.word_table.member(seg[:-1]):
                    split.append(seg[:-1])
                    benefit = True
                    continue

            initial_clitics = []
            final_clitics = []
            if self.has_initial:
                while True:
                    clitic = self.initial_clitic_regex.match(seg)
                    if clitic is None:
                        break
                    benefit = True
                    initial_clitics.append(clitic.group(0))
                    seg = seg[clitic.end(0) :]
                    if self.word_table.member(seg):
                        break
            if self.has_final:
                while True:
                    clitic = self.final_clitic_regex.search(seg)
                    if clitic is None:
                        break
                    benefit = True
                    final_clitics.append(clitic.group(0))
                    seg = seg[: clitic.start(0)]
                    if self.word_table.member(seg):
                        break
                final_clitics.reverse()
            split.extend([clean_initial_quote_regex.sub("", x) for x in initial_clitics])
            seg = clean_final_quote_regex.sub("", clean_initial_quote_regex.sub("", seg))
            if seg:
                split.append(seg)
            split.extend([clean_final_quote_regex.sub("", x) for x in final_clitics])
            if not benefit and self.word_table.member(seg):
                benefit = True
        if not benefit:
            return [item]
        return split

    def parse_graphemes(
        self,
        item: str,
    ) -> typing.Generator[str]:
        for word, regex in self.non_speech_regexes.items():
            if regex.match(item):
                yield word
                break
        else:
            for c in item:
                if self.grapheme_set is not None and c in self.grapheme_set:
                    yield c
                else:
                    yield self.oov_word

    def __call__(
        self,
        item: str,
    ) -> typing.List[str]:
        """
        Return the list of sub words if necessary
        taking into account clitic and compound markers

        Parameters
        ----------
        item: str
            Word to look up

        Returns
        -------
        list[str]
            List of subwords that are in the dictionary
        """
        if self.word_table and self.word_table.member(item):
            return [item]
        if self.cutoff_regex is not None and self.cutoff_regex.match(item):
            return [item]
        for regex in self.non_speech_regexes.values():
            if regex.match(item):
                return [item]
        return self.split_clitics(item)


class SimpleTokenizer:
    def __init__(
        self,
        word_break_markers: typing.List[str],
        punctuation: typing.List[str],
        clitic_markers: typing.List[str],
        compound_markers: typing.List[str],
        brackets: typing.List[typing.Tuple[str, str]],
        laughter_word: str = LAUGHTER_WORD,
        oov_word: str = OOV_WORD,
        bracketed_word: str = BRACKETED_WORD,
        cutoff_word: str = CUTOFF_WORD,
        ignore_case: bool = True,
        use_g2p: bool = False,
        clitic_set: typing.Iterable = None,
        grapheme_set: typing.Iterable = None,
        word_table: pywrapfst.SymbolTable = None,
    ):
        self.word_break_markers = word_break_markers
        self.word_table = word_table
        self.punctuation = punctuation
        self.clitic_markers = clitic_markers
        self.compound_markers = compound_markers
        self.brackets = brackets
        self.laughter_word = laughter_word
        self.oov_word = oov_word
        self.bracketed_word = bracketed_word

        initial_brackets = re.escape("".join(x[0] for x in self.brackets))
        final_brackets = re.escape("".join(x[1] for x in self.brackets))
        self.cutoff_identifier = re.sub(rf"[{initial_brackets}{final_brackets}]", "", cutoff_word)
        self.ignore_case = ignore_case
        self.use_g2p = use_g2p
        self.clitic_set = set()
        if clitic_set is not None:
            self.clitic_set.update(clitic_set)
        elif clitic_markers and self.word_table is not None:
            for i in range(self.word_table.num_symbols()):
                w = self.word_table.find(i)
                if w.startswith(clitic_markers[0]) or w.endswith(clitic_markers[0]):
                    self.clitic_set.add(w)

        self.grapheme_set = set()
        if grapheme_set is not None:
            self.grapheme_set.update(grapheme_set)

        self.clitic_marker = None
        self.clitic_cleanup_regex = None
        self.compound_regex = None
        self.bracket_regex = None
        self.cutoff_regex = None
        self.bracket_sanitize_regex = None
        self.laughter_regex = None
        self.word_break_regex = None
        self.clitic_quote_regex = None
        self.punctuation_regex = None
        self.initial_clitic_regex = None
        self.final_clitic_regex = None
        self.non_speech_regexes = {}
        self._compile_regexes()
        self.sanitize_function = SanitizeFunction(
            self.word_table,
            self.clitic_marker,
            self.clitic_cleanup_regex,
            self.clitic_quote_regex,
            self.punctuation_regex,
            self.word_break_regex,
            self.bracket_regex,
            self.bracket_sanitize_regex,
            self.cutoff_regex,
            self.ignore_case,
        )
        self.split_function = SplitWordsFunction(
            self.word_table,
            self.clitic_marker,
            self.initial_clitic_regex,
            self.final_clitic_regex,
            self.compound_regex,
            self.cutoff_regex,
            self.non_speech_regexes,
            self.oov_word,
            self.grapheme_set,
            always_split_compounds=self.use_g2p,
        )

    def _compile_regexes(self) -> None:
        """Compile regular expressions necessary for corpus parsing"""
        if len(self.clitic_markers) >= 1:
            other_clitic_markers = self.clitic_markers[1:]
            if other_clitic_markers:
                extra = ""
                if "-" in other_clitic_markers:
                    extra = "-"
                    other_clitic_markers = [x for x in other_clitic_markers if x != "-"]
                self.clitic_cleanup_regex = re.compile(
                    rf'[{extra}{"".join(other_clitic_markers)}]'
                )
            self.clitic_marker = self.clitic_markers[0]
        if self.compound_markers:
            extra = ""
            compound_markers = self.compound_markers
            if "-" in self.compound_markers:
                extra = "-"
                compound_markers = [x for x in compound_markers if x != "-"]
            self.compound_regex = re.compile(
                rf"(?<=\w)[{extra}{''.join(compound_markers)}](?:$|(?=\w))"
            )
        if self.brackets:
            left_brackets = re.escape("".join(x[0] for x in self.brackets))
            right_brackets = re.escape("".join(x[1] for x in self.brackets))
            self.cutoff_regex = re.compile(
                rf"^[{left_brackets}]({self.cutoff_identifier}|hes(itation)?)([-_](?P<word>[^{right_brackets}]+))?[{right_brackets}]$",
                flags=re.IGNORECASE,
            )
            self.bracket_regex = re.compile(rf"[{left_brackets}].*?[{right_brackets}]+")
            self.laughter_regex = re.compile(
                rf"[{left_brackets}](laugh(ing|ter)?|lachen|lg)[{right_brackets}]+",
                flags=re.IGNORECASE,
            )
        all_punctuation = set()
        non_word_character_set = set(self.punctuation)
        non_word_character_set -= {b for x in self.brackets for b in x}

        if self.clitic_markers:
            all_punctuation.update(self.clitic_markers)
        if self.compound_markers:
            all_punctuation.update(self.compound_markers)
        self.bracket_sanitize_regex = None
        if self.brackets:
            word_break_set = (
                non_word_character_set | set(self.clitic_markers) | set(self.compound_markers)
            )
            if self.word_break_markers:
                word_break_set |= set(self.word_break_markers)
            word_break_set = make_re_character_set_safe(word_break_set, [r"\s"])
            self.bracket_sanitize_regex = re.compile(f"(?<!^){word_break_set}(?!$)")

        word_break_character_set = make_re_character_set_safe(non_word_character_set, [r"\s"])
        self.word_break_regex = re.compile(rf"{word_break_character_set}+")
        punctuation_set = make_re_character_set_safe(all_punctuation)
        if all_punctuation:
            self.punctuation_regex = re.compile(rf"^{punctuation_set}+$")
        if len(self.clitic_markers) >= 1:
            non_clitic_punctuation = all_punctuation - set(self.clitic_markers)
            non_clitic_punctuation_set = make_re_character_set_safe(non_clitic_punctuation)
            non_punctuation_set = "[^" + punctuation_set[1:]
            self.clitic_quote_regex = re.compile(
                rf"((?<=\W)|(?<=^)){non_clitic_punctuation_set}*{self.clitic_marker}{non_clitic_punctuation_set}*(?P<word>{non_punctuation_set}+){non_clitic_punctuation_set}*{self.clitic_marker}{non_clitic_punctuation_set}*((?=\W)|(?=$))"
            )

        self.non_speech_regexes["<eps>"] = re.compile("<eps>")
        if self.laughter_regex is not None:
            self.non_speech_regexes[self.laughter_word] = self.laughter_regex
        if self.bracket_regex is not None:
            self.non_speech_regexes[self.bracketed_word] = self.bracket_regex

        if self.clitic_marker is not None:
            initial_clitics = sorted(x for x in self.clitic_set if x.endswith(self.clitic_marker))
            final_clitics = sorted(x for x in self.clitic_set if x.startswith(self.clitic_marker))
            if initial_clitics:
                self.initial_clitic_regex = re.compile(rf"^({'|'.join(initial_clitics)})(?=\w)")
            if final_clitics:
                self.final_clitic_regex = re.compile(rf"(?<=\w)({'|'.join(final_clitics)})$")

    def _dictionary_sanitize(self, text):
        words = self.sanitize_function(text)
        normalized_text = []
        normalized_character_text = []
        oovs = set()
        for w in words:
            for new_w in self.split_function(w):
                if not self.word_table.member(new_w):
                    oovs.add(new_w)
                normalized_text.append(self.split_function.to_str(new_w))
                if normalized_character_text:
                    if not self.clitic_marker or (
                        not normalized_text[-1].endswith(self.clitic_marker)
                        and not new_w.startswith(self.clitic_marker)
                    ):
                        normalized_character_text.append("<space>")
                for c in self.split_function.parse_graphemes(new_w):
                    normalized_character_text.append(c)
        normalized_text = " ".join(normalized_text)
        normalized_character_text = " ".join(normalized_character_text)
        return normalized_text, normalized_character_text, sorted(oovs)

    def _no_dictionary_sanitize(self, text):
        normalized_text = []
        normalized_character_text = []
        for w in self.sanitize_function(text):
            normalized_text.append(w)
            if normalized_character_text:
                normalized_character_text.append("<space>")
            for g in w:
                normalized_character_text.append(g)
        normalized_text = " ".join(normalized_text)
        normalized_character_text = " ".join(normalized_character_text)
        return normalized_text, normalized_character_text, []

    def __call__(self, text):
        """Run the function"""
        if self.word_table or self.grapheme_set:
            return self._dictionary_sanitize(text)
        else:
            return self._no_dictionary_sanitize(text)
