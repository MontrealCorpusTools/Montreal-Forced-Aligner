"""Mixins for dictionary parsing capabilities"""

from __future__ import annotations

import abc
import os
import re
import typing
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from montreal_forced_aligner.abc import DatabaseMixin
from montreal_forced_aligner.data import PhoneSetType

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict, ReversedMappingType

DEFAULT_PUNCTUATION = list(r'、。।，？!@<>→"”()“„–,.:;—¿?¡：）!\\&%#*~【】，…‥「」『』〝〟″⟨⟩♪・‹›«»～′$+=‘')

DEFAULT_WORD_BREAK_MARKERS = list(r'？!()，,.:;¡¿?“„"”&~%#—…‥、。【】$+=〝〟″‹›«»・⟨⟩「」『』')

DEFAULT_QUOTE_MARKERS = list("“„\"”〝〟″「」『』‚ʻʿ‘′'")

DEFAULT_CLITIC_MARKERS = list("'’‘")
DEFAULT_COMPOUND_MARKERS = list("-/")
DEFAULT_BRACKETS = [("[", "]"), ("{", "}"), ("<", ">"), ("(", ")"), ("＜", "＞")]

__all__ = ["SanitizeFunction", "SplitWordsFunction", "DictionaryMixin", "TemporaryDictionaryMixin"]


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
        punctuation: List[str],
        clitic_markers: List[str],
        compound_markers: List[str],
        brackets: List[Tuple[str, str]],
        ignore_case: bool = True,
        quote_markers: List[str] = None,
        word_break_markers: List[str] = None,
    ):
        self.base_clitic_marker = clitic_markers[0] if len(clitic_markers) else None
        self.all_punctuation = set()
        self.punctuation = punctuation
        self.clitic_markers = clitic_markers
        self.compound_markers = compound_markers
        self.quote_markers = quote_markers
        self.word_break_markers = word_break_markers
        self.brackets = brackets
        self.ignore_case = ignore_case
        non_word_character_set = set(punctuation)
        if self.clitic_markers:
            self.all_punctuation.update(self.clitic_markers)
        if self.compound_markers:
            self.all_punctuation.update(self.compound_markers)
        self.quote_regex = None
        if self.quote_markers:
            self.all_punctuation.update(self.quote_markers)
            unambiguous_quote_markers = (
                set(self.quote_markers) - set(self.clitic_markers) - set(self.compound_markers)
            )
            non_word_character_set.update(unambiguous_quote_markers)
            if self.base_clitic_marker and self.base_clitic_marker in self.quote_markers:
                self.quote_regex = re.compile(
                    rf"^{self.base_clitic_marker}+(?P<word>.*){self.base_clitic_marker}+$"
                )

        self.bracket_regex = None
        self.bracket_sanitize_regex = None
        if self.brackets:
            left_brackets = [x[0] for x in self.brackets]
            right_brackets = [x[1] for x in self.brackets]
            non_word_character_set -= set(left_brackets)
            non_word_character_set -= set(right_brackets)
            self.all_punctuation.update(left_brackets)
            self.all_punctuation.update(right_brackets)
            self.bracket_regex = re.compile(
                rf"[{re.escape(''.join(left_brackets))}]+.*?[{re.escape(''.join(right_brackets))}]+"
            )
            bracket_sanitize_set = (
                non_word_character_set | set(self.clitic_markers) | set(self.compound_markers)
            )
            extra = ""
            if "-" in bracket_sanitize_set:
                extra = "-"
                bracket_sanitize_set = [x for x in sorted(bracket_sanitize_set) if x != "-"]
            word_break_character_set = rf"[{extra}\s{re.escape(''.join(bracket_sanitize_set))}]"
            if self.word_break_markers:
                self.bracket_sanitize_regex = re.compile(word_break_character_set)
            else:
                self.bracket_sanitize_regex = re.compile(r"\s")
        extra = ""
        non_word_character_set = sorted(non_word_character_set)
        if "-" in self.word_break_markers:
            extra = "-"
            non_word_character_set = [x for x in non_word_character_set if x != "-"]
        word_character_set = rf"[^{extra}\s{''.join(non_word_character_set)}]"
        word_break_set = rf"[{extra}\s{''.join(non_word_character_set)}]"
        self.word_regex = re.compile(rf"{word_character_set}+")
        self.word_break_regex = re.compile(rf"{word_break_set}+")
        self.clitic_cleanup_regex = None
        if self.clitic_markers and len(self.clitic_markers) > 1:
            other_clitic_markers = self.clitic_markers[1:]
            if other_clitic_markers:
                extra = ""
                if "-" in other_clitic_markers:
                    extra = "-"
                    other_clitic_markers = [x for x in other_clitic_markers if x != "-"]
                self.clitic_cleanup_regex = re.compile(
                    rf'[{extra}{"".join(other_clitic_markers)}]'
                )
        non_word_character_set = sorted(self.all_punctuation)
        if "-" in self.all_punctuation:
            extra = "-"
            non_word_character_set = [x for x in non_word_character_set if x != "-"]
        self.punctuation_regex = re.compile(
            rf"^[{extra}{re.escape(''.join(non_word_character_set))}]+$"
        )

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
        if self.bracket_regex:
            for word_object in self.bracket_regex.finditer(text):
                word = word_object.group(0)
                new_word = self.bracket_sanitize_regex.sub("_", word)

                text = text.replace(word, new_word)

        if self.clitic_cleanup_regex:
            text = self.clitic_cleanup_regex.sub(self.base_clitic_marker, text)

        clitic_check = self.base_clitic_marker and self.base_clitic_marker in text
        words = self.word_break_regex.split(text)

        for w in words:
            if not w:
                continue
            if self.punctuation_regex.match(w):
                continue
            if clitic_check and w[0] == self.base_clitic_marker == w[-1]:
                w = w[1:-1]
            if w:
                yield w


class SplitWordsFunction:
    """
    Class for functions that splits words that have compound and clitic markers

    Parameters
    ----------
    clitic_markers: list[str]
        Characters that mark clitics
    compound_markers: list[str]
        Characters that mark compound words
    clitic_set: set[str]
        Set of clitic words
    brackets: list[tuple[str, str], optional
        Character tuples to treat as full brackets around words
    words_mapping: dict[str, int]
        Mapping of words to integer IDs
    specials_set: set[str]
        Set of special words
    oov_word : str
        What to label words not in the dictionary, defaults to None
    oov_word : str
        What to label words that are bracketed, defaults to None
    """

    def __init__(
        self,
        clitic_markers: List[str],
        compound_markers: List[str],
        clitic_set: Set[str],
        brackets: List[Tuple[str, str]],
        word_mapping: Optional[Dict[str, int]] = None,
        specials_set: Optional[Set[str]] = None,
        oov_word: Optional[str] = None,
        bracketed_word: Optional[str] = None,
        laughter_word: Optional[str] = None,
        laughter_regex: Optional[str] = None,
    ):
        self.clitic_marker = clitic_markers[0] if len(clitic_markers) else None
        self.compound_markers = compound_markers
        self.clitic_set = clitic_set
        self.specials_set = specials_set
        self.oov_word = oov_word
        self.bracketed_word = bracketed_word
        self.laughter_word = laughter_word
        if not word_mapping:
            word_mapping = None
        self.word_mapping = word_mapping
        self.compound_pattern = None
        self.clitic_pattern = None
        self.initial_clitic_pattern = None
        self.final_clitic_pattern = None
        self.bracket_pattern = None
        self.laughter_regex = re.compile(laughter_regex)
        if brackets:
            left_brackets = re.escape("".join(x[0] for x in brackets))
            right_brackets = "".join(x[1] for x in brackets)
            self.bracket_pattern = re.compile(rf"^[{left_brackets}].*[{right_brackets}]$")
        if compound_markers:
            extra = ""
            if "-" in compound_markers:
                extra = "-"
                compound_markers = [x for x in compound_markers if x != "-"]
            self.compound_pattern = re.compile(rf"[{extra}{''.join(compound_markers)}]")
        initial_clitics = sorted(x for x in self.clitic_set if x.endswith(self.clitic_marker))
        final_clitics = sorted(x for x in self.clitic_set if x.startswith(self.clitic_marker))
        self.has_initial = False
        self.has_final = False
        if initial_clitics:
            self.initial_clitic_pattern = re.compile(rf"^{'|'.join(initial_clitics)}(?=\w)")
            self.has_initial = True
        if final_clitics:
            self.final_clitic_pattern = re.compile(rf"(?<=\w){'|'.join(final_clitics)}$")
            self.has_final = True

    def to_int(self, normalized_text: str) -> int:
        """
        Convert normalized text to an integer ID

        Parameters
        ----------
        normalized_text:
            Word to convert

        Returns
        -------
        int
            Integer ID for the word
        """
        if normalized_text in self.word_mapping and normalized_text not in self.specials_set:
            return self.word_mapping[normalized_text]
        elif self.laughter_regex and self.laughter_regex.match(normalized_text):
            return self.word_mapping[self.laughter_word]
        elif self.bracket_pattern and self.bracket_pattern.match(normalized_text):
            return self.word_mapping[self.bracketed_word]
        else:
            return self.word_mapping[self.oov_word]

    def split_clitics(
        self,
        item: str,
    ) -> List[str]:
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
        if self.compound_pattern is not None and not item[-1] in self.compound_markers:
            s = self.compound_pattern.split(item)
        else:
            s = [item]
        benefit = False
        for seg in s:
            if not seg:
                continue
            if not self.clitic_marker or self.clitic_marker not in seg:
                split.append(seg)
                if not benefit and seg in self.word_mapping:
                    benefit = True
                continue
            elif seg.startswith(self.clitic_marker):
                if seg[1:] in self.word_mapping:
                    split.append(seg[1:])
                    benefit = True
                    continue
            elif seg.endswith(self.clitic_marker):
                if seg[:-1] in self.word_mapping:
                    split.append(seg[:-1])
                    benefit = True
                    continue

            initial_clitics = []
            final_clitics = []
            if self.has_initial:
                while True:
                    clitic = self.initial_clitic_pattern.match(seg)
                    if clitic is None:
                        break
                    benefit = True
                    initial_clitics.append(clitic.group(0))
                    seg = seg[clitic.end(0) :]
            if self.has_final:
                while True:
                    clitic = self.final_clitic_pattern.search(seg)
                    if clitic is None:
                        break
                    benefit = True
                    final_clitics.append(clitic.group(0))
                    seg = seg[: clitic.start(0)]
                final_clitics.reverse()
            split.extend(initial_clitics)
            split.append(seg)
            split.extend(final_clitics)
            if not benefit and seg in self.word_mapping:
                benefit = True
        if not benefit:
            return [item]
        return split

    def __call__(
        self,
        item: str,
    ) -> List[str]:
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
        if self.word_mapping is not None and item in self.word_mapping:
            return [item]
        if self.bracket_pattern and self.bracket_pattern.match(item):
            return [item]
        return self.split_clitics(item)


class DictionaryMixin:
    """
    Abstract class for MFA classes that use acoustic models

    Parameters
    ----------
    oov_word : str
        What to label words not in the dictionary, defaults to ``'<unk>'``
    position_dependent_phones : bool
        Specifies whether phones should be represented as dependent on their
        position in the word (beginning, middle or end), defaults to True
    num_silence_states : int
        Number of states to use for silence phones, defaults to 5
    num_non_silence_states : int
        Number of states to use for non-silence phones, defaults to 3
    shared_silence_phones : bool
        Specify whether to share states across all silence phones, defaults
        to False
    ignore_case: bool
        Flag for whether all items should be converted to lower case, defaults to True
    silence_probability : float
        Probability of optional silences following words, defaults to 0.5
    initial_silence_probability : float
        Probability of initial silence, defaults to 0.5
    final_silence_correction : float
        Correction term on final silence, defaults to None
    final_non_silence_correction : float
        Correction term on final non-silence, defaults to None
    punctuation: str, optional
        Punctuation to use when parsing text
    clitic_markers: str, optional
        Clitic markers to use when parsing text
    compound_markers: str, optional
        Compound markers to use when parsing text
    quote_markers: list[str], optional
        Quotation markers to use when parsing text
    word_break_markers: list[str], optional
        Word break markers to use when parsing text
    brackets: list[tuple[str, str], optional
        Character tuples to treat as full brackets around words
    clitic_set: set[str]
        Set of clitic words
    disambiguation_symbols: set[str]
        Set of disambiguation symbols
    max_disambiguation_symbol: int
        Maximum number of disambiguation symbols required, defaults to 0
    preserve_suprasegmentals: int
        Flag for whether to keep phones separated by tone and stress
    base_phone_mapping: dict[str, str]
        Mapping between phone symbols to make them share a base root for decision trees
    """

    positions: List[str] = ["_B", "_E", "_I", "_S"]

    def __init__(
        self,
        oov_word: str = "<unk>",
        silence_word: str = "<eps>",
        optional_silence_phone: str = "sil",
        oov_phone: str = "spn",
        other_noise_phone: Optional[str] = None,
        position_dependent_phones: bool = True,
        num_silence_states: int = 5,
        num_non_silence_states: int = 3,
        shared_silence_phones: bool = False,
        ignore_case: bool = True,
        silence_probability: float = 0.5,
        initial_silence_probability: float = 0.5,
        final_silence_correction: float = None,
        final_non_silence_correction: float = None,
        punctuation: List[str] = None,
        clitic_markers: List[str] = None,
        compound_markers: List[str] = None,
        quote_markers: List[str] = None,
        word_break_markers: List[str] = None,
        brackets: List[Tuple[str, str]] = None,
        non_silence_phones: Set[str] = None,
        disambiguation_symbols: Set[str] = None,
        clitic_set: Set[str] = None,
        max_disambiguation_symbol: int = 0,
        phone_set_type: typing.Union[str, PhoneSetType] = "UNKNOWN",
        preserve_suprasegmentals: bool = False,
        base_phone_mapping: Dict[str, str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.punctuation = DEFAULT_PUNCTUATION
        self.clitic_markers = DEFAULT_CLITIC_MARKERS
        self.compound_markers = DEFAULT_COMPOUND_MARKERS
        self.brackets = DEFAULT_BRACKETS
        self.quote_markers = DEFAULT_QUOTE_MARKERS
        self.word_break_markers = DEFAULT_WORD_BREAK_MARKERS
        if punctuation is not None:
            self.punctuation = punctuation
        if clitic_markers is not None:
            self.clitic_markers = clitic_markers
        if compound_markers is not None:
            self.compound_markers = compound_markers
        if brackets is not None:
            self.brackets = brackets
        if quote_markers is not None:
            self.quote_markers = quote_markers
        if word_break_markers is not None:
            self.word_break_markers = word_break_markers
        self.num_silence_states = num_silence_states
        self.num_non_silence_states = num_non_silence_states
        self.shared_silence_phones = shared_silence_phones
        self.silence_probability = silence_probability
        self.initial_silence_probability = initial_silence_probability
        self.final_silence_correction = final_silence_correction
        self.final_non_silence_correction = final_non_silence_correction
        self.ignore_case = ignore_case
        self.oov_word = oov_word
        self.silence_word = silence_word
        self.bracketed_word = "[bracketed]"
        self.laughter_word = "[laughter]"
        self.position_dependent_phones = position_dependent_phones
        self.optional_silence_phone = optional_silence_phone
        self.other_noise_phone = other_noise_phone
        self.oov_phone = oov_phone
        self.oovs_found = Counter()
        if non_silence_phones is None:
            non_silence_phones = set()
        self.non_silence_phones = non_silence_phones
        self.excluded_phones = set()
        self.excluded_pronunciation_count = 0
        self.max_disambiguation_symbol = max_disambiguation_symbol
        if disambiguation_symbols is None:
            disambiguation_symbols = set()
        self.disambiguation_symbols = disambiguation_symbols
        if clitic_set is None:
            clitic_set = set()
        self.clitic_set = clitic_set
        if not isinstance(phone_set_type, PhoneSetType):
            phone_set_type = PhoneSetType[phone_set_type]
        self.phone_set_type = phone_set_type
        self.preserve_suprasegmentals = preserve_suprasegmentals
        self.base_phone_mapping = base_phone_mapping

    @property
    def base_phones(self) -> Dict[str, Set[str]]:
        """Grouped phones by base phone"""
        base_phones = {}
        for p in self.non_silence_phones:
            b = self.get_base_phone(p)
            if b not in base_phones:
                base_phones[b] = set()
            base_phones[b].add(p)

        return base_phones

    def get_base_phone(self, phone: str) -> str:
        """
        Get the base phone, either through stripping diacritics, tone, and/or stress

        Parameters
        ----------
        phone: str
            Phone used in pronunciation dictionary

        Returns
        -------
        str
            Base phone
        """
        if self.preserve_suprasegmentals and (
            self is PhoneSetType.ARPA or self is PhoneSetType.PINYIN
        ):
            return phone
        elif self.preserve_suprasegmentals:
            pattern = self.phone_set_type.suprasegmental_phone_regex
        else:
            pattern = self.phone_set_type.base_phone_regex
        if self.phone_set_type.has_base_phone_regex:
            base_phone = pattern.sub("", phone)
            if self.base_phone_mapping and base_phone in self.base_phone_mapping:
                return self.base_phone_mapping[base_phone]
            return base_phone
        return phone

    @property
    def extra_questions_mapping(self) -> Dict[str, List[str]]:
        """Mapping of extra questions for the given phone set type"""
        mapping = {"silence_question": []}
        for p in sorted(self.silence_phones):
            mapping["silence_question"].append(p)
            if self.position_dependent_phones:
                mapping["silence_question"].extend([p + x for x in self.positions])
        for k, v in self.phone_set_type.extra_questions.items():
            if k not in mapping:
                mapping[k] = []
            if self.phone_set_type is PhoneSetType.ARPA:
                if self.position_dependent_phones:
                    for x in sorted(v):
                        mapping[k].extend([x + pos for pos in self.positions])
                else:
                    mapping[k] = sorted(v)
            elif self.phone_set_type == PhoneSetType.IPA:
                filtered_v = set()
                for x in self.non_silence_phones:
                    base_phone = self.get_base_phone(x)
                    if base_phone in v:
                        filtered_v.add(x)
                if len(filtered_v) < 2:
                    del mapping[k]
                    continue
                if self.position_dependent_phones:
                    for x in sorted(filtered_v):
                        mapping[k].extend([x + pos for pos in self.positions])
                else:
                    mapping[k] = sorted(filtered_v)
            elif self.phone_set_type is PhoneSetType.PINYIN:
                filtered_v = set()
                for x in self.non_silence_phones:
                    base_phone = self.get_base_phone(x)
                    if base_phone in v or x in v:
                        filtered_v.add(x)
                    elif x in v:
                        filtered_v.add(x)
                if len(filtered_v) < 2:
                    del mapping[k]
                    continue
                if self.position_dependent_phones:
                    for x in sorted(filtered_v):
                        mapping[k].extend([x + pos for pos in self.positions])
                else:
                    mapping[k] = sorted(filtered_v)
        if self.position_dependent_phones:
            phones = sorted(self.non_silence_phones)
            for pos in self.positions:
                mapping[f"non_silence{pos}"] = [x + pos for x in phones]
            silence_phones = sorted(self.silence_phones)
            for pos in [""] + self.positions:
                mapping[f"silence{pos}"] = [x + pos for x in silence_phones]
        return mapping

    @property
    def dictionary_options(self) -> MetaDict:
        """Dictionary options"""
        return {
            "punctuation": self.punctuation,
            "clitic_markers": self.clitic_markers,
            "clitic_set": self.clitic_set,
            "compound_markers": self.compound_markers,
            "brackets": self.brackets,
            "num_silence_states": self.num_silence_states,
            "num_non_silence_states": self.num_non_silence_states,
            "shared_silence_phones": self.shared_silence_phones,
            "silence_probability": self.silence_probability,
            "initial_silence_probability": self.initial_silence_probability,
            "final_silence_correction": self.final_silence_correction,
            "final_non_silence_correction": self.final_non_silence_correction,
            "oov_word": self.oov_word,
            "silence_word": self.silence_word,
            "position_dependent_phones": self.position_dependent_phones,
            "optional_silence_phone": self.optional_silence_phone,
            "oov_phone": self.oov_phone,
            "non_silence_phones": self.non_silence_phones,
            "max_disambiguation_symbol": self.max_disambiguation_symbol,
            "disambiguation_symbols": self.disambiguation_symbols,
            "phone_set_type": str(self.phone_set_type),
        }

    @property
    def silence_phones(self):
        """Silence phones"""
        if self.other_noise_phone is not None:
            return {self.optional_silence_phone, self.oov_phone, self.other_noise_phone}

        return {
            self.optional_silence_phone,
            self.oov_phone,
        }

    @property
    def context_independent_csl(self):
        """Context independent colon-separated list"""
        return ":".join(str(self.phone_mapping[x]) for x in self.kaldi_silence_phones)

    @property
    def specials_set(self):
        """Special words, like the ``oov_word`` ``silence_word``, ``<s>``, and ``</s>``"""
        return {self.silence_word, "<s>", "</s>"}

    @property
    def phone_mapping(self) -> Dict[str, int]:
        """Mapping of phones to integer IDs"""
        phone_mapping = {}
        i = 0
        phone_mapping["<eps>"] = i
        for p in self.kaldi_silence_phones:
            i += 1
            phone_mapping[p] = i
        for p in self.kaldi_non_silence_phones:
            i += 1
            phone_mapping[p] = i
        i = max(phone_mapping.values())
        for x in range(self.max_disambiguation_symbol + 2):
            p = f"#{x}"
            self.disambiguation_symbols.add(p)
            i += 1
            phone_mapping[p] = i
        return phone_mapping

    @property
    def silence_disambiguation_symbol(self) -> str:
        """
        Silence disambiguation symbol
        """
        return f"#{self.max_disambiguation_symbol + 1}"

    @property
    def reversed_phone_mapping(self) -> ReversedMappingType:
        """
        A mapping of integer ids to phones
        """
        mapping = {}
        for k, v in self.phone_mapping.items():
            mapping[v] = k
        return mapping

    @property
    def positional_silence_phones(self) -> List[str]:
        """
        List of silence phones with positions
        """
        silence_phones = []
        for p in sorted(self.silence_phones):
            silence_phones.append(p)
            for pos in self.positions:
                silence_phones.append(p + pos)
        return silence_phones

    def _generate_positional_list(self, phones: Set[str]) -> List[str]:
        """
        Helper function to generate positional list for phones along with any base phones for the phone set

        Parameters
        ----------
        phones: set[str]
            Set of phones

        Returns
        -------
        list[str]
            List of positional phones, sorted by base phone
        """
        positional_phones = []
        phones |= {self.get_base_phone(p) for p in phones}
        for p in sorted(phones):
            if p not in self.non_silence_phones:
                continue
            for pos in self.positions:
                pos_p = p + pos
                if pos_p not in positional_phones:
                    positional_phones.append(pos_p)
        return positional_phones

    def _generate_non_positional_list(self, phones: Set[str]) -> List[str]:
        """
        Helper function to generate non-positional list for phones with any base phones for the phone set

        Parameters
        ----------
        phones: set[str]
            Set of phones

        Returns
        -------
        list[str]
            List of non-positional phones, sorted by base phone
        """
        base_phones = set()
        for p in phones:
            base_phone = self.get_base_phone(p)
            base_phones.add(base_phone)

        return sorted(phones | base_phones)

    def _generate_phone_list(self, phones: Set[str]) -> List[str]:
        """
        Helper function to generate phone list

        Parameters
        ----------
        phones: set[str]
            Set of phones

        Returns
        -------
        list[str]
            List of positional or non-positional phones, sorted by base phone
        """
        if self.position_dependent_phones:
            return self._generate_positional_list(phones)
        return self._generate_non_positional_list(phones)

    @property
    def positional_non_silence_phones(self) -> List[str]:
        """
        List of non-silence phones with positions
        """
        return self._generate_positional_list(self.non_silence_phones)

    @property
    def kaldi_non_silence_phones(self):
        """Non silence phones in Kaldi format"""
        if self.position_dependent_phones:
            return self.positional_non_silence_phones
        return self._generate_non_positional_list(self.non_silence_phones)

    @property
    def kaldi_grouped_phones(self) -> Dict[str, List[str]]:
        """Non silence phones in Kaldi format"""
        groups = {}
        for p in sorted(self.non_silence_phones):
            base_phone = self.get_base_phone(p)
            if base_phone not in groups:
                if self.position_dependent_phones:
                    groups[base_phone] = [base_phone + pos for pos in self.positions]
                else:
                    groups[base_phone] = [base_phone]
            if self.position_dependent_phones:
                groups[base_phone].extend(
                    [p + pos for pos in self.positions if p + pos not in groups[base_phone]]
                )
            else:
                if p not in groups[base_phone]:
                    groups[base_phone].append(p)
        return groups

    @property
    def kaldi_silence_phones(self):
        """Silence phones in Kaldi format"""
        if self.position_dependent_phones:
            return self.positional_silence_phones
        return sorted(self.silence_phones)

    @property
    def optional_silence_csl(self) -> str:
        """
        Phone ID of the optional silence phone
        """
        return str(self.phone_mapping[self.optional_silence_phone])

    @property
    def silence_csl(self) -> str:
        """
        A colon-separated string of silence phone ids
        """
        return ":".join(map(str, (self.phone_mapping[x] for x in self.kaldi_silence_phones)))

    @property
    def non_silence_csl(self) -> str:
        """
        A colon-separated string of non-silence phone ids
        """
        return ":".join(map(str, (self.phone_mapping[x] for x in self.kaldi_non_silence_phones)))

    @property
    def phones(self) -> set:
        """
        The set of all phones (silence and non-silence)
        """
        return self.silence_phones | self.non_silence_phones

    def check_bracketed(self, word: str) -> bool:
        """
        Checks whether a given string is surrounded by brackets.

        Parameters
        ----------
        word : str
            Text to check for final brackets

        Returns
        -------
        bool
            True if the word is fully bracketed, false otherwise
        """
        for b in self.brackets:
            if re.match(rf"^{re.escape(b[0])}.*{re.escape(b[1])}$", word):
                return True
        return False

    def construct_sanitize_function(self) -> SanitizeFunction:
        """
        Construct a :class:`~montreal_forced_aligner.dictionary.mixins.SanitizeFunction` to use in multiprocessing jobs

        Returns
        -------
        :class:`~montreal_forced_aligner.dictionary.mixins.SanitizeFunction`
            Function for sanitizing text
        """
        f = SanitizeFunction(
            self.punctuation,
            self.clitic_markers,
            self.compound_markers,
            self.brackets,
            self.ignore_case,
            self.quote_markers,
            self.word_break_markers,
        )

        return f

    def sanitize(self, text: str) -> typing.Generator[str]:
        """
        Sanitize text according to punctuation and clitic markers

        Parameters
        ----------
        text: str
            Text to sanitize

        Returns
        -------
        Generator[str]
            Sanitized form
        """
        yield from self.construct_sanitize_function()(text)


class TemporaryDictionaryMixin(DictionaryMixin, DatabaseMixin, metaclass=abc.ABCMeta):
    """
    Mixin for dictionaries backed by a temporary directory
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._disambiguation_symbols_int_path = None
        self._phones_dir = None
        self._lexicon_fst_paths = {}

    @property
    def word_boundary_int_path(self) -> str:
        """Path to the word boundary integer IDs"""
        return os.path.join(self.dictionary_output_directory, "phones", "word_boundary.int")

    def _write_word_boundaries(self) -> None:
        """
        Write the word boundaries file to the temporary directory
        """
        boundary_path = os.path.join(
            self.dictionary_output_directory, "phones", "word_boundary.txt"
        )
        with open(boundary_path, "w", encoding="utf8") as f, open(
            self.word_boundary_int_path, "w", encoding="utf8"
        ) as intf:
            if self.position_dependent_phones:
                for p in sorted(self.phone_mapping.keys(), key=lambda x: self.phone_mapping[x]):
                    if p == "<eps>" or p.startswith("#"):
                        continue
                    cat = "nonword"
                    if p.endswith("_B"):
                        cat = "begin"
                    elif p.endswith("_S"):
                        cat = "singleton"
                    elif p.endswith("_I"):
                        cat = "internal"
                    elif p.endswith("_E"):
                        cat = "end"
                    f.write(" ".join([p, cat]) + "\n")
                    intf.write(" ".join([str(self.phone_mapping[p]), cat]) + "\n")

    def _get_grouped_phones(self) -> Dict[str, Set[str]]:
        """
        Group phones for use in Kaldi processing

        Returns
        -------
        dict[str, set[str]]
            Grouped phone by manner
        """
        phones = {
            "stops": set(),
            "fricatives": set(),
            "affricates": set(),
            "liquids": set(),
            "nasals": set(),
            "monophthongs": set(),
            "diphthongs": set(),
            "triphthongs": set(),
            "other": set(),
        }
        for p in self.non_silence_phones:
            base_phone = self.get_base_phone(p)
            if base_phone in self.phone_set_type.stops:
                phones["stops"].add(p)
            elif base_phone in self.phone_set_type.affricates:
                phones["affricates"].add(p)
            elif base_phone in (
                self.phone_set_type.laterals
                | self.phone_set_type.approximants
                | self.phone_set_type.nasal_approximants
            ):
                phones["liquids"].add(p)
            elif base_phone in (
                self.phone_set_type.fricatives
                | self.phone_set_type.lateral_fricatives
                | self.phone_set_type.sibilants
            ):
                phones["fricatives"].add(p)
            elif base_phone in self.phone_set_type.vowels:
                phones["monophthongs"].add(p)
            elif base_phone in self.phone_set_type.diphthong_phones:
                phones["diphthongs"].add(p)
            elif base_phone in self.phone_set_type.triphthong_phones:
                phones["triphthongs"].add(p)
            else:
                phones["other"].add(p)

        return phones

    def _write_topo(self) -> None:
        """
        Write the topo file to the temporary directory
        """

        sil_transp = 1 / (self.num_silence_states - 1)

        silence_lines = [
            "<TopologyEntry>",
            "<ForPhones>",
            " ".join(str(self.phone_mapping[x]) for x in self.kaldi_silence_phones),
            "</ForPhones>",
        ]
        for i in range(self.num_silence_states):
            if i == 0:  # Initial silence state
                transition_string = " ".join(
                    f"<Transition> {x} {sil_transp}" for x in range(self.num_silence_states - 1)
                )
                silence_lines.append(f"<State> {i} <PdfClass> {i} {transition_string} </State>")
            elif i < self.num_silence_states - 1:  # non-final silence states
                transition_string = " ".join(
                    f"<Transition> {x} {sil_transp}" for x in range(1, self.num_silence_states)
                )
                silence_lines.append(f"<State> {i} <PdfClass> {i} {transition_string} </State>")
            else:
                silence_lines.append(
                    f"<State> {i} <PdfClass> {i} <Transition> {i} 0.75 <Transition> {self.num_silence_states} 0.25 </State>"
                )
        silence_lines.append(f"<State> {self.num_silence_states} </State>")
        silence_lines.append("</TopologyEntry>")
        silence_topo_string = "\n".join(silence_lines)

        topo_sections = [silence_topo_string]
        topo_phones = self._get_grouped_phones()

        for phone_list in topo_phones.values():
            if not phone_list:
                continue
            non_silence_lines = [
                "<TopologyEntry>",
                "<ForPhones>",
                " ".join(
                    str(self.phone_mapping[x]) for x in self._generate_phone_list(phone_list)
                ),
                "</ForPhones>",
            ]
            # num_states = state_mapping[phone_type]
            num_states = self.num_non_silence_states

            for i in range(num_states):
                if i == 0:  # Initial non_silence state
                    transition_probability = 1 / self.num_non_silence_states
                    transition_string = " ".join(
                        f"<Transition> {x} {transition_probability}"
                        for x in range(1, self.num_non_silence_states + 1)
                    )
                    non_silence_lines.append(
                        f"<State> {i} <PdfClass> {i} {transition_string} </State>"
                    )
                elif i == num_states - 1:
                    non_silence_lines.append(
                        f"<State> {i} <PdfClass> {i} <Transition> {i+1} 1.0 </State>"
                    )
                else:
                    non_silence_lines.append(
                        f"<State> {i} <PdfClass> {i} <Transition> {i} 0.5 <Transition> {i+1} 0.5 </State>"
                    )
            non_silence_lines.append(f"<State> {num_states} </State>")
            non_silence_lines.append("</TopologyEntry>")
            non_silence_topo_string = "\n".join(non_silence_lines)
            topo_sections.append(non_silence_topo_string)

        with open(self.topo_path, "w") as f:
            f.write("<Topology>\n")
            for section in topo_sections:
                f.write(section + "\n\n")
            f.write("</Topology>\n")

    def _write_phone_sets(self) -> None:
        """
        Write phone symbol sets to the temporary directory
        """

        sets_file = os.path.join(self.dictionary_output_directory, "phones", "sets.txt")
        roots_file = os.path.join(self.dictionary_output_directory, "phones", "roots.txt")

        sets_int_file = os.path.join(self.dictionary_output_directory, "phones", "sets.int")
        roots_int_file = os.path.join(self.dictionary_output_directory, "phones", "roots.int")

        with open(sets_file, "w", encoding="utf8") as setf, open(
            roots_file, "w", encoding="utf8"
        ) as rootf, open(sets_int_file, "w", encoding="utf8") as setintf, open(
            roots_int_file, "w", encoding="utf8"
        ) as rootintf:

            # process silence phones
            if self.shared_silence_phones:
                phone_string = " ".join(self.kaldi_silence_phones)
                phone_int_string = " ".join(
                    str(self.phone_mapping[x]) for x in self.kaldi_silence_phones
                )
                setf.write(f"{phone_string}\n")
                setintf.write(f"{phone_int_string}\n")
                rootf.write(f"not-shared not-split {phone_string}\n")
                rootintf.write(f"not-shared not-split {phone_int_string}\n")
            else:
                for sp in self.silence_phones:
                    if self.position_dependent_phones:
                        mapped = [sp + x for x in [""] + self.positions]
                    else:
                        mapped = [sp]
                    phone_string = " ".join(mapped)
                    phone_int_string = " ".join(str(self.phone_mapping[x]) for x in mapped)
                    setf.write(f"{phone_string}\n")
                    setintf.write(f"{phone_int_string}\n")
                    rootf.write(f"shared split {phone_string}\n")
                    rootintf.write(f"shared split {phone_int_string}\n")

            # process nonsilence phones
            for group in self.kaldi_grouped_phones.values():

                phone_string = " ".join(group)
                phone_int_string = " ".join(str(self.phone_mapping[x]) for x in group)
                setf.write(f"{phone_string}\n")
                setintf.write(f"{phone_int_string}\n")
                rootf.write(f"shared split {phone_string}\n")
                rootintf.write(f"shared split {phone_int_string}\n")

    @property
    def phone_symbol_table_path(self):
        """Path to file containing phone symbols and their integer IDs"""
        return os.path.join(self.phones_dir, "phones.txt")

    @property
    def disambiguation_symbols_txt_path(self):
        """Path to the file containing phone disambiguation symbols"""
        return os.path.join(self.phones_dir, "disambiguation_symbols.txt")

    @property
    def disambiguation_symbols_int_path(self):
        """Path to the file containing integer IDs for phone disambiguation symbols"""
        if self._disambiguation_symbols_int_path is None:
            self._disambiguation_symbols_int_path = os.path.join(
                self.phones_dir, "disambiguation_symbols.int"
            )
        return self._disambiguation_symbols_int_path

    @property
    def phones_dir(self) -> str:
        """Directory for storing phone information"""
        if self._phones_dir is None:
            self._phones_dir = os.path.join(self.dictionary_output_directory, "phones")
        return self._phones_dir

    @property
    def topo_path(self) -> str:
        """Path to the dictionary's topology file"""
        return os.path.join(self.phones_dir, "topo")

    def _write_disambig(self) -> None:
        """
        Write disambiguation symbols to the temporary directory
        """
        disambig = self.disambiguation_symbols_txt_path
        disambig_int = self.disambiguation_symbols_int_path
        with open(disambig, "w", encoding="utf8") as outf, open(
            disambig_int, "w", encoding="utf8"
        ) as intf:
            for d in sorted(self.disambiguation_symbols, key=lambda x: self.phone_mapping[x]):
                outf.write(f"{d}\n")
                intf.write(f"{self.phone_mapping[d]}\n")
