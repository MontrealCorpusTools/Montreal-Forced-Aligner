"""Mixins for dictionary parsing capabilities"""

from __future__ import annotations

import abc
import os
import re
import typing
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from montreal_forced_aligner.abc import DatabaseMixin
from montreal_forced_aligner.data import PhoneSetType, PhoneType
from montreal_forced_aligner.db import Phone
from montreal_forced_aligner.helper import mfa_open

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict

DEFAULT_PUNCTUATION = list(r'、。।，？！!@<>→"”()“„–,.:;—¿?¡：）!\\&%#*~【】，…‥「」『』〝〟″⟨⟩♪・‹›«»～′$+=‘')

DEFAULT_WORD_BREAK_MARKERS = list(r'？！!()，,.:;¡¿?“„"”&~%#—…‥、。【】$+=〝〟″‹›«»・⟨⟩「」『』')

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
        clitic_marker: str,
        clitic_cleanup_regex: Optional[re.Pattern],
        clitic_quote_regex: Optional[re.Pattern],
        punctuation_regex: Optional[re.Pattern],
        word_break_regex: Optional[re.Pattern],
        bracket_regex: Optional[re.Pattern],
        bracket_sanitize_regex: Optional[re.Pattern],
        ignore_case: bool = True,
    ):
        self.clitic_marker = clitic_marker
        self.clitic_cleanup_regex = clitic_cleanup_regex
        self.clitic_quote_regex = clitic_quote_regex
        self.punctuation_regex = punctuation_regex
        self.word_break_regex = word_break_regex
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
        if self.bracket_regex:
            for word_object in self.bracket_regex.finditer(text):
                word = word_object.group(0)
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
    """

    def __init__(
        self,
        clitic_marker: str,
        initial_clitic_regex: Optional[re.Pattern],
        final_clitic_regex: Optional[re.Pattern],
        compound_regex: Optional[re.Pattern],
        non_speech_regexes: Dict[str, re.Pattern],
        oov_word: Optional[str] = None,
        word_mapping: Optional[Dict[str, int]] = None,
        grapheme_mapping: Optional[Dict[str, int]] = None,
    ):
        self.clitic_marker = clitic_marker
        self.compound_regex = compound_regex
        self.oov_word = oov_word
        self.specials_set = {self.oov_word, "<s>", "</s>"}
        if not word_mapping:
            word_mapping = None
        self.word_mapping = word_mapping
        if not grapheme_mapping:
            grapheme_mapping = None
        self.grapheme_mapping = grapheme_mapping
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
        for word, regex in self.non_speech_regexes.items():
            if regex.match(normalized_text):
                return word
        return normalized_text

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
        if self.compound_regex is not None:
            s = self.compound_regex.split(item)
        else:
            s = [item]
        if self.word_mapping is None:
            return [item]
        clean_initial_quote_regex = re.compile("^'")
        clean_final_quote_regex = re.compile("'$")
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
                    clitic = self.initial_clitic_regex.match(seg)
                    if clitic is None:
                        break
                    benefit = True
                    initial_clitics.append(clitic.group(0))
                    seg = seg[clitic.end(0) :]
                    if seg in self.word_mapping:
                        break
            if self.has_final:
                while True:
                    clitic = self.final_clitic_regex.search(seg)
                    if clitic is None:
                        break
                    benefit = True
                    final_clitics.append(clitic.group(0))
                    seg = seg[: clitic.start(0)]
                    if seg in self.word_mapping:
                        break
                final_clitics.reverse()
            split.extend([clean_initial_quote_regex.sub("", x) for x in initial_clitics])
            seg = clean_final_quote_regex.sub("", clean_initial_quote_regex.sub("", seg))
            if seg:
                split.append(seg)
            split.extend([clean_final_quote_regex.sub("", x) for x in final_clitics])
            if not benefit and seg in self.word_mapping:
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
            characters = list(item)
            for c in characters:
                if self.grapheme_mapping is not None and c in self.grapheme_mapping:
                    yield c
                else:
                    yield self.oov_word

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
        for regex in self.non_speech_regexes.values():
            if regex.match(item):
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
        position_dependent_phones: bool = False,
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
        use_cutoff_model: bool = False,
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
        self.clitic_marker = None
        if self.clitic_markers:
            self.clitic_marker = self.clitic_markers[0]

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
        if phone_set_type is None:
            phone_set_type = "UNKNOWN"
        if not isinstance(phone_set_type, PhoneSetType):
            phone_set_type = PhoneSetType[phone_set_type]
        self.phone_set_type = phone_set_type
        self.preserve_suprasegmentals = preserve_suprasegmentals
        self.base_phone_mapping = base_phone_mapping
        self.punctuation_regex = None
        self.compound_regex = None
        self.bracket_regex = None
        self.laughter_regex = None
        self.word_break_regex = None
        self.bracket_sanitize_regex = None
        self.use_cutoff_model = use_cutoff_model
        self._phone_groups = {}

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
            elif self.phone_set_type is PhoneSetType.IPA:
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
    def silence_phones(self) -> Set[str]:
        """Silence phones"""
        if self.other_noise_phone is not None:
            return {self.optional_silence_phone, self.oov_phone, self.other_noise_phone}

        return {
            self.optional_silence_phone,
            self.oov_phone,
        }

    @property
    def context_independent_csl(self) -> str:
        """Context independent colon-separated list"""
        return ":".join(str(self.phone_mapping[x]) for x in self.kaldi_silence_phones)

    @property
    def specials_set(self) -> Set[str]:
        """Special words, like the ``oov_word`` ``silence_word``, ``<s>``, and ``</s>``"""
        return {
            self.silence_word,
            self.oov_word,
            self.bracketed_word,
            self.laughter_word,
            "<s>",
            "</s>",
        }

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
    def reversed_phone_mapping(self) -> Dict[int, str]:
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
    def kaldi_non_silence_phones(self) -> List[str]:
        """Non silence phones in Kaldi format"""
        if self.position_dependent_phones:
            return self.positional_non_silence_phones
        return self._generate_non_positional_list(self.non_silence_phones)

    @property
    def phone_groups(self) -> typing.Dict[str, typing.List[str]]:
        if not self._phone_groups:
            for p in sorted(self.non_silence_phones):
                base_phone = self.get_base_phone(p)
                if base_phone not in self._phone_groups:
                    self._phone_groups[base_phone] = [base_phone]
                if p not in self._phone_groups[base_phone]:
                    self._phone_groups[base_phone].append(p)
        return self._phone_groups

    @property
    def kaldi_grouped_phones(self) -> Dict[str, List[str]]:
        """Non silence phones in Kaldi format"""
        groups = {}
        for k, v in self.phone_groups.items():
            if self.position_dependent_phones:
                groups[k] = [x + pos for pos in self.positions for x in v]
            else:
                groups[k] = v
        return {k: v for k, v in groups.items() if v}

    @property
    def kaldi_silence_phones(self) -> List[str]:
        """Silence phones in Kaldi format"""
        if self.position_dependent_phones:
            return self.positional_silence_phones
        return sorted(self.silence_phones)

    @property
    def optional_silence_csl(self) -> str:
        """
        Phone ID of the optional silence phone
        """
        try:
            return str(self.phone_mapping[self.optional_silence_phone])
        except Exception:
            return ""

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
        with mfa_open(boundary_path, "w") as f, mfa_open(self.word_boundary_int_path, "w") as intf:
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

        with mfa_open(self.topo_path, "w") as f:
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

        with mfa_open(sets_file, "w") as setf, mfa_open(roots_file, "w") as rootf, mfa_open(
            sets_int_file, "w"
        ) as setintf, mfa_open(roots_int_file, "w") as rootintf:

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
                group = sorted(group, key=lambda x: self.phone_mapping[x])
                phone_string = " ".join(group)
                phone_int_string = " ".join(str(self.phone_mapping[x]) for x in group)
                setf.write(f"{phone_string}\n")
                setintf.write(f"{phone_int_string}\n")
                rootf.write(f"shared split {phone_string}\n")
                rootintf.write(f"shared split {phone_int_string}\n")

    @property
    def phone_symbol_table_path(self) -> str:
        """Path to file containing phone symbols and their integer IDs"""
        return os.path.join(self.phones_dir, "phones.txt")

    @property
    def grapheme_symbol_table_path(self) -> str:
        """Path to file containing grapheme symbols and their integer IDs"""
        return os.path.join(self.phones_dir, "graphemes.txt")

    @property
    def disambiguation_symbols_txt_path(self) -> str:
        """Path to the file containing phone disambiguation symbols"""
        return os.path.join(self.phones_dir, "disambiguation_symbols.txt")

    @property
    def disambiguation_symbols_int_path(self) -> str:
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
        with self.session() as session, mfa_open(disambig, "w") as outf, mfa_open(
            disambig_int, "w"
        ) as intf:
            disambiguation_symbols = session.query(Phone.mapping_id, Phone.kaldi_label).filter(
                Phone.phone_type == PhoneType.disambiguation
            )
            for p_id, p in disambiguation_symbols:
                outf.write(f"{p}\n")
                intf.write(f"{p_id}\n")
        phone_disambig_path = os.path.join(self.phones_dir, "phone_disambig.txt")
        with mfa_open(phone_disambig_path, "w") as f:
            f.write(str(self.phone_mapping["#0"]))
