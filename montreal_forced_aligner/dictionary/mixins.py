"""Mixins for dictionary parsing capabilities"""

from __future__ import annotations

import abc
import os
import re
import typing
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from montreal_forced_aligner.abc import DatabaseMixin
from montreal_forced_aligner.data import PhoneSetType, PhoneType, WordType
from montreal_forced_aligner.db import Dictionary, Phone, Word
from montreal_forced_aligner.helper import mfa_open

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict

DEFAULT_PUNCTUATION = list(
    r'、。।，？！!@<>→"”()“„–,.:;—¿?¡：）|؟!\\&%#*،~【】，…‥「」『』〝〟″⟨⟩♪・‚‘‹›«»～′$+=‘۔―'
)

DEFAULT_WORD_BREAK_MARKERS = list(r'？！!()，,.:;¡¿?“„"”&~%#—…‥、。|【】$+=〝〟″‹›«»・⟨⟩،「」『』؟')

DEFAULT_QUOTE_MARKERS = list("“„\"”〝〟″「」『』‚ʻʿ‘′'")

DEFAULT_CLITIC_MARKERS = list("'’‘")
DEFAULT_COMPOUND_MARKERS = list("-‑/")
DEFAULT_BRACKETS = [("<", ">"), ("[", "]"), ("{", "}"), ("(", ")"), ("＜", "＞")]

__all__ = ["DictionaryMixin", "TemporaryDictionaryMixin"]


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
        cutoff_word: str = "<cutoff>",
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
        self.cutoff_word = cutoff_word
        if (self.cutoff_word[0], self.cutoff_word[-1]) not in self.brackets:
            self.cutoff_word = f"{self.brackets[0][0]}{self.cutoff_word}{self.brackets[0][1]}"
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
        self._topologies = {}

    @property
    def tokenizer(self):
        from montreal_forced_aligner.tokenization.simple import SimpleTokenizer

        word_table = None
        if hasattr(self, "session") and hasattr(self, "_default_dictionary_id"):
            with self.session() as session:
                d = session.get(Dictionary, self._default_dictionary_id)
                word_table = d.word_table
        tokenizer = SimpleTokenizer(
            word_table=word_table,
            word_break_markers=self.word_break_markers,
            punctuation=self.punctuation,
            clitic_markers=self.clitic_markers,
            compound_markers=self.compound_markers,
            brackets=self.brackets,
            laughter_word=self.laughter_word,
            oov_word=self.oov_word,
            bracketed_word=self.bracketed_word,
            cutoff_word=self.cutoff_word,
            ignore_case=self.ignore_case,
        )
        return tokenizer

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
        mapping = {}
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
    def specials_set(self) -> Set[str]:
        """Special words, like the ``oov_word`` ``silence_word``, ``<s>``, and ``</s>``"""
        return {
            self.silence_word,
            self.oov_word,
            self.bracketed_word,
            self.laughter_word,
            self.cutoff_word,
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
        if not hasattr(self, "acoustic_model"):
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
        if (
            not self._phone_groups
            and getattr(self, "phone_group_path", None)
            and hasattr(self, "load_phone_groups")
        ):
            self.load_phone_groups()
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
    def silence_symbols(self) -> typing.List[int]:
        """
        A colon-separated string of silence phone ids
        """
        return [self.phone_mapping[x] for x in self.kaldi_silence_phones]

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
        self._num_words = None
        self._num_speech_words = None

    @property
    def num_words(self) -> int:
        """Number of words (including OOVs and special symbols) in the dictionary"""
        if self._num_words is None:
            with self.session() as session:
                self._num_words = session.query(Word).count()
        return self._num_words

    @property
    def num_speech_words(self) -> int:
        """Number of speech words in the dictionary"""
        if self._num_speech_words is None:
            with self.session() as session:
                self._num_speech_words = (
                    session.query(Word).filter(Word.word_type.in_(WordType.speech_types())).count()
                )
        return self._num_speech_words

    @property
    def word_boundary_int_path(self) -> Path:
        """Path to the word boundary integer IDs"""
        return self.dictionary_output_directory.joinpath("phones", "word_boundary.int")

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
        if (
            not self._topologies
            and getattr(self, "topology_path", None)
            and hasattr(self, "load_phone_topologies")
        ):
            self.load_phone_topologies()
        sil_transp = 1 / (self.num_silence_states - 1)
        topo_groups = defaultdict(set)

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

        for k, v in self._topologies.items():
            min_states = v.get("min_states", 1)
            max_states = v.get("max_states", self.num_non_silence_states)
            topo_groups[(min_states, max_states)].add(k)
        for phone in self.non_silence_phones:
            if phone not in self._topologies:
                topo_groups[(1, self.num_non_silence_states)].add(phone)

        for (min_states, max_states), phone_list in topo_groups.items():
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
            num_states = max_states

            for i in range(num_states):
                if i == 0:  # Initial non_silence state
                    if min_states == max_states:
                        transition_string = f"<Transition> {i} 0.5 <Transition> {i + 1} 0.5"
                    else:
                        transition_probability = 1 / max_states
                        transition_string = " ".join(
                            f"<Transition> {x} {transition_probability}"
                            for x in range(min_states, max_states + 1)
                        )
                    non_silence_lines.append(
                        f"<State> {i} <PdfClass> {i} {transition_string} </State>"
                    )
                elif i == num_states - 1:
                    non_silence_lines.append(
                        f"<State> {i} <PdfClass> {i} <Transition> {i + 1} 1.0 </State>"
                    )
                else:
                    non_silence_lines.append(
                        f"<State> {i} <PdfClass> {i} <Transition> {i} 0.5 <Transition> {i + 1} 0.5 </State>"
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

    def shared_phones_set_symbols(self):
        phone_sets = []
        if self.shared_silence_phones:
            phone_sets.append([self.phone_mapping[x] for x in self.kaldi_silence_phones])
        else:
            for sp in self.silence_phones:
                if self.position_dependent_phones:
                    mapped = [sp + x for x in [""] + self.positions]
                else:
                    mapped = [sp]
                phone_sets.append([self.phone_mapping[x] for x in mapped])
        found_phones = set()
        for group in self.kaldi_grouped_phones.values():
            for x in group:
                if x in found_phones:
                    raise Exception(f"The phone {x} in multiple phone groups.")
            found_phones.update(group)
            group = sorted(self.phone_mapping[x] for x in group)
            phone_sets.append(group)
        missing_phones = self.non_silence_phones - found_phones
        if missing_phones:
            raise Exception(
                f"The following phones were missing from phone groups: {', '.join(missing_phones)}"
            )
        return phone_sets

    def shared_phones_roots(self):
        phone_sets = []
        if self.shared_silence_phones:
            phone_sets.append([self.phone_mapping[x] for x in self.kaldi_silence_phones])
        else:
            for sp in self.silence_phones:
                if self.position_dependent_phones:
                    mapped = [sp + x for x in [""] + self.positions]
                else:
                    mapped = [sp]
                phone_sets.append([self.phone_mapping[x] for x in mapped])
        for group in self.kaldi_grouped_phones.values():
            group = sorted(self.phone_mapping[x] for x in group)
            phone_sets.append(group)
        return phone_sets

    def _write_phone_sets(self) -> None:
        """
        Write phone symbol sets to the temporary directory
        """

        sets_file = self.dictionary_output_directory.joinpath("phones", "sets.txt")
        roots_file = self.dictionary_output_directory.joinpath("phones", "roots.txt")

        sets_int_file = self.dictionary_output_directory.joinpath("phones", "sets.int")
        roots_int_file = self.dictionary_output_directory.joinpath("phones", "roots.int")

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
    def phone_symbol_table_path(self) -> Path:
        """Path to file containing phone symbols and their integer IDs"""
        return self.phones_dir.joinpath("phones.txt")

    @property
    def grapheme_symbol_table_path(self) -> Path:
        """Path to file containing grapheme symbols and their integer IDs"""
        return self.phones_dir.joinpath("graphemes.txt")

    @property
    def disambiguation_symbols_txt_path(self) -> Path:
        """Path to the file containing phone disambiguation symbols"""
        return self.phones_dir.joinpath("disambiguation_symbols.txt")

    @property
    def disambiguation_symbols_int_path(self) -> Path:
        """Path to the file containing integer IDs for phone disambiguation symbols"""
        if self._disambiguation_symbols_int_path is None:
            self._disambiguation_symbols_int_path = self.phones_dir.joinpath(
                "disambiguation_symbols.int"
            )
        return self._disambiguation_symbols_int_path

    @property
    def phones_dir(self) -> Path:
        """Directory for storing phone information"""
        if self._phones_dir is None:
            self._phones_dir = self.dictionary_output_directory.joinpath("phones")
        return self._phones_dir

    @property
    def topo_path(self) -> Path:
        """Path to the dictionary's topology file"""
        return self.phones_dir.joinpath("topo")

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
        phone_disambig_path = self.phones_dir.joinpath("phone_disambig.txt")
        with mfa_open(phone_disambig_path, "w") as f:
            f.write(str(self.phone_mapping["#0"]))
