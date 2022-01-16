"""Mixins for dictionary parsing capabilities"""

from __future__ import annotations

import abc
import os
import re
import typing
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from montreal_forced_aligner.abc import TemporaryDirectoryMixin
from montreal_forced_aligner.data import PhoneSetType

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict, ReversedMappingType

DEFAULT_PUNCTUATION = list(r'、。।，？@<>"(),.:;¿?¡!\\&%#*~【】，…‥「」『』〝〟″⟨⟩♪・‹›«»～′$+=‘')

DEFAULT_CLITIC_MARKERS = list("'’")
DEFAULT_COMPOUND_MARKERS = list("-/")
DEFAULT_BRACKETS = [("[", "]"), ("{", "}"), ("<", ">"), ("(", ")"), ("＜", "＞")]

__all__ = ["SanitizeFunction", "DictionaryMixin"]


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
    """

    def __init__(
        self,
        punctuation: List[str],
        clitic_markers: List[str],
        compound_markers: List[str],
        brackets: List[Tuple[str, str]],
        ignore_case: bool = True,
    ):
        self.punctuation = punctuation
        self.clitic_markers = clitic_markers
        self.compound_markers = compound_markers
        self.brackets = brackets
        self.ignore_case = ignore_case

    def __call__(self, item):
        """
        Sanitize an item according to punctuation and clitic markers

        Parameters
        ----------
        item: str
            Word to sanitize

        Returns
        -------
        str
            Sanitized form
        """
        if self.ignore_case:
            item = item.lower()
        for c in self.clitic_markers:
            item = item.replace(c, self.clitic_markers[0])
        if not item:
            return item
        for b in self.brackets:
            if re.match(rf"^{re.escape(b[0])}.*{re.escape(b[1])}$", item):
                return item
        if self.punctuation:
            item = re.sub(rf"^[{re.escape(''.join(self.punctuation))}]+", "", item)
            item = re.sub(rf"[{re.escape(''.join(self.punctuation))}]+$", "", item)
        return item


class SplitWordsFunction:
    """
    Class for functions that splits words that have compound and clitic markers

    Parameters
    ----------
    clitic_markers: list[str]
        Characters that mark clitics
    compound_markers: list[str]
        Characters that mark compound words
    """

    def __init__(
        self,
        clitic_markers: List[str],
        compound_markers: List[str],
        clitic_set: Set[str],
        word_set: Optional[Set[str]] = None,
    ):
        self.clitic_markers = clitic_markers
        self.compound_markers = compound_markers
        self.clitic_set = clitic_set
        if not word_set:
            word_set = None
        self.word_set = word_set
        self.compound_pattern = None
        self.clitic_pattern = None
        self.initial_clitic_pattern = None
        self.final_clitic_pattern = None
        if self.compound_markers:
            self.compound_pattern = re.compile(rf"[{re.escape(''.join(self.compound_markers))}]")
        initial_clitics = sorted(x for x in self.clitic_set if x.endswith(self.clitic_markers[0]))
        final_clitics = sorted(x for x in self.clitic_set if x.startswith(self.clitic_markers[0]))
        if initial_clitics:
            groups = f"({'|'.join(initial_clitics)})?" * 4
            self.initial_clitic_pattern = re.compile(rf"^{groups}$")
        if final_clitics:
            groups = f"({'|'.join(final_clitics)})?" * 4
            self.final_clitic_pattern = re.compile(rf"^{groups}$")
        if initial_clitics and final_clitics:
            self.clitic_pattern = re.compile(
                rf"^(?P<initial>({'|'.join(initial_clitics)})*)(?P<word>.+?)(?P<final>({'|'.join(final_clitics)})*)$"
            )
        elif initial_clitics:
            self.clitic_pattern = re.compile(
                rf"^(?P<initial>({'|'.join(initial_clitics)})*)(?P<word>.+?)$"
            )
        elif final_clitics:
            self.clitic_pattern = re.compile(
                rf"^(?P<word>.+?)(?P<final>({'|'.join(final_clitics)})*)$"
            )

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
        if self.word_set is not None and item in self.word_set:
            return [item]
        split = []
        if self.compound_pattern:
            s = re.split(self.compound_pattern, item)
        else:
            s = [item]
        for seg in s:
            if self.clitic_pattern is None or self.clitic_markers[0] not in seg:
                split.append(seg)
                continue

            m = re.match(self.clitic_pattern, seg)
            if not m:
                continue
            try:
                if m.group("initial"):
                    for clitic in self.initial_clitic_pattern.match(m.group("initial")).groups():
                        if clitic is None:
                            continue
                        split.append(clitic)
            except IndexError:
                pass
            split.append(m.group("word"))
            try:
                if m.group("final"):
                    for clitic in self.final_clitic_pattern.match(m.group("final")).groups():
                        if clitic is None:
                            continue
                        split.append(clitic)
            except IndexError:
                pass
        if self.word_set is not None and not any(x in self.word_set for x in split):
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
        if self.word_set is not None and item in self.word_set:
            return [item]
        split = self.split_clitics(item)
        if self.word_set is None:
            return split
        oov_count = sum(1 for x in split if x not in self.word_set)
        if oov_count < len(
            split
        ):  # Only returned split item if it gains us any transcribed speech
            return split
        return [item]


class DictionaryMixin:
    """
    Abstract class for MFA classes that use acoustic models

    Parameters
    ----------
    oov_code : str
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
        to True
    silence_probability : float
        Probability of optional silences following words, defaults to 0.5
    punctuation: str, optional
        Punctuation to use when parsing text
    clitic_markers: str, optional
        Clitic markers to use when parsing text
    compound_markers: str, optional
        Compound markers to use when parsing text
    brackets: list[tuple[str, str], optional
        Character tuples to treat as full brackets around words
    clitic_set: set[str]
        Set of clitic words
    disambiguation_symbols: set[str]
        Set of disambiguation symbols
    max_disambiguation_symbol: int
        Maximum number of disambiguation symbols required, defaults to 0
    """

    positions: List[str] = ["_B", "_E", "_I", "_S"]

    def __init__(
        self,
        oov_word: str = "<unk>",
        silence_word: str = "<sil>",
        noise_word: str = "<noise>",
        optional_silence_phone: str = "sil",
        oov_phone: str = "spn",
        other_noise_phone: str = "noi",
        position_dependent_phones: bool = True,
        num_silence_states: int = 5,
        num_noise_states: int = 5,
        num_non_silence_states: int = 3,
        shared_silence_phones: bool = False,
        ignore_case: bool = True,
        silence_probability: float = 0.5,
        punctuation: List[str] = None,
        clitic_markers: List[str] = None,
        compound_markers: List[str] = None,
        brackets: List[Tuple[str, str]] = None,
        non_silence_phones: Set[str] = None,
        disambiguation_symbols: Set[str] = None,
        clitic_set: Set[str] = None,
        max_disambiguation_symbol: int = 0,
        phone_set_type: typing.Union[str, PhoneSetType] = "UNKNOWN",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.punctuation = DEFAULT_PUNCTUATION
        self.clitic_markers = DEFAULT_CLITIC_MARKERS
        self.compound_markers = DEFAULT_COMPOUND_MARKERS
        self.brackets = DEFAULT_BRACKETS
        if punctuation is not None:
            self.punctuation = punctuation
        if clitic_markers is not None:
            self.clitic_markers = clitic_markers
        if compound_markers is not None:
            self.compound_markers = compound_markers
        if brackets is not None:
            self.brackets = brackets

        self.num_silence_states = num_silence_states
        self.num_noise_states = num_noise_states
        self.num_non_silence_states = num_non_silence_states
        self.shared_silence_phones = shared_silence_phones
        self.silence_probability = silence_probability
        self.ignore_case = ignore_case
        self.oov_word = oov_word
        self.silence_word = silence_word
        self.noise_word = noise_word
        self.position_dependent_phones = position_dependent_phones
        self.optional_silence_phone = optional_silence_phone
        self.oov_phone = oov_phone
        self.oovs_found = Counter()
        self.other_noise_phone = other_noise_phone
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

    @property
    def base_phones(self) -> Dict[str, Set[str]]:
        base_phones = {}
        for p in self.non_silence_phones:
            b = self.phone_set_type.get_base_phone(p)
            if b not in base_phones:
                base_phones[b] = set()
            base_phones[b].add(p)

        return base_phones

    @property
    def extra_questions_mapping(self) -> Dict[str, List[str]]:
        """Mapping of extra questions for the given phone set type"""
        mapping = {}
        mapping["silence_question"] = []
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
                    base_phone = self.phone_set_type.get_base_phone(x)
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
                    base_phone = self.phone_set_type.get_base_phone(x)
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
            "oov_word": self.oov_word,
            "silence_word": self.silence_word,
            "position_dependent_phones": self.position_dependent_phones,
            "optional_silence_phone": self.optional_silence_phone,
            "oov_phone": self.oov_phone,
            "other_noise_phone": self.other_noise_phone,
            "non_silence_phones": self.non_silence_phones,
            "max_disambiguation_symbol": self.max_disambiguation_symbol,
            "disambiguation_symbols": self.disambiguation_symbols,
            "phone_set_type": str(self.phone_set_type),
        }

    @property
    def silence_phones(self):
        """Silence phones"""
        return {
            self.optional_silence_phone,
            self.oov_phone,
            self.other_noise_phone,
        }

    @property
    def context_independent_csl(self):
        """Context independent colon-separated list"""
        return ":".join(str(self.phone_mapping[x]) for x in self.silence_phones)

    @property
    def specials_set(self):
        """Special words, like the ``oov_word`` ``silence_word``, ``<eps>``, ``<s>``, and ``</s>``"""
        return {self.oov_word, self.silence_word, self.noise_word, "<eps>", "<s>", "</s>"}

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
        for p in sorted(phones):
            if p not in self.non_silence_phones:
                continue
            base_phone = self.phone_set_type.get_base_phone(p)
            for pos in self.positions:
                pos_p = base_phone + pos
                if pos_p not in positional_phones:
                    positional_phones.append(pos_p)
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
            base_phone = self.phone_set_type.get_base_phone(p)
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
    def kaldi_phones_for_topo(self):
        """Mappings of phones for generating topo file"""
        mapping = {}
        for p in sorted(self.non_silence_phones):
            base_phone = self.phone_set_type.get_base_phone(p)
            query_set = {p, base_phone}
            if any(x in self.phone_set_type.extra_short_phones for x in query_set):
                num_states = 1  # One state for extra short sounds
            elif any(x in self.phone_set_type.diphthong_phones for x in query_set):
                num_states = 5  # 5 states for diphthongs (onset of first target, steady state,
                # transition to next target, steady state, offset of second target)
            elif any(x in self.phone_set_type.triphthong_phones for x in query_set):
                num_states = 6  # 5 states for diphthongs (onset of first target, steady state,
                # transition to next target, steady state, offset of second target)
            elif any(x in self.phone_set_type.affricate_phones for x in query_set):
                num_states = 4  # 4 states for affricates (closure, burst, onset of frication, offset of frication)
            elif any(x in self.phone_set_type.stop_phones for x in query_set):
                num_states = 2  # Two states for stops (closure, burst), extra states added below for aspirated, ejectives
            else:
                num_states = self.num_non_silence_states
            if self.phone_set_type is PhoneSetType.IPA:
                if re.match(r"^.*[ʱʼʰʲʷⁿˠ][ː]?$", p):
                    num_states += 1
                if re.match(r"^.*̚$", p) and p not in self.phone_set_type.extra_short_phones:
                    num_states -= 1
            elif self.phone_set_type is PhoneSetType.PINYIN:
                if p in {"c", "ch", "q"}:
                    num_states += 1
            if num_states not in mapping:
                mapping[num_states] = []
            mapping[num_states].extend(
                [x for x in self._generate_phone_list({p}) if x not in mapping[num_states]]
            )
        if self.phone_set_type is PhoneSetType.ARPA:
            mapping[1] = [x for x in mapping[1] if "0" in x]
        return mapping

    @property
    def kaldi_grouped_phones(self) -> Dict[str, List[str]]:
        """Non silence phones in Kaldi format"""
        groups = {}
        for p in sorted(self.non_silence_phones):

            base_phone = self.phone_set_type.get_base_phone(p)
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

    def save_oovs_found(self, directory: str) -> None:
        """
        Save all out of vocabulary items to a file in the specified directory

        Parameters
        ----------
        directory : str
            Path to directory to save ``oovs_found.txt``
        """
        with open(os.path.join(directory, "oovs_found.txt"), "w", encoding="utf8") as f, open(
            os.path.join(directory, "oov_counts.txt"), "w", encoding="utf8"
        ) as cf:
            for oov in sorted(self.oovs_found.keys(), key=lambda x: (-self.oovs_found[x], x)):
                f.write(oov + "\n")
                cf.write(f"{oov}\t{self.oovs_found[oov]}\n")

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
            self.punctuation, self.clitic_markers, self.compound_markers, self.brackets
        )

        return f

    def sanitize(self, item: str) -> str:
        """
        Sanitize an item according to punctuation and clitic markers

        Parameters
        ----------
        item: str
            Word to sanitize

        Returns
        -------
        str
            Sanitized form
        """
        return self.construct_sanitize_function()(item)


class TemporaryDictionaryMixin(DictionaryMixin, TemporaryDirectoryMixin, metaclass=abc.ABCMeta):
    def _write_word_boundaries(self) -> None:
        """
        Write the word boundaries file to the temporary directory
        """
        boundary_path = os.path.join(
            self.dictionary_output_directory, "phones", "word_boundary.txt"
        )
        boundary_int_path = os.path.join(
            self.dictionary_output_directory, "phones", "word_boundary.int"
        )
        with open(boundary_path, "w", encoding="utf8") as f, open(
            boundary_int_path, "w", encoding="utf8"
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
        topo_phones = self.kaldi_phones_for_topo
        for num_states, phone_list in topo_phones.items():
            non_silence_lines = [
                "<TopologyEntry>",
                "<ForPhones>",
                " ".join(str(self.phone_mapping[x]) for x in phone_list),
                "</ForPhones>",
            ]
            for i in range(num_states):
                non_silence_lines.append(
                    f"<State> {i} <PdfClass> {i} <Transition> {i} 0.75 <Transition> {i+1} 0.25 </State>"
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

    def _write_phone_symbol_table(self) -> None:
        """
        Write the phone mapping to the temporary directory
        """
        with open(self.phone_symbol_table_path, "w", encoding="utf8") as f:
            for p, i in sorted(self.phone_mapping.items(), key=lambda x: x[1]):
                f.write(f"{p} {i}\n")

    @property
    def disambiguation_symbols_txt_path(self):
        """Path to the file containing phone disambiguation symbols"""
        return os.path.join(self.phones_dir, "disambiguation_symbols.txt")

    @property
    def disambiguation_symbols_int_path(self):
        """Path to the file containing integer IDs for phone disambiguation symbols"""
        return os.path.join(self.phones_dir, "disambiguation_symbols.int")

    @property
    def phones_dir(self) -> str:
        """Directory for storing phone information"""
        return os.path.join(self.dictionary_output_directory, "phones")

    @property
    def topo_path(self) -> str:
        """Path to the dictionary's topology file"""
        return os.path.join(self.phones_dir, "topo")

    def _write_extra_questions(self) -> None:
        """
        Write extra questions symbols to the temporary directory
        """
        phone_extra = os.path.join(self.phones_dir, "extra_questions.txt")
        phone_extra_int = os.path.join(self.phones_dir, "extra_questions.int")
        with open(phone_extra, "w", encoding="utf8") as outf, open(
            phone_extra_int, "w", encoding="utf8"
        ) as intf:
            for v in self.extra_questions_mapping.values():
                if not v:
                    continue
                outf.write(f"{' '.join(v)}\n")
                intf.write(f"{' '.join(str(self.phone_mapping[x]) for x in v)}\n")
        # error

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
