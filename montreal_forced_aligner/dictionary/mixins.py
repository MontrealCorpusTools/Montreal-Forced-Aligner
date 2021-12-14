"""Mixins for dictionary parsing capabilities"""

from __future__ import annotations

import abc
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from montreal_forced_aligner.abc import TemporaryDirectoryMixin
from montreal_forced_aligner.data import CtmInterval

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import (
        DictionaryEntryType,
        MappingType,
        MetaDict,
        ReversedMappingType,
        WordsType,
    )

DEFAULT_PUNCTUATION = list(r'、。।，@<>"(),.:;¿?¡!\\&%#*~【】，…‥「」『』〝〟″⟨⟩♪・‹›«»～′$+=‘')

DEFAULT_CLITIC_MARKERS = list("'’")
DEFAULT_COMPOUND_MARKERS = list("-/")
DEFAULT_STRIP_DIACRITICS = ["ː", "ˑ", "̩", "̆", "̑", "̯", "͡", "‿", "͜"]
DEFAULT_DIGRAPHS = ["[dt][szʒʃʐʑʂɕç]", "[aoɔe][ʊɪ]"]
DEFAULT_BRACKETS = [("[", "]"), ("{", "}"), ("<", ">"), ("(", ")")]

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
    ):
        self.punctuation = punctuation
        self.clitic_markers = clitic_markers
        self.compound_markers = compound_markers
        self.brackets = brackets

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
        punctuation: List[str],
        clitic_markers: List[str],
        compound_markers: List[str],
        brackets: List[Tuple[str, str]],
        clitic_set: Set[str],
        word_set: Optional[Set[str]] = None,
    ):
        self.punctuation = punctuation
        self.clitic_markers = clitic_markers
        self.compound_markers = compound_markers
        self.brackets = brackets
        self.sanitize_function = SanitizeFunction(
            punctuation, clitic_markers, compound_markers, brackets
        )
        self.clitic_set = clitic_set
        if not word_set:
            word_set = None
        self.word_set = word_set
        self.compound_pattern = re.compile(rf"[{re.escape(''.join(self.compound_markers))}]")
        initial_clitics = sorted(
            x for x in self.clitic_set if any(x.endswith(y) for y in self.clitic_markers)
        )
        final_clitics = sorted(
            x for x in self.clitic_set if any(x.startswith(y) for y in self.clitic_markers)
        )
        optional_initial_groups = f"({'|'.join(initial_clitics)})?" * 4
        optional_final_groups = f"({'|'.join(final_clitics)})?" * 4
        if initial_clitics and final_clitics:
            self.clitic_pattern = re.compile(
                rf"^(?:(?:{optional_initial_groups}(.+?))|(?:(.+?){optional_final_groups}))$"
            )
        elif initial_clitics:
            self.clitic_pattern = re.compile(rf"^(?:(?:{optional_initial_groups}(.+?))|(.+))$")
        elif final_clitics:
            self.clitic_pattern = re.compile(rf"^(?:(.+)|(?:(.+?){optional_final_groups}))$")
        else:
            self.clitic_pattern = None

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
        s = re.split(self.compound_pattern, item)
        for seg in s:
            if self.clitic_pattern is None:
                split.append(seg)
                continue

            m = re.match(self.clitic_pattern, seg)
            if not m:
                split.append(seg)
                continue
            for g in m.groups():
                if g is None:
                    continue
                split.append(g)
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
        sanitized = self.sanitize_function(item)
        if self.word_set is not None and sanitized in self.word_set:
            return [sanitized]
        split = self.split_clitics(sanitized)
        if self.word_set is None:
            return split
        oov_count = sum(1 for x in split if x not in self.word_set)
        if oov_count < len(
            split
        ):  # Only returned split item if it gains us any transcribed speech
            return split
        return [sanitized]


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
    multilingual_ipa: bool
        Flag for multilingual IPA mode, defaults to False
    strip_diacritics: list[str], optional
        Diacritics to strip in multilingual IPA mode
    digraphs: list[str], optional
        Digraphs to split up in multilingual IPA mode
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
        silence_probability: float = 0.5,
        punctuation: List[str] = None,
        clitic_markers: List[str] = None,
        compound_markers: List[str] = None,
        multilingual_ipa: bool = False,
        strip_diacritics: List[str] = None,
        digraphs: List[str] = None,
        brackets: List[Tuple[str, str]] = None,
        non_silence_phones: Set[str] = None,
        disambiguation_symbols: Set[str] = None,
        clitic_set: Set[str] = None,
        max_disambiguation_symbol: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.strip_diacritics = DEFAULT_STRIP_DIACRITICS
        self.digraphs = DEFAULT_DIGRAPHS
        self.punctuation = DEFAULT_PUNCTUATION
        self.clitic_markers = DEFAULT_CLITIC_MARKERS
        self.compound_markers = DEFAULT_COMPOUND_MARKERS
        self.brackets = DEFAULT_BRACKETS
        if strip_diacritics is not None:
            self.strip_diacritics = strip_diacritics
        if digraphs is not None:
            self.digraphs = digraphs
        if punctuation is not None:
            self.punctuation = punctuation
        if clitic_markers is not None:
            self.clitic_markers = clitic_markers
        if compound_markers is not None:
            self.compound_markers = compound_markers
        if brackets is not None:
            self.brackets = brackets

        self.multilingual_ipa = multilingual_ipa
        self.num_silence_states = num_silence_states
        self.num_noise_states = num_noise_states
        self.num_non_silence_states = num_non_silence_states
        self.shared_silence_phones = shared_silence_phones
        self.silence_probability = silence_probability
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

    @property
    def base_phone_regex(self) -> Optional[str]:
        """Regex pattern for extracting a base phone for the phone set"""
        return None

    @property
    def phone_set_type(self) -> str:
        """Phone set type, defaults to 'UNKNOWN', currently only 'ARPA' is supported"""
        return "UNKNOWN"

    @property
    def extra_questions_mapping(self) -> Dict[str, List[str]]:
        mapping = {}
        mapping["silence_question"] = []
        for p in sorted(self.silence_phones):
            mapping["silence_question"].append(p)
            if self.position_dependent_phones:
                mapping["silence_question"].extend([p + x for x in self.positions])
        if self.phone_set_type == "ARPA":
            mapping["non_silence_arpa_questions"] = []
            for p in self.kaldi_grouped_phones.keys():
                if self.position_dependent_phones:
                    mapping["non_silence_arpa_questions"].extend([p + x for x in self.positions])
                else:
                    mapping["non_silence_arpa_questions"].append(p)
            # extra stress questions
            for i in range(3):
                mapping[f"stress_{i}"] = []
                for p in self.kaldi_non_silence_phones:
                    if str(i) not in p:
                        continue
                    mapping[f"stress_{i}"].append(p)
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
            "strip_diacritics": self.strip_diacritics,
            "digraphs": self.digraphs,
            "punctuation": self.punctuation,
            "clitic_markers": self.clitic_markers,
            "clitic_set": self.clitic_set,
            "compound_markers": self.compound_markers,
            "brackets": self.brackets,
            "multilingual_ipa": self.multilingual_ipa,
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

    @property
    def positional_non_silence_phones(self) -> List[str]:
        """
        List of non-silence phones with positions
        """
        non_silence_phones = []
        for p in sorted(self.non_silence_phones):
            if self.phone_set_type == "ARPA":
                m = re.match(self.base_phone_regex, p)
                if m:
                    base_phone = m.group(0)
                    for pos in self.positions:
                        pos_p = base_phone + pos
                        if pos_p not in non_silence_phones:
                            non_silence_phones.append(pos_p)
            for pos in self.positions:
                pos_p = p + pos
                if pos_p not in non_silence_phones:
                    non_silence_phones.append(pos_p)
        return non_silence_phones

    @property
    def kaldi_non_silence_phones(self):
        """Non silence phones in Kaldi format"""
        if self.position_dependent_phones:
            return self.positional_non_silence_phones
        base_phones = set()
        if self.phone_set_type == "ARPA":
            for p in self.non_silence_phones:
                m = re.match(self.base_phone_regex, p)
                if m:
                    base_phone = m.groups()[0]
                    base_phones.add(base_phone)

        return sorted(self.non_silence_phones | base_phones)

    @property
    def kaldi_grouped_phones(self) -> Dict[str, List[str]]:
        """Non silence phones in Kaldi format"""
        groups = {}
        for p in sorted(self.non_silence_phones):
            if self.phone_set_type == "ARPA":
                m = re.match(self.base_phone_regex, p)
                if m:
                    base_phone = m.groups()[0]
                    if base_phone not in groups:
                        groups[base_phone] = []
                        if self.position_dependent_phones:
                            groups[base_phone] = [base_phone + pos for pos in self.positions]
                        else:
                            groups[base_phone] = [base_phone]
                    if base_phone == p:
                        continue
                    if self.position_dependent_phones:
                        groups[base_phone].extend([p + pos for pos in self.positions])
                    else:
                        groups[base_phone].append(p)
            else:
                if self.position_dependent_phones:
                    groups[p] = [p + pos for pos in self.positions]
                else:
                    groups[p] = [p]

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

    def parse_ipa(self, transcription: List[str]) -> Tuple[str, ...]:
        """
        Parse a transcription in a multilingual IPA format (strips out diacritics and splits digraphs).

        Parameters
        ----------
        transcription: list[str]
            Transcription to parse

        Returns
        -------
        tuple[str, ...]
            Parsed transcription
        """
        new_transcription = []
        for t in transcription:
            new_t = t
            for d in self.strip_diacritics:
                new_t = new_t.replace(d, "")
            if "g" in new_t:
                new_t = new_t.replace("g", "ɡ")

            found = False
            for digraph in self.digraphs:
                if re.match(rf"^{digraph}$", new_t):
                    found = True
            if found:
                new_transcription.extend(new_t)
                continue
            new_transcription.append(new_t)
        return tuple(new_transcription)


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
        topo_template = "<State> {cur_state} <PdfClass> {cur_state} <Transition> {cur_state} 0.75 <Transition> {next_state} 0.25 </State>"
        topo_sil_template = "<State> {cur_state} <PdfClass> {cur_state} {transitions} </State>"
        topo_transition_template = "<Transition> {} {}"

        sil_transp = 1 / (self.num_silence_states - 1)
        initial_transition = [
            topo_transition_template.format(x, sil_transp)
            for x in range(self.num_silence_states - 1)
        ]
        middle_transition = [
            topo_transition_template.format(x, sil_transp)
            for x in range(1, self.num_silence_states)
        ]
        final_transition = [
            topo_transition_template.format(self.num_silence_states - 1, 0.75),
            topo_transition_template.format(self.num_silence_states, 0.25),
        ]
        states = []
        for i in range(self.num_non_silence_states):
            states.append(topo_template.format(cur_state=i, next_state=i + 1))
        states.append(f"<State> {self.num_non_silence_states} </State>")
        non_silence_state_string = "\n".join(states)

        states = []
        for i in range(self.num_silence_states):
            if i == 0:
                transition = " ".join(initial_transition)
            elif i == self.num_silence_states - 1:
                transition = " ".join(final_transition)
            else:
                transition = " ".join(middle_transition)
            states.append(topo_sil_template.format(cur_state=i, transitions=transition))
        states.append(f"<State> {self.num_silence_states} </State>")
        silence_state_string = "\n".join(states)

        with open(self.topo_path, "w") as f:
            f.write(
                f"""
            <Topology>
            <TopologyEntry>
            <ForPhones>
            {' '.join(str(self.phone_mapping[x]) for x in self.kaldi_silence_phones)}
            </ForPhones>
            {silence_state_string}
            </TopologyEntry>


            <TopologyEntry>
            <ForPhones>
            {' '.join(str(self.phone_mapping[x]) for x in self.kaldi_non_silence_phones)}
            </ForPhones>
            {non_silence_state_string}
            </TopologyEntry>
            </Topology>
            """
            )

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


@dataclass
class DictionaryData:
    """
    Information required for parsing Kaldi-internal ids to text

    Attributes
    ----------
    dictionary_options: dict[str, Any]
        Options for the dictionary
    sanitize_function: :class:`~montreal_forced_aligner.dictionary.mixins.SanitizeFunction`
        Function to sanitize text
    split_function: :class:`~montreal_forced_aligner.dictionary.mixins.SplitWordsFunction`
        Function to split words into subwords
    words_mapping: MappingType
        Mapping from words to their integer IDs
    reversed_words_mapping: ReversedMappingType
        Mapping from integer IDs to words
    words: WordsType
        Words and their associated pronunciations
    """

    dictionary_options: MetaDict
    sanitize_function: SanitizeFunction
    split_function: SplitWordsFunction
    words_mapping: MappingType
    reversed_words_mapping: ReversedMappingType
    words: WordsType
    lookup_cache: Dict[str, List[str]]

    @property
    def oov_word(self) -> str:
        """Out of vocabulary code"""
        return self.dictionary_options["oov_word"]

    @property
    def oov_phone(self) -> str:
        """Out of vocabulary code"""
        return self.dictionary_options["oov_phone"]

    @property
    def other_noise_phone(self) -> str:
        """Out of vocabulary code"""
        return self.dictionary_options["other_noise_phone"]

    @property
    def optional_silence_phone(self) -> str:
        """Out of vocabulary code"""
        return self.dictionary_options["optional_silence_phone"]

    @property
    def oov_int(self) -> int:
        """Out of vocabulary integer ID"""
        return self.words_mapping[self.oov_word]

    @property
    def compound_markers(self) -> List[str]:
        """Characters that separate compound words"""
        return self.dictionary_options["compound_markers"]

    @property
    def clitic_markers(self) -> List[str]:
        """Characters that mark clitics"""
        return self.dictionary_options["clitic_markers"]

    @property
    def clitic_set(self) -> Set[str]:
        """Set of clitics"""
        return self.dictionary_options["clitic_set"]

    @property
    def punctuation(self) -> List[str]:
        """Characters to treat as punctuation"""
        return self.dictionary_options["punctuation"]

    @property
    def strip_diacritics(self) -> List[str]:
        """IPA diacritics to strip in multilingual IPA mode"""
        return self.dictionary_options["strip_diacritics"]

    @property
    def multilingual_ipa(self) -> bool:
        """Flag for multilingual IPA mode"""
        return self.dictionary_options["multilingual_ipa"]

    @property
    def silence_phones(self) -> Set[str]:
        """Silence phones"""
        return {
            self.dictionary_options["optional_silence_phone"],
            self.dictionary_options["oov_phone"],
            self.dictionary_options["other_noise_phone"],
        }

    def lookup(
        self,
        item: str,
    ) -> List[str]:
        """
        Look up a word and return the list of sub words if necessary
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
        if item in self.lookup_cache:
            return self.lookup_cache[item]
        if item in self.words:
            return [item]
        sanitized = self.sanitize_function(item)
        if sanitized in self.words:
            self.lookup_cache[item] = [sanitized]
            return [sanitized]
        split = self.split_function(sanitized)
        oov_count = sum(1 for x in split if x not in self.words)
        if oov_count < len(
            split
        ):  # Only returned split item if it gains us any transcribed speech
            self.lookup_cache[item] = split
            return split
        self.lookup_cache[item] = [sanitized]
        return [sanitized]

    def to_int(
        self,
        item: str,
    ) -> List[int]:
        """
        Convert a given word into integer IDs

        Parameters
        ----------
        item: str
            Word to look up

        Returns
        -------
        list[int]
            List of integer IDs corresponding to each subword
        """
        if item == "":
            return []
        sanitized = self.lookup(item)
        text_int = []
        for item in sanitized:
            if not item:
                continue
            if item not in self.words_mapping:
                text_int.append(self.oov_int)
            else:
                text_int.append(self.words_mapping[item])
        return text_int

    def check_word(self, item: str) -> bool:
        """
        Check whether a word is in the dictionary, takes into account sanitization and
        clitic and compound markers

        Parameters
        ----------
        item: str
            Word to check

        Returns
        -------
        bool
            True if the look up would not result in an OOV item
        """
        if item == "":
            return False
        if item in self.words:
            return True
        sanitized = self.sanitize_function(item)
        if sanitized in self.words:
            return True

        sanitized = self.split_function(sanitized)
        if all(s in self.words for s in sanitized):
            return True
        return False

    def map_to_original_pronunciation(
        self, phones: List[CtmInterval], subpronunciations: List[DictionaryEntryType]
    ) -> List[CtmInterval]:
        """
        Convert phone transcriptions from multilingual IPA mode to their original IPA transcription

        Parameters
        ----------
        phones: list[CtmInterval]
            List of aligned phones
        subpronunciations: list[DictionaryEntryType]
            Pronunciations of each sub word to reconstruct the transcriptions

        Returns
        -------
        list[CtmInterval]
            Intervals with their original IPA pronunciation rather than the internal simplified form
        """
        transcription = tuple(x.label for x in phones)
        new_phones = []
        mapping_ind = 0
        transcription_ind = 0
        for pronunciations in subpronunciations:
            pron = None
            if mapping_ind >= len(phones):
                break
            for p in pronunciations:
                if (
                    "original_pronunciation" in p
                    and transcription == p["pronunciation"] == p["original_pronunciation"]
                ) or (transcription == p["pronunciation"] and "original_pronunciation" not in p):
                    new_phones.extend(phones)
                    mapping_ind += len(phones)
                    break
                if (
                    p["pronunciation"]
                    == transcription[
                        transcription_ind : transcription_ind + len(p["pronunciation"])
                    ]
                    and pron is None
                ):
                    pron = p
            if mapping_ind >= len(phones):
                break
            if not pron:
                new_phones.extend(phones)
                mapping_ind += len(phones)
                break
            to_extend = phones[transcription_ind : transcription_ind + len(pron["pronunciation"])]
            transcription_ind += len(pron["pronunciation"])
            p = pron
            if (
                "original_pronunciation" not in p
                or p["pronunciation"] == p["original_pronunciation"]
            ):
                new_phones.extend(to_extend)
                mapping_ind += len(to_extend)
                break
            for pi in p["original_pronunciation"]:
                if pi == phones[mapping_ind].label:
                    new_phones.append(phones[mapping_ind])
                else:
                    modded_phone = pi
                    new_p = phones[mapping_ind].label
                    for diacritic in self.strip_diacritics:
                        modded_phone = modded_phone.replace(diacritic, "")
                    if modded_phone == new_p:
                        phones[mapping_ind].label = pi
                        new_phones.append(phones[mapping_ind])
                    elif mapping_ind != len(phones) - 1:
                        new_p = phones[mapping_ind].label + phones[mapping_ind + 1].label
                        if modded_phone == new_p:
                            new_phones.append(
                                CtmInterval(
                                    phones[mapping_ind].begin,
                                    phones[mapping_ind + 1].end,
                                    new_p,
                                    phones[mapping_ind].utterance,
                                )
                            )
                            mapping_ind += 1
                mapping_ind += 1
        return new_phones
