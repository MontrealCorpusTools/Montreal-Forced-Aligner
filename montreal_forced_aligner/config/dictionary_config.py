"""Class definitions for configuring pronunciation dictionaries"""
from __future__ import annotations

import re
from typing import Collection, Dict, List, Optional, Set, Tuple, Union

from .base_config import BaseConfig

DEFAULT_PUNCTUATION = list(r'、。।，@<>"(),.:;¿?¡!\\&%#*~【】，…‥「」『』〝〟″⟨⟩♪・‹›«»～′$+=‘')

DEFAULT_CLITIC_MARKERS = list("'’")
DEFAULT_COMPOUND_MARKERS = list("-/")
DEFAULT_STRIP_DIACRITICS = ["ː", "ˑ", "̩", "̆", "̑", "̯", "͡", "‿", "͜"]
DEFAULT_DIGRAPHS = ["[dt][szʒʃʐʑʂɕç]", "[aoɔe][ʊɪ]"]
DEFAULT_BRACKETS = [("[", "]"), ("{", "}"), ("<", ">"), ("(", ")")]

__all__ = ["DictionaryConfig"]


class DictionaryConfig(BaseConfig):
    """
    Class for storing configuration information about pronunciation dictionaries
        Path to a directory to store files for Kaldi
    oov_code : str, optional
        What to label words not in the dictionary, defaults to ``'<unk>'``
    position_dependent_phones : bool, optional
        Specifies whether phones should be represented as dependent on their
        position in the word (beginning, middle or end), defaults to True
    num_sil_states : int, optional
        Number of states to use for silence phones, defaults to 5
    num_nonsil_states : int, optional
        Number of states to use for non-silence phones, defaults to 3
    shared_silence_phones : bool, optional
        Specify whether to share states across all silence phones, defaults
        to True
    sil_prob : float, optional
        Probability of optional silences following words, defaults to 0.5
    word_set : Collection[str], optional
        Word set to limit output files
    debug: bool, optional
        Flag for whether to perform debug steps and prevent intermediate cleanup
    logger: :class:`~logging.Logger`, optional
        Logger to output information to
    punctuation: str, optional
        Punctuation to use when parsing text
    clitic_markers: str, optional
        Clitic markers to use when parsing text
    compound_markers: str, optional
        Compound markers to use when parsing text
    multilingual_ipa: bool, optional
        Flag for multilingual IPA mode, defaults to False
    strip_diacritics: List[str], optional
        Diacritics to strip in multilingual IPA mode
    digraphs: List[str], optional
        Digraphs to split up in multilingual IPA mode
    """

    topo_template = "<State> {cur_state} <PdfClass> {cur_state} <Transition> {cur_state} 0.75 <Transition> {next_state} 0.25 </State>"
    topo_sil_template = "<State> {cur_state} <PdfClass> {cur_state} {transitions} </State>"
    topo_transition_template = "<Transition> {} {}"
    positions: List[str] = ["_B", "_E", "_I", "_S"]

    def __init__(
        self,
        oov_word: str = "<unk>",
        silence_word: str = "!sil",
        nonoptional_silence_phone: str = "sil",
        optional_silence_phone: str = "sp",
        oov_phone: str = "spn",
        other_noise_phone: str = "spn",
        position_dependent_phones: bool = True,
        num_silence_states: int = 5,
        num_non_silence_states: int = 3,
        shared_silence_phones: bool = True,
        silence_probability: float = 0.5,
        debug: bool = False,
        punctuation: Optional[Union[str, Collection[str]]] = None,
        clitic_markers: Optional[Union[str, Collection[str]]] = None,
        compound_markers: Optional[Collection[str]] = None,
        multilingual_ipa: bool = False,
        strip_diacritics: Optional[Collection[str]] = None,
        digraphs: Optional[Collection[str]] = None,
        brackets: Optional[Collection[Tuple[str, str]]] = None,
    ):
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
        self.num_non_silence_states = num_non_silence_states
        self.shared_silence_phones = shared_silence_phones
        self.silence_probability = silence_probability
        self.oov_word = oov_word
        self.silence_word = silence_word
        self.position_dependent_phones = position_dependent_phones
        self.optional_silence_phone = optional_silence_phone
        self.nonoptional_silence_phone = nonoptional_silence_phone
        self.oov_phone = oov_phone
        self.other_noise_phone = other_noise_phone
        self.debug = debug
        self.non_silence_phones: Set[str] = set()
        self.max_disambiguation_symbol = 0
        self.disambiguation_symbols = set()
        self.clitic_set: Set[str] = set()

    @property
    def silence_phones(self):
        return {
            self.oov_phone,
            self.optional_silence_phone,
            self.nonoptional_silence_phone,
            self.other_noise_phone,
        }

    @property
    def specials_set(self):
        return {self.oov_word, self.silence_word, "<eps>", "<s>", "</s>"}

    def update(self, data: dict) -> None:
        for k, v in data.items():
            if not hasattr(self, k):
                continue
            if k == "phones":
                continue
            if k in ["punctuation", "clitic_markers", "compound_markers"]:
                if not v:
                    continue
                if "-" in v:
                    v = "-" + v.replace("-", "")
                if "]" in v and r"\]" not in v:
                    v = v.replace("]", r"\]")
            print(k, v)
            setattr(self, k, v)

    @property
    def phone_mapping(self) -> Dict[str, int]:
        phone_mapping = {}
        i = 0
        phone_mapping["<eps>"] = i
        if self.position_dependent_phones:
            for p in self.positional_silence_phones:
                i += 1
                phone_mapping[p] = i
            for p in self.positional_non_silence_phones:
                i += 1
                phone_mapping[p] = i
        else:
            for p in sorted(self.silence_phones):
                i += 1
                phone_mapping[p] = i
            for p in sorted(self.non_silence_phones):
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
            for pos in self.positions:
                non_silence_phones.append(p + pos)
        return non_silence_phones

    @property
    def kaldi_silence_phones(self):
        if self.position_dependent_phones:
            return self.positional_silence_phones
        return sorted(self.silence_phones)

    @property
    def kaldi_non_silence_phones(self):
        if self.position_dependent_phones:
            return self.positional_non_silence_phones
        return sorted(self.non_silence_phones)

    @property
    def optional_silence_csl(self) -> str:
        """
        Phone id of the optional silence phone
        """
        return str(self.phone_mapping[self.optional_silence_phone])

    @property
    def silence_csl(self) -> str:
        """
        A colon-separated list (as a string) of silence phone ids
        """
        return ":".join(map(str, (self.phone_mapping[x] for x in self.kaldi_silence_phones)))

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
            if word.startswith(b[0]) and word.endswith(b[-1]):
                return True
        return False

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
        for c in self.clitic_markers:
            item = item.replace(c, self.clitic_markers[0])
        if not item:
            return item
        if self.check_bracketed(item):
            return item
        sanitized = re.sub(rf"^[{''.join(self.punctuation)}]+", "", item)
        sanitized = re.sub(rf"[{''.join(self.punctuation)}]+$", "", sanitized)

        return sanitized

    def parse_ipa(self, transcription: List[str]) -> Tuple[str, ...]:
        """
        Parse a transcription in a multilingual IPA format (strips out diacritics and splits digraphs).

        Parameters
        ----------
        transcription: List[str]
            Transcription to parse

        Returns
        -------
        Tuple[str, ...]
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
