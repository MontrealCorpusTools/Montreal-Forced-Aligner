"""Mixins for dictionary parsing capabilities"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..abc import MetaDict

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
        punctuation: list[str],
        clitic_markers: list[str],
        compound_markers: list[str],
        brackets: list[tuple[str, str]],
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
        punctuation: list[str],
        clitic_markers: list[str],
        compound_markers: list[str],
        brackets: list[tuple[str, str]],
        clitic_set: set[str],
        word_set: Optional[set[str]] = None,
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

    def split_clitics(
        self,
        item: str,
    ) -> list[str]:
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
        if any(x in item for x in self.compound_markers):
            s = re.split(rf"[{''.join(self.compound_markers)}]", item)
            if any(x in item for x in self.clitic_markers):
                new_s = []
                for seg in s:
                    if any(x in seg for x in self.clitic_markers):
                        new_s.extend(self.split_clitics(seg))
                    else:
                        new_s.append(seg)
                s = new_s
            return s
        if any(
            x in item and not item.endswith(x) and not item.startswith(x)
            for x in self.clitic_markers
        ):
            initial, final = re.split(rf"[{''.join(self.clitic_markers)}]", item, maxsplit=1)
            if any(x in final for x in self.clitic_markers):
                final = self.split_clitics(final)
            else:
                final = [final]
            for clitic in self.clitic_markers:
                if initial + clitic in self.clitic_set:
                    return [initial + clitic] + final
                elif clitic + final[0] in self.clitic_set:
                    final[0] = clitic + final[0]
                    return [initial] + final
        return [item]

    def __call__(
        self,
        item: str,
    ) -> list[str]:
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

    positions: list[str] = ["_B", "_E", "_I", "_S"]

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
        punctuation: list[str] = None,
        clitic_markers: list[str] = None,
        compound_markers: list[str] = None,
        multilingual_ipa: bool = False,
        strip_diacritics: list[str] = None,
        digraphs: list[str] = None,
        brackets: list[tuple[str, str]] = None,
        non_silence_phones: set[str] = None,
        disambiguation_symbols: set[str] = None,
        clitic_set: set[str] = None,
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
        self.num_non_silence_states = num_non_silence_states
        self.shared_silence_phones = shared_silence_phones
        self.silence_probability = silence_probability
        self.oov_word = oov_word
        self.silence_word = silence_word
        self.position_dependent_phones = position_dependent_phones
        self.optional_silence_phone = optional_silence_phone
        self.nonoptional_silence_phone = nonoptional_silence_phone
        self.oov_phone = oov_phone
        self.oovs_found = Counter()
        self.other_noise_phone = other_noise_phone
        if non_silence_phones is None:
            non_silence_phones = set()
        self.non_silence_phones = non_silence_phones
        self.max_disambiguation_symbol = max_disambiguation_symbol
        if disambiguation_symbols is None:
            disambiguation_symbols = set()
        self.disambiguation_symbols = disambiguation_symbols
        if clitic_set is None:
            clitic_set = set()
        self.clitic_set = clitic_set

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
            "nonoptional_silence_phone": self.nonoptional_silence_phone,
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
            self.oov_phone,
            self.optional_silence_phone,
            self.nonoptional_silence_phone,
            self.other_noise_phone,
        }

    @property
    def specials_set(self):
        """Special words, like the ``oov_word`` ``silence_word``, ``<eps>``, ``<s>``, and ``</s>``"""
        return {self.oov_word, self.silence_word, "<eps>", "<s>", "</s>"}

    @property
    def phone_mapping(self) -> dict[str, int]:
        """Mapping of phones to integer IDs"""
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
    def positional_silence_phones(self) -> list[str]:
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
    def positional_non_silence_phones(self) -> list[str]:
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
    def kaldi_non_silence_phones(self):
        """Non silence phones in Kaldi format"""
        if self.position_dependent_phones:
            return self.positional_non_silence_phones
        return sorted(self.non_silence_phones)

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

    def parse_ipa(self, transcription: list[str]) -> tuple[str, ...]:
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
