"""Pronunciation dictionaries for use in alignment and transcription"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    IpaType = Optional[List[str]]
    PunctuationType = Optional[str]
    from logging import Logger
    from .corpus.classes import Speaker

import logging
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

import yaml

from .config.base_config import (
    DEFAULT_CLITIC_MARKERS,
    DEFAULT_COMPOUND_MARKERS,
    DEFAULT_DIGRAPHS,
    DEFAULT_PUNCTUATION,
    DEFAULT_STRIP_DIACRITICS,
)
from .exceptions import DictionaryError, DictionaryFileError, DictionaryPathError
from .utils import get_available_dictionaries, get_dictionary_path, thirdparty_binary

DictionaryEntryType = List[Dict[str, Union[Tuple[str], float, None, int]]]
ReversedMappingType = Dict[int, str]
WordsType = Dict[str, DictionaryEntryType]
MappingType = Dict[str, int]
MultiSpeakerMappingType = Dict[str, str]

__all__ = [
    "compile_graphemes",
    "sanitize",
    "check_format",
    "check_bracketed",
    "parse_ipa",
    "DictionaryData",
    "Dictionary",
    "MultispeakerDictionary",
]


def compile_graphemes(graphemes: Set[str]) -> re.Pattern:
    """
    Compiles the list of graphemes into a regular expression pattern.

    Parameters
    ----------
    graphemes: Set[str]
        Set of characters to treat as orthographic text

    Returns
    -------
    re.Pattern
        Compiled pattern that matches all graphemes
    """
    base = r"^\W*([{}]+)\W*"
    string = re.escape("".join(graphemes))
    try:
        return re.compile(base.format(string))
    except Exception:
        print(graphemes)
        raise


def check_bracketed(word: str, brackets: Optional[List[Tuple[str, str]]] = None) -> bool:
    """
    Checks whether a given string is surrounded by brackets.

    Parameters
    ----------
    word : str
        Text to check for final brackets
    brackets: List[Tuple[str, str]]], optional
        Brackets to check, defaults to [('[', ']'), ('{', '}'), ('<', '>'), ('(', ')')]

    Returns
    -------
    bool
        True if the word is fully bracketed, false otherwise
    """
    if brackets is None:
        brackets = [("[", "]"), ("{", "}"), ("<", ">"), ("(", ")")]
    for b in brackets:
        if word.startswith(b[0]) and word.endswith(b[-1]):
            return True
    return False


def sanitize(
    item: str, punctuation: Optional[str] = None, clitic_markers: Optional[str] = None
) -> str:
    """
    Sanitize an item according to punctuation and clitic markers

    Parameters
    ----------
    item: str
        Word to sanitize
    punctuation: str
        Characters to treat as punctuation
    clitic_markers: str
        Characters to treat as clitic markers, will be collapsed to the first marker

    Returns
    -------
    str
        Sanitized form
    """
    if punctuation is None:
        punctuation = DEFAULT_PUNCTUATION
    if clitic_markers is None:
        clitic_markers = DEFAULT_CLITIC_MARKERS
    for c in clitic_markers:
        item = item.replace(c, clitic_markers[0])
    if not item:
        return item
    if check_bracketed(item):
        return item
    sanitized = re.sub(rf"^[{punctuation}]+", "", item)
    sanitized = re.sub(rf"[{punctuation}]+$", "", sanitized)

    return sanitized


def check_format(path: str) -> Tuple[bool, bool]:
    """
    Check the pronunciation dictionary format

    Parameters
    ----------
    path: str
        Path of pronunciation dictionary

    Returns
    -------
    bool
        Flag for whether the dictionary has pronunciation probabilities
    bool
        Flag for whether the dictionary includes silence probabilities
    """
    count = 0
    pronunciation_probabilities = True
    silence_probabilities = True
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.split()
            _ = line.pop(0)  # word
            next_item = line.pop(0)
            if pronunciation_probabilities:
                try:
                    prob = float(next_item)
                    if prob > 1 or prob < 0:
                        raise ValueError
                except ValueError:
                    pronunciation_probabilities = False
            try:
                next_item = line.pop(0)
            except IndexError:
                silence_probabilities = False
            if silence_probabilities:
                try:
                    prob = float(next_item)
                    if prob > 1 or prob < 0:
                        raise ValueError
                except ValueError:
                    silence_probabilities = False
            count += 1
            if count > 10:
                break
    return pronunciation_probabilities, silence_probabilities


def parse_ipa(
    transcription: List[str], strip_diacritics: IpaType = None, digraphs: IpaType = None
) -> Tuple[str, ...]:
    """
    Parse a transcription in a multilingual IPA format (strips out diacritics and splits digraphs).

    Parameters
    ----------
    transcription: List[str]
        Transcription to parse
    strip_diacritics: List[str]
        List of diacritics to remove from characters in the transcription
    digraphs: List[str]
        List of digraphs to split up into separate characters

    Returns
    -------
    Tuple[str, ...]
        Parsed transcription
    """
    if strip_diacritics is None:
        strip_diacritics = DEFAULT_STRIP_DIACRITICS
    if digraphs is None:
        digraphs = DEFAULT_DIGRAPHS
    new_transcription = []
    for t in transcription:
        new_t = t
        for d in strip_diacritics:
            new_t = new_t.replace(d, "")
        if "g" in new_t:
            new_t = new_t.replace("g", "ɡ")

        found = False
        for digraph in digraphs:
            if re.match(r"^{}$".format(digraph), new_t):
                found = True
        if found:
            new_transcription.extend(new_t)
            continue
        new_transcription.append(new_t)
    return tuple(new_transcription)


class DictionaryData(NamedTuple):
    """
    Information required for parsing Kaldi-internal ids to text
    """

    silences: Set[str]
    multilingual_ipa: bool
    words_mapping: MappingType
    reversed_words_mapping: ReversedMappingType
    reversed_phone_mapping: ReversedMappingType
    punctuation: PunctuationType
    clitic_set: Set[str]
    clitic_markers: PunctuationType
    compound_markers: PunctuationType
    strip_diacritics: IpaType
    oov_int: int
    oov_code: str
    words: WordsType


class Dictionary(object):
    """
    Class containing information about a pronunciation dictionary

    Parameters
    ----------
    input_path : str
        Path to an input pronunciation dictionary
    output_directory : str
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
    has_multiple = False

    def __init__(
        self,
        input_path: str,
        output_directory: str,
        oov_code: str = "<unk>",
        position_dependent_phones: bool = True,
        num_sil_states: int = 5,
        num_nonsil_states: int = 3,
        shared_silence_phones: bool = True,
        sil_prob: float = 0.5,
        word_set: Optional[Collection[str]] = None,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        punctuation: PunctuationType = None,
        clitic_markers: PunctuationType = None,
        compound_markers: PunctuationType = None,
        multilingual_ipa: bool = False,
        strip_diacritics: IpaType = None,
        digraphs: IpaType = None,
    ):
        self.multilingual_ipa = multilingual_ipa
        self.strip_diacritics = DEFAULT_STRIP_DIACRITICS
        self.digraphs = DEFAULT_DIGRAPHS
        if strip_diacritics is not None:
            self.strip_diacritics = strip_diacritics
        if digraphs is not None:
            self.digraphs = digraphs
        self.punctuation = DEFAULT_PUNCTUATION
        self.clitic_markers = DEFAULT_CLITIC_MARKERS
        self.compound_markers = DEFAULT_COMPOUND_MARKERS
        if punctuation is not None:
            self.punctuation = punctuation
        if clitic_markers is not None:
            self.clitic_markers = clitic_markers
        if compound_markers is not None:
            self.compound_markers = compound_markers

        if not os.path.exists(input_path):
            raise (DictionaryPathError(input_path))
        if not os.path.isfile(input_path):
            raise (DictionaryFileError(input_path))
        self.input_path = input_path
        self.name = os.path.splitext(os.path.basename(input_path))[0]
        self.debug = debug
        self.output_directory = os.path.join(output_directory, self.name)
        os.makedirs(self.output_directory, exist_ok=True)
        self.log_file = os.path.join(self.output_directory, f"{self.name}.log")
        if logger is None:
            self.logger = logging.getLogger("dictionary_setup")
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_file, "w", "utf-8")
            handler.setFormatter = logging.Formatter("%(name)s %(message)s")
            self.logger.addHandler(handler)
            self.individual_logger = True
        else:
            self.logger = logger
            self.individual_logger = False
        self.num_sil_states = num_sil_states
        self.num_nonsil_states = num_nonsil_states
        self.shared_silence_phones = shared_silence_phones
        self.sil_prob = sil_prob
        self.oov_code = oov_code
        self.sil_code = "!sil"
        self.oovs_found = Counter()
        self.position_dependent_phones = position_dependent_phones

        self.words = {}
        self.nonsil_phones: Set[str] = set()
        self.sil_phones = {"sp", "spn", "sil"}
        self.optional_silence = "sp"
        self.nonoptional_silence = "sil"
        self.graphemes = set()
        self.max_disambiguation_symbol = 0
        self.disambiguation_symbols = set()
        self.all_words = defaultdict(list)
        self.clitic_set = set()
        self.specials_set = {self.oov_code, self.sil_code, "<eps>", "<s>", "</s>"}
        self.words[self.sil_code] = [{"pronunciation": ("sp",), "probability": 1}]
        self.words[self.oov_code] = [{"pronunciation": ("spn",), "probability": 1}]
        self.pronunciation_probabilities, self.silence_probabilities = check_format(input_path)
        progress = f'Parsing dictionary "{self.name}"'
        if self.pronunciation_probabilities:
            progress += " with pronunciation probabilities"
        else:
            progress += " without pronunciation probabilities"
        if self.silence_probabilities:
            progress += " with silence probabilities"
        else:
            progress += " without silence probabilities"
        self.logger.info(progress)
        with open(input_path, "r", encoding="utf8") as inf:
            for i, line in enumerate(inf):
                line = line.strip()
                if not line:
                    continue
                line = line.split()
                word = sanitize(line.pop(0).lower(), self.punctuation, self.clitic_markers)
                if not line:
                    raise DictionaryError(
                        f"Line {i} of {input_path} does not have a pronunciation."
                    )
                if word in ["!sil", oov_code]:
                    continue
                self.graphemes.update(word)
                prob = None
                if self.pronunciation_probabilities:
                    prob = float(line.pop(0))
                    if prob > 1 or prob < 0:
                        raise ValueError
                if self.silence_probabilities:
                    right_sil_prob = float(line.pop(0))
                    left_sil_prob = float(line.pop(0))
                    left_nonsil_prob = float(line.pop(0))
                else:
                    right_sil_prob = None
                    left_sil_prob = None
                    left_nonsil_prob = None
                if self.multilingual_ipa:
                    pron = parse_ipa(line, self.strip_diacritics, self.digraphs)
                else:
                    pron = tuple(line)
                pronunciation = {
                    "pronunciation": pron,
                    "probability": prob,
                    "disambiguation": None,
                    "right_sil_prob": right_sil_prob,
                    "left_sil_prob": left_sil_prob,
                    "left_nonsil_prob": left_nonsil_prob,
                }
                if self.multilingual_ipa:
                    pronunciation["original_pronunciation"] = tuple(line)
                if not any(x in self.sil_phones for x in pron):
                    self.nonsil_phones.update(pron)
                if word in self.words and pron in {x["pronunciation"] for x in self.words[word]}:
                    continue
                if word not in self.words:
                    self.words[word] = []
                self.words[word].append(pronunciation)
                # test whether a word is a clitic
                is_clitic = False
                for cm in self.clitic_markers:
                    if word.startswith(cm) or word.endswith(cm):
                        is_clitic = True
                if is_clitic:
                    self.clitic_set.add(word)
        if word_set is not None:
            word_set = {y for x in word_set for y in self._lookup(x)}
            word_set.add("!sil")
            word_set.add(self.oov_code)
        self.word_set = word_set
        if self.word_set is not None:
            self.word_set = self.word_set | self.clitic_set
        if not self.graphemes:
            raise DictionaryFileError(f"No words were found in the dictionary path {input_path}")
        self.word_pattern = compile_graphemes(self.graphemes)
        self.log_info()
        self.phone_mapping = {}
        self.words_mapping = {}

    def __hash__(self) -> Any:
        """Return the hash of a given dictionary"""
        return hash(self.input_path)

    @property
    def output_paths(self) -> Dict[str, str]:
        """
        Mapping of output directory for this dictionary
        """
        return {self.name: self.output_directory}

    @property
    def silences(self) -> Set[str]:
        """
        Set of symbols that correspond to silence
        """
        return {self.optional_silence, self.nonoptional_silence}

    def get_dictionary(self, speaker: Union[Speaker, str]) -> Dictionary:
        """
        Wrapper function to return this dictionary for any arbitrary speaker

        Parameters
        ----------
        speaker: Union[Speaker, str]
            Speaker to look up dictionary for

        Returns
        -------
        Dictionary
            This dictionary
        """
        return self

    def data(self, word_set: Optional[Collection[str]] = None) -> DictionaryData:
        """
        Generates a dictionary data for use in parsing utilities

        Parameters
        ----------
        word_set: Collection[str], optional
            Word set to limit data to

        Returns
        -------
        DictionaryData
            Data necessary for parsing text
        """

        def word_check(word):
            """Check whether a word should be included in the output"""
            if word in word_set:
                return True
            if word in self.clitic_set:
                return True
            if word in self.specials_set:
                return True
            return False

        if word_set:
            words_mapping = {k: v for k, v in self.words_mapping.items() if word_check(k)}
            reversed_word_mapping = {
                k: v for k, v in self.reversed_word_mapping.items() if word_check(v)
            }
            words = {k: v for k, v in self.words.items() if word_check(k)}
        else:
            words_mapping = self.words_mapping
            reversed_word_mapping = self.reversed_word_mapping
            words = self.words
        return DictionaryData(
            self.silences,
            self.multilingual_ipa,
            words_mapping,
            reversed_word_mapping,
            self.reversed_phone_mapping,
            self.punctuation,
            self.clitic_set,
            self.clitic_markers,
            self.compound_markers,
            self.strip_diacritics,
            self.oov_int,
            self.oov_code,
            words,
        )

    def cleanup_logger(self) -> None:
        """
        Clean up and detach logger from handles
        """
        if not self.individual_logger:
            return
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def log_info(self) -> None:
        """
        Dump debugging information to the logger
        """
        self.logger.debug(f'"{self.name}" DICTIONARY INFORMATION')
        if self.pronunciation_probabilities:
            self.logger.debug("Has pronunciation probabilities")
        else:
            self.logger.debug("Has NO pronunciation probabilities")
        if self.silence_probabilities:
            self.logger.debug("Has silence probabilities")
        else:
            self.logger.debug("Has NO silence probabilities")

        self.logger.debug(f"Grapheme set: {', '.join(sorted(self.graphemes))}")
        self.logger.debug(f"Phone set: {', '.join(sorted(self.nonsil_phones))}")
        self.logger.debug(f"Punctuation: {self.punctuation}")
        self.logger.debug(f"Clitic markers: {self.clitic_markers}")
        self.logger.debug(f"Clitic set: {', '.join(sorted(self.clitic_set))}")
        if self.multilingual_ipa:
            self.logger.debug(f"Strip diacritics: {', '.join(sorted(self.strip_diacritics))}")
            self.logger.debug(f"Digraphs: {', '.join(sorted(self.digraphs))}")

    def set_word_set(self, word_set: List[str]) -> None:
        """
        Limit output to a subset of overall words

        Parameters
        ----------
        word_set: List[str]
            Word set to limit generated files to
        """
        word_set = {y for x in word_set for y in self._lookup(x)}
        word_set.add(self.sil_code)
        word_set.add(self.oov_code)
        self.word_set = word_set | self.clitic_set
        self.generate_mappings()

    @property
    def actual_words(self) -> Dict[str, DictionaryEntryType]:
        """
        Mapping of words to integer IDs without Kaldi-internal words
        """
        return {
            k: v
            for k, v in self.words.items()
            if k not in [self.sil_code, self.oov_code, "<eps>"] and len(v)
        }

    def split_clitics(self, item: str) -> List[str]:
        """
        Split a word into subwords based on clitic and compound markers

        Parameters
        ----------
        item: str
            Word to split up

        Returns
        -------
        List[str]
            List of subwords
        """
        if item in self.words:
            return [item]
        if any(x in item for x in self.compound_markers):
            s = re.split(rf"[{self.compound_markers}]", item)
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
            initial, final = re.split(rf"[{self.clitic_markers}]", item, maxsplit=1)
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

    def __len__(self) -> int:
        """Return the number of pronunciations across all words"""
        return sum(len(x) for x in self.words.values())

    def exclude_for_alignment(self, word: str) -> bool:
        """
        Check for whether to exclude a word from alignment lexicons (if there is a word set in the dictionary,
        checks whether the given string is in the word set)

        Parameters
        ----------
        word: str
            Word to check

        Returns
        -------
        bool
            True if there is no word set on the dictionary, or if the word is in the given word set
        """
        if self.word_set is None:
            return False
        if word not in self.word_set and word not in self.clitic_set:
            return True
        return False

    def generate_mappings(self) -> None:
        """
        Generate phone and word mappings from text to integer IDs
        """
        if self.phone_mapping:
            return
        self.phone_mapping = {}
        i = 0
        self.phone_mapping["<eps>"] = i
        if self.position_dependent_phones:
            for p in self.positional_sil_phones:
                i += 1
                self.phone_mapping[p] = i
            for p in self.positional_nonsil_phones:
                i += 1
                self.phone_mapping[p] = i
        else:
            for p in sorted(self.sil_phones):
                i += 1
                self.phone_mapping[p] = i
            for p in sorted(self.nonsil_phones):
                i += 1
                self.phone_mapping[p] = i

        self.words_mapping = {}
        i = 0
        self.words_mapping["<eps>"] = i
        for w in sorted(self.words.keys()):
            if self.exclude_for_alignment(w):
                continue
            i += 1
            self.words_mapping[w] = i

        self.words_mapping["#0"] = i + 1
        self.words_mapping["<s>"] = i + 2
        self.words_mapping["</s>"] = i + 3
        self.oovs_found = Counter()
        self.add_disambiguation()

    def add_disambiguation(self) -> None:
        """
        Calculate disambiguation symbols for each pronunciation
        """
        subsequences = set()
        pronunciation_counts = defaultdict(int)

        for w, prons in self.words.items():
            if self.exclude_for_alignment(w):
                continue
            for p in prons:
                pronunciation_counts[p["pronunciation"]] += 1
                pron = p["pronunciation"][:-1]
                while pron:
                    subsequences.add(tuple(p))
                    pron = pron[:-1]
        last_used = defaultdict(int)
        for w, prons in sorted(self.words.items()):
            if self.exclude_for_alignment(w):
                continue
            for p in prons:
                if (
                    pronunciation_counts[p["pronunciation"]] == 1
                    and not p["pronunciation"] in subsequences
                ):
                    disambig = None
                else:
                    pron = p["pronunciation"]
                    last_used[pron] += 1
                    disambig = last_used[pron]
                p["disambiguation"] = disambig
        if last_used:
            self.max_disambiguation_symbol = max(last_used.values())
        else:
            self.max_disambiguation_symbol = 0
        self.disambiguation_symbols = set()
        i = max(self.phone_mapping.values())
        for x in range(self.max_disambiguation_symbol + 2):
            p = f"#{x}"
            self.disambiguation_symbols.add(p)
            i += 1
            self.phone_mapping[p] = i

    def create_utterance_fst(self, text: List[str], frequent_words: List[Tuple[str, int]]) -> str:
        """
        Create an FST for an utterance with frequent words as a unigram language model

        Parameters
        ----------
        text: List[str]
            Text of the utterance
        frequent_words: List[Tuple[str, int]]
            Frequent words to incorporate into the FST
        Returns
        -------
        str
            FST created from the utterance text and frequent words
        """
        num_words = len(text)
        word_probs = Counter(text)
        word_probs = {k: v / num_words for k, v in word_probs.items()}
        word_probs.update(frequent_words)
        fst_text = ""
        for k, v in word_probs.items():
            cost = -1 * math.log(v)
            w = self.to_int(k)[0]
            fst_text += f"0 0 {w} {w} {cost}\n"
        fst_text += f"0 {-1 * math.log(1 / num_words)}\n"
        return fst_text

    def to_int(self, item: str) -> List[int]:
        """
        Convert a given word into integer IDs

        Parameters
        ----------
        item: str
            Word to look up

        Returns
        -------
        List[int]
            List of integer IDs corresponding to each subword
        """
        if item == "":
            return []
        sanitized = self._lookup(item)
        text_int = []
        for item in sanitized:
            if not item:
                continue
            if item not in self.words_mapping:
                self.oovs_found.update([item])
                text_int.append(self.oov_int)
            else:
                text_int.append(self.words_mapping[item])
        return text_int

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

    def _lookup(self, item: str) -> List[str]:
        """
        Look up a word and return the list of sub words if necessary taking into account clitic and compound markers

        Parameters
        ----------
        item: str
            Word to look up

        Returns
        -------
        List[str]
            List of subwords that are in the dictionary
        """
        if item in self.words:
            return [item]
        sanitized = sanitize(item, self.punctuation, self.clitic_markers)
        if sanitized in self.words:
            return [sanitized]
        split = self.split_clitics(sanitized)
        oov_count = sum(1 for x in split if x not in self.words)
        if oov_count < len(
            split
        ):  # Only returned split item if it gains us any transcribed speech
            return split
        return [sanitized]

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
        sanitized = sanitize(item, self.punctuation, self.clitic_markers)
        if sanitized in self.words:
            return True

        sanitized = self.split_clitics(sanitized)
        if all(s in self.words for s in sanitized):
            return True
        return False

    @property
    def reversed_word_mapping(self) -> ReversedMappingType:
        """
        A mapping of integer ids to words
        """
        mapping = {}
        for k, v in self.words_mapping.items():
            mapping[v] = k
        return mapping

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
    def oov_int(self) -> int:
        """
        The integer id for out of vocabulary items
        """
        return self.words_mapping[self.oov_code]

    @property
    def positional_sil_phones(self) -> List[str]:
        """
        List of silence phones with positions
        """
        sil_phones = []
        for p in sorted(self.sil_phones):
            sil_phones.append(p)
            for pos in self.positions:
                sil_phones.append(p + pos)
        return sil_phones

    @property
    def positional_nonsil_phones(self) -> List[str]:
        """
        List of non-silence phones with positions
        """
        nonsil_phones = []
        for p in sorted(self.nonsil_phones):
            for pos in self.positions:
                nonsil_phones.append(p + pos)
        return nonsil_phones

    @property
    def optional_silence_csl(self) -> str:
        """
        Phone id of the optional silence phone
        """
        return str(self.phone_mapping[self.optional_silence])

    @property
    def silence_csl(self) -> str:
        """
        A colon-separated list (as a string) of silence phone ids
        """
        if self.position_dependent_phones:
            return ":".join(map(str, (self.phone_mapping[x] for x in self.positional_sil_phones)))
        else:
            return ":".join(map(str, (self.phone_mapping[x] for x in self.sil_phones)))

    @property
    def phones_dir(self) -> str:
        """
        Directory to store information Kaldi needs about phones
        """
        return os.path.join(self.output_directory, "phones")

    @property
    def phones(self) -> set:
        """
        The set of all phones (silence and non-silence)
        """
        return self.sil_phones | self.nonsil_phones

    @property
    def words_symbol_path(self) -> str:
        """
        Path of word to int mapping file for the dictionary
        """
        return os.path.join(self.output_directory, "words.txt")

    @property
    def disambig_path(self) -> str:
        """
        Path of disambiguated lexicon fst (L.fst)
        """
        return os.path.join(self.output_directory, "L_disambig.fst")

    def write(self, write_disambiguation: Optional[bool] = False) -> None:
        """
        Write the files necessary for Kaldi

        Parameters
        ----------
        write_disambiguation: bool, optional
            Flag for including disambiguation information
        """
        self.logger.info("Creating dictionary information...")
        os.makedirs(self.phones_dir, exist_ok=True)
        self.generate_mappings()
        self._write_graphemes()
        self._write_phone_map_file()
        self._write_phone_sets()
        self._write_phone_symbol_table()
        self._write_disambig()
        self._write_topo()
        self._write_word_boundaries()
        self._write_extra_questions()
        self._write_word_file()
        self._write_align_lexicon()
        self._write_fst_text(write_disambiguation=write_disambiguation)
        self._write_fst_binary(write_disambiguation=write_disambiguation)
        self.cleanup()

    def cleanup(self) -> None:
        """
        Clean up temporary files in the output directory
        """
        if not self.debug:
            if os.path.exists(os.path.join(self.output_directory, "temp.fst")):
                os.remove(os.path.join(self.output_directory, "temp.fst"))
            if os.path.exists(os.path.join(self.output_directory, "lexicon.text.fst")):
                os.remove(os.path.join(self.output_directory, "lexicon.text.fst"))

    def _write_graphemes(self) -> None:
        """
        Write graphemes to temporary directory
        """
        outfile = os.path.join(self.output_directory, "graphemes.txt")
        if os.path.exists(outfile):
            return
        with open(outfile, "w", encoding="utf8") as f:
            for char in sorted(self.graphemes):
                f.write(char + "\n")

    def export_lexicon(
        self,
        path: str,
        write_disambiguation: Optional[bool] = False,
        probability: Optional[bool] = False,
    ) -> None:
        """
        Export pronunciation dictionary to a text file

        Parameters
        ----------
        path: str
            Path to save dictionary
        write_disambiguation: bool, optional
            Flag for whether to include disambiguation information
        probability: bool, optional
            Flag for whether to include probabilities
        """
        with open(path, "w", encoding="utf8") as f:
            for w in sorted(self.words.keys()):
                for p in sorted(
                    self.words[w],
                    key=lambda x: (x["pronunciation"], x["probability"], x["disambiguation"]),
                ):
                    phones = " ".join(p["pronunciation"])
                    if write_disambiguation and p["disambiguation"] is not None:
                        phones += f" #{p['disambiguation']}"
                    if probability:
                        f.write(f"{w}\t{p['probability']}\t{phones}\n")
                    else:
                        f.write(f"{w}\t{phones}\n")

    def _write_phone_map_file(self) -> None:
        """
        Write the phone map to the temporary directory
        """
        outfile = os.path.join(self.output_directory, "phone_map.txt")
        if os.path.exists(outfile):
            return
        with open(outfile, "w", encoding="utf8") as f:
            for sp in self.sil_phones:
                if self.position_dependent_phones:
                    new_phones = [sp + x for x in ["", ""] + self.positions]
                else:
                    new_phones = [sp]
                f.write(" ".join(new_phones) + "\n")
            for nsp in self.nonsil_phones:
                if self.position_dependent_phones:
                    new_phones = [nsp + x for x in [""] + self.positions]
                else:
                    new_phones = [nsp]
                f.write(" ".join(new_phones) + "\n")

    def _write_phone_symbol_table(self) -> None:
        """
        Write the phone mapping to the temporary directory
        """
        outfile = os.path.join(self.output_directory, "phones.txt")
        if os.path.exists(outfile):
            return
        with open(outfile, "w", encoding="utf8") as f:
            for p, i in sorted(self.phone_mapping.items(), key=lambda x: x[1]):
                f.write(f"{p} {i}\n")

    def _write_word_boundaries(self) -> None:
        """
        Write the word boundaries file to the temporary directory
        """
        boundary_path = os.path.join(self.output_directory, "phones", "word_boundary.txt")
        boundary_int_path = os.path.join(self.output_directory, "phones", "word_boundary.int")
        if os.path.exists(boundary_path) and os.path.exists(boundary_int_path):
            return
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

    def _write_word_file(self) -> None:
        """
        Write the word mapping to the temporary directory
        """
        words_path = os.path.join(self.output_directory, "words.txt")
        if os.path.exists(words_path):
            return
        if sys.platform == "win32":
            newline = ""
        else:
            newline = None
        with open(words_path, "w", encoding="utf8", newline=newline) as f:
            for w, i in sorted(self.words_mapping.items(), key=lambda x: x[1]):
                f.write(f"{w} {i}\n")

    def _write_align_lexicon(self) -> None:
        """
        Write the alignment lexicon text file to the temporary directory
        """
        path = os.path.join(self.phones_dir, "align_lexicon.int")
        if os.path.exists(path):
            return

        with open(path, "w", encoding="utf8") as f:
            for w, i in self.words_mapping.items():
                if self.exclude_for_alignment(w):
                    continue
                if w not in self.words:  # special characters
                    continue
                for pron in sorted(
                    self.words[w],
                    key=lambda x: (x["pronunciation"], x["probability"], x["disambiguation"]),
                ):

                    phones = list(pron["pronunciation"])
                    if self.position_dependent_phones:
                        if len(phones) == 1:
                            phones[0] += "_S"
                        else:
                            for j in range(len(phones)):
                                if j == 0:
                                    phones[j] += "_B"
                                elif j == len(phones) - 1:
                                    phones[j] += "_E"
                                else:
                                    phones[j] += "_I"
                    p = " ".join(str(self.phone_mapping[x]) for x in phones)
                    f.write(f"{i} {i} {p}\n".format(i=i, p=p))

    def _write_topo(self) -> None:
        """
        Write the topo file to the temporary directory
        """
        filepath = os.path.join(self.output_directory, "topo")
        if os.path.exists(filepath):
            return
        sil_transp = 1 / (self.num_sil_states - 1)
        initial_transition = [
            self.topo_transition_template.format(x, sil_transp)
            for x in range(self.num_sil_states - 1)
        ]
        middle_transition = [
            self.topo_transition_template.format(x, sil_transp)
            for x in range(1, self.num_sil_states)
        ]
        final_transition = [
            self.topo_transition_template.format(self.num_sil_states - 1, 0.75),
            self.topo_transition_template.format(self.num_sil_states, 0.25),
        ]
        with open(filepath, "w") as f:
            f.write("<Topology>\n")
            f.write("<TopologyEntry>\n")
            f.write("<ForPhones>\n")
            if self.position_dependent_phones:
                phones = self.positional_nonsil_phones
            else:
                phones = sorted(self.nonsil_phones)
            f.write(f"{' '.join(str(self.phone_mapping[x]) for x in phones)}\n")
            f.write("</ForPhones>\n")
            states = [
                self.topo_template.format(cur_state=x, next_state=x + 1)
                for x in range(self.num_nonsil_states)
            ]
            f.write("\n".join(states))
            f.write(f"\n<State> {self.num_nonsil_states} </State>\n")
            f.write("</TopologyEntry>\n")

            f.write("<TopologyEntry>\n")
            f.write("<ForPhones>\n")
            if self.position_dependent_phones:
                phones = self.positional_sil_phones
            else:
                phones = self.sil_phones
            f.write(f"{' '.join(str(self.phone_mapping[x]) for x in phones)}\n")
            f.write("</ForPhones>\n")
            states = []
            for i in range(self.num_sil_states):
                if i == 0:
                    transition = " ".join(initial_transition)
                elif i == self.num_sil_states - 1:
                    transition = " ".join(final_transition)
                else:
                    transition = " ".join(middle_transition)
                states.append(self.topo_sil_template.format(cur_state=i, transitions=transition))
            f.write("\n".join(states))
            f.write(f"\n<State> {self.num_sil_states} </State>\n")
            f.write("</TopologyEntry>\n")
            f.write("</Topology>\n")

    def _write_phone_sets(self) -> None:
        """
        Write phone symbol sets to the temporary directory
        """
        sharesplit = ["shared", "split"]
        if not self.shared_silence_phones:
            sil_sharesplit = ["not-shared", "not-split"]
        else:
            sil_sharesplit = sharesplit

        sets_file = os.path.join(self.output_directory, "phones", "sets.txt")
        roots_file = os.path.join(self.output_directory, "phones", "roots.txt")

        sets_int_file = os.path.join(self.output_directory, "phones", "sets.int")
        roots_int_file = os.path.join(self.output_directory, "phones", "roots.int")
        if (
            os.path.exists(sets_file)
            and os.path.exists(roots_file)
            and os.path.exists(sets_int_file)
            and os.path.exists(roots_int_file)
        ):
            return

        with open(sets_file, "w", encoding="utf8") as setf, open(
            roots_file, "w", encoding="utf8"
        ) as rootf, open(sets_int_file, "w", encoding="utf8") as setintf, open(
            roots_int_file, "w", encoding="utf8"
        ) as rootintf:

            # process silence phones
            for i, sp in enumerate(self.sil_phones):
                if self.position_dependent_phones:
                    mapped = [sp + x for x in [""] + self.positions]
                else:
                    mapped = [sp]
                setf.write(" ".join(mapped) + "\n")
                setintf.write(" ".join(map(str, (self.phone_mapping[x] for x in mapped))) + "\n")
                if i == 0:
                    line = sil_sharesplit + mapped
                    lineint = sil_sharesplit + [self.phone_mapping[x] for x in mapped]
                else:
                    line = sharesplit + mapped
                    lineint = sharesplit + [self.phone_mapping[x] for x in mapped]
                rootf.write(" ".join(line) + "\n")
                rootintf.write(" ".join(map(str, lineint)) + "\n")

            # process nonsilence phones
            for nsp in sorted(self.nonsil_phones):
                if self.position_dependent_phones:
                    mapped = [nsp + x for x in self.positions]
                else:
                    mapped = [nsp]
                setf.write(" ".join(mapped) + "\n")
                setintf.write(" ".join(map(str, (self.phone_mapping[x] for x in mapped))) + "\n")
                line = sharesplit + mapped
                lineint = sharesplit + [self.phone_mapping[x] for x in mapped]
                rootf.write(" ".join(line) + "\n")
                rootintf.write(" ".join(map(str, lineint)) + "\n")

    def _write_extra_questions(self) -> None:
        """
        Write extra questions symbols to the temporary directory
        """
        phone_extra = os.path.join(self.phones_dir, "extra_questions.txt")
        phone_extra_int = os.path.join(self.phones_dir, "extra_questions.int")
        if os.path.exists(phone_extra) and os.path.exists(phone_extra_int):
            return
        with open(phone_extra, "w", encoding="utf8") as outf, open(
            phone_extra_int, "w", encoding="utf8"
        ) as intf:
            if self.position_dependent_phones:
                sils = sorted(self.positional_sil_phones)
            else:
                sils = sorted(self.sil_phones)
            outf.write(" ".join(sils) + "\n")
            intf.write(" ".join(map(str, (self.phone_mapping[x] for x in sils))) + "\n")

            if self.position_dependent_phones:
                nonsils = sorted(self.positional_nonsil_phones)
            else:
                nonsils = sorted(self.nonsil_phones)
            outf.write(" ".join(nonsils) + "\n")
            intf.write(" ".join(map(str, (self.phone_mapping[x] for x in nonsils))) + "\n")
            if self.position_dependent_phones:
                for p in self.positions:
                    line = [x + p for x in sorted(self.nonsil_phones)]
                    outf.write(" ".join(line) + "\n")
                    intf.write(" ".join(map(str, (self.phone_mapping[x] for x in line))) + "\n")
                for p in [""] + self.positions:
                    line = [x + p for x in sorted(self.sil_phones)]
                    outf.write(" ".join(line) + "\n")
                    intf.write(" ".join(map(str, (self.phone_mapping[x] for x in line))) + "\n")

    def _write_disambig(self) -> None:
        """
        Write disambiguation symbols to the temporary directory
        """
        disambig = os.path.join(self.phones_dir, "disambiguation_symbols.txt")
        disambig_int = os.path.join(self.phones_dir, "disambiguation_symbols.int")
        if os.path.exists(disambig) and os.path.exists(disambig_int):
            return
        with open(disambig, "w", encoding="utf8") as outf, open(
            disambig_int, "w", encoding="utf8"
        ) as intf:
            for d in sorted(self.disambiguation_symbols, key=lambda x: self.phone_mapping[x]):
                outf.write(f"{d}\n")
                intf.write(f"{self.phone_mapping[d]}\n")

    def _write_fst_binary(self, write_disambiguation: Optional[bool] = False) -> None:
        """
        Write the binary fst file to the temporary directory

        Parameters
        ----------
        write_disambiguation: bool, optional
            Flag for including disambiguation symbols
        """
        if write_disambiguation:
            lexicon_fst_path = os.path.join(self.output_directory, "lexicon_disambig.text.fst")
            output_fst = os.path.join(self.output_directory, "L_disambig.fst")
        else:
            lexicon_fst_path = os.path.join(self.output_directory, "lexicon.text.fst")
            output_fst = os.path.join(self.output_directory, "L.fst")
        if os.path.exists(output_fst):
            return

        phones_file_path = os.path.join(self.output_directory, "phones.txt")
        words_file_path = os.path.join(self.output_directory, "words.txt")

        log_path = os.path.join(self.output_directory, "fst.log")
        temp_fst_path = os.path.join(self.output_directory, "temp.fst")
        with open(log_path, "w") as log_file:
            compile_proc = subprocess.Popen(
                [
                    thirdparty_binary("fstcompile"),
                    f"--isymbols={phones_file_path}",
                    f"--osymbols={words_file_path}",
                    "--keep_isymbols=false",
                    "--keep_osymbols=false",
                    lexicon_fst_path,
                    temp_fst_path,
                ],
                stderr=log_file,
            )
            compile_proc.communicate()
            if write_disambiguation:
                temp2_fst_path = os.path.join(self.output_directory, "temp2.fst")
                phone_disambig_path = os.path.join(self.output_directory, "phone_disambig.txt")
                word_disambig_path = os.path.join(self.output_directory, "word_disambig.txt")
                with open(phone_disambig_path, "w") as f:
                    f.write(str(self.phone_mapping["#0"]))
                with open(word_disambig_path, "w") as f:
                    f.write(str(self.words_mapping["#0"]))
                selfloop_proc = subprocess.Popen(
                    [
                        thirdparty_binary("fstaddselfloops"),
                        phone_disambig_path,
                        word_disambig_path,
                        temp_fst_path,
                        temp2_fst_path,
                    ],
                    stderr=log_file,
                )
                selfloop_proc.communicate()
                arc_sort_proc = subprocess.Popen(
                    [
                        thirdparty_binary("fstarcsort"),
                        "--sort_type=olabel",
                        temp2_fst_path,
                        output_fst,
                    ],
                    stderr=log_file,
                )
            else:
                arc_sort_proc = subprocess.Popen(
                    [
                        thirdparty_binary("fstarcsort"),
                        "--sort_type=olabel",
                        temp_fst_path,
                        output_fst,
                    ],
                    stderr=log_file,
                )
            arc_sort_proc.communicate()

    def _write_fst_text(self, write_disambiguation: Optional[bool] = False) -> None:
        """
        Write the text fst file to the temporary directory

        Parameters
        ----------
        write_disambiguation: bool, optional
            Flag for including disambiguation symbols
        """
        if write_disambiguation:
            lexicon_fst_path = os.path.join(self.output_directory, "lexicon_disambig.text.fst")
            sildisambig = f"#{self.max_disambiguation_symbol + 1}"
        else:
            lexicon_fst_path = os.path.join(self.output_directory, "lexicon.text.fst")
        if os.path.exists(lexicon_fst_path):
            return
        if self.sil_prob != 0:
            silphone = self.optional_silence
            nonoptsil = self.nonoptional_silence

            silcost = -1 * math.log(self.sil_prob)
            nosilcost = -1 * math.log(1.0 - self.sil_prob)
            startstate = 0
            loopstate = 1
            silstate = 2
        else:
            loopstate = 0
            nextstate = 1

        with open(lexicon_fst_path, "w", encoding="utf8") as outf:
            if self.sil_prob != 0:
                outf.write(
                    "\t".join(map(str, [startstate, loopstate, "<eps>", "<eps>", nosilcost]))
                    + "\n"
                )  # no silence

                outf.write(
                    "\t".join(map(str, [startstate, loopstate, nonoptsil, "<eps>", silcost]))
                    + "\n"
                )  # silence
                outf.write(
                    "\t".join(map(str, [silstate, loopstate, silphone, "<eps>"])) + "\n"
                )  # no cost
                nextstate = 3
                if write_disambiguation:
                    disambigstate = 3
                    nextstate = 4
                    outf.write(
                        "\t".join(
                            map(str, [startstate, disambigstate, silphone, "<eps>", silcost])
                        )
                        + "\n"
                    )  # silence.
                    outf.write(
                        "\t".join(map(str, [silstate, disambigstate, silphone, "<eps>", silcost]))
                        + "\n"
                    )  # no cost.
                    outf.write(
                        "\t".join(map(str, [disambigstate, loopstate, sildisambig, "<eps>"]))
                        + "\n"
                    )  # silence disambiguation symbol.

            for w in sorted(self.words.keys()):
                if self.exclude_for_alignment(w):
                    continue
                for pron in sorted(
                    self.words[w],
                    key=lambda x: (x["pronunciation"], x["probability"], x["disambiguation"]),
                ):
                    phones = list(pron["pronunciation"])
                    prob = pron["probability"]
                    disambig_symbol = pron["disambiguation"]
                    if self.position_dependent_phones:
                        if len(phones) == 1:
                            phones[0] += "_S"
                        else:
                            for i in range(len(phones)):
                                if i == 0:
                                    phones[i] += "_B"
                                elif i == len(phones) - 1:
                                    phones[i] += "_E"
                                else:
                                    phones[i] += "_I"
                    if not self.pronunciation_probabilities:
                        pron_cost = 0
                    else:
                        if prob is None:
                            prob = 1.0
                        elif not prob:
                            prob = 0.001  # Dithering to ensure low probability entries
                        pron_cost = -1 * math.log(prob)

                    pron_cost_string = ""
                    if pron_cost != 0:
                        pron_cost_string = f"\t{pron_cost}"

                    s = loopstate
                    word_or_eps = w
                    local_nosilcost = nosilcost + pron_cost
                    local_silcost = silcost + pron_cost
                    while len(phones) > 0:
                        p = phones.pop(0)
                        if len(phones) > 0 or (
                            write_disambiguation and disambig_symbol is not None
                        ):
                            ns = nextstate
                            nextstate += 1
                            outf.write(
                                "\t".join(map(str, [s, ns, p, word_or_eps]))
                                + pron_cost_string
                                + "\n"
                            )
                            word_or_eps = "<eps>"
                            pron_cost_string = ""
                            s = ns
                        elif self.sil_prob == 0:
                            ns = loopstate
                            outf.write(
                                "\t".join(map(str, [s, ns, p, word_or_eps]))
                                + pron_cost_string
                                + "\n"
                            )
                            word_or_eps = "<eps>"
                            pron_cost_string = ""
                            s = ns
                        else:
                            outf.write(
                                "\t".join(
                                    map(str, [s, loopstate, p, word_or_eps, local_nosilcost])
                                )
                                + "\n"
                            )
                            outf.write(
                                "\t".join(map(str, [s, silstate, p, word_or_eps, local_silcost]))
                                + "\n"
                            )
                    if write_disambiguation and disambig_symbol is not None:
                        outf.write(
                            "\t".join(
                                map(
                                    str,
                                    [
                                        s,
                                        loopstate,
                                        f"#{disambig_symbol}",
                                        word_or_eps,
                                        local_nosilcost,
                                    ],
                                )
                            )
                            + "\n"
                        )
                        outf.write(
                            "\t".join(
                                map(
                                    str,
                                    [
                                        s,
                                        silstate,
                                        f"#{disambig_symbol}",
                                        word_or_eps,
                                        local_silcost,
                                    ],
                                )
                            )
                            + "\n"
                        )

            outf.write(f"{loopstate}\t0\n")


class MultispeakerDictionary(Dictionary):
    """
    Class containing information about a pronunciation dictionary with different dictionaries per speaker

    Parameters
    ----------
    input_path : str
        Path to an input pronunciation dictionary
    output_directory : str
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

    has_multiple = True

    def __init__(
        self,
        input_path: str,
        output_directory: str,
        oov_code: Optional[str] = "<unk>",
        position_dependent_phones: Optional[bool] = True,
        num_sil_states: Optional[int] = 5,
        num_nonsil_states: Optional[int] = 3,
        shared_silence_phones: Optional[bool] = True,
        sil_prob: Optional[float] = 0.5,
        word_set: Optional[List[str]] = None,
        debug: Optional[bool] = False,
        logger: Optional[Logger] = None,
        punctuation: PunctuationType = None,
        clitic_markers: PunctuationType = None,
        compound_markers: PunctuationType = None,
        multilingual_ipa: Optional[bool] = False,
        strip_diacritics: IpaType = None,
        digraphs: IpaType = None,
    ):
        self.multilingual_ipa = multilingual_ipa
        self.strip_diacritics = DEFAULT_STRIP_DIACRITICS
        self.digraphs = DEFAULT_DIGRAPHS
        if strip_diacritics is not None:
            self.strip_diacritics = strip_diacritics
        if digraphs is not None:
            self.digraphs = digraphs
        self.punctuation = DEFAULT_PUNCTUATION
        self.clitic_markers = DEFAULT_CLITIC_MARKERS
        self.compound_markers = DEFAULT_COMPOUND_MARKERS
        if punctuation is not None:
            self.punctuation = punctuation
        if clitic_markers is not None:
            self.clitic_markers = clitic_markers
        if compound_markers is not None:
            self.compound_markers = compound_markers
        self.input_path = input_path
        self.debug = debug
        self.output_directory = os.path.join(output_directory, "dictionary")
        os.makedirs(self.output_directory, exist_ok=True)
        self.log_file = os.path.join(self.output_directory, "dictionary.log")
        if logger is None:
            self.logger = logging.getLogger("dictionary_setup")
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_file, "w", "utf-8")
            handler.setFormatter = logging.Formatter("%(name)s %(message)s")
            self.logger.addHandler(handler)
        else:
            self.logger = logger
        self.num_sil_states = num_sil_states
        self.num_nonsil_states = num_nonsil_states
        self.shared_silence_phones = shared_silence_phones
        self.sil_prob = sil_prob
        self.oov_code = oov_code
        self.sil_code = "!sil"
        self.oovs_found = Counter()
        self.position_dependent_phones = position_dependent_phones
        self.max_disambiguation_symbol = 0
        self.disambiguation_symbols = set()
        self.optional_silence = "sp"
        self.nonoptional_silence = "sil"

        if word_set is not None:
            word_set = {sanitize(x, self.punctuation, self.clitic_markers) for x in word_set}
            word_set.add("!sil")
            word_set.add(self.oov_code)
        self.word_set = word_set

        if not os.path.exists(input_path):
            raise (DictionaryPathError(input_path))
        if not os.path.isfile(input_path):
            raise (DictionaryFileError(input_path))

        self.speaker_mapping = {}
        self.dictionary_mapping = {}
        self.logger.info("Parsing multispeaker dictionary file")
        available_langs = get_available_dictionaries()
        with open(input_path, "r", encoding="utf8") as f:
            data = yaml.safe_load(f)
            for speaker, path in data.items():
                if path in available_langs:
                    path = get_dictionary_path(path)
                dictionary_name = os.path.splitext(os.path.basename(path))[0]
                self.speaker_mapping[speaker] = dictionary_name
                if dictionary_name not in self.dictionary_mapping:
                    self.dictionary_mapping[dictionary_name] = Dictionary(
                        path,
                        output_directory,
                        oov_code=self.oov_code,
                        position_dependent_phones=self.position_dependent_phones,
                        word_set=self.word_set,
                        num_sil_states=self.num_sil_states,
                        num_nonsil_states=self.num_nonsil_states,
                        shared_silence_phones=self.shared_silence_phones,
                        sil_prob=self.sil_prob,
                        debug=self.debug,
                        logger=self.logger,
                        punctuation=self.punctuation,
                        clitic_markers=self.clitic_markers,
                        compound_markers=self.compound_markers,
                        multilingual_ipa=self.multilingual_ipa,
                        strip_diacritics=self.strip_diacritics,
                        digraphs=self.digraphs,
                    )

        self.nonsil_phones = set()
        self.sil_phones = {"sp", "spn", "sil"}
        self.words = set()
        self.clitic_set = set()
        for d in self.dictionary_mapping.values():
            self.nonsil_phones.update(d.nonsil_phones)
            self.sil_phones.update(d.sil_phones)
            self.words.update(d.words)
            self.clitic_set.update(d.clitic_set)
        self.words_mapping = {}
        self.phone_mapping = {}

    @property
    def silences(self) -> set:
        """
        Set of silence phones
        """
        return {self.optional_silence, self.nonoptional_silence}

    def get_dictionary_name(self, speaker: Union[str, Speaker]) -> str:
        """
        Get the dictionary name for a given speaker

        Parameters
        ----------
        speaker: Union[Speaker, str]
            Speaker to look up

        Returns
        -------
        str
            Dictionary name for the speaker
        """
        if not isinstance(speaker, str):
            speaker = speaker.name
        if speaker not in self.speaker_mapping:
            return self.speaker_mapping["default"]
        return self.speaker_mapping[speaker]

    def get_dictionary(self, speaker: Union[Speaker, str]) -> Dictionary:
        """
        Get a dictionary for a given speaker

        Parameters
        ----------
        speaker: Union[Speaker, str]
            Speaker to look up

        Returns
        -------
        Dictionary
            Dictionary for the speaker
        """
        return self.dictionary_mapping[self.get_dictionary_name(speaker)]

    def generate_mappings(self) -> None:
        """
        Generate phone and word mappings from text to integer IDs
        """
        self.phone_mapping = {}
        i = 0
        self.phone_mapping["<eps>"] = i
        if self.position_dependent_phones:
            for p in self.positional_sil_phones:
                i += 1
                self.phone_mapping[p] = i
            for p in self.positional_nonsil_phones:
                i += 1
                self.phone_mapping[p] = i
        else:
            for p in sorted(self.sil_phones):
                i += 1
                self.phone_mapping[p] = i
            for p in sorted(self.nonsil_phones):
                i += 1
                self.phone_mapping[p] = i

        self.words_mapping = {}
        i = 0
        self.words_mapping["<eps>"] = i
        for w in sorted(self.words):
            if self.exclude_for_alignment(w):
                continue
            i += 1
            self.words_mapping[w] = i

        self.words_mapping["#0"] = i + 1
        self.words_mapping["<s>"] = i + 2
        self.words_mapping["</s>"] = i + 3
        self.words.update(["<eps>", "#0", "<s>", "</s>"])
        self.oovs_found = Counter()
        self.max_disambiguation_symbol = 0
        for d in self.dictionary_mapping.values():
            d.generate_mappings()
            if d.max_disambiguation_symbol > self.max_disambiguation_symbol:
                self.max_disambiguation_symbol = d.max_disambiguation_symbol
        i = max(self.phone_mapping.values())
        self.disambiguation_symbols = set()
        for x in range(self.max_disambiguation_symbol + 2):
            p = f"#{x}"
            self.disambiguation_symbols.add(p)
            i += 1
            self.phone_mapping[p] = i

    def write(self, write_disambiguation: Optional[bool] = False) -> None:
        """
        Write all child dictionaries to the temporary directory

        Parameters
        ----------
        write_disambiguation: bool, optional
            Flag to use disambiguation symbols in the output
        """
        os.makedirs(self.phones_dir, exist_ok=True)
        self.generate_mappings()
        for d in self.dictionary_mapping.values():
            d.phone_mapping = self.phone_mapping
            d.write(write_disambiguation)

    @property
    def output_paths(self) -> Dict[str, str]:
        """
        Mapping of output directory for child dictionaries
        """
        return {d.name: d.output_directory for d in self.dictionary_mapping.values()}


if TYPE_CHECKING:
    DictionaryType = Union[MultispeakerDictionary, Dictionary]
