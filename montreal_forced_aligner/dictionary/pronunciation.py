"""Pronunciation dictionaries for use in alignment and transcription"""

from __future__ import annotations

import collections
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Collection, Dict, List, Optional, Set

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import (
        ReversedMappingType,
    )

from montreal_forced_aligner.data import Pronunciation, WordData
from montreal_forced_aligner.dictionary.mixins import SplitWordsFunction, TemporaryDictionaryMixin
from montreal_forced_aligner.exceptions import (
    DictionaryError,
    DictionaryFileError,
    KaldiProcessingError,
)
from montreal_forced_aligner.models import DictionaryModel
from montreal_forced_aligner.utils import thirdparty_binary

__all__ = ["PronunciationDictionaryMixin", "PronunciationDictionary"]


class PronunciationDictionaryMixin(TemporaryDictionaryMixin):
    """
    Abstract mixin class containing information about a pronunciation dictionary

    Parameters
    ----------
    dictionary_path : str
        Path to pronunciation dictionary
    root_dictionary : :class:`~montreal_forced_aligner.dictionary.mixins.TemporaryDictionaryMixin`, optional
        Optional root dictionary to take phone information from

    See Also
    --------
    :class:`~montreal_forced_aligner.dictionary.mixins.DictionaryMixin`
        For dictionary parsing parameters
    :class:`~montreal_forced_aligner.abc.TemporaryDirectoryMixin`
        For temporary directory parameters

    Attributes
    ----------
    dictionary_model: DictionaryModel
        Dictionary model to load
    words: WordsType
        Words mapped to their pronunciations
    graphemes: set[str]
        Set of graphemes in the dictionary
    words_mapping: MappingType
        Mapping of words to integer IDs
    lexicon_word_set: set[str]
        Word set to limit output of lexicon files
    """

    def __init__(self, dictionary_path, root_dictionary=None, **kwargs):
        super().__init__(**kwargs)
        self.dictionary_model = DictionaryModel(
            dictionary_path, phone_set_type=self.phone_set_type
        )
        self.name = self.dictionary_model.name

        self.phone_set_type = self.dictionary_model.phone_set_type
        self.root_dictionary = root_dictionary
        pretrained = False
        if self.non_silence_phones:
            pretrained = True
        os.makedirs(self.dictionary_output_directory, exist_ok=True)
        self.words = {}
        self.graphemes = set()
        eps_word = WordData(self.silence_word, {Pronunciation((self.optional_silence_phone,))})
        self.words[self.silence_word] = eps_word
        self.phone_counts = collections.Counter()
        sanitize = False
        self.silence_words = set()
        clitic_cleanup_regex = None
        if len(self.clitic_markers) > 1:
            sanitize = True
            clitic_cleanup_regex = re.compile(rf'[{"".join(self.clitic_markers[1:])}]')
        with open(self.dictionary_model.path, "r", encoding="utf8") as inf:
            for i, line in enumerate(inf):
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    word, line = line.split("\t", maxsplit=1)
                    line = line.strip().split()
                else:
                    line = line.split()
                    word = line.pop(0)
                if self.ignore_case:
                    word = word.lower()
                if " " in word:
                    continue
                if sanitize:
                    word = clitic_cleanup_regex.sub(self.clitic_markers[0], word)
                if not line:
                    raise DictionaryError(
                        f"Line {i} of {self.dictionary_model.path} does not have a pronunciation."
                    )
                if word in self.specials_set:
                    continue
                self.graphemes.update(word)
                prob = 1
                if self.dictionary_model.pronunciation_probabilities:
                    prob = float(line.pop(0))
                    if prob > 1 or prob < 0:
                        raise ValueError
                silence_after_prob = None
                silence_before_correct = None
                non_silence_before_correct = None
                if self.dictionary_model.silence_probabilities:
                    silence_after_prob = float(line.pop(0))
                    silence_before_correct = float(line.pop(0))
                    non_silence_before_correct = float(line.pop(0))
                pron = tuple(line)
                if pretrained:
                    difference = set(pron) - self.non_silence_phones - self.silence_phones
                    if difference:
                        self.excluded_phones.update(difference)
                        self.excluded_pronunciation_count += 1
                        continue
                pronunciation = Pronunciation(
                    pronunciation=pron,
                    probability=prob,
                    silence_after_probability=silence_after_prob,
                    silence_before_correction=silence_before_correct,
                    non_silence_before_correction=non_silence_before_correct,
                )

                if word in self.words and pronunciation in self.words[word]:
                    continue
                if not set(pron) - self.silence_phones:
                    self.silence_words.add(word)
                else:
                    self.non_silence_phones.update(pron)
                    self.phone_counts.update(pron)
                if word not in self.words:
                    self.words[word] = WordData(word, set())

                self.words[word].add_pronunciation(pronunciation)
                # test whether a word is a clitic
                if not self.clitic_markers or self.clitic_markers[0] not in word:
                    continue
                if word.startswith(self.clitic_markers[0]) or word.endswith(
                    self.clitic_markers[0]
                ):
                    self.clitic_set.add(word)
        if self.oov_word not in self.words:
            oov_word = WordData(self.oov_word, {Pronunciation(pronunciation=(self.oov_phone,))})
            self.words[self.oov_word] = oov_word
        for w in ["laugh", "laughter", "lachen", "lg"]:
            for initial_bracket, final_bracket in self.brackets:
                orth = f"{initial_bracket}{w}{final_bracket}"
                if orth not in self.words:
                    self.words[orth] = WordData(
                        orth, {Pronunciation(pronunciation=(self.oov_phone,))}
                    )
                self.silence_words.add(orth)
        self.bracketed_word = "[bracketed]"
        if self.bracketed_word not in self.words:
            self.words[self.bracketed_word] = WordData(
                self.bracketed_word, {Pronunciation(pronunciation=(self.oov_phone,))}
            )
        self.silence_words.add(self.bracketed_word)
        self.lexicon_word_set = set(self.words.keys())
        self.non_silence_phones -= self.silence_phones

        self.bracket_regex = None
        if self.brackets:
            left_brackets = [x[0] for x in self.brackets]
            right_brackets = [x[1] for x in self.brackets]
            self.bracket_regex = re.compile(
                rf"[{re.escape(''.join(left_brackets))}].*?[{re.escape(''.join(right_brackets))}]+"
            )

        self.words_mapping = {}
        self._to_int_cache = {}
        if not self.graphemes:
            raise DictionaryFileError(
                f"No words were found in the dictionary path {self.dictionary_model.path}"
            )

    def __hash__(self) -> Any:
        """Return the hash of a given dictionary"""
        return hash(self.name)

    @property
    def dictionary_output_directory(self) -> str:
        """Temporary directory to store all dictionary information"""
        return os.path.join(self.temporary_directory, self.name)

    @property
    def silences(self) -> Set[str]:
        """
        Set of symbols that correspond to silence
        """
        return self.silence_phones

    @property
    def actual_words(self) -> Dict[str, WordData]:
        """Words in the dictionary stripping out Kaldi's internal words"""
        return {
            k: v
            for k, v in self.words.items()
            if k not in self.specials_set and k not in self.silence_words
        }

    def construct_split_words_function(self) -> SplitWordsFunction:
        """
        Construct a :class:`~montreal_forced_aligner.dictionary.mixins.SplitWordsFunction` to use in multiprocessing jobs

        Returns
        -------
        :class:`~montreal_forced_aligner.dictionary.mixins.SplitWordsFunction`
            Function for sanitizing text
        """
        f = SplitWordsFunction(
            self.clitic_markers,
            self.compound_markers,
            self.clitic_set,
            self.brackets,
            set(self.words.keys()),
        )

        return f

    def set_lexicon_word_set(self, word_set: Collection[str]) -> None:
        """
        Limit lexicon output to a subset of overall words

        Parameters
        ----------
        word_set: Collection[str]
            Word set to limit generated files to
        """
        self.lexicon_word_set = {self.silence_word, self.oov_word}
        self.lexicon_word_set.update(self.clitic_set)
        self.lexicon_word_set.update(word_set)

        self.generate_mappings()

    def split_clitics(self, item: str) -> List[str]:
        """
        Split a word into subwords based on clitic and compound markers

        Parameters
        ----------
        item: str
            Word to split up

        Returns
        -------
        list[str]
            List of subwords
        """

        return self.construct_split_words_function()(item)

    def __bool__(self):
        """Check that the dictionary contains words"""
        return bool(self.words)

    def __len__(self) -> int:
        """Return the number of pronunciations across all words"""
        return sum(len(x.pronunciations) for x in self.words.values())

    def generate_mappings(self) -> None:
        """
        Generate word mappings from text to integer IDs
        """
        if self.words_mapping:
            return
        self.words_mapping = {}
        i = 0
        self.words_mapping[self.silence_word] = i
        for w in sorted(self.words.keys()):
            # if self.exclude_for_alignment(w):
            #    continue
            if w == self.silence_word:
                continue
            i += 1
            self.words_mapping[w] = i

        self.words_mapping["#0"] = i + 1
        self.words_mapping["<s>"] = i + 2
        self.words_mapping["</s>"] = i + 3
        self.add_disambiguation()

    def add_disambiguation(self) -> None:
        """
        Calculate disambiguation symbols for each pronunciation
        """
        subsequences = set()
        pronunciation_counts = defaultdict(int)

        for word in self.words.values():
            for p in word:
                pronunciation_counts[p.pronunciation] += 1
                pron = p.pronunciation[:-1]
                while pron:
                    subsequences.add(tuple(p.pronunciation))
                    pron = pron[:-1]
        last_used = defaultdict(int)
        for _, prons in sorted(self.words.items()):
            for p in prons:
                if (
                    pronunciation_counts[p.pronunciation] == 1
                    and p.pronunciation not in subsequences
                ):
                    disambig = None
                else:
                    pron = p.pronunciation
                    last_used[pron] += 1
                    disambig = last_used[pron]
                p.disambiguation = disambig
        if last_used:
            self.max_disambiguation_symbol = max(
                self.max_disambiguation_symbol, max(last_used.values())
            )

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
        if item in self.words:
            return [item]
        split = self.construct_split_words_function()(item)
        oov_count = sum(1 for x in split if x not in self.words)
        if oov_count < len(
            split
        ):  # Only returned split item if it gains us any transcribed speech
            return split
        return [item]

    def to_int(self, item: str, normalized=False) -> List[int]:
        """
        Convert a given word into integer IDs

        Parameters
        ----------
        item: str
            Word to look up
        normalized: bool
            Flag for whether to use pre-existing normalized text

        Returns
        -------
        list[int]
            List of integer IDs corresponding to each subword
        """
        if item == "":
            return []
        if normalized:
            if item in self.words_mapping and item not in self.specials_set:
                return [self.words_mapping[item]]
            elif self.bracket_regex and self.bracket_regex.match(item):
                return [self.bracketed_int]
            else:
                return [self.oov_int]
        if item in self._to_int_cache:
            return self._to_int_cache[item]
        sanitized = self.lookup(item)
        text_int = []
        for item in sanitized:
            if not item:
                continue
            if item in self.words_mapping and item not in self.specials_set:
                text_int.append(self.words_mapping[item])
            elif self.bracket_regex and self.bracket_regex.match(item):
                text_int.append(self.bracketed_int)
            else:
                text_int.append(self.oov_int)
        self._to_int_cache[item] = text_int
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

        sanitized = self.construct_split_words_function()(item)
        for s in sanitized:
            if s not in self.words:
                return False
        return True

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
    def phone_mapping(self) -> Dict[str, int]:
        """Mapping of phones to integer IDs"""
        if self.root_dictionary is not None:
            return self.root_dictionary.phone_mapping
        return super().phone_mapping

    @property
    def silence_disambiguation_symbol(self) -> str:
        """Mapping of phones to integer IDs"""
        if self.root_dictionary is not None:
            return self.root_dictionary.silence_disambiguation_symbol
        return super().silence_disambiguation_symbol

    @property
    def oov_int(self) -> int:
        """
        The integer id for out of vocabulary items
        """
        return self.words_mapping[self.oov_word]

    @property
    def bracketed_int(self) -> int:
        """
        The integer id for bracketed items
        """
        return self.words_mapping[self.bracketed_word]

    @property
    def phones_dir(self) -> str:
        """
        Directory to store information Kaldi needs about phones
        """
        if self.root_dictionary is not None:
            return self.root_dictionary.phones_dir
        return super().phones_dir

    @property
    def phone_symbol_table_path(self) -> str:
        """Path to file containing phone symbols and their integer IDs"""
        if self.root_dictionary is not None:
            return self.root_dictionary.phone_symbol_table_path
        return super().phone_symbol_table_path

    @property
    def words_symbol_path(self) -> str:
        """
        Path of word to int mapping file for the dictionary
        """
        return os.path.join(self.dictionary_output_directory, "words.txt")

    @property
    def lexicon_fst_path(self) -> str:
        """
        Path of disambiguated lexicon fst (L.fst)
        """
        return os.path.join(self.dictionary_output_directory, "L.fst")

    @property
    def lexicon_disambig_fst_path(self) -> str:
        """
        Path of disambiguated lexicon fst (L.fst)
        """
        return os.path.join(self.dictionary_output_directory, "L.disambig_fst")

    def write(
        self,
        write_disambiguation: bool = False,
        debug=False,
    ) -> None:
        """
        Write the files necessary for Kaldi

        Parameters
        ----------
        write_disambiguation: bool, optional
            Flag for including disambiguation information
        debug: bool, optional
            Flag for whether to keep temporary files, defaults to False
        """
        if self.root_dictionary is None:
            self.generate_mappings()
            os.makedirs(self.phones_dir, exist_ok=True)
            self._write_word_boundaries()
            self._write_phone_symbol_table()
            self._write_disambig()
        if debug:
            self.export_lexicon(os.path.join(self.dictionary_output_directory, "lexicon.txt"))
        self._write_graphemes()
        self._write_word_file()
        self._write_probabilistic_fst_text(write_disambiguation)
        self._write_fst_binary(write_disambiguation=write_disambiguation)
        if not debug:
            self.cleanup()

    def write_subset_fst(self, path: str, word_set: Set[str]):
        self._write_probabilistic_fst_text(disambiguation=False, path=path, word_set=word_set)
        self._write_fst_binary(write_disambiguation=False, path=path)

    def cleanup(self) -> None:
        """
        Clean up temporary files in the output directory
        """
        if os.path.exists(os.path.join(self.dictionary_output_directory, "temp.fst")):
            os.remove(os.path.join(self.dictionary_output_directory, "temp.fst"))
        if os.path.exists(os.path.join(self.dictionary_output_directory, "lexicon.text.fst")):
            os.remove(os.path.join(self.dictionary_output_directory, "lexicon.text.fst"))

    def _write_graphemes(self) -> None:
        """
        Write graphemes to temporary directory
        """
        outfile = os.path.join(self.dictionary_output_directory, "graphemes.txt")
        with open(outfile, "w", encoding="utf8") as f:
            for char in sorted(self.graphemes):
                f.write(char + "\n")

    @property
    def silence_probability_info(self) -> Dict[str, float]:
        return {
            "silence_probability": self.silence_probability,
            "initial_silence_probability": self.initial_silence_probability,
            "final_silence_correction": self.final_silence_correction,
            "final_non_silence_correction": self.final_non_silence_correction,
        }

    def export_lexicon(
        self,
        path: str,
        write_disambiguation: Optional[bool] = False,
        probability: Optional[bool] = False,
        silence_probabilities: Optional[bool] = False,
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
        silence_probabilities: bool, optional
            Flag for whether to include per pronunciation silence probabilities, only valid
            when ``probability`` is set to True
        """
        with open(path, "w", encoding="utf8") as f:
            for w in sorted(self.words.keys()):
                if w == self.silence_word:
                    continue
                for p in sorted(self.words[w]):
                    phones = " ".join(p.pronunciation)
                    if write_disambiguation and p.disambiguation is not None:
                        phones += f" #{p.disambiguation}"
                    probability_string = ""
                    if probability:
                        probability_string = f"{p.probability}"

                        if silence_probabilities:
                            if not p.silence_before_correction and w in self.silence_words:
                                continue
                            extra_probs = [
                                p.silence_after_probability,
                                p.silence_before_correction,
                                p.non_silence_before_correction,
                            ]
                            for x in extra_probs:
                                probability_string += f"\t{x if x else 0.0}"
                    if probability:
                        f.write(f"{w}\t{probability_string}\t{phones}\n")
                    else:
                        if w in self.silence_words:
                            continue
                        f.write(f"{w}\t{phones}\n")

    def _write_word_file(self) -> None:
        """
        Write the word mapping to the temporary directory
        """
        words_path = os.path.join(self.dictionary_output_directory, "words.txt")
        if sys.platform == "win32":
            newline = ""
        else:
            newline = None
        with open(words_path, "w", encoding="utf8", newline=newline) as f:
            for w, i in sorted(self.words_mapping.items(), key=lambda x: x[1]):
                f.write(f"{w} {i}\n")

    def _write_fst_binary(
        self, write_disambiguation: Optional[bool] = False, path: Optional[str] = None
    ) -> None:
        """
        Write the binary fst file to the temporary directory

        See Also
        --------
        :kaldi_src:`fstaddselfloops`
            Relevant Kaldi binary
        :openfst_src:`fstcompile`
            Relevant OpenFst binary
        :openfst_src:`fstarcsort`
            Relevant OpenFst binary

        Parameters
        ----------
        write_disambiguation: bool, optional
            Flag for including disambiguation symbols
        """
        text_ext = ".text_fst"
        binary_ext = ".fst"
        word_disambig_path = os.path.join(self.dictionary_output_directory, "word_disambig.txt")
        phone_disambig_path = os.path.join(self.dictionary_output_directory, "phone_disambig.txt")
        with open(phone_disambig_path, "w") as f:
            f.write(str(self.phone_mapping["#0"]))
        with open(word_disambig_path, "w") as f:
            f.write(str(self.words_mapping["#0"]))
        if write_disambiguation:
            text_ext = ".disambig_text_fst"
            binary_ext = ".disambig_fst"
        if path is not None:
            text_path = path.replace(".fst", text_ext)
            binary_path = path.replace(".fst", binary_ext)
        else:
            text_path = os.path.join(self.dictionary_output_directory, "lexicon" + text_ext)
            binary_path = os.path.join(self.dictionary_output_directory, "L" + binary_ext)

        words_file_path = os.path.join(self.dictionary_output_directory, "words.txt")

        log_path = os.path.join(
            self.dictionary_output_directory, os.path.basename(binary_path) + ".log"
        )
        with open(log_path, "w") as log_file:
            log_file.write(f"Phone isymbols: {self.phone_symbol_table_path}\n")
            log_file.write(f"Word osymbols: {words_file_path}\n")
            compile_proc = subprocess.Popen(
                [
                    thirdparty_binary("fstcompile"),
                    f"--isymbols={self.phone_symbol_table_path}",
                    f"--osymbols={words_file_path}",
                    "--keep_isymbols=false",
                    "--keep_osymbols=false",
                    "--keep_state_numbering=true",
                    text_path,
                ],
                stderr=log_file,
                stdout=subprocess.PIPE,
            )
            if write_disambiguation:
                selfloop_proc = subprocess.Popen(
                    [
                        thirdparty_binary("fstaddselfloops"),
                        phone_disambig_path,
                        word_disambig_path,
                    ],
                    stdin=compile_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                )
                arc_sort_proc = subprocess.Popen(
                    [
                        thirdparty_binary("fstarcsort"),
                        "--sort_type=olabel",
                        "-",
                        binary_path,
                    ],
                    stdin=selfloop_proc.stdout,
                    stderr=log_file,
                )
            else:
                arc_sort_proc = subprocess.Popen(
                    [
                        thirdparty_binary("fstarcsort"),
                        "--sort_type=olabel",
                        "-",
                        binary_path,
                    ],
                    stdin=compile_proc.stdout,
                    stderr=log_file,
                )
            arc_sort_proc.communicate()
            if arc_sort_proc.returncode != 0:
                raise KaldiProcessingError([log_path])

    def _write_probabilistic_fst_text(
        self, disambiguation=False, path: Optional[str] = None, word_set: Optional[Set[str]] = None
    ) -> None:
        """
        Write the L.fst text file to the temporary directory
        """
        base_ext = ".text_fst"
        if disambiguation:
            base_ext = ".disambig_text_fst"
        if path is not None:
            path = path.replace(".fst", base_ext)
        else:
            path = os.path.join(self.dictionary_output_directory, "lexicon" + base_ext)
        start_state = 0
        non_silence_state = 1  # Also loop state
        silence_state = 2
        next_state = 3
        if disambiguation:
            sil_disambiguation_symbol = self.silence_disambiguation_symbol
        else:
            sil_disambiguation_symbol = "<eps>"
        optional_silence_phone = self.optional_silence_phone
        oov_phone = self.oov_phone
        if self.position_dependent_phones:
            oov_phone += "_S"
        initial_silence_cost = -1 * math.log(self.initial_silence_probability)
        initial_non_silence_cost = -1 * math.log(1.0 - (self.initial_silence_probability))
        if self.final_silence_correction is None or self.final_non_silence_correction is None:
            final_silence_cost = 0
            final_non_silence_cost = 0
        else:
            final_silence_cost = -math.log(self.final_silence_correction)
            final_non_silence_cost = -math.log(self.final_non_silence_correction)
        base_silence_before_cost = 0.0
        base_non_silence_before_cost = 0.0
        base_silence_following_cost = -math.log(self.silence_probability)
        base_non_silence_following_cost = -math.log(1 - self.silence_probability)
        with open(path, "w", encoding="utf8") as outf:
            if self.dictionary_model.silence_probabilities:
                outf.write(
                    f"{start_state}\t{non_silence_state}\t{sil_disambiguation_symbol}\t{self.silence_word}\t{initial_non_silence_cost}\n"
                )  # initial no silence

                outf.write(
                    f"{start_state}\t{silence_state}\t{self.optional_silence_phone}\t{self.silence_word}\t{initial_silence_cost}\n"
                )  # initial silence
            else:
                outf.write(
                    f"{start_state}\t{non_silence_state}\t<eps>\t<eps>\t{initial_non_silence_cost}\n"
                )  # initial no silence

                outf.write(
                    f"{start_state}\t{non_silence_state}\t<eps>\t<eps>\t{initial_silence_cost}\n"
                )  # initial silence

                if disambiguation:
                    sil_disambiguation_state = next_state
                    next_state += 1
                    outf.write(
                        f"{silence_state}\t{sil_disambiguation_state}\t{self.optional_silence_phone}\t{self.silence_word}\t0.0\n"
                    )
                    outf.write(
                        f"{sil_disambiguation_state}\t{non_silence_state}\t{sil_disambiguation_symbol}\t{self.silence_word}\t0.0\n"
                    )
                else:
                    outf.write(
                        f"{silence_state}\t{non_silence_state}\t{self.optional_silence_phone}\t{self.silence_word}\t0.0\n"
                    )

            for word in sorted(self.words.keys()):
                if word_set is not None:
                    if word not in word_set:
                        continue
                for pron in sorted(self.words[word]):
                    phones = list(pron.pronunciation)
                    if phones == [self.optional_silence_phone]:
                        if pron.pronunciation[0] == self.optional_silence_phone:  # silences
                            outf.write(
                                f"{non_silence_state}\t{non_silence_state}\t{phones[0]}\t{word}\t0.0\n"
                            )
                            outf.write(
                                f"{start_state}\t{non_silence_state}\t{phones[0]}\t{word}\t0.0\n"
                            )  # initial silence
                        continue
                    prob = pron.probability
                    if self.dictionary_model.silence_probabilities:
                        silence_following_prob = pron.silence_after_probability
                        silence_before_correction = pron.silence_before_correction
                        non_silence_before_correction = pron.non_silence_before_correction
                        silence_before_cost = 0.0
                        non_silence_before_cost = 0.0
                        if not silence_following_prob:
                            silence_following_prob = self.silence_probability

                        silence_following_cost = -math.log(silence_following_prob)
                        non_silence_following_cost = -math.log(1 - (silence_following_prob))
                        if silence_before_correction:
                            silence_before_cost = -math.log(silence_before_correction)
                        if non_silence_before_correction:
                            non_silence_before_cost = -math.log(non_silence_before_correction)
                    else:
                        silence_before_cost = base_silence_before_cost
                        non_silence_before_cost = base_non_silence_before_cost
                        silence_following_cost = base_silence_following_cost
                        non_silence_following_cost = base_non_silence_following_cost
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
                    if prob == 0:
                        prob = 0.01  # Dithering to ensure low probability entries
                    elif prob is None:
                        prob = 1
                    pron_cost = abs(math.log(prob))
                    if disambiguation and pron.disambiguation:
                        phones += [f"#{pron.disambiguation}"]

                    if self.dictionary_model.silence_probabilities:
                        new_state = next_state
                        outf.write(
                            f"{non_silence_state}\t{new_state}\t{phones[0]}\t{word}\t{pron_cost+non_silence_before_cost}\n"
                        )
                        outf.write(
                            f"{silence_state}\t{new_state}\t{phones[0]}\t{word}\t{pron_cost+silence_before_cost}\n"
                        )

                        next_state += 1
                        current_state = new_state
                        for i in range(1, len(phones)):
                            new_state = next_state
                            next_state += 1
                            outf.write(f"{current_state}\t{new_state}\t{phones[i]}\t<eps>\n")
                            current_state = new_state
                        outf.write(
                            f"{current_state}\t{non_silence_state}\t{sil_disambiguation_symbol}\t<eps>\t{non_silence_following_cost}\n"
                        )
                        outf.write(
                            f"{current_state}\t{silence_state}\t{optional_silence_phone}\t<eps>\t{silence_following_cost}\n"
                        )
                    else:
                        current_state = non_silence_state
                        for i in range(len(phones) - 1):
                            w_or_eps = word if i == 0 else "<eps>"
                            cost = pron_cost if i == 0 else 0.0
                            outf.write(
                                f"{current_state}\t{next_state}\t{phones[i]}\t{w_or_eps}\t{cost}\n"
                            )
                            current_state = next_state
                            next_state += 1

                        i = len(phones) - 1
                        p = phones[i] if i >= 0 else "<eps>"
                        w = word if i <= 0 else "<eps>"
                        sil_cost = silence_following_cost + (pron_cost if i <= 0 else 0.0)
                        non_sil_cost = non_silence_following_cost + (pron_cost if i <= 0 else 0.0)
                        outf.write(
                            f"{current_state}\t{non_silence_state}\t{p}\t{w}\t{non_sil_cost}\n"
                        )
                        outf.write(f"{current_state}\t{silence_state}\t{p}\t{w}\t{sil_cost}\n")

            if self.dictionary_model.silence_probabilities:
                outf.write(f"{silence_state}\t{final_silence_cost}\n")
                outf.write(f"{non_silence_state}\t{final_non_silence_cost}\n")
            else:
                outf.write(f"{non_silence_state}\t0.0\n")


class PronunciationDictionary(PronunciationDictionaryMixin):
    """
    Class for processing pronunciation dictionaries

    See Also
    --------
    :class:`~montreal_forced_aligner.dictionary.pronunciation.PronunciationDictionaryMixin`
        For acoustic model training parsing parameters
    """

    @property
    def data_source_identifier(self) -> str:
        """Dictionary name"""
        return f"{self.name}"

    @property
    def identifier(self) -> str:
        """Dictionary name"""
        return f"{self.data_source_identifier}"

    @property
    def output_directory(self) -> str:
        """Temporary directory for the dictionary"""
        return os.path.join(self.temporary_directory, self.identifier)
