"""Pronunciation dictionaries for use in alignment and transcription"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Collection, Optional

if TYPE_CHECKING:
    from .multispeaker import MultispeakerDictionaryMixin
    from ..abc import DictionaryEntryType, MappingType, ReversedMappingType, WordsType, MetaDict
    from ..data import CtmType
    from .mixins import SanitizeFunction

from ..abc import TemporaryDirectoryMixin
from ..data import CtmInterval
from ..exceptions import DictionaryError, DictionaryFileError
from ..models import DictionaryModel
from ..utils import thirdparty_binary
from .mixins import DictionaryMixin

__all__ = [
    "PronunciationDictionaryMixin",
]


class PronunciationDictionaryMixin(DictionaryMixin, TemporaryDirectoryMixin):
    """
    Abstract mixin class containing information about a pronunciation dictionary

    Parameters
    ----------
    dictionary_path : str
        Path to pronunciation dictionary

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

    def __init__(self, dictionary_path, **kwargs):
        super().__init__(**kwargs)
        self.dictionary_model = DictionaryModel(dictionary_path)
        os.makedirs(self.dictionary_output_directory, exist_ok=True)
        self.words = {}
        self.graphemes = set()
        self.words[self.silence_word] = [
            {"pronunciation": (self.nonoptional_silence_phone,), "probability": 1}
        ]
        self.words[self.oov_word] = [{"pronunciation": (self.oov_phone,), "probability": 1}]

        with open(self.dictionary_model.path, "r", encoding="utf8") as inf:
            for i, line in enumerate(inf):
                line = line.strip()
                if not line:
                    continue
                line = line.split()
                word = self.sanitize(line.pop(0).lower())
                if not line:
                    raise DictionaryError(
                        f"Line {i} of {self.dictionary_model.path} does not have a pronunciation."
                    )
                if word in [self.silence_word, self.oov_word]:
                    continue
                self.graphemes.update(word)
                prob = 1
                if self.dictionary_model.pronunciation_probabilities:
                    prob = float(line.pop(0))
                    if prob > 1 or prob < 0:
                        raise ValueError
                if self.dictionary_model.silence_probabilities:
                    right_sil_prob = float(line.pop(0))
                    left_sil_prob = float(line.pop(0))
                    left_nonsil_prob = float(line.pop(0))
                else:
                    right_sil_prob = None
                    left_sil_prob = None
                    left_nonsil_prob = None
                if self.multilingual_ipa:
                    pron = self.parse_ipa(line)
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
                if not any(x in self.silence_phones for x in pron):
                    self.non_silence_phones.update(pron)
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
        self.words_mapping = {}
        self.lexicon_word_set = None
        if not self.graphemes:
            raise DictionaryFileError(
                f"No words were found in the dictionary path {self.dictionary_model.path}"
            )

    @property
    def name(self) -> str:
        """Name of the dictionary"""
        return self.dictionary_model.name

    def __hash__(self) -> Any:
        """Return the hash of a given dictionary"""
        return hash(self.dictionary_model.path)

    @property
    def dictionary_output_directory(self) -> str:
        """Temporary directory to store all dictionary information"""
        return os.path.join(self.temporary_directory, self.name)

    @property
    def output_paths(self) -> dict[str, str]:
        """
        Mapping of output directory for this dictionary
        """
        return {self.name: self.dictionary_output_directory}

    @property
    def silences(self) -> set[str]:
        """
        Set of symbols that correspond to silence
        """
        return self.silence_phones

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
            self.dictionary_options,
            self.construct_sanitize_function(),
            words_mapping,
            reversed_word_mapping,
            words,
        )

    def set_lexicon_word_set(self, word_set: Collection[str]) -> None:
        """
        Limit lexicon output to a subset of overall words

        Parameters
        ----------
        word_set: Collection[str]
            Word set to limit generated files to
        """
        word_set = {y for x in word_set for y in self._lookup(x)}
        word_set.add(self.silence_word)
        word_set.add(self.oov_word)
        self.lexicon_word_set = word_set | self.clitic_set
        self.generate_mappings()

    @property
    def actual_words(self) -> dict[str, "DictionaryEntryType"]:
        """
        Mapping of words to integer IDs without Kaldi-internal words
        """
        return {k: v for k, v in self.words.items() if k not in self.specials_set and len(v)}

    def split_clitics(self, item: str) -> list[str]:
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
        return self.data().split_clitics(item)

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
        if self.lexicon_word_set is None:
            return False
        if word not in self.lexicon_word_set and word not in self.clitic_set:
            return True
        return False

    def generate_mappings(self) -> None:
        """
        Generate word mappings from text to integer IDs
        """
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
            self.max_disambiguation_symbol = max(
                self.max_disambiguation_symbol, max(last_used.values())
            )

    def create_utterance_fst(self, text: list[str], frequent_words: list[tuple[str, int]]) -> str:
        """
        Create an FST for an utterance with frequent words as a unigram language model

        Parameters
        ----------
        text: list[str]
            Text of the utterance
        frequent_words: list[tuple[str, int]]
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

    def to_int(self, item: str) -> list[int]:
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
        return self.data().to_int(item)

    def _lookup(self, item: str) -> list[str]:
        """
        Look up a word and return the list of sub words if necessary taking into account clitic and compound markers

        Parameters
        ----------
        item: str
            Word to look up

        Returns
        -------
        list[str]
            List of subwords that are in the dictionary
        """
        return self.data().lookup(item)

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
        return self.data().check_word(item)

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
        return self.words_mapping[self.oov_word]

    @property
    def phones_dir(self) -> str:
        """
        Directory to store information Kaldi needs about phones
        """
        return os.path.join(os.path.dirname(self.dictionary_output_directory), "phones")

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
        return os.path.join(self.dictionary_output_directory, "L_disambig.fst")

    def write(
        self,
        write_disambiguation: bool = False,
        debug=False,
        multispeaker_dictionary: Optional[MultispeakerDictionaryMixin] = None,
    ) -> None:
        """
        Write the files necessary for Kaldi

        Parameters
        ----------
        write_disambiguation: bool, optional
            Flag for including disambiguation information
        debug: bool, optional
            Flag for whether to keep temporary files, defaults to False
        multispeaker_dictionary: MultispeakerDictionaryMixin, optional
            Main dictionary that has phone mapping IDs
        """
        self._write_graphemes()
        self._write_word_file()
        self._write_align_lexicon(multispeaker_dictionary)
        if write_disambiguation:
            self._write_fst_text_disambiguated(multispeaker_dictionary)
        else:
            self._write_basic_fst_text()
        self._write_fst_binary(multispeaker_dictionary, write_disambiguation=write_disambiguation)
        if not debug:
            self.cleanup()

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

    def _write_align_lexicon(
        self, multispeaker_dictionary: Optional[MultispeakerDictionaryMixin] = None
    ) -> None:
        """
        Write the alignment lexicon text file to the temporary directory
        """
        if multispeaker_dictionary is None:
            phone_mapping = self.phone_mapping
        else:
            phone_mapping = multispeaker_dictionary.phone_mapping
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
                    p = " ".join(str(phone_mapping[x]) for x in phones)
                    f.write(f"{i} {i} {p}\n".format(i=i, p=p))

    def _write_fst_binary(
        self,
        multispeaker_dictionary: Optional[MultispeakerDictionaryMixin] = None,
        write_disambiguation: Optional[bool] = False,
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
        multispeaker_dictionary: MultispeakerDictionaryMixin, optional
            Main dictionary with phone mappings
        write_disambiguation: bool, optional
            Flag for including disambiguation symbols
        """
        if write_disambiguation:
            lexicon_fst_path = os.path.join(
                self.dictionary_output_directory, "lexicon_disambig.text.fst"
            )
            output_fst = os.path.join(self.dictionary_output_directory, "L_disambig.fst")
        else:
            lexicon_fst_path = os.path.join(self.dictionary_output_directory, "lexicon.text.fst")
            output_fst = os.path.join(self.dictionary_output_directory, "L.fst")
        if multispeaker_dictionary is not None:
            phone_mapping = multispeaker_dictionary.phone_mapping
            phones_file_path = multispeaker_dictionary.phone_symbol_table_path
        else:
            phone_mapping = self.phone_mapping
            phones_file_path = os.path.join(self.dictionary_output_directory, "phones.txt")
        words_file_path = os.path.join(self.dictionary_output_directory, "words.txt")

        log_path = os.path.join(self.dictionary_output_directory, "fst.log")
        temp_fst_path = os.path.join(self.dictionary_output_directory, "temp.fst")
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
                temp2_fst_path = os.path.join(self.dictionary_output_directory, "temp2.fst")
                word_disambig_path = os.path.join(
                    self.dictionary_output_directory, "word_disambig0.txt"
                )
                phone_disambig_path = os.path.join(
                    self.dictionary_output_directory, "phone_disambig0.txt"
                )
                with open(phone_disambig_path, "w") as f:
                    f.write(str(phone_mapping["#0"]))
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

    def _write_basic_fst_text(self) -> None:
        """
        Write the L.fst text file to the temporary directory
        """
        sil_disambiguation = None
        nonoptional_silence = None
        optional_silence_phone = None
        lexicon_fst_path = os.path.join(self.dictionary_output_directory, "lexicon.text.fst")
        start_state = 0
        silence_state = 0
        silence_cost = 0
        no_silence_cost = 0
        loop_state = 0
        next_state = 1
        if self.silence_probability:
            optional_silence_phone = self.optional_silence_phone
            nonoptional_silence = self.nonoptional_silence_phone

            silence_cost = -1 * math.log(self.silence_probability)
            no_silence_cost = -1 * math.log(1.0 - self.silence_probability)
            loop_state = 1
            silence_state = 2

        with open(lexicon_fst_path, "w", encoding="utf8") as outf:
            if self.silence_probability:
                outf.write(
                    "\t".join(
                        map(str, [start_state, loop_state, "<eps>", "<eps>", no_silence_cost])
                    )
                    + "\n"
                )  # no silence

                outf.write(
                    "\t".join(
                        map(
                            str,
                            [start_state, loop_state, nonoptional_silence, "<eps>", silence_cost],
                        )
                    )
                    + "\n"
                )  # silence
                if sil_disambiguation is None:
                    outf.write(
                        "\t".join(
                            map(str, [silence_state, loop_state, optional_silence_phone, "<eps>"])
                        )
                        + "\n"
                    )  # no cost
                    next_state = 3
                else:
                    silence_disambiguation_state = next_state
                    next_state += 1
                    outf.write(
                        "\t".join(
                            map(
                                str,
                                [
                                    silence_state,
                                    silence_disambiguation_state,
                                    optional_silence_phone,
                                    "<eps>",
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
                                    silence_disambiguation_state,
                                    loop_state,
                                    sil_disambiguation,
                                    "<eps>",
                                ],
                            )
                        )
                        + "\n"
                    )

            for w in sorted(self.words.keys()):
                if self.exclude_for_alignment(w):
                    continue
                for pron in sorted(
                    self.words[w],
                    key=lambda x: (x["pronunciation"], x["probability"], x["disambiguation"]),
                ):
                    phones = list(pron["pronunciation"])
                    prob = pron["probability"]
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
                    if not prob:
                        prob = 0.001  # Dithering to ensure low probability entries
                    pron_cost = abs(math.log(prob))

                    current_state = loop_state
                    word_or_eps = w
                    local_no_silence_cost = no_silence_cost + pron_cost
                    local_silcost = no_silence_cost + pron_cost
                    for i, p in enumerate(phones):
                        if i < len(phones) - 1:
                            outf.write(
                                f"{current_state}\t{next_state}\t{p}\t{word_or_eps}\t{pron_cost}\n"
                            )
                            word_or_eps = "<eps>"
                            pron_cost = 0
                            current_state = next_state
                            next_state += 1
                        else:  # transition on last phone to loop state
                            if self.silence_probability:
                                outf.write(
                                    f"{current_state}\t{loop_state}\t{p}\t{word_or_eps}\t{local_no_silence_cost}\n"
                                )
                                outf.write(
                                    f"{current_state}\t{silence_state}\t{p}\t{word_or_eps}\t{local_silcost}\n"
                                )
                            else:
                                outf.write(
                                    f"{current_state}\t{loop_state}\t{p}\t{word_or_eps}\t{pron_cost}\n"
                                )
                                word_or_eps = "<eps>"

            outf.write(f"{loop_state}\t0\n")

    def _write_fst_text_disambiguated(
        self, multispeaker_dictionary: Optional[MultispeakerDictionaryMixin] = None
    ) -> None:
        """
        Write the text L_disambig.fst file to the temporary directory

        Parameters
        ----------
        multispeaker_dictionary: MultispeakerDictionaryMixin, optional
            Main dictionary with phone mappings
        """
        lexicon_fst_path = os.path.join(
            self.dictionary_output_directory, "lexicon_disambig.text.fst"
        )
        if multispeaker_dictionary is not None:
            sil_disambiguation = f"#{multispeaker_dictionary.max_disambiguation_symbol + 1}"
        else:
            sil_disambiguation = f"#{self.max_disambiguation_symbol + 1}"
        assert self.silence_probability
        start_state = 0
        loop_state = 1
        silence_state = 2
        next_state = 3

        silence_phone = self.nonoptional_silence_phone

        silence_cost = -1 * math.log(self.silence_probability)
        no_silence_cost = -1 * math.log(1 - self.silence_probability)

        with open(lexicon_fst_path, "w", encoding="utf8") as outf:
            outf.write(
                f"{start_state}\t{loop_state}\t<eps>\t<eps>\t{no_silence_cost}\n"
            )  # no silence
            outf.write(
                f"{start_state}\t{silence_state}\t<eps>\t<eps>\t{silence_cost}\n"
            )  # silence
            silence_disambiguation_state = next_state
            next_state += 1

            outf.write(
                f"{silence_state}\t{silence_disambiguation_state}\t{silence_phone}\t<eps>\t0.0\n"
            )  # silence disambig
            outf.write(
                f"{silence_disambiguation_state}\t{loop_state}\t{sil_disambiguation}\t<eps>\t0.0\n"
            )  # silence disambig

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
                    if not prob:
                        prob = 0.001  # Dithering to ensure low probability entries
                    pron_cost = abs(math.log(prob))
                    if disambig_symbol:
                        phones += [f"#{disambig_symbol}"]

                    current_state = loop_state
                    for i in range(0, len(phones) - 1):
                        p = phones[i]
                        outf.write(
                            f"{current_state}\t{next_state}\t{p}\t{w if i == 0 else '<eps>'}\t{pron_cost if i == 0 else 0.0}\n"
                        )
                        current_state = next_state
                        next_state += 1

                    i = len(phones) - 1

                    local_no_silence_cost = no_silence_cost + pron_cost
                    local_silcost = silence_cost + pron_cost
                    if i <= 0:
                        local_silcost = silence_cost
                        local_no_silence_cost = no_silence_cost
                    outf.write(
                        f"{current_state}\t{loop_state}\t{phones[i] if i >= 0 else '<eps>'}\t{w if i <= 0 else '<eps>'}\t{local_no_silence_cost}\n"
                    )
                    outf.write(
                        f"{current_state}\t{silence_state}\t{phones[i] if i >= 0 else '<eps>'}\t{w if i <= 0 else '<eps>'}\t{local_silcost}\n"
                    )

            outf.write(f"{loop_state}\t0.0\n")


class PronunciationDictionary(PronunciationDictionaryMixin):
    """
    Class for processing pronunciation dictionaries
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


@dataclass
class DictionaryData:
    """
    Information required for parsing Kaldi-internal ids to text

    Attributes
    ----------
    dictionary_options: MetaDict
        Options for the dictionary
    sanitize_function: SanitizeFunction
        Function to sanitize text
    words_mapping: MappingType
        Mapping from words to their integer IDs
    reversed_words_mapping: ReversedMappingType
        Mapping from integer IDs to words
    words: WordsType
        Words and their associated pronunciations
    """

    dictionary_options: MetaDict
    sanitize_function: SanitizeFunction
    words_mapping: MappingType
    reversed_words_mapping: ReversedMappingType
    words: WordsType

    @property
    def oov_word(self) -> str:
        """Out of vocabulary code"""
        return self.dictionary_options["oov_word"]

    @property
    def oov_int(self) -> int:
        """Out of vocabulary integer ID"""
        return self.words_mapping[self.oov_word]

    @property
    def compound_markers(self) -> list[str]:
        """Characters that separate compound words"""
        return self.dictionary_options["compound_markers"]

    @property
    def clitic_markers(self) -> list[str]:
        """Characters that mark clitics"""
        return self.dictionary_options["clitic_markers"]

    @property
    def clitic_set(self) -> set[str]:
        """Set of clitics"""
        return self.dictionary_options["clitic_set"]

    @property
    def punctuation(self) -> list[str]:
        """Characters to treat as punctuation"""
        return self.dictionary_options["punctuation"]

    @property
    def strip_diacritics(self) -> list[str]:
        """IPA diacritics to strip in multilingual IPA mode"""
        return self.dictionary_options["strip_diacritics"]

    @property
    def multilingual_ipa(self) -> bool:
        """Flag for multilingual IPA mode"""
        return self.dictionary_options["multilingual_ipa"]

    @property
    def silence_phones(self) -> set[str]:
        """Silence phones"""
        return {
            self.dictionary_options["oov_phone"],
            self.dictionary_options["optional_silence_phone"],
            self.dictionary_options["nonoptional_silence_phone"],
            self.dictionary_options["other_noise_phone"],
        }

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
        if item in self.words:
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

    def lookup(
        self,
        item: str,
    ) -> list[str]:
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
        sanitized = self.sanitize_function(item)
        if sanitized in self.words:
            return [sanitized]
        split = self.split_clitics(sanitized)
        oov_count = sum(1 for x in split if x not in self.words)
        if oov_count < len(
            split
        ):  # Only returned split item if it gains us any transcribed speech
            return split
        return [sanitized]

    def to_int(
        self,
        item: str,
    ) -> list[int]:
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

        sanitized = self.split_clitics(sanitized)
        if all(s in self.words for s in sanitized):
            return True
        return False

    def map_to_original_pronunciation(
        self, phones: CtmType, subpronunciations: list[DictionaryEntryType]
    ) -> CtmType:
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
