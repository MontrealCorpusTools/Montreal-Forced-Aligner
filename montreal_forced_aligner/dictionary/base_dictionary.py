"""Pronunciation dictionaries for use in alignment and transcription"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Collection, Dict, List, Optional, Set, Tuple, Union

if TYPE_CHECKING:
    from ..abc import ReversedMappingType, DictionaryEntryType

from ..abc import Dictionary
from ..config.dictionary_config import DictionaryConfig
from ..exceptions import DictionaryError, DictionaryFileError
from ..models import DictionaryModel
from ..utils import thirdparty_binary
from .data import DictionaryData

__all__ = [
    "PronunciationDictionary",
]


class PronunciationDictionary(Dictionary):
    """
    Class containing information about a pronunciation dictionary

    Parameters
    ----------
    dictionary_model : :class:`~montreal_forced_aligner.models.DictionaryModel`
        MFA Dictionary model
    output_directory : str
        Path to a directory to store files for Kaldi
    config: DictionaryConfig
        Configuration for generating lexicons
    word_set : Collection[str], optional
        Word set to limit output files
    logger: :class:`~logging.Logger`, optional
        Logger to output information to
    """

    topo_template = "<State> {cur_state} <PdfClass> {cur_state} <Transition> {cur_state} 0.75 <Transition> {next_state} 0.25 </State>"
    topo_sil_template = "<State> {cur_state} <PdfClass> {cur_state} {transitions} </State>"
    topo_transition_template = "<Transition> {} {}"
    positions: List[str] = ["_B", "_E", "_I", "_S"]

    def __init__(
        self,
        dictionary_model: Union[DictionaryModel, str],
        output_directory: str,
        config: Optional[DictionaryConfig] = None,
        word_set: Optional[Collection[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if isinstance(dictionary_model, str):
            dictionary_model = DictionaryModel(dictionary_model)
        if config is None:
            config = DictionaryConfig()
        super().__init__(dictionary_model, config)
        self.output_directory = os.path.join(output_directory, self.name)
        os.makedirs(self.output_directory, exist_ok=True)
        self.log_file = os.path.join(self.output_directory, f"{self.name}.log")
        if logger is None:
            self.logger = logging.getLogger("dictionary_setup")
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_file, "w", "utf-8")
            handler.setFormatter = logging.Formatter("%(name)s %(message)s")
            self.logger.addHandler(handler)
        else:
            self.logger = logger
        self.oovs_found = Counter()

        self.words = {}
        self.graphemes = set()
        self.all_words = defaultdict(list)
        self.words[self.config.silence_word] = [
            {"pronunciation": (self.config.nonoptional_silence_phone,), "probability": 1}
        ]
        self.words[self.config.oov_word] = [
            {"pronunciation": (self.config.oov_phone,), "probability": 1}
        ]

        progress = f'Parsing dictionary "{self.name}"'
        if self.dictionary_model.pronunciation_probabilities:
            progress += " with pronunciation probabilities"
        else:
            progress += " without pronunciation probabilities"
        if self.dictionary_model.silence_probabilities:
            progress += " with silence probabilities"
        else:
            progress += " without silence probabilities"
        self.logger.info(progress)
        with open(self.dictionary_model.path, "r", encoding="utf8") as inf:
            for i, line in enumerate(inf):
                line = line.strip()
                if not line:
                    continue
                line = line.split()
                word = self.config.sanitize(line.pop(0).lower())
                if not line:
                    raise DictionaryError(
                        f"Line {i} of {self.dictionary_model.path} does not have a pronunciation."
                    )
                if word in [self.config.silence_word, self.config.oov_word]:
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
                if self.config.multilingual_ipa:
                    pron = self.config.parse_ipa(line)
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
                if self.config.multilingual_ipa:
                    pronunciation["original_pronunciation"] = tuple(line)
                if not any(x in self.config.silence_phones for x in pron):
                    self.config.non_silence_phones.update(pron)
                if word in self.words and pron in {x["pronunciation"] for x in self.words[word]}:
                    continue
                if word not in self.words:
                    self.words[word] = []
                self.words[word].append(pronunciation)
                # test whether a word is a clitic
                is_clitic = False
                for cm in self.config.clitic_markers:
                    if word.startswith(cm) or word.endswith(cm):
                        is_clitic = True
                if is_clitic:
                    self.config.clitic_set.add(word)
        self.words_mapping = {}
        if word_set is not None:
            word_set = {y for x in word_set for y in self._lookup(x)}
            word_set.add(self.config.silence_word)
            word_set.add(self.config.oov_word)
        self.word_set = word_set
        if self.word_set is not None:
            self.word_set = self.word_set | self.config.clitic_set
        if not self.graphemes:
            raise DictionaryFileError(
                f"No words were found in the dictionary path {self.dictionary_model.path}"
            )

    def __hash__(self) -> Any:
        """Return the hash of a given dictionary"""
        return hash(self.dictionary_model.path)

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
        return self.config.silence_phones

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
            if word in self.config.clitic_set:
                return True
            if word in self.config.specials_set:
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
            self.config,
            words_mapping,
            reversed_word_mapping,
            self.reversed_phone_mapping,
            words,
        )

    def set_word_set(self, word_set: Collection[str]) -> None:
        """
        Limit output to a subset of overall words

        Parameters
        ----------
        word_set: Collection[str]
            Word set to limit generated files to
        """
        word_set = {y for x in word_set for y in self._lookup(x)}
        word_set.add(self.config.silence_word)
        word_set.add(self.config.oov_word)
        self.word_set = word_set | self.config.clitic_set
        self.generate_mappings()

    @property
    def actual_words(self) -> Dict[str, "DictionaryEntryType"]:
        """
        Mapping of words to integer IDs without Kaldi-internal words
        """
        return {
            k: v for k, v in self.words.items() if k not in self.config.specials_set and len(v)
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
        if self.word_set is None:
            return False
        if word not in self.word_set and word not in self.config.clitic_set:
            return True
        return False

    @property
    def phone_mapping(self) -> Dict[str, int]:
        return self.config.phone_mapping

    def generate_mappings(self) -> None:
        """
        Generate phone and word mappings from text to integer IDs
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
            self.config.max_disambiguation_symbol = max(
                self.config.max_disambiguation_symbol, max(last_used.values())
            )

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
        return self.data().to_int(item)

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
        return self.words_mapping[self.config.oov_word]

    @property
    def phones_dir(self) -> str:
        """
        Directory to store information Kaldi needs about phones
        """
        return os.path.join(self.output_directory, "phones")

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
        if write_disambiguation:
            self._write_fst_text_disambiguated()
        else:
            self._write_basic_fst_text()
        self._write_fst_binary(write_disambiguation=write_disambiguation)
        self.cleanup()

    def cleanup(self) -> None:
        """
        Clean up temporary files in the output directory
        """
        if not self.config.debug:
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
            for sp in self.config.silence_phones:
                if self.config.position_dependent_phones:
                    new_phones = [sp + x for x in ["", ""] + self.positions]
                else:
                    new_phones = [sp]
                f.write(" ".join(new_phones) + "\n")
            for nsp in self.config.non_silence_phones:
                if self.config.position_dependent_phones:
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
            if self.config.position_dependent_phones:
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
                    if self.config.position_dependent_phones:
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
        sil_transp = 1 / (self.config.num_silence_states - 1)
        initial_transition = [
            self.topo_transition_template.format(x, sil_transp)
            for x in range(self.config.num_silence_states - 1)
        ]
        middle_transition = [
            self.topo_transition_template.format(x, sil_transp)
            for x in range(1, self.config.num_silence_states)
        ]
        final_transition = [
            self.topo_transition_template.format(self.config.num_silence_states - 1, 0.75),
            self.topo_transition_template.format(self.config.num_silence_states, 0.25),
        ]
        with open(filepath, "w") as f:
            f.write("<Topology>\n")
            f.write("<TopologyEntry>\n")
            f.write("<ForPhones>\n")
            phones = self.config.kaldi_non_silence_phones
            f.write(f"{' '.join(str(self.phone_mapping[x]) for x in phones)}\n")
            f.write("</ForPhones>\n")
            states = [
                self.topo_template.format(cur_state=x, next_state=x + 1)
                for x in range(self.config.num_non_silence_states)
            ]
            f.write("\n".join(states))
            f.write(f"\n<State> {self.config.num_non_silence_states} </State>\n")
            f.write("</TopologyEntry>\n")

            f.write("<TopologyEntry>\n")
            f.write("<ForPhones>\n")

            phones = self.config.kaldi_silence_phones
            f.write(f"{' '.join(str(self.phone_mapping[x]) for x in phones)}\n")
            f.write("</ForPhones>\n")
            states = []
            for i in range(self.config.num_silence_states):
                if i == 0:
                    transition = " ".join(initial_transition)
                elif i == self.config.num_silence_states - 1:
                    transition = " ".join(final_transition)
                else:
                    transition = " ".join(middle_transition)
                states.append(self.topo_sil_template.format(cur_state=i, transitions=transition))
            f.write("\n".join(states))
            f.write(f"\n<State> {self.config.num_silence_states} </State>\n")
            f.write("</TopologyEntry>\n")
            f.write("</Topology>\n")

    def _write_phone_sets(self) -> None:
        """
        Write phone symbol sets to the temporary directory
        """
        sharesplit = ["shared", "split"]
        if not self.config.shared_silence_phones:
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
            for i, sp in enumerate(self.config.silence_phones):
                if self.config.position_dependent_phones:
                    mapped = [sp + x for x in [""] + self.positions]
                else:
                    mapped = [sp]
                setf.write(" ".join(mapped) + "\n")
                setintf.write(" ".join(map(str, (self.phone_mapping[x] for x in mapped))) + "\n")
                if i == 0:
                    line = sil_sharesplit + mapped
                    lineint = sil_sharesplit + [str(self.phone_mapping[x]) for x in mapped]
                else:
                    line = sharesplit + mapped
                    lineint = sharesplit + [str(self.phone_mapping[x]) for x in mapped]
                rootf.write(" ".join(line) + "\n")
                rootintf.write(" ".join(lineint) + "\n")

            # process nonsilence phones
            for nsp in sorted(self.config.non_silence_phones):
                if self.config.position_dependent_phones:
                    mapped = [nsp + x for x in self.positions]
                else:
                    mapped = [nsp]
                setf.write(" ".join(mapped) + "\n")
                setintf.write(" ".join(map(str, (self.phone_mapping[x] for x in mapped))) + "\n")
                line = sharesplit + mapped
                lineint = sharesplit + [str(self.phone_mapping[x]) for x in mapped]
                rootf.write(" ".join(line) + "\n")
                rootintf.write(" ".join(lineint) + "\n")

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
            silences = self.config.kaldi_silence_phones
            outf.write(" ".join(silences) + "\n")
            intf.write(" ".join(str(self.phone_mapping[x]) for x in silences) + "\n")

            non_silences = self.config.kaldi_non_silence_phones
            outf.write(" ".join(non_silences) + "\n")
            intf.write(" ".join(str(self.phone_mapping[x]) for x in non_silences) + "\n")
            if self.config.position_dependent_phones:
                for p in self.positions:
                    line = [x + p for x in sorted(self.config.non_silence_phones)]
                    outf.write(" ".join(line) + "\n")
                    intf.write(" ".join(str(self.phone_mapping[x]) for x in line) + "\n")
                for p in [""] + self.positions:
                    line = [x + p for x in sorted(self.config.silence_phones)]
                    outf.write(" ".join(line) + "\n")
                    intf.write(" ".join(str(self.phone_mapping[x]) for x in line) + "\n")

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
            for d in sorted(
                self.config.disambiguation_symbols, key=lambda x: self.phone_mapping[x]
            ):
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

    def _write_basic_fst_text(self) -> None:
        """
        Write the L.fst text file to the temporary directory
        """
        sil_disambiguation = None
        nonoptional_silence = None
        optional_silence_phone = None
        lexicon_fst_path = os.path.join(self.output_directory, "lexicon.text.fst")
        start_state = 0
        silence_state = 0
        silence_cost = 0
        no_silence_cost = 0
        loop_state = 0
        next_state = 1
        if self.config.silence_probability:
            optional_silence_phone = self.config.optional_silence_phone
            nonoptional_silence = self.config.nonoptional_silence_phone

            silence_cost = -1 * math.log(self.config.silence_probability)
            no_silence_cost = -1 * math.log(1.0 - self.config.silence_probability)
            loop_state = 1
            silence_state = 2

        with open(lexicon_fst_path, "w", encoding="utf8") as outf:
            if self.config.silence_probability:
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
                    if self.config.position_dependent_phones:
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
                            if self.config.silence_probability:
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

    def _write_fst_text_disambiguated(self) -> None:
        """
        Write the text L_disambig.fst file to the temporary directory
        """
        lexicon_fst_path = os.path.join(self.output_directory, "lexicon_disambig.text.fst")
        sil_disambiguation = f"#{self.config.max_disambiguation_symbol + 1}"
        assert self.config.silence_probability
        start_state = 0
        loop_state = 1
        silence_state = 2
        next_state = 3

        silence_phone = self.config.nonoptional_silence_phone

        silence_cost = -1 * math.log(self.config.silence_probability)
        no_silence_cost = -1 * math.log(1 - self.config.silence_probability)

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
                    if self.config.position_dependent_phones:
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
