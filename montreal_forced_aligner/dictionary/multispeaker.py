"""Pronunciation dictionaries for use in alignment and transcription"""

from __future__ import annotations

import abc
import collections
import math
import os
import re
import subprocess
import sys
import typing
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import sqlalchemy.orm.session
from sqlalchemy.orm import selectinload

from montreal_forced_aligner.data import PhoneType, WordType
from montreal_forced_aligner.db import (
    DictBundle,
    Dictionary,
    OovWord,
    Phone,
    Pronunciation,
    Speaker,
    Utterance,
    Word,
)
from montreal_forced_aligner.dictionary.mixins import (
    SanitizeFunction,
    SplitWordsFunction,
    TemporaryDictionaryMixin,
)
from montreal_forced_aligner.exceptions import (
    DictionaryError,
    DictionaryFileError,
    KaldiProcessingError,
)
from montreal_forced_aligner.helper import split_phone_position
from montreal_forced_aligner.models import DictionaryModel, PhoneSetType
from montreal_forced_aligner.utils import thirdparty_binary

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from dataclassy import dataclass

__all__ = [
    "MultispeakerDictionaryMixin",
    "MultispeakerDictionary",
    "MultispeakerSanitizationFunction",
]


@dataclass
class MultispeakerSanitizationFunction:
    """
    Function for sanitizing text based on a multispeaker dictionary

    Parameters
    ----------
    speaker_mapping: dict[str, str]
        Mapping of speakers to dictionary names
    sanitize_function: :class:`~montreal_forced_aligner.dictionary.mixins.SanitizeFunction`
        Function to use for stripping punctuation
    split_functions: dict[str, :class:`~montreal_forced_aligner.dictionary.mixins.SplitWordsFunction`]
        Mapping of dictionary ids to functions for splitting compounds and clitics into separate words
    """

    speaker_mapping: Dict[str, int]
    sanitize_function: SanitizeFunction
    split_functions: Dict[int, SplitWordsFunction]

    def get_dict_id_for_speaker(self, speaker_name: str) -> int:
        """
        Get the dictionary id of the speaker

        Parameters
        ----------
        speaker_name: str
            Speaker to look up

        Returns
        -------
        int
            Dictionary id
        """
        if speaker_name not in self.speaker_mapping:
            speaker_name = "default"
        return self.speaker_mapping[speaker_name]

    def get_functions_for_speaker(
        self, speaker_name: str
    ) -> Tuple[SanitizeFunction, SplitWordsFunction]:
        """
        Look up functions based on speaker name

        Parameters
        ----------
        speaker_name
            Speaker to get functions for

        Returns
        -------
        :class:`~montreal_forced_aligner.dictionary.mixins.SanitizeFunction`
            Function for sanitizing text
        :class:`~montreal_forced_aligner.dictionary.mixins.SplitWordsFunction`
            Function for splitting up words
        """
        try:
            dict_id = self.get_dict_id_for_speaker(speaker_name)
            split_function = self.split_functions[dict_id]
        except KeyError:
            split_function = None
        return self.sanitize_function, split_function


class MultispeakerDictionaryMixin(TemporaryDictionaryMixin, metaclass=abc.ABCMeta):
    """
    Mixin class containing information about a pronunciation dictionary with different dictionaries per speaker

    Parameters
    ----------
    dictionary_path : str
        Dictionary path
    kwargs : kwargs
        Extra parameters to passed to parent classes (see below)

    See Also
    --------
    :class:`~montreal_forced_aligner.dictionary.mixins.DictionaryMixin`
        For dictionary parsing parameters
    :class:`~montreal_forced_aligner.abc.TemporaryDirectoryMixin`
        For temporary directory parameters


    Attributes
    ----------
    dictionary_model: :class:`~montreal_forced_aligner.models.DictionaryModel`
        Dictionary model
    dictionary_lookup: dict[str, int]
        Mapping of dictionary names to ids
    """

    def __init__(self, dictionary_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.dictionary_model = DictionaryModel(
            dictionary_path, phone_set_type=self.phone_set_type
        )
        self._num_dictionaries = None
        self.dictionary_lookup = {}
        self._phone_mapping = None
        self._words_mappings = {}
        self._default_dictionary_id = None
        self._dictionary_base_names = None

    @property
    def dictionary_base_names(self) -> Dict[int, str]:
        """Mapping of base file names for pronunciation dictionaries"""
        if self._dictionary_base_names is None:
            with self.session() as session:
                dictionaries = session.query(Dictionary.id, Dictionary.name).all()
                self._dictionary_base_names = {}
                for d_id, d_name in dictionaries:
                    base_name = d_name
                    if any(d_name == x[1] and d_id != x[0] for x in dictionaries):
                        base_name += f"_{d_id}"
                    base_name += ".dict"
                    self._dictionary_base_names[d_id] = base_name
        return self._dictionary_base_names

    def word_mapping(self, dictionary_id: int = 1) -> Dict[str, int]:
        """
        Get the word mapping for a specified dictionary id

        Parameters
        ----------
        dictionary_id: int, optional
            Database ID for dictionary, defaults to 1

        Returns
        -------
        dict[str, int]
            Mapping from words to their integer IDs for Kaldi processing
        """
        if dictionary_id not in self._words_mappings:
            self._words_mappings[dictionary_id] = {}
            with self.session() as session:
                words = session.query(Word.word, Word.mapping_id).filter(
                    Word.dictionary_id == dictionary_id
                )
                for w, index in words:
                    self._words_mappings[dictionary_id][w] = index
                self._words_mappings[dictionary_id]["#0"] = index + 1
                self._words_mappings[dictionary_id]["<s>"] = index + 2
                self._words_mappings[dictionary_id]["</s>"] = index + 3
        return self._words_mappings[dictionary_id]

    def reversed_word_mapping(self, dictionary_id: int = 1) -> Dict[int, str]:
        """
        Get the reversed word mapping for a specified dictionary id

        Parameters
        ----------
        dictionary_id: int, optional
            Database ID for dictionary, defaults to 1

        Returns
        -------
        dict[int, str]
            Mapping from integer IDs to words for Kaldi processing
        """
        mapping = {}
        for k, v in self.word_mapping(dictionary_id).items():
            mapping[v] = k
        return mapping

    @property
    def num_dictionaries(self) -> int:
        """Number of pronunciation dictionaries"""
        return len(self.dictionary_lookup)

    @property
    def sanitize_function(self) -> MultispeakerSanitizationFunction:
        """Sanitization function for the dictionary"""
        sanitize_function = SanitizeFunction(
            self.punctuation,
            self.clitic_markers,
            self.compound_markers,
            self.brackets,
            self.ignore_case,
            self.quote_markers,
            self.word_break_markers,
        )
        split_functions = {}
        with self.session() as session:
            dictionaries = session.query(
                Dictionary.id, Dictionary.default, Dictionary.laughter_regex
            )
            speaker_mapping = {
                x[0]: x[1] for x in session.query(Speaker.name, Speaker.dictionary_id)
            }
            for dict_id, default, laughter_regex in dictionaries:
                if default:
                    speaker_mapping["default"] = dict_id

                clitic_set = set(
                    x[0]
                    for x in session.query(Word.word)
                    .filter(Word.word_type == WordType.clitic)
                    .filter(Word.dictionary_id == dict_id)
                )
                split_functions[dict_id] = SplitWordsFunction(
                    self.clitic_markers,
                    self.compound_markers,
                    clitic_set,
                    self.brackets,
                    self.word_mapping(dict_id),
                    self.specials_set,
                    self.oov_word,
                    self.bracketed_word,
                    self.laughter_word,
                    laughter_regex,
                )
        return MultispeakerSanitizationFunction(
            speaker_mapping, sanitize_function, split_functions
        )

    def dictionary_setup(self):
        """Set up the dictionary for processing"""
        exist_check = os.path.exists(self.db_path)
        if not exist_check:
            self.initialize_database()
        auto_set = {PhoneSetType.AUTO, PhoneSetType.UNKNOWN, "AUTO", "UNKNOWN"}
        if not isinstance(self.phone_set_type, PhoneSetType):
            self.phone_set_type = PhoneSetType[self.phone_set_type]

        os.makedirs(self.dictionary_output_directory, exist_ok=True)
        pretrained = False
        if self.non_silence_phones:
            pretrained = True
        bracket_regex = None
        laughter_regex = None
        if self.brackets:
            left_brackets = [x[0] for x in self.brackets]
            right_brackets = [x[1] for x in self.brackets]
            bracket_regex = re.compile(
                rf"[{re.escape(''.join(left_brackets))}].*?[{re.escape(''.join(right_brackets))}]+"
            )
            laughter_regex = re.compile(
                rf"[{re.escape(''.join(left_brackets))}](laugh(ing|ter)?|lachen|lg)[{re.escape(''.join(right_brackets))}]+",
                flags=re.IGNORECASE,
            )
        self._speaker_ids = getattr(self, "_speaker_ids", {})
        dictionary_id_cache = {}
        with self.session() as session:
            for speaker_id, speaker_name, dictionary_id, dict_name, path in (
                session.query(
                    Speaker.id, Speaker.name, Dictionary.id, Dictionary.name, Dictionary.path
                )
                .join(Speaker.dictionary)
                .filter(Dictionary.default == False)  # noqa
            ):
                self._speaker_ids[speaker_name] = speaker_id
                dictionary_id_cache[path] = dictionary_id
                self.dictionary_lookup[dict_name] = dictionary_id
            dictionary = (
                session.query(Dictionary).filter(Dictionary.default == True).first()  # noqa
            )
            if dictionary:
                self._default_dictionary_id = dictionary.id
                dictionary_id_cache[dictionary.path] = self._default_dictionary_id
                self.dictionary_lookup[dictionary.name] = dictionary.id
            word_primary_key = 1
            pronunciation_primary_key = 1
            word_objs = []
            pron_objs = []
            speaker_objs = []
            phone_counts = collections.Counter()
            graphemes = set()
            self._current_speaker_index = getattr(self, "_current_speaker_index", 1)
            for (
                dictionary_model,
                speakers,
            ) in self.dictionary_model.load_dictionary_paths().values():
                if dictionary_model.path not in dictionary_id_cache:
                    word_cache = {}
                    pronunciation_cache = set()
                    subsequences = set()
                    pronunciation_counts = collections.defaultdict(int)
                    if self.phone_set_type not in auto_set:
                        if (
                            self.phone_set_type != dictionary_model.phone_set_type
                            and dictionary_model.phone_set_type not in auto_set
                        ):
                            raise DictionaryError(
                                f"Mismatch found in phone sets: {self.phone_set_type} vs {dictionary_model.phone_set_type}"
                            )
                    else:
                        self.phone_set_type = dictionary_model.phone_set_type
                    sanitize = False
                    clitic_cleanup_regex = None
                    clitic_marker = None
                    if len(self.clitic_markers) >= 1:
                        sanitize = True
                        clitic_cleanup_regex = re.compile(rf'[{"".join(self.clitic_markers[1:])}]')
                        clitic_marker = self.clitic_markers[0]

                    dictionary = Dictionary(
                        name=dictionary_model.name,
                        path=dictionary_model.path,
                        phone_set_type=self.phone_set_type,
                        root_temp_directory=self.dictionary_output_directory,
                        position_dependent_phones=self.position_dependent_phones,
                        clitic_marker=clitic_marker,
                        bracket_regex=bracket_regex.pattern,
                        laughter_regex=laughter_regex.pattern,
                        default="default" in speakers,
                        max_disambiguation_symbol=0,
                        silence_word=self.silence_word,
                        oov_word=self.oov_word,
                        bracketed_word=self.bracketed_word,
                        laughter_word=self.laughter_word,
                        optional_silence_phone=self.optional_silence_phone,
                    )
                    session.add(dictionary)
                    session.flush()
                    dictionary_id_cache[dictionary_model.path] = dictionary.id
                    if dictionary.default:
                        self._default_dictionary_id = dictionary.id
                    self._words_mappings[dictionary.id] = {}
                    current_index = 0
                    word_objs.append(
                        {
                            "id": word_primary_key,
                            "mapping_id": current_index,
                            "word": self.silence_word,
                            "word_type": WordType.silence,
                            "dictionary_id": dictionary.id,
                        }
                    )
                    self._words_mappings[dictionary.id][self.silence_word] = current_index
                    current_index += 1

                    pron_objs.append(
                        {
                            "id": pronunciation_primary_key,
                            "pronunciation": self.optional_silence_phone,
                            "probability": 1.0,
                            "disambiguation": None,
                            "silence_after_probability": None,
                            "silence_before_correction": None,
                            "non_silence_before_correction": None,
                            "word_id": word_primary_key,
                        }
                    )
                    word_primary_key += 1
                    pronunciation_primary_key += 1

                    special_words = {self.oov_word: WordType.oov}
                    if bracket_regex is not None:
                        special_words[self.bracketed_word] = WordType.bracketed
                    if laughter_regex is not None:
                        special_words[self.laughter_word] = WordType.laughter
                    specials_found = set()
                    with open(dictionary_model.path, "r", encoding="utf8") as inf:
                        for i, line in enumerate(inf):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                if "\t" in line:
                                    word, line = line.split("\t", maxsplit=1)
                                    line = line.strip().split()
                                else:
                                    line = line.split()
                                    word = line.pop(0)
                            except Exception as e:
                                raise DictionaryError(
                                    f'Error parsing line {i} of {dictionary_model.path}: "{line}" had error "{e}"'
                                )
                            if self.ignore_case:
                                word = word.lower()
                            if " " in word:
                                continue
                            if sanitize:
                                word = clitic_cleanup_regex.sub(clitic_marker, word)
                            if not line:
                                raise DictionaryError(
                                    f"Line {i} of {dictionary_model.path} does not have a pronunciation."
                                )
                            if word in self.specials_set:
                                continue
                            graphemes.update(word)
                            prob = 1
                            if dictionary_model.pronunciation_probabilities:
                                prob = float(line.pop(0))
                                if prob > 1 or prob < 0:
                                    raise ValueError
                            silence_after_prob = None
                            silence_before_correct = None
                            non_silence_before_correct = None
                            if dictionary_model.silence_probabilities:
                                silence_after_prob = float(line.pop(0))
                                silence_before_correct = float(line.pop(0))
                                non_silence_before_correct = float(line.pop(0))
                            pron = tuple(line)
                            if pretrained:
                                difference = (
                                    set(pron) - self.non_silence_phones - self.silence_phones
                                )
                                if difference:
                                    self.excluded_phones.update(difference)
                                    self.excluded_pronunciation_count += 1
                                    continue
                            if word not in word_cache:
                                if pron == (self.optional_silence_phone,):
                                    wt = WordType.silence
                                elif dictionary.clitic_marker is not None and (
                                    word.startswith(dictionary.clitic_marker)
                                    or word.endswith(dictionary.clitic_marker)
                                ):
                                    wt = WordType.clitic
                                else:
                                    wt = WordType.speech
                                    for special_w, special_wt in special_words.items():
                                        if word == special_w:
                                            wt = special_wt
                                            specials_found.add(special_w)
                                            break
                                word_objs.append(
                                    {
                                        "id": word_primary_key,
                                        "mapping_id": current_index,
                                        "word": word,
                                        "word_type": wt,
                                        "dictionary_id": dictionary.id,
                                    }
                                )
                                self._words_mappings[dictionary.id][word] = current_index
                                current_index += 1
                                word_cache[word] = word_primary_key
                                word_primary_key += 1
                            pron_string = " ".join(pron)
                            if (word, pron_string) in pronunciation_cache:
                                continue

                            if (
                                not pretrained
                                and word_objs[word_cache[word] - 1]["word_type"] is WordType.speech
                            ):
                                self.non_silence_phones.update(pron)
                            pron_objs.append(
                                {
                                    "id": pronunciation_primary_key,
                                    "pronunciation": pron_string,
                                    "probability": prob,
                                    "disambiguation": None,
                                    "silence_after_probability": silence_after_prob,
                                    "silence_before_correction": silence_before_correct,
                                    "non_silence_before_correction": non_silence_before_correct,
                                    "word_id": word_cache[word],
                                }
                            )
                            pronunciation_primary_key += 1
                            pronunciation_cache.add((word, pron_string))
                            phone_counts.update(pron)
                            pronunciation_counts[pron] += 1
                            pron = pron[:-1]
                            while pron:
                                subsequences.add(tuple(pron))
                                pron = pron[:-1]

                    for w, wt in special_words.items():
                        if w in specials_found:
                            continue
                        word_objs.append(
                            {
                                "id": word_primary_key,
                                "mapping_id": current_index,
                                "word": w,
                                "word_type": wt,
                                "dictionary_id": dictionary.id,
                            }
                        )
                        self._words_mappings[dictionary.id][w] = current_index
                        current_index += 1
                        pron = tuple(self.oov_phone)
                        pron_objs.append(
                            {
                                "id": pronunciation_primary_key,
                                "pronunciation": self.oov_phone,
                                "probability": 1.0,
                                "disambiguation": None,
                                "silence_after_probability": None,
                                "silence_before_correction": None,
                                "non_silence_before_correction": None,
                                "word_id": word_primary_key,
                            }
                        )
                        pronunciation_primary_key += 1
                        word_primary_key += 1
                        pronunciation_counts[pron] += 1
                    self._words_mappings[dictionary.id]["#0"] = current_index
                    self._words_mappings[dictionary.id]["<s>"] = current_index + 1
                    self._words_mappings[dictionary.id]["</s>"] = current_index + 2

                    last_used = collections.defaultdict(int)
                    for p in pron_objs:
                        pron = tuple(p["pronunciation"].split())
                        if not (pronunciation_counts[pron] == 1 and pron not in subsequences):
                            last_used[pron] += 1
                            p["disambiguation"] = last_used[pron]
                    if last_used:
                        dictionary.max_disambiguation_symbol = max(
                            dictionary.max_disambiguation_symbol, max(last_used.values())
                        )
                    if not graphemes:
                        raise DictionaryFileError(
                            f"No words were found in the dictionary path {dictionary_model.path}"
                        )
                    self.max_disambiguation_symbol = max(
                        self.max_disambiguation_symbol, dictionary.max_disambiguation_symbol
                    )
                for speaker in speakers:
                    if speaker != "default":
                        if speaker not in self._speaker_ids:
                            speaker_objs.append(
                                {
                                    "id": self._current_speaker_index,
                                    "name": speaker,
                                    "dictionary_id": dictionary.id,
                                }
                            )
                            self._speaker_ids[speaker] = self._current_speaker_index
                            self._current_speaker_index += 1
                self.dictionary_lookup[dictionary.name] = dictionary.id
                session.commit()

            self.non_silence_phones -= self.silence_phones
            phone_objs = []
            if self.non_silence_phones:
                max_phone_ind = session.query(sqlalchemy.func.max(Phone.mapping_id)).scalar()
                i = 0
                if max_phone_ind is not None:
                    session.query(Phone).delete()
                    session.commit()
                phone_objs.append(
                    {
                        "id": i + 1,
                        "mapping_id": i,
                        "phone": "<eps>",
                        "phone_type": PhoneType.silence,
                    }
                )
                for p in self.kaldi_silence_phones:
                    i += 1
                    phone_objs.append(
                        {"id": i + 1, "mapping_id": i, "phone": p, "phone_type": PhoneType.silence}
                    )
                for p in self.kaldi_non_silence_phones:
                    i += 1
                    phone_objs.append(
                        {
                            "id": i + 1,
                            "mapping_id": i,
                            "phone": p,
                            "phone_type": PhoneType.non_silence,
                        }
                    )
                for x in range(self.max_disambiguation_symbol + 2):
                    p = f"#{x}"
                    self.disambiguation_symbols.add(p)
                    i += 1
                    phone_objs.append(
                        {
                            "id": i + 1,
                            "mapping_id": i,
                            "phone": p,
                            "phone_type": PhoneType.disambiguation,
                        }
                    )
            else:
                phones = [
                    x[0]
                    for x in session.query(Phone.phone).filter(
                        Phone.phone_type == PhoneType.non_silence
                    )
                ]
                if self.position_dependent_phones:
                    phones = [split_phone_position(x)[0] for x in phones]
                self.non_silence_phones.update(phones)
                phones = [
                    x[0]
                    for x in session.query(Phone.phone).filter(
                        Phone.phone_type == PhoneType.disambiguation
                    )
                ]
                self.disambiguation_symbols.update(phones)
        with self.session() as session:
            with session.bind.begin() as conn:
                if word_objs:
                    conn.execute(sqlalchemy.insert(Word.__table__), word_objs)
                    conn.execute(sqlalchemy.insert(Pronunciation.__table__), pron_objs)
                if speaker_objs:
                    conn.execute(sqlalchemy.insert(Speaker.__table__), speaker_objs)
                if phone_objs:
                    conn.execute(sqlalchemy.insert(Phone.__table__), phone_objs)

                session.commit()
        return graphemes, phone_counts

    def _write_probabilistic_fst_text(
        self,
        session: sqlalchemy.orm.session.Session,
        dictionary: Dictionary,
        silence_disambiguation_symbol=None,
        path: typing.Optional[str] = None,
    ) -> None:
        """
        Write the L.fst or L_disambig.fst text file to the temporary directory

        Parameters
        ----------
        session: :class:`~sqlalchemy.orm.session.Session`
            Session to use for querying
        dictionary: :class:`~montreal_forced_aligner.db.Dictionary`
            Dictionary for generating L.fst
        silence_disambiguation_symbol: str, optional
            Symbol to use for disambiguating silence for L_disambig.fst
        path: str, optional
            Full path to write L.fst to
        """
        base_ext = ".text_fst"
        disambiguation = False
        if silence_disambiguation_symbol is not None:
            base_ext = ".disambig_text_fst"
            disambiguation = True
        if path is not None:
            path = path.replace(".fst", base_ext)
        else:
            path = os.path.join(dictionary.temp_directory, "lexicon" + base_ext)
        start_state = 0
        non_silence_state = 1  # Also loop state
        silence_state = 2
        next_state = 3
        if silence_disambiguation_symbol is None:
            silence_disambiguation_symbol = "<eps>"

        initial_silence_cost = -1 * math.log(self.initial_silence_probability)
        initial_non_silence_cost = -1 * math.log(1.0 - (self.initial_silence_probability))
        if self.final_silence_correction is None or self.final_non_silence_correction is None:
            final_silence_cost = "0"
            final_non_silence_cost = "0"
        else:
            final_silence_cost = str(-math.log(self.final_silence_correction))
            final_non_silence_cost = str(-math.log(self.final_non_silence_correction))
        base_silence_before_cost = 0.0
        base_non_silence_before_cost = 0.0
        base_silence_following_cost = -math.log(self.silence_probability)
        base_non_silence_following_cost = -math.log(1 - self.silence_probability)
        with open(path, "w", encoding="utf8") as outf:
            if self.final_non_silence_correction is not None:
                outf.write(
                    f"{start_state}\t{non_silence_state}\t{silence_disambiguation_symbol}\t{self.silence_word}\t{initial_non_silence_cost}\n"
                )  # initial no silence

                outf.write(
                    f"{start_state}\t{silence_state}\t{self.optional_silence_phone}\t{self.silence_word}\t{initial_silence_cost}\n"
                )  # initial silence
            else:
                outf.write(
                    f"{start_state}\t{non_silence_state}\t<eps>\t<eps>\t{initial_non_silence_cost}\n"
                )  # initial no silence

                outf.write(
                    f"{start_state}\t{silence_state}\t<eps>\t<eps>\t{initial_silence_cost}\n"
                )  # initial silence

                if disambiguation:
                    sil_disambiguation_state = next_state
                    next_state += 1
                    outf.write(
                        f"{silence_state}\t{sil_disambiguation_state}\t{self.optional_silence_phone}\t{self.silence_word}\t0.0\n"
                    )
                    outf.write(
                        f"{sil_disambiguation_state}\t{non_silence_state}\t{silence_disambiguation_symbol}\t{self.silence_word}\t0.0\n"
                    )
                else:
                    outf.write(
                        f"{silence_state}\t{non_silence_state}\t{self.optional_silence_phone}\t{self.silence_word}\t0.0\n"
                    )
            silence_query = (
                session.query(Word.word)
                .filter(Word.word_type == WordType.silence)
                .filter(Word.dictionary_id == dictionary.id)
            )
            for (word,) in silence_query:
                if word == self.silence_word:
                    continue
                outf.write(
                    f"{non_silence_state}\t{non_silence_state}\t{self.optional_silence_phone}\t{word}\t0.0\n"
                )
                outf.write(
                    f"{start_state}\t{non_silence_state}\t{self.optional_silence_phone}\t{word}\t0.0\n"
                )  # initial silence
            columns = [Word.word, Pronunciation.pronunciation, Pronunciation.probability]
            if self.final_non_silence_correction is not None:
                columns.extend(
                    [
                        Pronunciation.silence_after_probability,
                        Pronunciation.silence_before_correction,
                        Pronunciation.non_silence_before_correction,
                    ]
                )
            if disambiguation:
                columns.append(Pronunciation.disambiguation)
            bn = DictBundle("pronunciation_data", *columns)
            pronunciation_query = (
                session.query(bn)
                .join(Pronunciation.word)
                .filter(Word.dictionary_id == dictionary.id)
                .filter(Word.word_type != WordType.silence)
            )
            for row in pronunciation_query:
                data = row.pronunciation_data
                phones = data["pronunciation"].split()
                probability = data["probability"]
                if self.final_non_silence_correction is not None:
                    silence_after_probability = data["silence_after_probability"]
                    silence_before_correction = data["silence_before_correction"]
                    non_silence_before_correction = data["non_silence_before_correction"]
                    silence_before_cost = 0.0
                    non_silence_before_cost = 0.0
                    if not silence_after_probability:
                        silence_after_probability = self.silence_probability

                    silence_following_cost = -math.log(silence_after_probability)
                    non_silence_following_cost = -math.log(1 - (silence_after_probability))
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
                        phones[0] += "_B"
                        phones[-1] += "_E"
                        for i in range(1, len(phones) - 1):
                            phones[i] += "_I"
                if probability == 0:
                    probability = 0.01  # Dithering to ensure low probability entries
                elif probability is None:
                    probability = 1
                pron_cost = abs(math.log(probability))
                if disambiguation and data["disambiguation"] is not None:
                    phones += [f"#{data['disambiguation']}"]

                if self.final_non_silence_correction is not None:
                    new_state = next_state
                    outf.write(
                        f"{non_silence_state}\t{new_state}\t{phones[0]}\t{data['word']}\t{pron_cost+non_silence_before_cost}\n"
                    )
                    outf.write(
                        f"{silence_state}\t{new_state}\t{phones[0]}\t{data['word']}\t{pron_cost+silence_before_cost}\n"
                    )

                    next_state += 1
                    current_state = new_state
                    for i in range(1, len(phones)):
                        new_state = next_state
                        next_state += 1
                        outf.write(f"{current_state}\t{new_state}\t{phones[i]}\t<eps>\n")
                        current_state = new_state
                    outf.write(
                        f"{current_state}\t{non_silence_state}\t{silence_disambiguation_symbol}\t<eps>\t{non_silence_following_cost}\n"
                    )
                    outf.write(
                        f"{current_state}\t{silence_state}\t{self.optional_silence_phone}\t<eps>\t{silence_following_cost}\n"
                    )
                else:
                    current_state = non_silence_state
                    for i in range(len(phones) - 1):
                        w_or_eps = data["word"] if i == 0 else "<eps>"
                        cost = pron_cost if i == 0 else 0.0
                        outf.write(
                            f"{current_state}\t{next_state}\t{phones[i]}\t{w_or_eps}\t{cost}\n"
                        )
                        current_state = next_state
                        next_state += 1

                    i = len(phones) - 1
                    p = phones[i] if i >= 0 else "<eps>"
                    w = data["word"] if i <= 0 else "<eps>"
                    sil_cost = silence_following_cost + (pron_cost if i <= 0 else 0.0)
                    non_sil_cost = non_silence_following_cost + (pron_cost if i <= 0 else 0.0)
                    outf.write(f"{current_state}\t{non_silence_state}\t{p}\t{w}\t{non_sil_cost}\n")
                    outf.write(f"{current_state}\t{silence_state}\t{p}\t{w}\t{sil_cost}\n")

            if self.final_non_silence_correction is not None:
                outf.write(f"{silence_state}\t{final_silence_cost}\n")
                outf.write(f"{non_silence_state}\t{final_non_silence_cost}\n")
            else:
                outf.write(f"{non_silence_state}\t0.0\n")

    def export_lexicon(
        self,
        dictionary_id: int,
        path: str,
        write_disambiguation: typing.Optional[bool] = False,
        probability: typing.Optional[bool] = False,
        silence_probabilities: typing.Optional[bool] = False,
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
        with open(path, "w", encoding="utf8") as f, self.session() as session:
            columns = [Word.word, Pronunciation.pronunciation]
            if write_disambiguation:
                columns.append(Pronunciation.disambiguation)
            if probability:
                columns.append(Pronunciation.probability)
                if silence_probabilities:
                    columns.append(Pronunciation.silence_after_probability)
                    columns.append(Pronunciation.silence_before_correction)
                    columns.append(Pronunciation.non_silence_before_correction)
            bn = DictBundle("pronunciation_data", *columns)
            pronunciations = (
                session.query(bn)
                .join(Pronunciation.word)
                .filter(Word.dictionary_id == dictionary_id)
            )
            for row in pronunciations:
                data = row.pronunciation_data
                phones = data["pronunciation"]
                if write_disambiguation and data["disambiguation"] is not None:
                    phones += f" #{data['disambiguation']}"
                probability_string = ""
                if probability:
                    probability_string = f"{data['probability']}"

                    if silence_probabilities:
                        extra_probs = [
                            data["silence_after_probability"],
                            data["silence_before_correction"],
                            data["non_silence_before_correction"],
                        ]
                        for x in extra_probs:
                            probability_string += f"\t{x if x else 0.0}"
                if probability:
                    f.write(f"{data['word']}\t{probability_string}\t{phones}\n")
                else:
                    f.write(f"{data['word']}\t{phones}\n")

    @property
    def phone_disambig_path(self):
        """Path to file containing phone symbols and their integer IDs"""
        return os.path.join(self.phones_dir, "phone_disambig.txt")

    def _write_fst_binary(
        self,
        dictionary: Dictionary,
        write_disambiguation: bool = False,
        path: typing.Optional[str] = None,
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
        dictionary: :class:`~montreal_forced_aligner.db.Dictionary`
            Dictionary object
        write_disambiguation: bool, optional
            Flag for including disambiguation symbols
        path: str, optional
            Full path to write compiled L.fst to
        """
        text_ext = ".text_fst"
        binary_ext = ".fst"
        word_disambig_path = os.path.join(dictionary.temp_directory, "word_disambig.txt")
        with open(word_disambig_path, "w") as f:
            f.write(str(self.word_mapping(dictionary.id)["#0"]))
        if write_disambiguation:
            text_ext = ".disambig_text_fst"
            binary_ext = ".disambig_fst"
        if path is not None:
            text_path = path.replace(".fst", text_ext)
            binary_path = path.replace(".fst", binary_ext)
        else:
            text_path = os.path.join(dictionary.temp_directory, "lexicon" + text_ext)
            binary_path = os.path.join(dictionary.temp_directory, "L" + binary_ext)

        words_file_path = os.path.join(dictionary.temp_directory, "words.txt")

        log_path = os.path.join(dictionary.temp_directory, os.path.basename(binary_path) + ".log")
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
                        self.phone_disambig_path,
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

    def actual_words(
        self, session: sqlalchemy.orm.session.Session, dictionary_id=None
    ) -> typing.Dict[str, Word]:
        """Words in the dictionary stripping out Kaldi's internal words"""
        words = (
            session.query(Word)
            .options(selectinload(Word.pronunciations))
            .filter(Word.word_type == WordType.speech)
        )
        if dictionary_id is not None:
            words = words.filter(Word.dictionary_id == dictionary_id)
        return {x.word: x for x in words}

    @property
    def phone_mapping(self) -> Dict[str, int]:
        """Mapping of phone symbols to integer IDs for Kaldi processing"""
        if self._phone_mapping is None:
            with self.session() as session:
                phones = session.query(Phone.phone, Phone.mapping_id).order_by(Phone.id).all()
                self._phone_mapping = {x[0]: x[1] for x in phones}
        return self._phone_mapping

    def _write_phone_symbol_table(self) -> None:
        """
        Write the phone mapping to the temporary directory
        """
        with open(self.phone_symbol_table_path, "w", encoding="utf8") as f:
            for p, i in self.phone_mapping.items():
                f.write(f"{p} {i}\n")

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

    @property
    def name(self) -> str:
        """Name of the dictionary"""
        return self.dictionary_model.name

    def save_oovs_found(self, directory: str) -> None:
        """
        Save all out of vocabulary items to a file in the specified directory

        Parameters
        ----------
        directory : str
            Path to directory to save ``oovs_found.txt``
        """

        with self.session() as session:
            dictionaries = (
                session.query(Dictionary).options(selectinload(Dictionary.oov_words)).all()
            )
            for dictionary in dictionaries:
                with open(
                    os.path.join(directory, f"oovs_found_{dictionary.name}_{dictionary.id}.txt"),
                    "w",
                    encoding="utf8",
                ) as f, open(
                    os.path.join(directory, f"oov_counts_{dictionary.name}_{dictionary.id}.txt"),
                    "w",
                    encoding="utf8",
                ) as cf:
                    for oov in dictionary.oov_words:
                        f.write(oov.word + "\n")
                        cf.write(f"{oov.word}\t{oov.count}\n")

    def calculate_oovs_found(self) -> None:
        """Sum the counts of oovs found in pronunciation dictionaries"""

        with self.session() as session:
            dictionaries = (
                session.query(Dictionary).options(selectinload(Dictionary.oov_words)).all()
            )
            oov_words = {}
            for dictionary in dictionaries:
                dict_id = dictionary.id
                oov_words = {}
                utterances = (
                    session.query(Utterance.oovs)
                    .join(Utterance.speaker)
                    .join(Speaker.dictionary)
                    .filter(Speaker.dictionary_id == dict_id)
                    .filter(Utterance.oovs != "")
                )
                for u in utterances:
                    for w in u.oovs.split():
                        if w not in oov_words:
                            oov_words[w] = {"word": w, "count": 0, "dictionary_id": dict_id}
                        oov_words[w]["count"] += 1
            session.bulk_insert_mappings(OovWord, oov_words.values())
            session.commit()
        self.save_oovs_found(self.output_directory)

    def write_lexicon_information(self, write_disambiguation: Optional[bool] = False) -> None:
        """
        Write all child dictionaries to the temporary directory

        Parameters
        ----------
        write_disambiguation: bool, optional
            Flag to use disambiguation symbols in the output
        """
        os.makedirs(self.phones_dir, exist_ok=True)
        phone_disambig_path = os.path.join(self.phones_dir, "phone_disambig.txt")
        with open(phone_disambig_path, "w") as f:
            f.write(str(self.phone_mapping["#0"]))
        self._write_word_boundaries()
        self._write_phone_symbol_table()
        self._write_disambig()
        silence_disambiguation_symbol = None
        if write_disambiguation:
            silence_disambiguation_symbol = self.silence_disambiguation_symbol

        debug = getattr(self, "debug", False)
        with self.session() as session:
            dictionaries = session.query(Dictionary)
            for d in dictionaries:
                os.makedirs(d.temp_directory, exist_ok=True)
                if debug:
                    self.export_lexicon(d.id, os.path.join(d.temp_directory, "lexicon.txt"))
                self._write_word_file(d)
                self._write_probabilistic_fst_text(session, d, silence_disambiguation_symbol)
                self._write_fst_binary(
                    d, write_disambiguation=silence_disambiguation_symbol is not None
                )
                if not debug:
                    if os.path.exists(os.path.join(d.temp_directory, "temp.fst")):
                        os.remove(os.path.join(d.temp_directory, "temp.fst"))
                    if os.path.exists(os.path.join(d.temp_directory, "lexicon.text.fst")):
                        os.remove(os.path.join(d.temp_directory, "lexicon.text.fst"))

    def write_training_information(self) -> None:
        """Write phone information needed for training"""
        self._write_topo()
        self._write_phone_sets()
        self._write_extra_questions()

    def _write_word_file(self, dictionary: Dictionary) -> None:
        """
        Write the word mapping to the temporary directory
        """
        if sys.platform == "win32":
            newline = ""
        else:
            newline = None
        with open(dictionary.words_symbol_path, "w", encoding="utf8", newline=newline) as f:
            for w, i in self.word_mapping(dictionary.id).items():
                f.write(f"{w} {i}\n")


class MultispeakerDictionary(MultispeakerDictionaryMixin):
    """
    Class for processing multi- and single-speaker pronunciation dictionaries

    See Also
    --------
    :class:`~montreal_forced_aligner.dictionary.multispeaker.MultispeakerDictionaryMixin`
        For dictionary parsing parameters
    """

    @property
    def data_source_identifier(self) -> str:
        """Name of the dictionary"""
        return f"{self.name}"

    @property
    def identifier(self) -> str:
        """Name of the dictionary"""
        return f"{self.data_source_identifier}"

    @property
    def output_directory(self) -> str:
        """Root temporary directory to store all dictionary information"""
        return os.path.join(self.temporary_directory, self.identifier)
