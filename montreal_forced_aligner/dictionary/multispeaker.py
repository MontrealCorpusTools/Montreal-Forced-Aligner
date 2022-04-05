"""Pronunciation dictionaries for use in alignment and transcription"""

from __future__ import annotations

import abc
import os
from typing import TYPE_CHECKING, Collection, Dict, Optional, Tuple

from sqlalchemy.orm import Session, joinedload, load_only

from montreal_forced_aligner.corpus.db import Dictionary, Speaker, Utterance
from montreal_forced_aligner.dictionary.mixins import (
    SanitizeFunction,
    SplitWordsFunction,
    TemporaryDictionaryMixin,
)
from montreal_forced_aligner.dictionary.pronunciation import PronunciationDictionary
from montreal_forced_aligner.exceptions import DictionaryError
from montreal_forced_aligner.models import DictionaryModel, PhoneSetType

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
        Mapping of dictionary names to functions for splitting compounds and clitics into separate words
    """

    speaker_mapping: Dict[str, str]
    sanitize_function: SanitizeFunction
    split_functions: Dict[str, SplitWordsFunction]

    def get_dict_name_for_speaker(self, speaker_name: str) -> str:
        """
        Get the dictionary name of the speaker

        Parameters
        ----------
        speaker_name: str
            Speaker to look up

        Returns
        -------
        str
            Dictionary name
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
            dict_name = self.get_dict_name_for_speaker(speaker_name)
            split_function = self.split_functions[dict_name]
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
    speaker_mapping: dict[str, str]
        Mapping of speaker names to dictionary names
    dictionary_mapping: dict[str, :class:`~montreal_forced_aligner.dictionary.pronunciation.PronunciationDictionary`]
        Mapping of dictionary names to pronunciation dictionary
    """

    def __init__(self, dictionary_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.dictionary_model = DictionaryModel(
            dictionary_path, phone_set_type=self.phone_set_type
        )
        self.speaker_mapping = {}
        self.dictionary_mapping: Dict[str, PronunciationDictionary] = {}
        self.dictionary_ids = {}

    @property
    def num_dictionaries(self) -> int:
        """Number of pronunciation dictionaries"""
        return len(self.dictionary_mapping)

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
        for dictionary_name, dictionary in self.dictionary_mapping.items():
            split_functions[dictionary_name] = SplitWordsFunction(
                self.clitic_markers,
                self.compound_markers,
                dictionary.clitic_set,
                self.brackets,
                dictionary.words_mapping,
                self.specials_set,
                dictionary.oov_word,
                dictionary.bracketed_word,
            )
        return MultispeakerSanitizationFunction(
            self.speaker_mapping, sanitize_function, split_functions
        )

    @property
    def lexicon_fst_paths(self):
        """Path to the file containing phone disambiguation symbols"""
        if not self._lexicon_fst_paths:
            self._lexicon_fst_paths = {}
            for dict_name, dictionary in self.dictionary_mapping.items():
                self._lexicon_fst_paths[dict_name] = dictionary.lexicon_fst_path
        return self._lexicon_fst_paths

    def dictionary_setup(self):
        """Set up the dictionary for processing"""
        auto_set = {PhoneSetType.AUTO, PhoneSetType.UNKNOWN, "AUTO", "UNKNOWN"}
        if not isinstance(self.phone_set_type, PhoneSetType):
            self.phone_set_type = PhoneSetType[self.phone_set_type]

        options = self.dictionary_options
        pretrained = False
        if self.non_silence_phones:
            pretrained = True

        for speaker, dictionary in self.dictionary_model.load_dictionary_paths().items():
            self.speaker_mapping[speaker] = dictionary.name
            if dictionary.name not in self.dictionary_mapping:
                if not pretrained:
                    options["non_silence_phones"] = set()
                self.dictionary_mapping[dictionary.name] = PronunciationDictionary(
                    dictionary_path=dictionary.path,
                    temporary_directory=self.dictionary_output_directory,
                    root_dictionary=self,
                    **options,
                )
                if self.phone_set_type not in auto_set:
                    if (
                        self.phone_set_type
                        != self.dictionary_mapping[dictionary.name].phone_set_type
                    ):
                        raise DictionaryError(
                            f"Mismatch found in phone sets: {self.phone_set_type} vs {self.dictionary_mapping[dictionary.name].phone_set_type}"
                        )
                else:
                    self.phone_set_type = self.dictionary_mapping[dictionary.name].phone_set_type

                self.excluded_phones.update(
                    self.dictionary_mapping[dictionary.name].excluded_phones
                )
                self.excluded_pronunciation_count += self.dictionary_mapping[
                    dictionary.name
                ].excluded_pronunciation_count
        for dictionary in self.dictionary_mapping.values():
            self.non_silence_phones.update(dictionary.non_silence_phones)
        for dictionary in self.dictionary_mapping.values():
            dictionary.non_silence_phones = self.non_silence_phones
        if hasattr(self, "db_path"):
            from montreal_forced_aligner.corpus.db import Dictionary, Phone, Word

            with Session(self.db_engine) as session:
                session.query(Phone).delete()
                session.flush()
                for phone, id in self.phone_mapping.items():
                    session.merge(Phone(id=id, phone=phone))
                for dict_name, dictionary in self.dictionary_mapping.items():
                    d = session.query(Dictionary).filter(Dictionary.name == dict_name).first()
                    insert = False
                    if d is None:
                        d = Dictionary(name=dict_name)
                        session.add(d)
                        insert = True
                    d.phones_directory = self.phones_dir
                    d.lexicon_fst_path = dictionary.lexicon_fst_path
                    d.lexicon_disambig_fst_path = dictionary.lexicon_disambig_fst_path
                    d.words_path = dictionary.words_symbol_path
                    d.word_boundary_int_path = self.word_boundary_int_path
                    if self.clitic_markers:
                        d.clitic_marker = self.clitic_markers[0]
                    d.silence_word = self.silence_word
                    d.optional_silence_phone = self.optional_silence_phone
                    d.oov_word = self.oov_word
                    d.position_dependent_phones = self.position_dependent_phones
                    d.bracketed_word = dictionary.bracketed_word
                    session.flush()
                    dictionary.generate_mappings()
                    word_mapping = []
                    self.dictionary_ids[d.name] = d.id
                    if insert:
                        for word, word_id in dictionary.words_mapping.items():
                            pronunciations = []
                            if word not in dictionary.words:
                                continue
                            for pron in dictionary.words[word]:
                                pronunciations.append(" ".join(pron.pronunciation))
                            word_mapping.append(
                                {
                                    "id": word_id,
                                    "word": word,
                                    "dictionary_id": d.id,
                                    "pronunciations": ";".join(pronunciations),
                                }
                            )
                        session.bulk_insert_mappings(Word, word_mapping)
                session.commit()

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

        for dict_name, dictionary in self.dictionary_mapping.items():
            with open(
                os.path.join(directory, f"oovs_found_{dict_name}.txt"), "w", encoding="utf8"
            ) as f, open(
                os.path.join(directory, f"oov_counts_{dict_name}.txt"), "w", encoding="utf8"
            ) as cf:
                for oov in sorted(
                    dictionary.oovs_found.keys(), key=lambda x: (-dictionary.oovs_found[x], x)
                ):
                    f.write(oov + "\n")
                    cf.write(f"{oov}\t{dictionary.oovs_found[oov]}\n")

    def calculate_oovs_found(self) -> None:
        """Sum the counts of oovs found in pronunciation dictionaries"""

        if hasattr(self, "session"):
            with self.session() as session:
                utterances = (
                    session.query(Utterance)
                    .join(Utterance.speaker)
                    .join(Speaker.dictionary)
                    .options(
                        joinedload(Utterance.speaker, innerjoin=True)
                        .joinedload(Speaker.dictionary, innerjoin=True)
                        .load_only(Dictionary.name),
                        load_only(Utterance.oovs),
                    )
                )
                for u in utterances:
                    dict_name = u.speaker.dictionary.name
                    self.dictionary_mapping[dict_name].oovs_found.update(u.oovs.split())
        for dictionary in self.dictionary_mapping.values():
            self.oovs_found.update(dictionary.oovs_found)
        self.save_oovs_found(self.output_directory)

    @property
    def default_dictionary(self) -> PronunciationDictionary:
        """Default pronunciation dictionary"""
        return self.get_dictionary("default")

    def get_dictionary_name(self, speaker_name: str) -> str:
        """
        Get the dictionary name for a given speaker

        Parameters
        ----------
        speaker_name: str
            Speaker to look up

        Returns
        -------
        str
            Dictionary name for the speaker
        """

        if speaker_name not in self.speaker_mapping:
            return self.speaker_mapping["default"]
        return self.speaker_mapping[speaker_name]

    def get_dictionary(self, speaker_name: str) -> PronunciationDictionary:
        """
        Get a dictionary for a given speaker

        Parameters
        ----------
        speaker_name: str
            Speaker to look up

        Returns
        -------
        :class:`~montreal_forced_aligner.dictionary.PronunciationDictionary`
            Pronunciation dictionary for the speaker
        """
        return self.dictionary_mapping[self.get_dictionary_name(speaker_name)]

    def write_lexicon_information(self, write_disambiguation: Optional[bool] = False) -> None:
        """
        Write all child dictionaries to the temporary directory

        Parameters
        ----------
        write_disambiguation: bool, optional
            Flag to use disambiguation symbols in the output
        """
        os.makedirs(self.phones_dir, exist_ok=True)
        for d in self.dictionary_mapping.values():
            d.generate_mappings()
            if d.max_disambiguation_symbol > self.max_disambiguation_symbol:
                self.max_disambiguation_symbol = d.max_disambiguation_symbol
        self._write_word_boundaries()
        self._write_phone_symbol_table()
        self._write_disambig()
        for d in self.dictionary_mapping.values():
            d.write(write_disambiguation, debug=getattr(self, "debug", False))

    def set_lexicon_word_set(self, word_set: Collection[str]) -> None:
        """
        Limit output to a subset of overall words

        Parameters
        ----------
        word_set: Collection[str]
            Word set to limit generated files to
        """
        for d in self.dictionary_mapping.values():
            d.set_lexicon_word_set(word_set)

    @property
    def output_paths(self) -> Dict[str, str]:
        """
        Mapping of output directory for child directories
        """
        return {d.name: d.dictionary_output_directory for d in self.dictionary_mapping.values()}


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
