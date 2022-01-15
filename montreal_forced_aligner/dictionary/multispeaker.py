"""Pronunciation dictionaries for use in alignment and transcription"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Collection, Dict, Optional, Tuple, Union

from montreal_forced_aligner.dictionary.mixins import (
    SanitizeFunction,
    SplitWordsFunction,
    TemporaryDictionaryMixin,
)
from montreal_forced_aligner.dictionary.pronunciation import PronunciationDictionary
from montreal_forced_aligner.exceptions import DictionaryError
from montreal_forced_aligner.models import DictionaryModel, PhoneSetType

if TYPE_CHECKING:
    from montreal_forced_aligner.corpus.classes import Speaker


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

    def get_dict_name_for_speaker(self, speaker_name):
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

    @property
    def sanitize_function(self) -> MultispeakerSanitizationFunction:
        """Sanitization function for the dictionary"""
        sanitize_function = SanitizeFunction(
            self.punctuation,
            self.clitic_markers,
            self.compound_markers,
            self.brackets,
            self.ignore_case,
        )
        split_functions = {}
        for dictionary_name, dictionary in self.dictionary_mapping.items():
            split_functions[dictionary_name] = SplitWordsFunction(
                self.clitic_markers,
                self.compound_markers,
                dictionary.clitic_set,
                set(dictionary.actual_words.keys()),
            )
        return MultispeakerSanitizationFunction(
            self.speaker_mapping, sanitize_function, split_functions
        )

    def dictionary_setup(self):
        """Setup the dictionary for processing"""
        auto_set = {PhoneSetType.AUTO, PhoneSetType.UNKNOWN, "AUTO", "UNKNOWN"}
        if not isinstance(self.phone_set_type, PhoneSetType):
            self.phone_set_type = PhoneSetType[self.phone_set_type]
        for speaker, dictionary in self.dictionary_model.load_dictionary_paths().items():
            self.speaker_mapping[speaker] = dictionary.name
            if dictionary.name not in self.dictionary_mapping:
                self.dictionary_mapping[dictionary.name] = PronunciationDictionary(
                    dictionary_path=dictionary.path,
                    temporary_directory=self.dictionary_output_directory,
                    root_dictionary=self,
                    **self.dictionary_options,
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

                self.non_silence_phones.update(
                    self.dictionary_mapping[dictionary.name].non_silence_phones
                )
                self.excluded_phones.update(
                    self.dictionary_mapping[dictionary.name].excluded_phones
                )
                self.excluded_pronunciation_count += self.dictionary_mapping[
                    dictionary.name
                ].excluded_pronunciation_count
        for dictionary in self.dictionary_mapping.values():
            dictionary.non_silence_phones = self.non_silence_phones

    @property
    def name(self) -> str:
        """Name of the dictionary"""
        return self.dictionary_model.name

    def calculate_oovs_found(self) -> None:
        """Sum the counts of oovs found in pronunciation dictionaries"""
        for dictionary in self.dictionary_mapping.values():
            self.oovs_found.update(dictionary.oovs_found)
        self.save_oovs_found(self.output_directory)

    @property
    def default_dictionary(self) -> PronunciationDictionary:
        """Default pronunciation dictionary"""
        return self.get_dictionary("default")

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

    def get_dictionary(self, speaker: Union[Speaker, str]) -> PronunciationDictionary:
        """
        Get a dictionary for a given speaker

        Parameters
        ----------
        speaker: Union[Speaker, str]
            Speaker to look up

        Returns
        -------
        :class:`~montreal_forced_aligner.dictionary.PronunciationDictionary`
            Pronunciation dictionary for the speaker
        """
        return self.dictionary_mapping[self.get_dictionary_name(speaker)]

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
        self._write_phone_sets()
        self._write_phone_symbol_table()
        self._write_disambig()
        self._write_topo()
        self._write_extra_questions()
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
