"""Pronunciation dictionaries for use in alignment and transcription"""

from __future__ import annotations

import logging
import os
from collections import Counter
from typing import TYPE_CHECKING, Collection, Dict, Optional, Union

from ..abc import Dictionary
from ..config.dictionary_config import DictionaryConfig
from ..models import DictionaryModel
from .base_dictionary import PronunciationDictionary

if TYPE_CHECKING:

    from ..corpus.classes import Speaker


__all__ = [
    "MultispeakerDictionary",
]


class MultispeakerDictionary(Dictionary):
    """
    Class containing information about a pronunciation dictionary with different dictionaries per speaker

    Parameters
    ----------
    dictionary_model : DictionaryModel
        Multispeaker dictionary
    output_directory : str
        Path to a directory to store files for Kaldi
    config: DictionaryConfig, optional
        Configuration for generating lexicons
    word_set : Collection[str], optional
        Word set to limit output files
    logger: :class:`~logging.Logger`, optional
        Logger to output information to
    """

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

        self.speaker_mapping = {}
        self.dictionary_mapping = {}

        for speaker, dictionary in self.dictionary_model.load_dictionary_paths().items():
            self.speaker_mapping[speaker] = dictionary.name
            if dictionary.name not in self.dictionary_mapping:
                self.dictionary_mapping[dictionary.name] = PronunciationDictionary(
                    dictionary,
                    self.output_directory,
                    config,
                    word_set=word_set,
                    logger=self.logger,
                )

    @property
    def phones_dir(self):
        return self.get_dictionary("default").phones_dir

    @property
    def topo_path(self):
        return os.path.join(self.get_dictionary("default").output_directory, "topo")

    @property
    def oovs_found(self) -> Counter[str, int]:
        oovs = Counter()
        for dictionary in self.dictionary_mapping.values():
            oovs.update(dictionary.oovs_found)
        return oovs

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
    def silences(self) -> set:
        """
        Set of silence phones
        """
        return self.config.silence_phones

    @property
    def default_dictionary(self) -> PronunciationDictionary:
        """Default PronunciationDictionary"""
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
            PronunciationDictionary name for the speaker
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
            PronunciationDictionary for the speaker
        """
        return self.dictionary_mapping[self.get_dictionary_name(speaker)]

    def write(self, write_disambiguation: Optional[bool] = False) -> None:
        """
        Write all child dictionaries to the temporary directory

        Parameters
        ----------
        write_disambiguation: bool, optional
            Flag to use disambiguation symbols in the output
        """
        for d in self.dictionary_mapping.values():
            d.write(write_disambiguation)

    def set_word_set(self, word_set: Collection[str]) -> None:
        """
        Limit output to a subset of overall words

        Parameters
        ----------
        word_set: Collection[str]
            Word set to limit generated files to
        """
        for d in self.dictionary_mapping.values():
            d.set_word_set(word_set)

    @property
    def output_paths(self) -> Dict[str, str]:
        """
        Mapping of output directory for child dictionaries
        """
        return {d.name: d.output_directory for d in self.dictionary_mapping.values()}
