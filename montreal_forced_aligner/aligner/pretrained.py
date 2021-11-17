"""Class definitions for aligning with pretrained acoustic models"""
from __future__ import annotations

import os
from collections import Counter
from typing import TYPE_CHECKING, Optional

from ..multiprocessing import generate_pronunciations
from .base import BaseAligner

if TYPE_CHECKING:
    from logging import Logger

    from ..config import AlignConfig
    from ..corpus import Corpus
    from ..dictionary import MultispeakerDictionary
    from ..models import AcousticModel

__all__ = ["PretrainedAligner"]


class PretrainedAligner(BaseAligner):
    """
    Class for aligning a dataset using a pretrained acoustic model

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.Corpus`
        Corpus object for the dataset
    dictionary : :class:`~montreal_forced_aligner.dictionary.MultispeakerDictionary`
        Dictionary object for the pronunciation dictionary
    acoustic_model : :class:`~montreal_forced_aligner.models.AcousticModel`
        Archive containing the acoustic model and pronunciation dictionary
    align_config : :class:`~montreal_forced_aligner.config.AlignConfig`
        Configuration for alignment
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    debug: bool
        Flag for debug mode, default is False
    verbose: bool
        Flag for verbose mode, default is False
    logger: :class:`~logging.Logger`
        Logger to use
    """

    def __init__(
        self,
        corpus: Corpus,
        dictionary: MultispeakerDictionary,
        acoustic_model: AcousticModel,
        align_config: AlignConfig,
        temp_directory: Optional[str] = None,
        debug: bool = False,
        verbose: bool = False,
        logger: Optional[Logger] = None,
    ):
        super().__init__(
            corpus,
            dictionary,
            align_config,
            temp_directory,
            debug,
            verbose,
            logger,
            acoustic_model=acoustic_model,
        )
        self.data_directory = corpus.split_directory
        log_dir = os.path.join(self.align_directory, "log")
        os.makedirs(log_dir, exist_ok=True)
        self.align_config.logger = self.logger
        self.logger.info("Done with setup!")

    @property
    def model_directory(self) -> str:
        """Model directory"""
        return os.path.join(self.temp_directory, "model")

    def setup(self) -> None:
        """Set up aligner"""
        self.dictionary.config.non_silence_phones = self.acoustic_model.meta["phones"]
        super(PretrainedAligner, self).setup()
        self.acoustic_model.export_model(self.align_directory)

    @property
    def ali_paths(self):
        """Alignment archive paths"""
        jobs = [x.align_arguments(self) for x in self.corpus.jobs]
        ali_paths = []
        for j in jobs:
            ali_paths.extend(j.ali_paths.values())
        return ali_paths

    def generate_pronunciations(
        self, output_path: str, calculate_silence_probs: bool = False, min_count: int = 1
    ) -> None:
        """
        Generate pronunciation probabilities for the dictionary

        Parameters
        ----------
        output_path: str
            Path to save new dictionary
        calculate_silence_probs: bool
            Flag for whether to calculate silence probabilities, default is False
        min_count: int
            Specifies the minimum count of words to include in derived probabilities, default is 1
        """
        pron_counts, utt_mapping = generate_pronunciations(self)
        for dict_name, dictionary in self.dictionary.dictionary_mapping.items():
            counts = pron_counts[dict_name]
            mapping = utt_mapping[dict_name]
            if calculate_silence_probs:
                sil_before_counts = Counter()
                nonsil_before_counts = Counter()
                sil_after_counts = Counter()
                nonsil_after_counts = Counter()
                sils = ["<s>", "</s>", "<eps>"]
                for v in mapping.values():
                    for i, w in enumerate(v):
                        if w in sils:
                            continue
                        prev_w = v[i - 1]
                        next_w = v[i + 1]
                        if prev_w in sils:
                            sil_before_counts[w] += 1
                        else:
                            nonsil_before_counts[w] += 1
                        if next_w in sils:
                            sil_after_counts[w] += 1
                        else:
                            nonsil_after_counts[w] += 1

            dictionary.pronunciation_probabilities = True
            for word, prons in dictionary.words.items():
                if word not in counts:
                    for p in prons:
                        p["probability"] = 1
                else:
                    total = 0
                    best_pron = 0
                    best_count = 0
                    for p in prons:
                        p["probability"] = min_count
                        if p["pronunciation"] in counts[word]:
                            p["probability"] += counts[word][p["pronunciation"]]
                        total += p["probability"]
                        if p["probability"] > best_count:
                            best_pron = p["pronunciation"]
                            best_count = p["probability"]
                    for p in prons:
                        if p["pronunciation"] == best_pron:
                            p["probability"] = 1
                        else:
                            p["probability"] /= total
                    dictionary.words[word] = prons
            dictionary.export_lexicon(output_path, probability=True)
