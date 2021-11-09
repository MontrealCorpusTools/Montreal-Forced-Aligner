"""Class definitions for Speaker classification in MFA"""
from __future__ import annotations

import logging
import os
import shutil
from typing import TYPE_CHECKING, Optional

import numpy as np

from .config import TEMP_DIR
from .corpus.classes import Speaker
from .exceptions import KaldiProcessingError
from .helper import load_scp
from .multiprocessing import extract_ivectors
from .utils import log_kaldi_errors

if TYPE_CHECKING:
    from .config import SpeakerClassificationConfig
    from .corpus import TranscribeCorpus
    from .models import IvectorExtractor, MetaDict

__all__ = ["SpeakerClassifier"]


class SpeakerClassifier:
    """
    Class for performing speaker classification

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.base.Corpus`
        Corpus object for the dataset
    ivector_extractor : :class:`~montreal_forced_aligner.models.IvectorExtractor`
        Configuration for alignment
    classification_config : :class:`~montreal_forced_aligner.config.speaker_classification_config.SpeakerClassificationConfig`
        Configuration for alignment
    compute_segments: bool, optional
        Flag for whether segments should be created
    num_speakers: int, optional
        Number of speakers in the corpus, if known
    cluster: bool, optional
        Flag for whether speakers should be clustered instead of classified
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    call_back : callable, optional
        Specifies a call back function for diarization
    debug : bool
        Flag for running in debug mode, defaults to false
    verbose : bool
        Flag for running in verbose mode, defaults to false
    """

    def __init__(
        self,
        corpus: TranscribeCorpus,
        ivector_extractor: IvectorExtractor,
        classification_config: SpeakerClassificationConfig,
        compute_segments: Optional[bool] = False,
        num_speakers: Optional[int] = None,
        cluster: Optional[bool] = False,
        temp_directory: Optional[str] = None,
        debug: Optional[bool] = False,
        verbose: Optional[bool] = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.corpus = corpus
        self.ivector_extractor = ivector_extractor
        self.feature_config = self.ivector_extractor.feature_config
        self.classification_config = classification_config

        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = temp_directory
        os.makedirs(self.temp_directory, exist_ok=True)
        self.debug = debug
        self.compute_segments = compute_segments
        self.verbose = verbose
        if logger is None:
            self.log_file = os.path.join(self.temp_directory, "speaker_classifier.log")
            self.logger = logging.getLogger("speaker_classifier")
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_file, "w", "utf-8")
            handler.setFormatter = logging.Formatter("%(name)s %(message)s")
            self.logger.addHandler(handler)
        else:
            self.logger = logger
        self.classifier = None
        self.speaker_labels = {}
        self.ivectors = {}
        self.num_speakers = num_speakers
        self.cluster = cluster
        self.uses_voiced = False
        self.uses_cmvn = True
        self.uses_splices = False
        self.setup()

    @property
    def classify_directory(self) -> str:
        """Temporary directory for speaker classification"""
        return os.path.join(self.temp_directory, "speaker_classification")

    @property
    def data_directory(self) -> str:
        """Corpus data directory"""
        return self.corpus.split_directory

    @property
    def working_directory(self) -> str:
        """Current working directory for the speaker classifier"""
        return self.classify_directory

    @property
    def ie_path(self) -> str:
        """Path for the IvectorExtractor model file"""
        return os.path.join(self.working_directory, "final.ie")

    @property
    def speaker_classification_model_path(self) -> str:
        """Path for the speaker classification model"""
        return os.path.join(self.working_directory, "speaker_classifier.mdl")

    @property
    def speaker_labels_path(self) -> str:
        """Path for the speaker labels file"""
        return os.path.join(self.working_directory, "speaker_labels.txt")

    @property
    def model_path(self) -> str:
        """Path for the acoustic model file"""
        return os.path.join(self.working_directory, "final.mdl")

    @property
    def dubm_path(self) -> str:
        """Path for the DUBM model"""
        return os.path.join(self.working_directory, "final.dubm")

    @property
    def working_log_directory(self) -> str:
        """Current log directory"""
        return self.log_directory

    @property
    def ivector_options(self) -> MetaDict:
        """Ivector configuration options"""
        data = self.ivector_extractor.meta
        data["silence_weight"] = 0.0
        data["posterior_scale"] = 0.1
        data["max_count"] = 100
        data["sil_phones"] = None
        return data

    @property
    def log_directory(self) -> str:
        """Log directory"""
        return os.path.join(self.classify_directory, "log")

    @property
    def use_mp(self) -> bool:
        """Flag for whether to use multiprocessing"""
        return self.classification_config.use_mp

    def setup(self) -> None:
        """
        Sets up the corpus and speaker classifier

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        done_path = os.path.join(self.classify_directory, "done")
        if os.path.exists(done_path):
            self.logger.info("Classification already done, skipping initialization.")
            return
        dirty_path = os.path.join(self.classify_directory, "dirty")
        if os.path.exists(dirty_path):
            shutil.rmtree(self.classify_directory)
        log_dir = os.path.join(self.classify_directory, "log")
        os.makedirs(log_dir, exist_ok=True)
        self.ivector_extractor.export_model(self.classify_directory)
        try:
            self.corpus.initialize_corpus(None, self.feature_config)
            extract_ivectors(self)
        except Exception as e:
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise

    def load_ivectors(self) -> None:
        """
        Load ivectors from the temporary directory
        """
        self.ivectors = {}
        for j in self.corpus.jobs:
            ivectors_args = j.extract_ivector_arguments(self)
            for ivectors_path in ivectors_args.ivector_paths.values():
                ivec = load_scp(ivectors_path)
                for utt, ivector in ivec.items():
                    ivector = [float(x) for x in ivector]
                    self.ivectors[utt] = ivector

    def cluster_utterances(self) -> None:
        """
        Cluster utterances based on their ivectors
        """
        from sklearn.cluster import KMeans

        if not self.ivectors:
            self.load_ivectors()
        x = []
        for v in self.ivectors.values():
            x.append(v)
        x = np.array(x)
        clust = KMeans(self.num_speakers).fit(x)
        y = clust.labels_
        for i, u in enumerate(self.ivectors.keys()):
            speaker_name = y[i]
            utterance = self.corpus.utterances[u]
            if speaker_name not in self.corpus.speakers:
                self.corpus.speakers[speaker_name] = Speaker(speaker_name)
            utterance.set_speaker(self.corpus.speakers[speaker_name])

    def export_classification(self, output_directory: str) -> None:
        """
        Export files with their new speaker labels

        Parameters
        ----------
        output_directory: str
            Output directory to save files
        """
        backup_output_directory = None
        if not self.classification_config.overwrite:
            backup_output_directory = os.path.join(self.classify_directory, "output")
            os.makedirs(backup_output_directory, exist_ok=True)

        for file in self.corpus.files.values():
            file.save(output_directory, backup_output_directory)
