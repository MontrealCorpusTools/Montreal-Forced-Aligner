"""
Speaker classification
======================


"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

import yaml

from .abc import FileExporterMixin, TopLevelMfaWorker
from .corpus.ivector_corpus import IvectorCorpusMixin
from .exceptions import KaldiProcessingError
from .helper import load_scp
from .models import IvectorExtractorModel
from .utils import log_kaldi_errors

if TYPE_CHECKING:
    from argparse import Namespace

    from .abc import MetaDict
__all__ = ["SpeakerClassifier"]


class SpeakerClassifier(
    IvectorCorpusMixin, TopLevelMfaWorker, FileExporterMixin
):  # pragma: no cover
    """
    Class for performing speaker classification, not currently very functional, but
    is planned to be expanded in the future

    Parameters
    ----------
    ivector_extractor_path : str
        Path to ivector extractor model
    num_speakers: int, optional
        Number of speakers in the corpus, if known
    cluster: bool, optional
        Flag for whether speakers should be clustered instead of classified
    """

    def __init__(
        self, ivector_extractor_path: str, num_speakers: int = 0, cluster: bool = True, **kwargs
    ):
        self.ivector_extractor = IvectorExtractorModel(ivector_extractor_path)
        kwargs.update(self.ivector_extractor.parameters)
        super().__init__(**kwargs)
        self.classifier = None
        self.speaker_labels = {}
        self.ivectors = {}
        self.num_speakers = num_speakers
        self.cluster = cluster

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[str] = None,
        args: Optional[Namespace] = None,
        unknown_args: Optional[List[str]] = None,
    ) -> MetaDict:
        """
        Parse parameters for speaker classification from a config path or command-line arguments

        Parameters
        ----------
        config_path: str
            Config path
        args: :class:`~argparse.Namespace`
            Command-line arguments from argparse
        unknown_args: list[str], optional
            Extra command-line arguments

        Returns
        -------
        dict[str, Any]
            Configuration parameters
        """
        global_params = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf8") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
                for k, v in data.items():
                    if k == "features":
                        if "type" in v:
                            v["feature_type"] = v["type"]
                            del v["type"]
                        global_params.update(v)
                    else:
                        if v is None and k in {
                            "punctuation",
                            "compound_markers",
                            "clitic_markers",
                        }:
                            v = []
                        global_params[k] = v
        global_params.update(cls.parse_args(args, unknown_args))
        return global_params

    @property
    def workflow_identifier(self) -> str:
        """Speaker classification identifier"""
        return "speaker_classification"

    @property
    def ie_path(self) -> str:
        """Path for the ivector extractor model file"""
        return os.path.join(self.working_directory, "final.ie")

    @property
    def model_path(self) -> str:
        """Path for the acoustic model file"""
        return os.path.join(self.working_directory, "final.mdl")

    @property
    def dubm_path(self) -> str:
        """Path for the DUBM model"""
        return os.path.join(self.working_directory, "final.dubm")

    def setup(self) -> None:
        """
        Sets up the corpus and speaker classifier

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """

        self.check_previous_run()
        done_path = os.path.join(self.working_directory, "done")
        if os.path.exists(done_path):
            self.logger.info("Classification already done, skipping initialization.")
            return
        log_dir = os.path.join(self.working_directory, "log")
        os.makedirs(log_dir, exist_ok=True)
        try:
            self.load_corpus()
            self.ivector_extractor.export_model(self.working_directory)
            self.extract_ivectors()
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise

    def load_ivectors(self) -> None:
        """
        Load ivectors from the temporary directory
        """
        self.ivectors = {}
        for ivectors_args in self.extract_ivectors_arguments():
            for ivectors_path in ivectors_args.ivector_paths.values():
                ivec = load_scp(ivectors_path)
                for utt, ivector in ivec.items():
                    ivector = [float(x) for x in ivector]
                    self.ivectors[utt] = ivector

    def cluster_utterances(self) -> None:
        """
        Cluster utterances based on their ivectors
        """
        self.logger.error(
            "Speaker diarization functionality is currently under construction and not working in the current version."
        )
        raise NotImplementedError(
            "Speaker diarization functionality is currently under construction and not working in the current version."
        )

    def export_files(self, output_directory: str) -> None:
        """
        Export files with their new speaker labels

        Parameters
        ----------
        output_directory: str
            Output directory to save files
        """
        backup_output_directory = None
        if not self.overwrite:
            backup_output_directory = os.path.join(self.working_directory, "output")
            os.makedirs(backup_output_directory, exist_ok=True)

        for file in self.files:
            file.save(output_directory, backup_output_directory)
