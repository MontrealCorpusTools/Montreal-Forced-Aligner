"""Classes for corpora that use ivectors as features"""
import os

from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusMixin
from montreal_forced_aligner.corpus.features import (
    ExtractIvectorsArguments,
    IvectorConfigMixin,
    extract_ivectors_func,
)
from montreal_forced_aligner.utils import run_mp, run_non_mp

__all__ = ["IvectorCorpusMixin"]


class IvectorCorpusMixin(AcousticCorpusMixin, IvectorConfigMixin):
    """
    Abstract corpus mixin for corpora that extract ivectors

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusMixin`
        For dictionary and corpus parsing parameters
    :class:`~montreal_forced_aligner.corpus.features.IvectorConfigMixin`
        For ivector extraction parameters

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def ie_path(self):
        """Ivector extractor ie path"""
        raise NotImplementedError

    @property
    def dubm_path(self):
        """DUBM model path"""
        raise

    def write_corpus_information(self) -> None:
        """
        Output information to the temporary directory for later loading
        """
        super().write_corpus_information()
        self._write_utt2spk()

    def _write_utt2spk(self):
        """Write feats scp file for Kaldi"""
        with open(
            os.path.join(self.corpus_output_directory, "utt2spk.scp"), "w", encoding="utf8"
        ) as f:
            for utterance in self.utterances.values():
                f.write(f"{utterance.name} {utterance.speaker.name}\n")

    def extract_ivectors_arguments(self) -> list[ExtractIvectorsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.corpus.features.extract_ivectors_func`

        Returns
        -------
        list[ExtractIvectorsArguments]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        return [
            ExtractIvectorsArguments(
                os.path.join(self.working_log_directory, f"extract_ivectors.{j.name}.log"),
                j.current_dictionary_names,
                feat_strings[j.name],
                self.ivector_options,
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                self.ie_path,
                j.construct_path_dictionary(self.working_directory, "ivectors", "scp"),
                j.construct_path_dictionary(self.working_directory, "weights", "ark"),
                self.model_path,
                self.dubm_path,
            )
            for j in self.jobs
        ]

    def extract_ivectors(self) -> None:
        """
        Multiprocessing function that extracts job_name-vectors.

        See Also
        --------
        :func:`~montreal_forced_aligner.corpus.features.extract_ivectors_func`
            Multiprocessing helper function for each job
        :meth:`.IvectorCorpusMixin.extract_ivectors_arguments`
            Job method for generating arguments for helper function
        :kaldi_steps_sid:`extract_ivectors`
            Reference Kaldi script
        """

        log_dir = self.working_log_directory
        os.makedirs(log_dir, exist_ok=True)

        jobs = self.extract_ivectors_arguments()
        if self.use_mp:
            run_mp(extract_ivectors_func, jobs, log_dir)
        else:
            run_non_mp(extract_ivectors_func, jobs, log_dir)
