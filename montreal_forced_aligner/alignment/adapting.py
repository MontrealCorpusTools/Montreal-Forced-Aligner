"""Class definitions for adapting acoustic models"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, NamedTuple

from montreal_forced_aligner.abc import AdapterMixin
from montreal_forced_aligner.alignment.pretrained import PretrainedAligner
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.utils import log_kaldi_errors, run_mp, run_non_mp, thirdparty_binary

if TYPE_CHECKING:
    from montreal_forced_aligner.models import MetaDict


__all__ = ["AdaptingAligner"]


class MapAccStatsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.alignment.adapting.map_acc_stats_func`"""

    log_path: str
    dictionaries: list[str]
    feature_strings: dict[str, str]
    model_path: str
    ali_paths: dict[str, str]
    acc_paths: dict[str, str]


def map_acc_stats_func(
    log_path: str,
    dictionaries: list[str],
    feature_strings: dict[str, str],
    model_path: str,
    ali_paths: dict[str, str],
    acc_paths: dict[str, str],
) -> None:
    """
    Multiprocessing function for accumulating mapped stats for adapting acoustic models to new
    domains

    See Also
    --------
    :meth:`.AdaptingAligner.train_map`
        Main function that calls this function in parallel
    :meth:`.AdaptingAligner.map_acc_stats_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`gmm-acc-stats-ali`
        Relevant Kaldi binary

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: list[str]
        List of dictionary names
    feature_strings: dict[str, str]
        Dictionary of feature strings per dictionary name
    model_path: str
        Path to the acoustic model file
    ali_paths: dict[str, str]
        Dictionary of alignment archives per dictionary name
    acc_paths: dict[str, str]
        Dictionary of accumulated stats files per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            acc_path = acc_paths[dict_name]
            ali_path = ali_paths[dict_name]
            acc_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-acc-stats-ali"),
                    model_path,
                    feature_string,
                    f"ark,s,cs:{ali_path}",
                    acc_path,
                ],
                stderr=log_file,
                env=os.environ,
            )
            acc_proc.communicate()


class AdaptingAligner(PretrainedAligner, AdapterMixin):
    """
    Adapt an acoustic model to a new dataset

    Parameters
    ----------
    mapping_tau: int
        Tau to use in mapping stats between new domain data and pretrained model

    See Also
    --------
    :class:`~montreal_forced_aligner.alignment.pretrained.PretrainedAligner`
        For dictionary, corpus, and alignment parameters
    :class:`~montreal_forced_aligner.abc.AdapterMixin`
        For adapting parameters

    Attributes
    ----------
    initialized: bool
        Flag for whether initialization is complete
    adaptation_done: bool
        Flag for whether adaptation is complete
    """

    def __init__(self, mapping_tau: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.mapping_tau = mapping_tau
        self.initialized = False
        self.adaptation_done = False

    def map_acc_stats_arguments(self, alignment=False) -> list[MapAccStatsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.alignment.adapting.map_acc_stats_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.adapting.MapAccStatsArguments`]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        if alignment:
            model_path = self.alignment_model_path
        else:
            model_path = self.model_path
        return [
            MapAccStatsArguments(
                os.path.join(self.working_log_directory, f"map_acc_stats.{j.name}.log"),
                j.current_dictionary_names,
                feat_strings[j.name],
                model_path,
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                j.construct_path_dictionary(self.working_directory, "map", "acc"),
            )
            for j in self.jobs
        ]

    @property
    def workflow_identifier(self) -> str:
        """Adaptation identifier"""
        return "adapt_acoustic_model"

    @property
    def align_directory(self) -> str:
        """Align directory"""
        return os.path.join(self.output_directory, "adapted_align")

    @property
    def working_directory(self) -> str:
        """Current working directory"""
        if self.adaptation_done:
            return self.align_directory
        return self.workflow_directory

    @property
    def working_log_directory(self) -> str:
        """Current log directory"""
        return os.path.join(self.working_directory, "log")

    @property
    def model_path(self):
        """Current acoustic model path"""
        if not self.adaptation_done:
            return os.path.join(self.working_directory, "0.mdl")
        return os.path.join(self.working_directory, "final.mdl")

    @property
    def next_model_path(self):
        """Mapped acoustic model path"""
        return os.path.join(self.working_directory, "final.mdl")

    def train_map(self) -> None:
        """
        Trains an adapted acoustic model through mapping model states and update those with
        enough data.

        See Also
        --------
        :func:`~montreal_forced_aligner.alignment.adapting.map_acc_stats_func`
            Multiprocessing helper function for each job
        :meth:`.AdaptingAligner.map_acc_stats_arguments`
            Job method for generating arguments for the helper function
        :kaldi_src:`gmm-sum-accs`
            Relevant Kaldi binary
        :kaldi_src:`gmm-ismooth-stats`
            Relevant Kaldi binary
        :kaldi_src:`gmm-est`
            Relevant Kaldi binary
        :kaldi_steps:`train_map`
            Reference Kaldi script

        """
        begin = time.time()
        initial_mdl_path = os.path.join(self.working_directory, "0.mdl")
        final_mdl_path = os.path.join(self.working_directory, "final.mdl")
        log_directory = self.working_log_directory
        os.makedirs(log_directory, exist_ok=True)

        jobs = self.map_acc_stats_arguments()
        if self.use_mp:
            run_mp(map_acc_stats_func, jobs, log_directory)
        else:
            run_non_mp(map_acc_stats_func, jobs, log_directory)
        log_path = os.path.join(self.working_log_directory, "map_model_est.log")
        occs_path = os.path.join(self.working_directory, "final.occs")
        with open(log_path, "w", encoding="utf8") as log_file:
            acc_files = []
            for j in jobs:
                acc_files.extend(j.acc_paths.values())
            sum_proc = subprocess.Popen(
                [thirdparty_binary("gmm-sum-accs"), "-"] + acc_files,
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            ismooth_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-ismooth-stats"),
                    "--smooth-from-model",
                    f"--tau={self.mapping_tau}",
                    initial_mdl_path,
                    "-",
                    "-",
                ],
                stderr=log_file,
                stdin=sum_proc.stdout,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            est_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-est"),
                    "--update-flags=m",
                    f"--write-occs={occs_path}",
                    "--remove-low-count-gaussians=false",
                    initial_mdl_path,
                    "-",
                    final_mdl_path,
                ],
                stdin=ismooth_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            est_proc.communicate()
        if self.uses_speaker_adaptation:
            initial_alimdl_path = os.path.join(self.working_directory, "0.alimdl")
            final_alimdl_path = os.path.join(self.working_directory, "0.alimdl")
            if os.path.exists(initial_alimdl_path):
                self.speaker_independent = True
                jobs = self.map_acc_stats_arguments(alignment=True)
                if self.use_mp:
                    run_mp(map_acc_stats_func, jobs, log_directory)
                else:
                    run_non_mp(map_acc_stats_func, jobs, log_directory)

                log_path = os.path.join(self.working_log_directory, "map_model_est.log")
                with open(log_path, "w", encoding="utf8") as log_file:
                    acc_files = []
                for j in jobs:
                    acc_files.extend(j.acc_paths)
                    sum_proc = subprocess.Popen(
                        [thirdparty_binary("gmm-sum-accs"), "-"] + acc_files,
                        stderr=log_file,
                        stdout=subprocess.PIPE,
                        env=os.environ,
                    )
                    ismooth_proc = subprocess.Popen(
                        [
                            thirdparty_binary("gmm-ismooth-stats"),
                            "--smooth-from-model",
                            f"--tau={self.mapping_tau}",
                            initial_alimdl_path,
                            "-",
                            "-",
                        ],
                        stderr=log_file,
                        stdin=sum_proc.stdout,
                        stdout=subprocess.PIPE,
                        env=os.environ,
                    )
                    est_proc = subprocess.Popen(
                        [
                            thirdparty_binary("gmm-est"),
                            "--update-flags=m",
                            "--remove-low-count-gaussians=false",
                            initial_alimdl_path,
                            "-",
                            final_alimdl_path,
                        ],
                        stdin=ismooth_proc.stdout,
                        stderr=log_file,
                        env=os.environ,
                    )
                    est_proc.communicate()

        self.logger.debug(f"Mapping models took {time.time() - begin}")

    def adapt(self) -> None:
        """Run the adaptation"""
        self.setup()
        dirty_path = os.path.join(self.working_directory, "dirty")
        done_path = os.path.join(self.working_directory, "done")
        if os.path.exists(done_path):
            self.logger.info("Adaptation already done, skipping.")
            return
        self.logger.info("Generating initial alignments...")
        for f in ["final.mdl", "final.alimdl"]:
            p = os.path.join(self.working_directory, f)
            if not os.path.exists(p):
                continue
            os.rename(p, os.path.join(self.working_directory, f.replace("final", "0")))
        self.align()
        os.makedirs(self.align_directory, exist_ok=True)
        try:
            self.logger.info("Adapting pretrained model...")
            self.train_map()
            self.export_model(os.path.join(self.working_log_directory, "acoustic_model.zip"))
            shutil.copyfile(
                os.path.join(self.working_directory, "final.mdl"),
                os.path.join(self.align_directory, "final.mdl"),
            )
            shutil.copyfile(
                os.path.join(self.working_directory, "final.occs"),
                os.path.join(self.align_directory, "final.occs"),
            )
            shutil.copyfile(
                os.path.join(self.working_directory, "tree"),
                os.path.join(self.align_directory, "tree"),
            )
            if os.path.exists(os.path.join(self.working_directory, "final.alimdl")):
                shutil.copyfile(
                    os.path.join(self.working_directory, "final.alimdl"),
                    os.path.join(self.align_directory, "final.alimdl"),
                )
            if os.path.exists(os.path.join(self.working_directory, "lda.mat")):
                shutil.copyfile(
                    os.path.join(self.working_directory, "lda.mat"),
                    os.path.join(self.align_directory, "lda.mat"),
                )
            self.adaptation_done = True
        except Exception as e:
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise
        with open(done_path, "w"):
            pass

    @property
    def meta(self) -> MetaDict:
        """Acoustic model metadata"""
        from datetime import datetime

        from ..utils import get_mfa_version

        data = {
            "phones": sorted(self.non_silence_phones),
            "version": get_mfa_version(),
            "architecture": self.acoustic_model.meta["architecture"],
            "train_date": str(datetime.now()),
            "features": self.feature_options,
            "multilingual_ipa": self.multilingual_ipa,
        }
        if self.multilingual_ipa:
            data["strip_diacritics"] = self.strip_diacritics
            data["digraphs"] = self.digraphs
        return data

    def export_model(self, output_model_path: str) -> None:
        """
        Output an acoustic model to the specified path

        Parameters
        ----------
        output_model_path : str
            Path to save adapted acoustic model
        """
        directory, filename = os.path.split(output_model_path)
        basename, _ = os.path.splitext(filename)
        acoustic_model = AcousticModel.empty(basename, root_directory=self.working_log_directory)
        acoustic_model.add_meta_file(self)
        acoustic_model.add_model(self.align_directory)
        if directory:
            os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(output_model_path)
        acoustic_model.dump(output_model_path)
