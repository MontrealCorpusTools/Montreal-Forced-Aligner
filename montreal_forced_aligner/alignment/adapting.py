"""Class definitions for adapting acoustic models"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import subprocess
import time
from queue import Empty
from typing import TYPE_CHECKING, List

import tqdm

from montreal_forced_aligner.abc import AdapterMixin
from montreal_forced_aligner.alignment.multiprocessing import AccStatsArguments, AccStatsFunction
from montreal_forced_aligner.alignment.pretrained import PretrainedAligner
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.utils import (
    KaldiProcessWorker,
    Stopped,
    log_kaldi_errors,
    thirdparty_binary,
)

if TYPE_CHECKING:
    from montreal_forced_aligner.models import MetaDict


__all__ = ["AdaptingAligner"]


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

    def map_acc_stats_arguments(self, alignment=False) -> List[AccStatsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsArguments`]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        if alignment:
            model_path = self.alignment_model_path
        else:
            model_path = self.model_path
        return [
            AccStatsArguments(
                os.path.join(self.working_log_directory, f"map_acc_stats.{j.name}.log"),
                j.current_dictionary_names,
                feat_strings[j.name],
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                j.construct_path_dictionary(self.working_directory, "map", "acc"),
                model_path,
            )
            for j in self.jobs
        ]

    def acc_stats(self, alignment=False):
        arguments = self.map_acc_stats_arguments(alignment)
        if alignment:
            initial_mdl_path = os.path.join(self.working_directory, "0.alimdl")
            final_mdl_path = os.path.join(self.working_directory, "0.alimdl")
        else:
            initial_mdl_path = os.path.join(self.working_directory, "0.mdl")
            final_mdl_path = os.path.join(self.working_directory, "final.mdl")
        if not os.path.exists(initial_mdl_path):
            return
        self.logger.info("Accumulating statistics...")
        with tqdm.tqdm(total=self.num_utterances) as pbar:
            if self.use_mp:
                manager = mp.Manager()
                error_dict = manager.dict()
                return_queue = manager.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = AccStatsFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, error_dict, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        num_utterances, errors = return_queue.get(timeout=1)
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(num_utterances + errors)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in arguments:
                    function = AccStatsFunction(args)
                    for num_utterances, errors in function.run():
                        pbar.update(num_utterances + errors)
        log_path = os.path.join(self.working_log_directory, "map_model_est.log")
        occs_path = os.path.join(self.working_directory, "final.occs")
        with open(log_path, "w", encoding="utf8") as log_file:
            acc_files = []
            for j in arguments:
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
        :class:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsFunction`
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
        log_directory = self.working_log_directory
        os.makedirs(log_directory, exist_ok=True)
        self.acc_stats(alignment=False)

        if self.uses_speaker_adaptation:
            self.acc_stats(alignment=True)

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
            "phone_set_type": str(self.phone_set_type),
        }
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
