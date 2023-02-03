"""Class definitions for adapting acoustic models"""
from __future__ import annotations

import logging
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
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.data import WorkflowType
from montreal_forced_aligner.db import CorpusWorkflow
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.helper import mfa_open
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

logger = logging.getLogger("mfa")


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
        self.initialized = False
        self.adaptation_done = False
        super().__init__(**kwargs)
        self.mapping_tau = mapping_tau

    def map_acc_stats_arguments(self, alignment=False) -> List[AccStatsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsArguments`]
            Arguments for processing
        """
        if alignment:
            model_path = self.alignment_model_path
        else:
            model_path = self.model_path
        arguments = []
        for j in self.jobs:
            feat_strings = {}
            for d_id in j.dictionary_ids:
                feat_strings[d_id] = j.construct_feature_proc_string(
                    self.working_directory,
                    d_id,
                    self.feature_options["uses_splices"],
                    self.feature_options["splice_left_context"],
                    self.feature_options["splice_right_context"],
                    self.feature_options["uses_speaker_adaptation"],
                )
            arguments.append(
                AccStatsArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(self.working_log_directory, f"map_acc_stats.{j.id}.log"),
                    j.dictionary_ids,
                    feat_strings,
                    j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                    j.construct_path_dictionary(self.working_directory, "map", "acc"),
                    model_path,
                )
            )
        return arguments

    def acc_stats(self, alignment: bool = False) -> None:
        """
        Accumulate stats for the mapped model

        Parameters
        ----------
        alignment: bool
            Flag for whether to accumulate stats for the mapped alignment model
        """
        arguments = self.map_acc_stats_arguments(alignment)
        if alignment:
            initial_mdl_path = os.path.join(self.working_directory, "unadapted.alimdl")
            final_mdl_path = os.path.join(self.working_directory, "final.alimdl")
        else:
            initial_mdl_path = os.path.join(self.working_directory, "unadapted.mdl")
            final_mdl_path = os.path.join(self.working_directory, "final.mdl")
        logger.info("Accumulating statistics...")
        with tqdm.tqdm(total=self.num_current_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = AccStatsFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    num_utterances, errors = result
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
        with mfa_open(log_path, "w") as log_file:
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
    def align_directory(self) -> str:
        """Align directory"""
        return os.path.join(self.output_directory, "adapted_align")

    @property
    def working_log_directory(self) -> str:
        """Current log directory"""
        return os.path.join(self.working_directory, "log")

    @property
    def model_path(self) -> str:
        """Current acoustic model path"""
        if self.current_workflow.workflow_type == WorkflowType.acoustic_model_adaptation:
            return os.path.join(self.working_directory, "unadapted.mdl")
        return os.path.join(self.working_directory, "final.mdl")

    @property
    def alignment_model_path(self) -> str:
        """Current acoustic model path"""
        if self.current_workflow.workflow_type == WorkflowType.acoustic_model_adaptation:
            path = os.path.join(self.working_directory, "unadapted.alimdl")
            if os.path.exists(path) and not getattr(self, "uses_speaker_adaptation", False):
                return path
            return self.model_path
        return super().alignment_model_path

    @property
    def next_model_path(self) -> str:
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

        logger.debug(f"Mapping models took {time.time() - begin:.3f} seconds")

    def adapt(self) -> None:
        """Run the adaptation"""
        logger.info("Generating initial alignments...")
        self.align()
        alignment_workflow = self.current_workflow
        self.create_new_current_workflow(WorkflowType.acoustic_model_adaptation)
        for f in ["final.mdl", "final.alimdl"]:
            shutil.copyfile(
                os.path.join(alignment_workflow.working_directory, f),
                os.path.join(self.working_directory, f.replace("final", "unadapted")),
            )
        shutil.copyfile(
            os.path.join(alignment_workflow.working_directory, "tree"),
            os.path.join(self.working_directory, "tree"),
        )
        shutil.copyfile(
            os.path.join(alignment_workflow.working_directory, "lda.mat"),
            os.path.join(self.working_directory, "lda.mat"),
        )
        for j in self.jobs:
            old_paths = j.construct_path_dictionary(
                alignment_workflow.working_directory, "ali", "ark"
            )
            new_paths = j.construct_path_dictionary(self.working_directory, "ali", "ark")
            for k, v in old_paths.items():
                shutil.copyfile(v, new_paths[k])
            old_paths = j.construct_path_dictionary(
                alignment_workflow.working_directory, "trans", "ark"
            )
            new_paths = j.construct_path_dictionary(self.working_directory, "trans", "ark")
            for k, v in old_paths.items():
                shutil.copyfile(v, new_paths[k])
        os.makedirs(self.align_directory, exist_ok=True)
        try:
            logger.info("Adapting pretrained model...")
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
            wf = self.current_workflow
            with self.session() as session:
                session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update(
                    {"done": True}
                )
                session.commit()
        except Exception as e:
            wf = self.current_workflow
            with self.session() as session:
                session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update(
                    {"dirty": True}
                )
                session.commit()
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise

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
        acoustic_model.add_model(self.working_directory)
        if directory:
            os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(output_model_path)
        acoustic_model.dump(output_model_path)
