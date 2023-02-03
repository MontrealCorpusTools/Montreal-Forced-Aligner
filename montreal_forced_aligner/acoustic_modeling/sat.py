"""Class definitions for Speaker Adapted Triphone trainer"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import time
import typing
from queue import Empty
from typing import Dict, List

import tqdm

from montreal_forced_aligner.acoustic_modeling.triphone import TriphoneTrainer
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.data import MfaArguments
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import (
    KaldiFunction,
    KaldiProcessWorker,
    Stopped,
    log_kaldi_errors,
    parse_logs,
    thirdparty_binary,
)

__all__ = ["SatTrainer", "AccStatsTwoFeatsFunction", "AccStatsTwoFeatsArguments"]


logger = logging.getLogger("mfa")


class AccStatsTwoFeatsArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.acoustic_modeling.sat.AccStatsTwoFeatsFunction`"""

    dictionaries: List[str]
    ali_paths: Dict[str, str]
    acc_paths: Dict[str, str]
    model_path: str
    feature_strings: Dict[str, str]
    si_feature_strings: Dict[str, str]


class AccStatsTwoFeatsFunction(KaldiFunction):
    """
    Multiprocessing function for accumulating stats across speaker-independent and
    speaker-adapted features

    See Also
    --------
    :meth:`.SatTrainer.create_align_model`
        Main function that calls this function in parallel
    :meth:`.SatTrainer.acc_stats_two_feats_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`ali-to-post`
        Relevant Kaldi binary
    :kaldi_src:`gmm-acc-stats-twofeats`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.acoustic_modeling.sat.AccStatsTwoFeatsArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(r"^LOG \(gmm-acc-stats-twofeats.* Average like for this file.*")

    def __init__(self, args: AccStatsTwoFeatsArguments):
        super().__init__(args)
        self.dictionaries = args.dictionaries
        self.ali_paths = args.ali_paths
        self.acc_paths = args.acc_paths
        self.model_path = args.model_path
        self.feature_strings = args.feature_strings
        self.si_feature_strings = args.si_feature_strings

    def _run(self) -> typing.Generator[bool]:
        """Run the function"""

        with mfa_open(self.log_path, "w") as log_file:
            for dict_id in self.dictionaries:
                ali_path = self.ali_paths[dict_id]
                acc_path = self.acc_paths[dict_id]
                feature_string = self.feature_strings[dict_id]
                si_feature_string = self.si_feature_strings[dict_id]
                ali_to_post_proc = subprocess.Popen(
                    [thirdparty_binary("ali-to-post"), f"ark:{ali_path}", "ark:-"],
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )
                acc_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-acc-stats-twofeats"),
                        self.model_path,
                        feature_string,
                        si_feature_string,
                        "ark,s,cs:-",
                        acc_path,
                    ],
                    stderr=subprocess.PIPE,
                    encoding="utf8",
                    stdin=ali_to_post_proc.stdout,
                    env=os.environ,
                )
                for line in acc_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield True
                self.check_call(acc_proc)


class SatTrainer(TriphoneTrainer):
    """
    Speaker adapted trainer (SAT), inherits from TriphoneTrainer

    Parameters
    ----------
    subset : int
        Number of utterances to use, defaults to 10000
    num_leaves : int
        Number of states in the decision tree, defaults to 2500
    max_gaussians : int
        Number of gaussians in the decision tree, defaults to 15000
    power : float
        Exponent for number of gaussians according to occurrence counts, defaults to 0.2

    See Also
    --------
    :class:`~montreal_forced_aligner.acoustic_modeling.triphone.TriphoneTrainer`
        For acoustic model training parsing parameters

    Attributes
    ----------
    fmllr_iterations : list
        List of iterations to perform fMLLR calculation
    """

    def __init__(
        self,
        subset: int = 10000,
        num_leaves: int = 2500,
        max_gaussians: int = 15000,
        power: float = 0.2,
        quick: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.subset = subset
        self.num_leaves = num_leaves
        self.max_gaussians = max_gaussians
        self.power = power
        self.fmllr_iterations = []
        self.quick = quick
        self.graph_batch_size = 0

    def acc_stats_two_feats_arguments(self) -> List[AccStatsTwoFeatsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.sat.AccStatsTwoFeatsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.sat.AccStatsTwoFeatsArguments`]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            feat_strings = {}
            si_feat_strings = {}
            for d_id in j.dictionary_ids:
                feat_strings[d_id] = j.construct_feature_proc_string(
                    self.working_directory,
                    d_id,
                    self.feature_options["uses_splices"],
                    self.feature_options["splice_left_context"],
                    self.feature_options["splice_right_context"],
                    self.feature_options["uses_speaker_adaptation"],
                )
                si_feat_strings[d_id] = j.construct_feature_proc_string(
                    self.working_directory,
                    d_id,
                    self.feature_options["uses_splices"],
                    self.feature_options["splice_left_context"],
                    self.feature_options["splice_right_context"],
                    False,
                )
            arguments.append(
                AccStatsTwoFeatsArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(self.working_log_directory, f"acc_stats_two_feats.{j.id}.log"),
                    j.dictionary_ids,
                    j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                    j.construct_path_dictionary(self.working_directory, "two_feat_acc", "ark"),
                    self.model_path,
                    feat_strings,
                    si_feat_strings,
                )
            )
        return arguments

    def calc_fmllr(self) -> None:
        """Calculate fMLLR transforms for the current iteration"""
        self.worker.calc_fmllr(iteration=self.iteration)

    def compute_calculated_properties(self) -> None:
        """Generate realignment iterations, initial gaussians, and fMLLR iterations based on configuration"""
        super().compute_calculated_properties()
        self.fmllr_iterations = []
        if not self.quick:
            self.fmllr_iterations = [2, 4, 6, 12]
        else:
            self.realignment_iterations = [10, 15]
            self.fmllr_iterations = [2, 6, 12]
            self.graph_batch_size = 750
            self.final_gaussian_iteration = self.num_iterations - 5
            self.power = 0.0
            self.initial_gaussians = int(self.max_gaussians / 2)
            if self.initial_gaussians < self.num_leaves:
                self.initial_gaussians = self.num_leaves

    def _trainer_initialization(self) -> None:
        """Speaker adapted training initialization"""
        if self.initialized:
            self.uses_speaker_adaptation = True
            self.worker.uses_speaker_adaptation = True
            return
        if os.path.exists(os.path.join(self.previous_aligner.working_directory, "lda.mat")):
            shutil.copyfile(
                os.path.join(self.previous_aligner.working_directory, "lda.mat"),
                os.path.join(self.working_directory, "lda.mat"),
            )
        for j in self.jobs:
            for path in j.construct_path_dictionary(
                self.previous_aligner.working_directory, "trans", "ark"
            ).values():
                if os.path.exists(path):
                    break
            else:
                continue
            break
        else:
            self.uses_speaker_adaptation = False
            self.worker.uses_speaker_adaptation = False
            self.calc_fmllr()
        self.uses_speaker_adaptation = True
        self.worker.uses_speaker_adaptation = True
        for j in self.jobs:
            transform_paths = j.construct_path_dictionary(
                self.previous_aligner.working_directory, "trans", "ark"
            )
            output_paths = j.construct_path_dictionary(self.working_directory, "trans", "ark")
            for k, path in transform_paths.items():
                shutil.copy(path, output_paths[k])
        self.tree_stats()
        self._setup_tree(init_from_previous=self.quick, initial_mix_up=self.quick)

        self.convert_alignments()

        self.compile_train_graphs()
        os.rename(self.model_path, self.next_model_path)

        self.iteration = 1
        parse_logs(self.working_log_directory)

    def finalize_training(self) -> None:
        """
        Finalize training and create a speaker independent model for initial alignment

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        try:
            self.create_align_model()
            self.uses_speaker_adaptation = True
            super().finalize_training()
            assert self.alignment_model_path.endswith("final.alimdl")
            assert os.path.exists(self.alignment_model_path)
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise

    def train_iteration(self) -> None:
        """
        Run a single training iteration
        """
        if os.path.exists(self.next_model_path):
            if self.iteration <= self.final_gaussian_iteration:
                self.increment_gaussians()
            self.iteration += 1
            return
        if self.iteration in self.realignment_iterations:
            self.align_iteration()
        if self.iteration in self.fmllr_iterations:

            self.calc_fmllr()

        self.acc_stats()

        if self.iteration <= self.final_gaussian_iteration:
            self.increment_gaussians()
        self.iteration += 1

    @property
    def alignment_model_path(self) -> str:
        """Alignment model path"""
        path = self.model_path.replace(".mdl", ".alimdl")
        if os.path.exists(path):
            return path
        return self.model_path

    def create_align_model(self) -> None:
        """
        Create alignment model for speaker-adapted training that will use speaker-independent
        features in later aligning.

        See Also
        --------
        :func:`~montreal_forced_aligner.acoustic_modeling.sat.AccStatsTwoFeatsFunction`
            Multiprocessing helper function for each job
        :meth:`.SatTrainer.acc_stats_two_feats_arguments`
            Job method for generating arguments for the helper function
        :kaldi_src:`gmm-est`
            Relevant Kaldi binary
        :kaldi_src:`gmm-sum-accs`
            Relevant Kaldi binary
        :kaldi_steps:`train_sat`
            Reference Kaldi script
        """
        logger.info("Creating alignment model for speaker-independent features...")
        begin = time.time()

        arguments = self.acc_stats_two_feats_arguments()
        with tqdm.tqdm(total=self.num_current_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = AccStatsTwoFeatsFunction(args)
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
                    pbar.update(1)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in arguments:
                    function = AccStatsTwoFeatsFunction(args)
                    for _ in function.run():
                        pbar.update(1)

        log_path = os.path.join(self.working_log_directory, "align_model_est.log")
        with mfa_open(log_path, "w") as log_file:

            acc_files = []
            for x in arguments:
                acc_files.extend(x.acc_paths.values())
            sum_proc = subprocess.Popen(
                [thirdparty_binary("gmm-sum-accs"), "-"] + acc_files,
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            est_command = [
                thirdparty_binary("gmm-est"),
                "--remove-low-count-gaussians=false",
            ]
            if not self.quick:
                est_command.append(f"--power={self.power}")
            else:
                est_command.append(
                    f"--write-occs={os.path.join(self.working_directory, 'final.occs')}"
                )
            est_command.extend(
                [
                    self.model_path,
                    "-",
                    self.model_path.replace(".mdl", ".alimdl"),
                ]
            )
            est_proc = subprocess.Popen(
                est_command,
                stdin=sum_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            est_proc.communicate()
        parse_logs(self.working_log_directory)
        if not GLOBAL_CONFIG.debug:
            for f in acc_files:
                os.remove(f)
        logger.debug(f"Alignment model creation took {time.time() - begin:.3f} seconds")
