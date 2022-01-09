"""Class definitions for Speaker Adapted Triphone trainer"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
import shutil
import subprocess
import time
from queue import Empty
from typing import Dict, List, NamedTuple

import tqdm

from montreal_forced_aligner.acoustic_modeling.triphone import TriphoneTrainer
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.utils import (
    KaldiFunction,
    KaldiProcessWorker,
    Stopped,
    log_kaldi_errors,
    parse_logs,
    thirdparty_binary,
)

__all__ = ["SatTrainer", "AccStatsTwoFeatsFunction", "AccStatsTwoFeatsArguments"]


class AccStatsTwoFeatsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.acoustic_modeling.sat.AccStatsTwoFeatsFunction`"""

    log_path: str
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

    done_pattern = re.compile(
        r"^LOG \(gmm-acc-stats-twofeats.*Done (?P<utterances>\d+) files, (?P<no_posteriors>\d+) with no posteriors, (?P<no_second_features>\d+) with no second features, (?P<errors>\d+) with other errors.$"
    )

    def __init__(self, args: AccStatsTwoFeatsArguments):
        self.log_path = args.log_path
        self.dictionaries = args.dictionaries
        self.ali_paths = args.ali_paths
        self.acc_paths = args.acc_paths
        self.model_path = args.model_path
        self.feature_strings = args.feature_strings
        self.si_feature_strings = args.si_feature_strings

    def run(self):
        """Run the function"""

        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_name in self.dictionaries:
                ali_path = self.ali_paths[dict_name]
                acc_path = self.acc_paths[dict_name]
                feature_string = self.feature_strings[dict_name]
                si_feature_string = self.si_feature_strings[dict_name]
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
                        yield 1, 0, 0, 0
                    else:
                        m = self.done_pattern.match(line.strip())
                        if m:
                            yield int(m.group("utterances")), int(m.group("no_posteriors")), int(
                                m.group("no_second_features")
                            ), int(m.group("errors"))


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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.subset = subset
        self.num_leaves = num_leaves
        self.max_gaussians = max_gaussians
        self.power = power
        self.fmllr_iterations = []

    def acc_stats_two_feats_arguments(self) -> List[AccStatsTwoFeatsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.sat.AccStatsTwoFeatsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.sat.AccStatsTwoFeatsArguments`]
            Arguments for processing
        """
        feat_strings = self.worker.construct_feature_proc_strings()
        si_feat_strings = self.worker.construct_feature_proc_strings(speaker_independent=True)
        return [
            AccStatsTwoFeatsArguments(
                os.path.join(self.working_log_directory, f"acc_stats_two_feats.{j.name}.log"),
                j.current_dictionary_names,
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                j.construct_path_dictionary(self.working_directory, "two_feat_acc", "ark"),
                self.model_path,
                feat_strings[j.name],
                si_feat_strings[j.name],
            )
            for j in self.jobs
        ]

    def calc_fmllr(self) -> None:
        self.worker.calc_fmllr()

    def compute_calculated_properties(self) -> None:
        """Generate realignment iterations, initial gaussians, and fMLLR iterations based on configuration"""
        super().compute_calculated_properties()
        self.fmllr_iterations = []
        max_fmllr_iter = int(self.num_iterations / 2) - 1
        for i in range(1, max_fmllr_iter):
            if i < max_fmllr_iter / 2 and i % 2 == 0:
                self.fmllr_iterations.append(i)
        self.fmllr_iterations.append(max_fmllr_iter)

    def _trainer_initialization(self) -> None:
        """Speaker adapted training initialization"""
        self.speaker_independent = False
        if os.path.exists(os.path.join(self.working_directory, "1.mdl")):
            return
        if os.path.exists(os.path.join(self.previous_aligner.working_directory, "lda.mat")):
            shutil.copyfile(
                os.path.join(self.previous_aligner.working_directory, "lda.mat"),
                os.path.join(self.working_directory, "lda.mat"),
            )
        self.tree_stats()
        self._setup_tree()

        self.compile_train_graphs()

        self.convert_alignments()
        os.rename(self.model_path, self.next_model_path)

        self.iteration = 1

        if os.path.exists(os.path.join(self.previous_aligner.working_directory, "trans.0.ark")):
            for j in self.jobs:
                for path in j.construct_path_dictionary(
                    self.previous_aligner.working_directory, "trans", "ark"
                ).values():
                    shutil.copy(
                        path,
                        path.replace(
                            self.previous_aligner.working_directory, self.working_directory
                        ),
                    )
        else:
            self.worker.current_trainer = self
            self.calc_fmllr()
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
            super().finalize_training()
            shutil.copy(
                os.path.join(self.working_directory, f"{self.num_iterations+1}.alimdl"),
                os.path.join(self.working_directory, "final.alimdl"),
            )
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise

    def train_iteration(self) -> None:
        """
        Run a single training iteration
        """
        if os.path.exists(self.next_model_path):
            self.iteration += 1
            return
        if self.iteration in self.realignment_iterations:
            self.align_utterances()
            if self.debug:
                self.compute_alignment_improvement()
        if self.iteration in self.fmllr_iterations:
            self.calc_fmllr()

        self.acc_stats()
        parse_logs(self.working_log_directory)
        if self.iteration < self.final_gaussian_iteration:
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
        self.logger.info("Creating alignment model for speaker-independent features...")
        begin = time.time()

        arguments = self.acc_stats_two_feats_arguments()
        with tqdm.tqdm(total=self.num_utterances) as pbar:
            if self.use_mp:
                manager = mp.Manager()
                error_dict = manager.dict()
                return_queue = manager.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = AccStatsTwoFeatsFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, error_dict, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        (
                            num_utterances,
                            no_posteriors,
                            no_second_features,
                            errors,
                        ) = return_queue.get(timeout=1)
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(num_utterances + no_posteriors + no_second_features + errors)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in arguments:
                    function = AccStatsTwoFeatsFunction(args)
                    for (
                        num_utterances,
                        no_posteriors,
                        no_second_features,
                        errors,
                    ) in function.run():
                        pbar.update(num_utterances + no_posteriors + no_second_features + errors)

        log_path = os.path.join(self.working_log_directory, "align_model_est.log")
        with open(log_path, "w", encoding="utf8") as log_file:

            acc_files = []
            for x in arguments:
                acc_files.extend(x.acc_paths.values())
            sum_proc = subprocess.Popen(
                [thirdparty_binary("gmm-sum-accs"), "-"] + acc_files,
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            est_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-est"),
                    "--remove-low-count-gaussians=false",
                    f"--power={self.power}",
                    self.model_path,
                    "-",
                    self.model_path.replace(".mdl", ".alimdl"),
                ],
                stdin=sum_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            est_proc.communicate()
        parse_logs(self.working_log_directory)
        if not self.debug:
            for f in acc_files:
                os.remove(f)
        self.logger.debug(f"Alignment model creation took {time.time() - begin}")
