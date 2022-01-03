"""Class definitions for TriphoneTrainer"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
import subprocess
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, NamedTuple

import tqdm

from montreal_forced_aligner.acoustic_modeling.base import AcousticModelTrainingMixin
from montreal_forced_aligner.utils import (
    KaldiFunction,
    KaldiProcessWorker,
    Stopped,
    parse_logs,
    run_mp,
    run_non_mp,
    thirdparty_binary,
)

if TYPE_CHECKING:
    from ..abc import MetaDict


__all__ = [
    "TriphoneTrainer",
    "TreeStatsArguments",
    "ConvertAlignmentsFunction",
    "ConvertAlignmentsArguments",
]


class TreeStatsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.acoustic_modeling.triphone.tree_stats_func`"""

    log_path: str
    dictionaries: List[str]
    ci_phones: str
    model_path: str
    feature_strings: Dict[str, str]
    ali_paths: Dict[str, str]
    treeacc_paths: Dict[str, str]


class ConvertAlignmentsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.acoustic_modeling.triphone.ConvertAlignmentsFunction`"""

    log_path: str
    dictionaries: List[str]
    model_path: str
    tree_path: str
    align_model_path: str
    ali_paths: Dict[str, str]
    new_ali_paths: Dict[str, str]


class ConvertAlignmentsFunction(KaldiFunction):
    """
    Multiprocessing function for converting alignments from a previous trainer

    See Also
    --------
    :meth:`.TriphoneTrainer.convert_alignments`
        Main function that calls this function in parallel
    :meth:`.TriphoneTrainer.convert_alignments_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`convert-ali`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.acoustic_modeling.triphone.ConvertAlignmentsArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(
        r"^LOG.*Succeeded converting alignments for (?P<utterances>\d+) files, failed for (?P<failed>\d+)$"
    )

    def __init__(self, args: ConvertAlignmentsArguments):
        self.log_path = args.log_path
        self.dictionaries = args.dictionaries
        self.model_path = args.model_path
        self.tree_path = args.tree_path
        self.align_model_path = args.align_model_path
        self.ali_paths = args.ali_paths
        self.new_ali_paths = args.new_ali_paths

    def run(self):
        """Run the function"""
        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_name in self.dictionaries:
                ali_path = self.ali_paths[dict_name]
                new_ali_path = self.new_ali_paths[dict_name]
                convert_proc = subprocess.Popen(
                    [
                        thirdparty_binary("convert-ali"),
                        self.align_model_path,
                        self.model_path,
                        self.tree_path,
                        f"ark:{ali_path}",
                        f"ark:{new_ali_path}",
                    ],
                    stderr=subprocess.PIPE,
                    encoding="utf8",
                    env=os.environ,
                )
                for line in convert_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield int(m.group("utterances")), int(m.group("failed"))


def tree_stats_func(
    log_path: str,
    dictionaries: List[str],
    ci_phones: str,
    model_path: str,
    feature_strings: Dict[str, str],
    ali_paths: Dict[str, str],
    treeacc_paths: Dict[str, str],
) -> None:
    """
    Multiprocessing function for calculating tree stats for training

    See Also
    --------
    :meth:`.TriphoneTrainer.tree_stats`
        Main function that calls this function in parallel
    :meth:`.TriphoneTrainer.tree_stats_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`acc-tree-stats`
        Relevant Kaldi binary

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: list[str]
        List of dictionary names
    ci_phones: str
        Colon-separated list of context-independent phones
    model_path: str
        Path to the acoustic model file
    feature_strings: dict[str, str]
        Dictionary of feature strings per dictionary name
    ali_paths: dict[str, str]
        Dictionary of alignment archives per dictionary name
    treeacc_paths: dict[str, str]
        Dictionary of accumulated tree stats files per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            ali_path = ali_paths[dict_name]
            treeacc_path = treeacc_paths[dict_name]
            subprocess.call(
                [
                    thirdparty_binary("acc-tree-stats"),
                    f"--ci-phones={ci_phones}",
                    model_path,
                    feature_string,
                    f"ark:{ali_path}",
                    treeacc_path,
                ],
                stderr=log_file,
            )


class TriphoneTrainer(AcousticModelTrainingMixin):
    """
    Triphone trainer

    Parameters
    ----------
    subset : int
        Number of utterances to use, defaults to 5000
    num_iterations : int
        Number of training iterations to perform, defaults to 35
    num_leaves : int
        Number of states in the decision tree, defaults to 1000
    max_gaussians : int
        Number of gaussians in the decision tree, defaults to 10000
    cluster_threshold : int
        For build-tree control final bottom-up clustering of leaves, defaults to 100

    See Also
    --------
    :class:`~montreal_forced_aligner.acoustic_modeling.base.AcousticModelTrainingMixin`
        For acoustic model training parsing parameters
    """

    def __init__(
        self,
        subset: int = 5000,
        num_iterations: int = 35,
        num_leaves: int = 1000,
        max_gaussians: int = 10000,
        cluster_threshold: int = -1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.subset = subset
        self.num_iterations = num_iterations
        self.num_leaves = num_leaves
        self.max_gaussians = max_gaussians
        self.cluster_threshold = cluster_threshold

    def tree_stats_arguments(self) -> List[TreeStatsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.triphone.tree_stats_func`


        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.triphone.TreeStatsArguments`]
            Arguments for processing
        """
        feat_strings = self.worker.construct_feature_proc_strings()
        alignment_model_path = os.path.join(self.previous_aligner.working_directory, "final.mdl")
        return [
            TreeStatsArguments(
                os.path.join(self.working_log_directory, f"acc_tree.{j.name}.log"),
                j.current_dictionary_names,
                self.worker.context_independent_csl,
                alignment_model_path,
                feat_strings[j.name],
                j.construct_path_dictionary(self.previous_aligner.working_directory, "ali", "ark"),
                j.construct_path_dictionary(self.working_directory, "tree", "acc"),
            )
            for j in self.jobs
        ]

    def convert_alignments_arguments(self) -> List[ConvertAlignmentsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.triphone.ConvertAlignmentsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.triphone.ConvertAlignmentsArguments`]
            Arguments for processing
        """
        return [
            ConvertAlignmentsArguments(
                os.path.join(self.working_log_directory, f"convert_alignments.{j.name}.log"),
                j.current_dictionary_names,
                self.model_path,
                self.tree_path,
                self.previous_aligner.alignment_model_path,
                j.construct_path_dictionary(self.previous_aligner.working_directory, "ali", "ark"),
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
            )
            for j in self.jobs
        ]

    def convert_alignments(self) -> None:
        """
        Multiprocessing function that converts alignments from previous training

        See Also
        --------
        :func:`~montreal_forced_aligner.acoustic_modeling.triphone.ConvertAlignmentsFunction`
            Multiprocessing helper function for each job
        :meth:`.TriphoneTrainer.convert_alignments_arguments`
            Job method for generating arguments for the helper function
        :kaldi_steps:`train_deltas`
            Reference Kaldi script
        :kaldi_steps:`train_lda_mllt`
            Reference Kaldi script
        :kaldi_steps:`train_sat`
            Reference Kaldi script

        """
        self.log_info("Converting alignments...")
        arguments = self.convert_alignments_arguments()
        with tqdm.tqdm(total=self.num_utterances) as pbar:
            if self.use_mp:
                manager = mp.Manager()
                error_dict = manager.dict()
                return_queue = manager.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = ConvertAlignmentsFunction(args)
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
                    function = ConvertAlignmentsFunction(args)
                    for num_utterances, errors in function.run():
                        pbar.update(num_utterances + errors)

    def acoustic_model_training_params(self) -> MetaDict:
        """Configuration parameters"""
        return {
            "num_iterations": self.num_iterations,
            "num_leaves": self.num_leaves,
            "max_gaussians": self.max_gaussians,
            "cluster_threshold": self.cluster_threshold,
        }

    def compute_calculated_properties(self) -> None:
        """Generate realignment iterations and initial gaussians based on configuration"""
        for i in range(0, self.num_iterations, 10):
            if i == 0:
                continue
            self.realignment_iterations.append(i)
        self.initial_gaussians = self.num_leaves
        self.current_gaussians = self.num_leaves

    @property
    def train_type(self) -> str:
        """Training identifier"""
        return "tri"

    @property
    def phone_type(self) -> str:
        """Phone type"""
        return "triphone"

    def _trainer_initialization(self) -> None:
        """Triphone training initialization"""
        self.tree_stats()
        self._setup_tree()

        self.compile_train_graphs()

        self.convert_alignments()
        os.rename(self.model_path, self.next_model_path)

    def tree_stats(self) -> None:
        """
        Multiprocessing function that computes stats for decision tree training.

        See Also
        --------
        :func:`~montreal_forced_aligner.acoustic_modeling.triphone.tree_stats_func`
            Multiprocessing helper function for each job
        :meth:`.TriphoneTrainer.tree_stats_arguments`
            Job method for generating arguments for the helper function
        :kaldi_src:`sum-tree-stats`
            Relevant Kaldi binary
        :kaldi_steps:`train_deltas`
            Reference Kaldi script
        :kaldi_steps:`train_lda_mllt`
            Reference Kaldi script
        :kaldi_steps:`train_sat`
            Reference Kaldi script

        """

        jobs = self.tree_stats_arguments()

        if self.use_mp:
            run_mp(tree_stats_func, jobs, self.working_log_directory)
        else:
            run_non_mp(tree_stats_func, jobs, self.working_log_directory)

        tree_accs = []
        for x in jobs:
            tree_accs.extend(x.treeacc_paths.values())
        log_path = os.path.join(self.working_log_directory, "sum_tree_acc.log")
        with open(log_path, "w", encoding="utf8") as log_file:
            subprocess.call(
                [
                    thirdparty_binary("sum-tree-stats"),
                    os.path.join(self.working_directory, "treeacc"),
                ]
                + tree_accs,
                stderr=log_file,
            )
        if not self.debug:
            for f in tree_accs:
                os.remove(f)

    def _setup_tree(self) -> None:
        """
        Set up the tree for the triphone model

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        log_path = os.path.join(self.working_log_directory, "questions.log")
        tree_path = os.path.join(self.working_directory, "tree")
        treeacc_path = os.path.join(self.working_directory, "treeacc")
        sets_int_path = os.path.join(self.worker.phones_dir, "sets.int")
        roots_int_path = os.path.join(self.worker.phones_dir, "roots.int")
        extra_question_int_path = os.path.join(self.worker.phones_dir, "extra_questions.int")
        topo_path = self.worker.topo_path
        questions_path = os.path.join(self.working_directory, "questions.int")
        questions_qst_path = os.path.join(self.working_directory, "questions.qst")
        with open(log_path, "w") as log_file:
            subprocess.call(
                [
                    thirdparty_binary("cluster-phones"),
                    treeacc_path,
                    sets_int_path,
                    questions_path,
                ],
                stderr=log_file,
            )

        with open(extra_question_int_path, "r") as inf, open(questions_path, "a") as outf:
            for line in inf:
                outf.write(line)

        log_path = os.path.join(self.working_log_directory, "compile_questions.log")
        with open(log_path, "w") as log_file:
            subprocess.call(
                [
                    thirdparty_binary("compile-questions"),
                    topo_path,
                    questions_path,
                    questions_qst_path,
                ],
                stderr=log_file,
            )

        log_path = os.path.join(self.working_log_directory, "build_tree.log")
        with open(log_path, "w") as log_file:
            subprocess.call(
                [
                    thirdparty_binary("build-tree"),
                    "--verbose=1",
                    f"--max-leaves={self.initial_gaussians}",
                    f"--cluster-thresh={self.cluster_threshold}",
                    treeacc_path,
                    roots_int_path,
                    questions_qst_path,
                    topo_path,
                    tree_path,
                ],
                stderr=log_file,
            )

        log_path = os.path.join(self.working_log_directory, "init_model.log")
        occs_path = os.path.join(self.working_directory, "0.occs")
        mdl_path = self.model_path
        with open(log_path, "w") as log_file:
            subprocess.call(
                [
                    thirdparty_binary("gmm-init-model"),
                    f"--write-occs={occs_path}",
                    tree_path,
                    treeacc_path,
                    topo_path,
                    mdl_path,
                ],
                stderr=log_file,
            )

        log_path = os.path.join(self.working_log_directory, "mixup.log")
        with open(log_path, "w") as log_file:
            subprocess.call(
                [
                    thirdparty_binary("gmm-mixup"),
                    f"--mix-up={self.initial_gaussians}",
                    mdl_path,
                    occs_path,
                    mdl_path,
                ],
                stderr=log_file,
            )
        os.remove(treeacc_path)
        os.rename(occs_path, self.next_occs_path)
        parse_logs(self.working_log_directory)
