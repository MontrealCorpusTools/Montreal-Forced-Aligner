"""Class definition for BaseTrainer"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
import shutil
import statistics
import subprocess
import time
from abc import abstractmethod
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple

import tqdm

from montreal_forced_aligner.abc import MfaWorker, ModelExporterMixin, TrainerMixin
from montreal_forced_aligner.alignment import AlignMixin
from montreal_forced_aligner.alignment.multiprocessing import AccStatsArguments, AccStatsFunction
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusPronunciationMixin
from montreal_forced_aligner.corpus.features import FeatureConfigMixin
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.helper import align_phones
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.textgrid import process_ctm_line
from montreal_forced_aligner.utils import (
    KaldiProcessWorker,
    Stopped,
    log_kaldi_errors,
    parse_logs,
    run_mp,
    run_non_mp,
    thirdparty_binary,
)

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict
    from montreal_forced_aligner.corpus.classes import UtteranceCollection
    from montreal_forced_aligner.corpus.multiprocessing import Job
    from montreal_forced_aligner.textgrid import CtmInterval


__all__ = ["AcousticModelTrainingMixin"]


class AlignmentImprovementArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.acoustic_modeling.base.compute_alignment_improvement_func`"""

    log_path: str
    dictionaries: List[str]
    model_path: str
    text_int_paths: Dict[str, str]
    word_boundary_paths: Dict[str, str]
    ali_paths: Dict[str, str]
    frame_shift: int
    reversed_phone_mappings: Dict[int, str]
    positions: List[str]
    phone_ctm_paths: Dict[str, str]


def compute_alignment_improvement_func(
    log_path: str,
    dictionaries: List[str],
    model_path: str,
    text_int_paths: Dict[str, str],
    word_boundary_paths: Dict[str, str],
    ali_paths: Dict[str, str],
    frame_shift: int,
    reversed_phone_mappings: Dict[int, str],
    positions: List[str],
    phone_ctm_paths: Dict[str, str],
) -> None:
    """
    Multiprocessing function for computing alignment improvement over training

    See Also
    --------
    :meth:`.AcousticModelTrainingMixin.compute_alignment_improvement`
        Main function that calls this function in parallel
    :meth:`.AcousticModelTrainingMixin.alignment_improvement_arguments`
        Job method for generating arguments for the helper function
    :kaldi_src:`linear-to-nbest`
        Relevant Kaldi binary
    :kaldi_src:`lattice-determinize-pruned`
        Relevant Kaldi binary
    :kaldi_src:`lattice-align-words`
        Relevant Kaldi binary
    :kaldi_src:`lattice-to-phone-lattice`
        Relevant Kaldi binary
    :kaldi_src:`nbest-to-ctm`
        Relevant Kaldi binary

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: list[str]
        List of dictionary names
    model_path: str
        Path to the acoustic model file
    text_int_paths: dict[str, str]
        Dictionary of text int files per dictionary name
    word_boundary_paths: dict[str, str]
        Dictionary of word boundary files per dictionary name
    ali_paths: dict[str, str]
        Dictionary of alignment archives per dictionary name
    frame_shift: int
        Frame shift of feature generation, in ms
    reversed_phone_mappings: dict[str, dict[int, str]]
        Mapping of phone IDs to phone labels per dictionary name
    positions: dict[str, list[str]]
        Positions per dictionary name
    phone_ctm_paths: dict[str, str]
        Dictionary of phone ctm files per dictionary name
    """
    try:

        frame_shift = frame_shift / 1000
        with open(log_path, "w", encoding="utf8") as log_file:
            for dict_name in dictionaries:
                text_int_path = text_int_paths[dict_name]
                ali_path = ali_paths[dict_name]
                phone_ctm_path = phone_ctm_paths[dict_name]
                word_boundary_path = word_boundary_paths[dict_name]
                if os.path.exists(phone_ctm_path):
                    continue

                lin_proc = subprocess.Popen(
                    [
                        thirdparty_binary("linear-to-nbest"),
                        f"ark:{ali_path}",
                        f"ark:{text_int_path}",
                        "",
                        "",
                        "ark:-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                det_proc = subprocess.Popen(
                    [thirdparty_binary("lattice-determinize-pruned"), "ark:-", "ark:-"],
                    stdin=lin_proc.stdout,
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )
                align_proc = subprocess.Popen(
                    [
                        thirdparty_binary("lattice-align-words"),
                        word_boundary_path,
                        model_path,
                        "ark:-",
                        "ark:-",
                    ],
                    stdin=det_proc.stdout,
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )
                phone_proc = subprocess.Popen(
                    [thirdparty_binary("lattice-to-phone-lattice"), model_path, "ark:-", "ark:-"],
                    stdin=align_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                nbest_proc = subprocess.Popen(
                    [
                        thirdparty_binary("nbest-to-ctm"),
                        f"--frame-shift={frame_shift}",
                        "ark:-",
                        phone_ctm_path,
                    ],
                    stdin=phone_proc.stdout,
                    stderr=log_file,
                    env=os.environ,
                )
                nbest_proc.communicate()
                mapping = reversed_phone_mappings
                actual_lines = []
                with open(phone_ctm_path, "r", encoding="utf8") as f:
                    for line in f:
                        line = line.strip()
                        if line == "":
                            continue
                        line = line.split(" ")
                        utt = line[0]
                        begin = float(line[2])
                        duration = float(line[3])
                        end = begin + duration
                        label = line[4]
                        try:
                            label = mapping[int(label)]
                        except KeyError:
                            pass
                        for p in positions:
                            if label.endswith(p):
                                label = label[: -1 * len(p)]
                        actual_lines.append([utt, begin, end, label])
                with open(phone_ctm_path, "w", encoding="utf8") as f:
                    for line in actual_lines:
                        f.write(f"{' '.join(map(str, line))}\n")
    except Exception as e:
        raise (Exception(str(e)))


def compare_alignments(
    alignments_one: Dict[str, List[CtmInterval]],
    alignments_two: Dict[str, List[CtmInterval]],
    silence_phone: str,
) -> Tuple[Optional[int], Optional[float]]:
    """
    Compares two sets of alignments for difference

    See Also
    --------
    :meth:`.AcousticModelTrainingMixin.compute_alignment_improvement`
        Main function that calls this function

    Parameters
    ----------
    alignments_one: dict[str, list[tuple[float, float, str]]]
        First set of alignments
    alignments_two: dict[str, list[tuple[float, float, str]]]
        Second set of alignments
    silence_phone: str
        Label of optional silence phone

    Returns
    -------
    Optional[int]
        Difference in number of aligned files
    Optional[float]
        Mean boundary difference between the two alignments
    """
    utterances_aligned_diff = len(alignments_two) - len(alignments_one)
    utts_one = set(alignments_one.keys())
    utts_two = set(alignments_two.keys())
    common_utts = utts_one.intersection(utts_two)
    differences = []
    for u in common_utts:
        one_alignment = alignments_one[u]
        two_alignment = alignments_two[u]
        avg_overlap_diff, phone_error_rate = align_phones(
            one_alignment, two_alignment, silence_phone
        )
        if avg_overlap_diff is None:
            return None, None
        differences.append(avg_overlap_diff)
    if differences:
        mean_difference = statistics.mean(differences)
    else:
        mean_difference = None
    return utterances_aligned_diff, mean_difference


class AcousticModelTrainingMixin(
    AlignMixin, TrainerMixin, FeatureConfigMixin, MfaWorker, ModelExporterMixin
):
    """
    Base trainer class for training acoustic models and ivector extractors

    Parameters
    ----------
    identifier : str
        Identifier for the trainer
    worker: :class:`~montreal_forced_aligner.corpus.acoustic.AcousticCorpusPronunciationMixin`
        Top-level worker
    num_iterations : int
        Number of iterations, defaults to 40
    subset : int
        Number of utterances to use, defaults to 0 which will use the whole corpus
    max_gaussians : int
        Total number of gaussians, defaults to 1000
    boost_silence : float
        Factor by which to boost silence during alignment, defaults to 1.25
    power : float
        Exponent for number of gaussians according to occurrence counts, defaults to 0.25
    initial_gaussians : int
        Initial number of gaussians, defaults to 0

    See Also
    --------
    :class:`~montreal_forced_aligner.alignment.mixins.AlignMixin`
        For alignment parameters
    :class:`~montreal_forced_aligner.abc.TrainerMixin`
        For training parameters
    :class:`~montreal_forced_aligner.corpus.features.FeatureConfigMixin`
        For feature generation parameters
    :class:`~montreal_forced_aligner.abc.MfaWorker`
        For MFA processing parameters
    :class:`~montreal_forced_aligner.abc.ModelExporterMixin`
        For model export parameters

    Attributes
    ----------
    realignment_iterations : list
        Iterations to perform alignment
    """

    architecture = "gmm-hmm"

    def __init__(
        self,
        identifier: str,
        worker: AcousticCorpusPronunciationMixin,
        num_iterations: int = 40,
        subset: int = 0,
        max_gaussians: int = 1000,
        boost_silence: float = 1.25,
        power: float = 0.25,
        initial_gaussians: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.identifier = identifier
        self.worker = worker
        self.num_iterations = num_iterations
        self.subset = subset
        self.max_gaussians = max_gaussians
        self.power = power
        self.initial_gaussians = initial_gaussians
        self.current_gaussians = initial_gaussians
        self.boost_silence = boost_silence
        self.training_complete = False
        self.realignment_iterations = []  # Gets set later

    def acc_stats_arguments(self) -> List[AccStatsArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsArguments`]
            Arguments for processing
        """
        feat_strings = self.worker.construct_feature_proc_strings()
        return [
            AccStatsArguments(
                os.path.join(self.working_directory, "log", f"acc.{self.iteration}.{j.name}.log"),
                j.current_dictionary_names,
                feat_strings[j.name],
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                j.construct_path_dictionary(self.working_directory, str(self.iteration), "acc"),
                self.model_path,
            )
            for j in self.jobs
        ]

    def alignment_improvement_arguments(self) -> List[AlignmentImprovementArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.base.compute_alignment_improvement_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.base.AlignmentImprovementArguments`]
            Arguments for processing
        """
        positions = self.positions
        phone_mapping = self.reversed_phone_mapping
        return [
            AlignmentImprovementArguments(
                os.path.join(self.working_log_directory, f"alignment_analysis.{j.name}.log"),
                j.current_dictionary_names,
                self.model_path,
                j.construct_path_dictionary(self.data_directory, "text", "int.scp"),
                j.word_boundary_int_files(),
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                self.frame_shift,
                phone_mapping,
                positions,
                j.construct_path_dictionary(
                    self.working_directory, f"phone.{self.iteration}", "ctm"
                ),
            )
            for j in self.jobs
        ]

    @property
    def previous_aligner(self) -> AcousticCorpusPronunciationMixin:
        """Previous aligner seeding training"""
        return self.worker

    @property
    def utterances(self) -> UtteranceCollection:
        return self.worker.utterances

    def log_debug(self, message: str) -> None:
        """
        Log a debug message. This function is a wrapper around the worker's :meth:`logging.Logger.debug`

        Parameters
        ----------
        message: str
            Debug message to log
        """
        self.worker.log_debug(message)

    def log_error(self, message: str) -> None:
        """
        Log an info message. This function is a wrapper around the worker's :meth:`logging.Logger.info`

        Parameters
        ----------
        message: str
            Info message to log
        """
        self.worker.log_error(message)

    def log_warning(self, message: str) -> None:
        """
        Log a warning message. This function is a wrapper around the worker's :meth:`logging.Logger.warning`

        Parameters
        ----------
        message: str
            Warning message to log
        """
        self.worker.log_warning(message)

    def log_info(self, message: str) -> None:
        """
        Log an error message. This function is a wrapper around the worker's :meth:`logging.Logger.error`

        Parameters
        ----------
        message: str
            Error message to log
        """
        self.worker.log_info(message)

    @property
    def logger(self) -> logging.Logger:
        """Top-level worker's logger"""
        return self.worker.logger

    @property
    def jobs(self) -> List[Job]:
        """Top-level worker's job objects"""
        return self.worker.jobs

    @property
    def disambiguation_symbols_int_path(self) -> str:
        """Path to the disambiguation int file"""
        return self.worker.disambiguation_symbols_int_path

    def construct_feature_proc_strings(
        self, speaker_independent: bool = False
    ) -> List[Dict[str, str]]:
        """Top-level worker's feature strings"""
        return self.worker.construct_feature_proc_strings(speaker_independent)

    def construct_base_feature_string(self, all_feats: bool = False) -> str:
        """Top-level worker's base feature string"""
        return self.worker.construct_base_feature_string(all_feats)

    @property
    def data_directory(self) -> str:
        """Get the current data directory based on subset"""
        return self.worker.data_directory

    @property
    def corpus_output_directory(self) -> str:
        """Directory of the corpus"""
        return self.worker.corpus_output_directory

    @property
    def num_utterances(self) -> int:
        """Number of utterances of the corpus"""
        if self.subset:
            return self.subset
        return self.worker.num_utterances

    def initialize_training(self) -> None:
        """Initialize training"""
        self.compute_calculated_properties()
        self.current_gaussians = 0
        begin = time.time()
        dirty_path = os.path.join(self.working_directory, "dirty")
        done_path = os.path.join(self.working_directory, "done")
        if os.path.exists(dirty_path):  # if there was an error, let's redo from scratch
            shutil.rmtree(self.working_directory)
        self.logger.info(f"Initializing training for {self.identifier}...")
        if os.path.exists(done_path):
            self.training_complete = True
            return
        os.makedirs(self.working_directory, exist_ok=True)
        os.makedirs(self.working_log_directory, exist_ok=True)
        if self.subset and self.subset >= len(self.worker.utterances):
            self.logger.warning(
                "Subset specified is larger than the dataset, "
                "using full corpus for this training block."
            )
            self.subset = 0
            self.worker.current_subset = 0
        try:
            self._trainer_initialization()
            parse_logs(self.working_log_directory)
        except Exception as e:
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise
        self.iteration = 1
        self.worker.current_trainer = self
        self.logger.info("Initialization complete!")
        self.logger.debug(
            f"Initialization for {self.identifier} took {time.time() - begin} seconds"
        )

    @abstractmethod
    def _trainer_initialization(self) -> None:
        """Descendant classes will override this for their own training initialization"""
        ...

    def acoustic_model_training_params(self) -> MetaDict:
        """Configuration parameters"""
        return {
            "subset": self.subset,
            "num_iterations": self.num_iterations,
            "max_gaussians": self.max_gaussians,
            "power": self.power,
            "initial_gaussians": self.initial_gaussians,
        }

    @property
    def working_directory(self) -> str:
        """Training directory"""
        return os.path.join(self.worker.output_directory, self.identifier)

    @property
    def working_log_directory(self) -> str:
        """Training log directory"""
        return os.path.join(self.working_directory, "log")

    @property
    def model_path(self) -> str:
        """Current acoustic model path"""
        if self.training_complete:
            return self.next_model_path
        return os.path.join(self.working_directory, f"{self.iteration}.mdl")

    @property
    def alignment_model_path(self) -> str:
        """Alignment model path"""
        return self.model_path

    @property
    def next_model_path(self):
        """Next iteration's acoustic model path"""
        if self.training_complete:
            return os.path.join(self.working_directory, "final.mdl")
        return os.path.join(self.working_directory, f"{self.iteration + 1}.mdl")

    @property
    def next_occs_path(self):
        """Next iteration's occs file path"""
        if self.training_complete:
            return os.path.join(self.working_directory, "final.occs")
        return os.path.join(self.working_directory, f"{self.iteration + 1}.occs")

    @abstractmethod
    def compute_calculated_properties(self) -> None:
        """Compute any calculated properties such as alignment iterations"""
        ...

    def increment_gaussians(self):
        """Increment the current number of gaussians"""
        self.current_gaussians += self.gaussian_increment

    def acc_stats(self):
        """
        Multiprocessing function that accumulates stats for GMM training.

        See Also
        --------
        :class:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsFunction`
            Multiprocessing helper function for each job
        :meth:`.AcousticModelTrainingMixin.acc_stats_arguments`
            Job method for generating arguments for the helper function
        :kaldi_src:`gmm-sum-accs`
            Relevant Kaldi binary
        :kaldi_src:`gmm-est`
            Relevant Kaldi binary
        :kaldi_steps:`train_mono`
            Reference Kaldi script
        :kaldi_steps:`train_deltas`
            Reference Kaldi script
        """
        self.logger.info("Accumulating statistics...")
        arguments = self.acc_stats_arguments()
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

        log_path = os.path.join(self.working_log_directory, f"update.{self.iteration}.log")
        with open(log_path, "w") as log_file:
            acc_files = []
            for a in arguments:
                acc_files.extend(a.acc_paths.values())
            sum_proc = subprocess.Popen(
                [thirdparty_binary("gmm-sum-accs"), "-"] + acc_files,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            est_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-est"),
                    f"--write-occs={self.next_occs_path}",
                    f"--mix-up={self.current_gaussians}",
                    f"--power={self.power}",
                    self.model_path,
                    "-",
                    self.next_model_path,
                ],
                stdin=sum_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            est_proc.communicate()
        avg_like_pattern = re.compile(
            r"Overall avg like per frame \(Gaussian only\) = (?P<like>[-.,\d]+) over (?P<frames>[.\d+e]) frames"
        )
        average_logdet_pattern = re.compile(
            r"Overall average logdet is (?P<logdet>[-.,\d]+) over (?P<frames>[.\d+e]) frames"
        )
        avg_like_sum = 0
        avg_like_frames = 0
        average_logdet_sum = 0
        average_logdet_frames = 0
        for a in arguments:
            with open(a.log_path, "r", encoding="utf8") as f:
                for line in f:
                    m = re.search(avg_like_pattern, line)
                    if m:
                        like = float(m.group("like"))
                        frames = float(m.group("frames"))
                        avg_like_sum += like * frames
                        avg_like_frames += frames
                    m = re.search(average_logdet_pattern, line)
                    if m:
                        logdet = float(m.group("logdet"))
                        frames = float(m.group("frames"))
                        average_logdet_sum += logdet * frames
                        average_logdet_frames += frames
        if avg_like_frames:
            log_like = avg_like_sum / avg_like_frames
            if average_logdet_frames:
                log_like += average_logdet_sum / average_logdet_frames
            self.logger.debug(f"Likelihood for iteration {self.iteration}: {log_like}")

        if not self.debug:
            for f in acc_files:
                os.remove(f)

    def parse_iteration_alignments(
        self, iteration: Optional[int] = None
    ) -> Dict[str, List[CtmInterval]]:
        """
        Function to parse phone CTMs in a given iteration

        Parameters
        ----------
        iteration: int, optional
            Iteration to compute over

        Returns
        -------
        dict[str, list[CtmInterval]]
            Per utterance CtmIntervals
        """
        data = {}
        for j in self.alignment_improvement_arguments():
            for phone_ctm_path in j.phone_ctm_paths.values():
                if iteration is not None:
                    phone_ctm_path = phone_ctm_path.replace(
                        f"phone.{self.iteration}", f"phone.{iteration}"
                    )
                with open(phone_ctm_path, "r", encoding="utf8") as f:
                    for line in f:
                        line = line.strip()
                        if line == "":
                            continue
                        interval = process_ctm_line(line)
                        if interval.utterance not in data:
                            data[interval.utterance] = []
                        data[interval.utterance].append(interval)
        return data

    def compute_alignment_improvement(self) -> None:
        """
        Computes aligner improvements in terms of number of aligned files and phone boundaries
        for debugging purposes
        """
        jobs = self.alignment_improvement_arguments()
        if self.use_mp:
            run_mp(compute_alignment_improvement_func, jobs, self.working_log_directory)
        else:
            run_non_mp(compute_alignment_improvement_func, jobs, self.working_log_directory)

        alignment_diff_path = os.path.join(self.working_directory, "train_change.csv")
        if self.iteration == 0 or self.iteration not in self.realignment_iterations:
            return
        ind = self.realignment_iterations.index(self.iteration)
        if ind != 0:
            previous_iteration = self.realignment_iterations[ind - 1]
        else:
            previous_iteration = 0
        try:
            previous_alignments = self.parse_iteration_alignments(previous_iteration)
        except FileNotFoundError:
            return
        current_alignments = self.parse_iteration_alignments()
        utterance_aligned_diff, mean_difference = compare_alignments(
            previous_alignments, current_alignments, self.optional_silence_phone
        )
        if utterance_aligned_diff:
            self.log_warning(
                "Cannot compare alignments, install the biopython package to use this functionality."
            )
            return
        if not os.path.exists(alignment_diff_path):
            with open(alignment_diff_path, "w", encoding="utf8") as f:
                f.write(
                    "iteration,number_aligned,number_previously_aligned,"
                    "difference_in_utts_aligned,mean_boundary_change\n"
                )
        if self.iteration in self.realignment_iterations:
            with open(alignment_diff_path, "a", encoding="utf8") as f:
                f.write(
                    f"{self.iteration},{len(current_alignments)},{len(previous_alignments)},"
                    f"{utterance_aligned_diff},{mean_difference}\n"
                )

    def train_iteration(self):
        """Perform an iteration of training"""
        if os.path.exists(self.next_model_path):
            self.iteration += 1
            return
        if self.iteration in self.realignment_iterations:
            self.align_utterances()
            self.logger.debug(
                f"Analyzing information for alignment in iteration {self.iteration}..."
            )
            self.compile_information()
            if self.debug:
                self.compute_alignment_improvement()
        self.acc_stats()

        parse_logs(self.working_log_directory)
        if self.iteration < self.final_gaussian_iteration:
            self.increment_gaussians()
        self.iteration += 1

    def train(self):
        """
        Train the model

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        done_path = os.path.join(self.working_directory, "done")
        dirty_path = os.path.join(self.working_directory, "dirty")
        if os.path.exists(done_path):
            self.logger.info(f"{self.identifier} training already done, skipping initialization.")
            return
        try:
            self.initialize_training()
            begin = time.time()
            for iteration in range(1, self.num_iterations + 1):
                self.log_info(
                    f"{self.identifier} - Iteration {iteration} of {self.num_iterations}"
                )
                self.iteration = iteration
                self.train_iteration()
            self.finalize_training()
        except Exception as e:
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise
        with open(done_path, "w"):
            pass
        self.logger.info("Training complete!")
        self.logger.debug(f"Training took {time.time() - begin} seconds")

    @property
    def exported_model_path(self) -> str:
        """Model path to export to once training is complete"""
        return os.path.join(self.working_log_directory, "acoustic_model.zip")

    def finalize_training(self) -> None:
        """
        Finalize the training, renaming all final iteration model files as "final", and exporting
        the model to be used in the next round alignment

        """
        shutil.copy(
            os.path.join(self.working_directory, f"{self.num_iterations+1}.mdl"),
            os.path.join(self.working_directory, "final.mdl"),
        )
        shutil.copy(
            os.path.join(self.working_directory, f"{self.num_iterations+1}.occs"),
            os.path.join(self.working_directory, "final.occs"),
        )
        self.export_model(self.exported_model_path)
        if not self.debug:
            for i in range(1, self.num_iterations + 1):
                model_path = os.path.join(self.working_directory, f"{i}.mdl")
                try:
                    os.remove(model_path)
                except FileNotFoundError:
                    pass
                try:
                    os.remove(os.path.join(self.working_directory, f"{i}.occs"))
                except FileNotFoundError:
                    pass
        self.training_complete = True
        self.worker.current_trainer = None

    @property
    def final_gaussian_iteration(self) -> int:
        """Final iteration to increase gaussians"""
        return self.num_iterations - 10

    @property
    def gaussian_increment(self) -> int:
        """Amount by which gaussians should be increases each iteration"""
        return int((self.max_gaussians - self.initial_gaussians) / self.final_gaussian_iteration)

    @property
    def train_type(self) -> str:
        """Training type, not implemented for BaseTrainer"""
        raise NotImplementedError

    @property
    def phone_type(self) -> str:
        """Phone type, not implemented for BaseTrainer"""
        raise NotImplementedError

    @property
    def meta(self) -> MetaDict:
        """Generate metadata for the acoustic model that was trained"""
        from datetime import datetime

        from ..utils import get_mfa_version

        data = {
            "phones": sorted(self.non_silence_phones),
            "version": get_mfa_version(),
            "architecture": self.architecture,
            "train_date": str(datetime.now()),
            "training": {
                "audio_duration": sum(x.duration for x in self.worker.utterances),
                "num_speakers": self.worker.num_speakers,
                "num_utterances": self.worker.num_utterances,
                "num_oovs": sum(self.worker.oovs_found.values()),
                "average_log_likelihood": statistics.mean(
                    x.alignment_log_likelihood
                    for x in self.worker.utterances
                    if x.alignment_log_likelihood
                ),
            },
            "features": self.feature_options,
            "phone_set_type": str(self.worker.phone_set_type),
        }
        return data

    def export_model(self, output_model_path: str) -> None:
        """
        Export an acoustic model to the specified path

        Parameters
        ----------
        output_model_path : str
            Path to save acoustic model
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
