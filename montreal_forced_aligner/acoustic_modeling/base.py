"""Class definition for BaseTrainer"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import time
from abc import abstractmethod
from queue import Empty
from typing import TYPE_CHECKING, Dict, List

import sqlalchemy.engine
import tqdm
from sqlalchemy.orm import Session

from montreal_forced_aligner.abc import MfaWorker, ModelExporterMixin, TrainerMixin
from montreal_forced_aligner.alignment import AlignMixin
from montreal_forced_aligner.alignment.multiprocessing import AccStatsArguments, AccStatsFunction
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusPronunciationMixin
from montreal_forced_aligner.corpus.features import FeatureConfigMixin
from montreal_forced_aligner.db import Utterance
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.utils import (
    KaldiProcessWorker,
    Stopped,
    log_kaldi_errors,
    parse_logs,
    thirdparty_binary,
)

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict
    from montreal_forced_aligner.corpus.multiprocessing import Job


__all__ = ["AcousticModelTrainingMixin"]


class AcousticModelTrainingMixin(
    AlignMixin, TrainerMixin, FeatureConfigMixin, MfaWorker, ModelExporterMixin
):
    """
    Base trainer class for training acoustic models and ivector extractors

    Parameters
    ----------
    identifier : str
        Identifier for the trainer
    worker: :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusPronunciationMixin`
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
        optional: bool = False,
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
        self.boost_silence = boost_silence
        self.training_complete = False
        self.optional = optional
        self.realignment_iterations = []  # Gets set later
        self.final_gaussian_iteration = 0  # Gets set later

    @property
    def db_path(self):
        return self.worker.db_path

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
                j.name,
                self.db_path,
                os.path.join(self.working_directory, "log", f"acc.{self.iteration}.{j.name}.log"),
                j.dictionary_ids,
                feat_strings[j.name],
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                j.construct_path_dictionary(self.working_directory, str(self.iteration), "acc"),
                self.model_path,
            )
            for j in self.jobs
        ]

    @property
    def previous_aligner(self) -> AcousticCorpusPronunciationMixin:
        """Previous aligner seeding training"""
        return self.worker

    def utterances(self, session: Session = None) -> sqlalchemy.orm.Query:
        """
        Get all utterances in the trainer's root worker

        Parameters
        ----------
        session: sqlalchemy.orm.Session, optional
           Session to use in querying

        Returns
        -------
        sqlalchemy.orm.Query
            Utterance query
        """
        return self.worker.utterances(session)

    def log_debug(self, message: str = "") -> None:
        """
        Log a debug message. This function is a wrapper around the worker's :meth:`logging.Logger.debug`

        Parameters
        ----------
        message: str
            Debug message to log
        """
        self.worker.log_debug(message)

    def log_error(self, message: str = "") -> None:
        """
        Log an info message. This function is a wrapper around the worker's :meth:`logging.Logger.info`

        Parameters
        ----------
        message: str
            Info message to log
        """
        self.worker.log_error(message)

    def log_warning(self, message: str = "") -> None:
        """
        Log a warning message. This function is a wrapper around the worker's :meth:`logging.Logger.warning`

        Parameters
        ----------
        message: str
            Warning message to log
        """
        self.worker.log_warning(message)

    def log_info(self, message: str = "") -> None:
        """
        Log an error message. This function is a wrapper around the worker's :meth:`logging.Logger.error`

        Parameters
        ----------
        message: str
            Error message to log
        """
        self.worker.log_info(message)

    @property
    def jobs(self) -> List[Job]:
        """Top-level worker's job objects"""
        return self.worker.jobs

    @property
    def db_engine(self) -> sqlalchemy.engine.Engine:
        """Top-level worker's DB engine"""
        return self.worker.db_engine

    def session(self, **kwargs) -> sqlalchemy.orm.session.Session:
        """Top-level worker's DB engine"""
        return self.worker.session(**kwargs)

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
    def num_current_utterances(self) -> int:
        """Number of utterances of the corpus"""
        if self.subset:
            return self.subset
        return self.worker.num_utterances

    def initialize_training(self) -> None:
        """Initialize training"""
        self.compute_calculated_properties()
        self.current_gaussians = self.initial_gaussians
        begin = time.time()
        dirty_path = os.path.join(self.working_directory, "dirty")
        done_path = os.path.join(self.working_directory, "done")
        if os.path.exists(dirty_path):  # if there was an error, let's redo from scratch
            shutil.rmtree(self.working_directory)
        if os.path.exists(done_path):
            self.log_info(f"{self.identifier} training already done, skipping initialization.")
            self.training_complete = True
            return
        self.log_info(f"Initializing training for {self.identifier}...")
        os.makedirs(self.working_directory, exist_ok=True)
        os.makedirs(self.working_log_directory, exist_ok=True)
        if self.subset and self.subset >= self.worker.num_utterances:
            self.log_warning(
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

                logger = logging.getLogger(self.identifier)
                log_kaldi_errors(e.error_logs, logger)
                e.update_log_file(logger)
            raise
        self.iteration = 1
        self.worker.current_trainer = self
        self.log_info("Initialization complete!")
        self.log_debug(f"Initialization for {self.identifier} took {time.time() - begin} seconds")

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
        self.log_info("Accumulating statistics...")
        arguments = self.acc_stats_arguments()
        with tqdm.tqdm(
            total=self.num_current_utterances, disable=getattr(self, "quiet", False)
        ) as pbar:
            if self.use_mp:
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
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    if isinstance(result, KaldiProcessingError):
                        error_dict[result.job_name] = result
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
            est_command = [
                thirdparty_binary("gmm-est"),
                f"--write-occs={self.next_occs_path}",
                f"--mix-up={self.current_gaussians}",
            ]
            if self.power > 0:
                est_command.append(f"--power={self.power}")
            est_command.extend(
                [
                    self.model_path,
                    "-",
                    self.next_model_path,
                ]
            )
            est_proc = subprocess.Popen(
                est_command,
                stdin=sum_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            est_proc.communicate()
        avg_like_pattern = re.compile(
            r"Overall avg like per frame.* = (?P<like>[-.,\d]+) over (?P<frames>[.\d+e]+) frames"
        )
        average_logdet_pattern = re.compile(
            r"Overall average logdet is (?P<logdet>[-.,\d]+) over (?P<frames>[.\d+e]+) frames"
        )
        avg_like_sum = 0
        avg_like_frames = 0
        average_logdet_sum = 0
        average_logdet_frames = 0
        for a in arguments:
            with open(a.log_path, "r", encoding="utf8") as f:
                for line in f:
                    m = avg_like_pattern.search(line)
                    if m:
                        like = float(m.group("like"))
                        frames = float(m.group("frames"))
                        avg_like_sum += like * frames
                        avg_like_frames += frames
                    m = average_logdet_pattern.search(line)
                    if m:
                        logdet = float(m.group("logdet"))
                        frames = float(m.group("frames"))
                        average_logdet_sum += logdet * frames
                        average_logdet_frames += frames
        if avg_like_frames:
            log_like = avg_like_sum / avg_like_frames
            if average_logdet_frames:
                log_like += average_logdet_sum / average_logdet_frames
            self.log_debug(f"Likelihood for iteration {self.iteration}: {log_like}")

        if not self.debug:
            for f in acc_files:
                os.remove(f)

    def align_iteration(self):
        begin = time.time()
        self.align_utterances()
        self.log_debug(
            f"Generating alignments for iteration {self.iteration} took {time.time()-begin} seconds"
        )
        self.log_debug(f"Analyzing information for alignment in iteration {self.iteration}...")
        begin = time.time()
        self.compile_information()
        self.log_debug(
            f"Analyzing iteration {self.iteration} alignments took {time.time()-begin} seconds"
        )

    def train_iteration(self):
        """Perform an iteration of training"""
        if os.path.exists(self.next_model_path):
            self.iteration += 1
            return
        if self.iteration in self.realignment_iterations:
            self.align_iteration()
        self.acc_stats()

        parse_logs(self.working_log_directory)
        if self.iteration <= self.final_gaussian_iteration:
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
        try:
            self.initialize_training()
            if self.training_complete:
                return
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

                logger = logging.getLogger(self.identifier)
                log_kaldi_errors(e.error_logs, logger)
                e.update_log_file(logger)
            raise
        with open(done_path, "w"):
            pass
        self.log_info("Training complete!")
        self.log_debug(f"Training took {time.time() - begin} seconds")

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
        ali_model_path = os.path.join(self.working_directory, f"{self.num_iterations+1}.alimdl")
        if os.path.exists(ali_model_path):
            shutil.copy(
                ali_model_path,
                os.path.join(self.working_directory, "final.alimdl"),
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
    def gaussian_increment(self) -> int:
        """Amount by which gaussians should be increased each iteration"""
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

        from sqlalchemy import func

        from ..utils import get_mfa_version

        with self.worker.session() as session:
            summary = session.query(
                func.count(Utterance.id),
                func.sum(Utterance.duration),
                func.avg(Utterance.alignment_log_likelihood / Utterance.num_frames),
            ).filter(
                Utterance.alignment_log_likelihood != None  # noqa
            )
            utterance_count, duration, average_log_likelihood = summary.first()
        data = {
            "phones": sorted(self._generate_non_positional_list(self.non_silence_phones)),
            "version": get_mfa_version(),
            "architecture": self.architecture,
            "train_date": str(datetime.now()),
            "training": {
                "audio_duration": duration,
                "num_speakers": self.worker.num_speakers,
                "num_utterances": utterance_count,
                "num_oovs": sum(self.worker.oovs_found.values()),
                "average_log_likelihood": average_log_likelihood,
            },
            "features": self.feature_options,
            "oov_phone": self.worker.oov_phone,
            "optional_silence_phone": self.worker.optional_silence_phone,
            "phone_set_type": str(self.worker.phone_set_type),
            "silence_probability": self.worker.silence_probability,
            "initial_silence_probability": self.worker.initial_silence_probability,
            "final_silence_correction": self.worker.final_silence_correction,
            "final_non_silence_correction": self.worker.final_non_silence_correction,
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
