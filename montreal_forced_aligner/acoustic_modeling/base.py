"""Class definition for BaseTrainer"""
from __future__ import annotations

import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List

import sqlalchemy.engine
from _kalpy.gmm import AccumAmDiagGmm
from _kalpy.matrix import DoubleVector
from kalpy.gmm.utils import read_gmm_model, write_gmm_model
from kalpy.utils import kalpy_logger
from sqlalchemy.orm import Session

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import MfaWorker, ModelExporterMixin, TrainerMixin
from montreal_forced_aligner.alignment import AlignMixin
from montreal_forced_aligner.alignment.multiprocessing import AccStatsArguments, AccStatsFunction
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusPronunciationMixin
from montreal_forced_aligner.corpus.features import FeatureConfigMixin
from montreal_forced_aligner.data import PhoneType
from montreal_forced_aligner.db import CorpusWorkflow, Phone, Utterance
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.utils import log_kaldi_errors, parse_logs, run_kaldi_function

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict
    from montreal_forced_aligner.db import Job


__all__ = ["AcousticModelTrainingMixin"]


logger = logging.getLogger("mfa")


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
        boost_silence: float = 1.0,
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
    def db_string(self) -> str:
        """Root worker's database connection string"""
        return self.worker.db_string

    def acc_stats_arguments(self) -> List[AccStatsArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsArguments`]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            arguments.append(
                AccStatsArguments(
                    j.id,
                    self.session if config.USE_THREADING else self.db_string,
                    self.working_log_directory.joinpath(f"acc.{self.iteration}.{j.id}.log"),
                    self.working_directory,
                    self.model_path,
                )
            )
        return arguments

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

    @property
    def jobs(self) -> List[Job]:
        """Top-level worker's job objects"""
        return self.worker.jobs

    @property
    def db_engine(self) -> sqlalchemy.engine.Engine:
        """Top-level worker's database engine"""
        return self.worker.db_engine

    def session(self, **kwargs) -> sqlalchemy.orm.session.Session:
        """Top-level worker's database session"""
        return self.worker.session(**kwargs)

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

    @property
    def workflow(self):
        with self.session() as session:
            wf = (
                session.query(CorpusWorkflow)
                .filter(CorpusWorkflow.name == self.identifier)
                .first()
            )
        return wf

    def initialize_training(self) -> None:
        """Initialize training"""
        begin = time.time()
        logger.info(f"Initializing training for {self.identifier}...")
        if self.subset and self.subset >= self.worker.num_utterances:
            logger.warning(
                "Subset specified is larger than the dataset, "
                "using full corpus for this training block."
            )
            self.subset = 0
            self.worker.current_subset = 0
        self.working_log_directory.mkdir(parents=True, exist_ok=True)
        self._trainer_initialization()
        self.iteration = 1
        self.worker.current_trainer = self
        self.compute_calculated_properties()
        self.current_gaussians = self.initial_gaussians
        logger.info("Initialization complete!")
        logger.debug(
            f"Initialization for {self.identifier} took {time.time() - begin:.3f} seconds"
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
    def working_directory(self) -> Path:
        """Training directory"""
        return self.worker.output_directory.joinpath(self.identifier)

    @property
    def working_log_directory(self) -> Path:
        """Training log directory"""
        return self.working_directory.joinpath("log")

    @property
    def model_path(self) -> Path:
        """Current acoustic model path"""
        if self.workflow.done:
            return self.next_model_path
        return self.working_directory.joinpath(f"{self.iteration}.mdl")

    @property
    def alignment_model_path(self) -> Path:
        """Alignment model path"""
        return self.model_path

    @property
    def next_model_path(self) -> Path:
        """Next iteration's acoustic model path"""
        if self.workflow.done:
            return self.working_directory.joinpath("final.mdl")
        return self.working_directory.joinpath(f"{self.iteration + 1}.mdl")

    @abstractmethod
    def compute_calculated_properties(self) -> None:
        """Compute any calculated properties such as alignment iterations"""
        ...

    def increment_gaussians(self) -> None:
        """Increment the current number of gaussians"""
        self.current_gaussians += self.gaussian_increment

    def acc_stats(self) -> None:
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
        logger.info("Accumulating statistics...")
        arguments = self.acc_stats_arguments()

        transition_model, acoustic_model = read_gmm_model(self.model_path)
        transition_accs = DoubleVector()
        gmm_accs = AccumAmDiagGmm()
        transition_model.InitStats(transition_accs)
        gmm_accs.init(acoustic_model)
        for result in run_kaldi_function(
            AccStatsFunction, arguments, total_count=self.num_current_utterances
        ):
            if isinstance(result, tuple):
                job_transition_accs, job_gmm_accs = result

                transition_accs.AddVec(1.0, job_transition_accs)
                gmm_accs.Add(1.0, job_gmm_accs)

        log_path = self.working_log_directory.joinpath(f"update.{self.iteration}.log")
        with kalpy_logger("kalpy.train", log_path) as train_logger:
            train_logger.debug(f"Model path: {self.model_path}")
            train_logger.debug(f"Next model path: {self.next_model_path}")
            train_logger.debug(f"Current gaussians: {self.current_gaussians}")
            train_logger.debug(f"Power: {self.power}")
            objf_impr, count = transition_model.mle_update(transition_accs)
            train_logger.debug(
                f"Transition model update: Overall {objf_impr / count} "
                f"log-like improvement per frame over {count} frames."
            )
            objf_impr, count = acoustic_model.mle_update(
                gmm_accs, mixup=self.current_gaussians, power=self.power
            )
            train_logger.debug(
                f"GMM update: Overall {objf_impr / count} "
                f"objective function improvement per frame over {count} frames."
            )
            tot_like = gmm_accs.TotLogLike()
            tot_t = gmm_accs.TotCount()
            train_logger.debug(
                f"Average Likelihood per frame for iteration {self.iteration} = {tot_like / tot_t} "
                f"over {tot_t} frames."
            )
            logger.debug(f"Log likelihood for iteration {self.iteration}: {tot_like / tot_t}")
            write_gmm_model(str(self.next_model_path), transition_model, acoustic_model)

    def align_iteration(self) -> None:
        """Run alignment for a training iteration"""
        begin = time.time()
        self.align_utterances(training=True)
        logger.debug(
            f"Generating alignments for iteration {self.iteration} took {time.time() - begin} seconds"
        )

    @property
    def initialized(self) -> bool:
        return (
            self.working_directory.joinpath("1.mdl").exists()
            or self.working_directory.joinpath("final.mdl").exists()
            or self.working_directory.joinpath("done").exists()
        )

    def train_iteration(self) -> None:
        """Perform an iteration of training"""
        if self.next_model_path.exists():
            self.iteration += 1
            if self.iteration <= self.final_gaussian_iteration:
                self.increment_gaussians()
            return
        if self.iteration in self.realignment_iterations:
            self.align_iteration()
        self.acc_stats()

        parse_logs(self.working_log_directory)
        if self.iteration <= self.final_gaussian_iteration:
            self.increment_gaussians()
        self.iteration += 1

    def train(self) -> None:
        """
        Train the model

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        self.working_log_directory.mkdir(parents=True, exist_ok=True)
        wf = self.worker.current_workflow
        if wf.done:
            return
        try:
            self.initialize_training()

            begin = time.time()
            for iteration in range(1, self.num_iterations + 1):
                logger.info(f"{self.identifier} - Iteration {iteration} of {self.num_iterations}")
                self.iteration = iteration
                self.train_iteration()
            self.finalize_training()
        except Exception as e:
            if not isinstance(e, KeyboardInterrupt):
                with self.session() as session:
                    session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update(
                        {"dirty": True}
                    )
                    session.commit()
                if isinstance(e, KaldiProcessingError):
                    log_kaldi_errors(e.error_logs)
                    e.update_log_file()
            raise
        logger.info("Training complete!")
        logger.debug(f"Training took {time.time() - begin:.3f} seconds")

    @property
    def exported_model_path(self) -> Path:
        """Model path to export to once training is complete"""
        return self.working_log_directory.joinpath("acoustic_model.zip")

    def finalize_training(self) -> None:
        """
        Finalize the training, renaming all final iteration model files as "final", and exporting
        the model to be used in the next round alignment

        """
        self.working_directory.joinpath(f"{self.num_iterations + 1}.mdl").rename(
            self.working_directory.joinpath("final.mdl")
        )
        ali_model_path = self.working_directory.joinpath(f"{self.num_iterations + 1}.alimdl")
        if ali_model_path.exists():
            ali_model_path.rename(self.working_directory.joinpath("final.alimdl"))
        self.export_model(self.exported_model_path)
        if not config.DEBUG:
            for i in range(1, self.num_iterations + 1):
                model_path = self.working_directory.joinpath(f"{i}.mdl")
                try:
                    model_path.unlink(missing_ok=True)
                except FileNotFoundError:
                    pass
            for file in self.working_directory.iterdir():
                if any(file.name.startswith(x) for x in ["fsts.", "trans.", "ali."]):
                    file.unlink(missing_ok=True)
        wf = self.worker.current_workflow
        with self.session() as session:
            session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update({"done": True})
            session.commit()
        self.worker.current_trainer = None

    @property
    def dictionary_base_names(self):
        return self.worker.dictionary_base_names

    @property
    def lexicon_compilers(self):
        return self.worker.lexicon_compilers

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
    def use_g2p(self):
        return self.worker.use_g2p

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
                func.sum(Utterance.alignment_log_likelihood) / func.sum(Utterance.num_frames),
            ).filter(
                Utterance.alignment_log_likelihood != None  # noqa
            )
            utterance_count, duration, average_log_likelihood = summary.first()
        try:
            default_dict = self.worker.dictionary_base_names[self.worker._default_dictionary_id]
        except KeyError:
            from montreal_forced_aligner.db import Dictionary

            with self.session() as session:
                default_dict = (
                    session.query(Dictionary.name)
                    .filter(Dictionary.default == True)  # noqa
                    .first()[0]
                )
        non_silence_phones = self.non_silence_phones
        if not non_silence_phones:
            phone_mapping = {}
            with self.worker.session() as session:
                query = session.query(
                    Phone.kaldi_label, Phone.phone, Phone.mapping_id, Phone.phone_type
                ).filter(Phone.phone_type != PhoneType.disambiguation)
                for kaldi_label, phone, m_id, phone_type in query:
                    if phone_type is PhoneType.non_silence:
                        non_silence_phones.add(phone)
                    phone_mapping[kaldi_label] = m_id
        else:
            phone_mapping = self.phone_mapping

        data = {
            "phones": sorted(self._generate_non_positional_list(non_silence_phones)),
            "phone_mapping": {k: v for k, v in phone_mapping.items() if not k.startswith("#")},
            "phone_groups": self.worker.phone_groups,
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
            "dictionaries": {
                "names": sorted(self.worker.dictionary_base_names.values()),
                "default": default_dict,
                "silence_word": self.worker.silence_word,
                "use_g2p": self.worker.use_g2p,
                "oov_word": self.worker.oov_word,
                "bracketed_word": self.worker.bracketed_word,
                "laughter_word": self.worker.laughter_word,
                "clitic_marker": self.worker.clitic_marker,
                "position_dependent_phones": self.worker.position_dependent_phones,
            },
            "language": str(self.worker.language),
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

    def export_model(self, output_model_path: Path) -> None:
        """
        Export an acoustic model to the specified path

        Parameters
        ----------
        output_model_path : str
            Path to save acoustic model
        """
        directory = output_model_path.parent

        acoustic_model = AcousticModel.empty(
            output_model_path.stem, root_directory=self.working_log_directory
        )
        acoustic_model.add_meta_file(self.worker)
        acoustic_model.add_model(self.working_directory)
        acoustic_model.add_model(self.worker.phones_dir)
        acoustic_model.add_pronunciation_models(
            self.working_directory, self.worker.dictionary_base_names.values()
        )
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
        acoustic_model.dump(output_model_path)
