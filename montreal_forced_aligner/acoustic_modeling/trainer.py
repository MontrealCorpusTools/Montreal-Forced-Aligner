"""Class definitions for trainable aligners"""
from __future__ import annotations

import collections
import json
import logging
import os
import shutil
import time
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from _kalpy.hmm import AlignmentToPosterior
from _kalpy.matrix import DoubleVector
from kalpy.gmm.data import AlignmentArchive
from kalpy.gmm.utils import read_gmm_model
from sqlalchemy.orm import joinedload, subqueryload

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import (
    KaldiFunction,
    MetaDict,
    ModelExporterMixin,
    TopLevelMfaWorker,
)
from montreal_forced_aligner.data import MfaArguments, WorkflowType
from montreal_forced_aligner.db import (
    CorpusWorkflow,
    Dictionary,
    Job,
    PhoneInterval,
    Speaker,
    Utterance,
    WordInterval,
    bulk_update,
)
from montreal_forced_aligner.exceptions import ConfigError, KaldiProcessingError
from montreal_forced_aligner.helper import load_configuration, mfa_open, parse_old_features
from montreal_forced_aligner.models import AcousticModel, DictionaryModel
from montreal_forced_aligner.transcription.transcriber import TranscriberMixin
from montreal_forced_aligner.utils import log_kaldi_errors, run_kaldi_function

if TYPE_CHECKING:
    from dataclasses import dataclass

    from montreal_forced_aligner.acoustic_modeling.base import AcousticModelTrainingMixin
    from montreal_forced_aligner.acoustic_modeling.pronunciation_probabilities import (
        PronunciationProbabilityTrainer,
    )
else:
    from dataclassy import dataclass

__all__ = ["TrainableAligner", "TransitionAccFunction", "TransitionAccArguments"]


logger = logging.getLogger("mfa")


@dataclass
class TransitionAccArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.acoustic_modeling.trainer.TransitionAccFunction`"""

    working_directory: Path
    model_path: Path


class TransitionAccFunction(KaldiFunction):
    """
    Multiprocessing function to accumulate transition stats

    See Also
    --------
    :kaldi_src:`ali-to-post`
        Relevant Kaldi binary
    :kaldi_src:`post-to-tacc`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.acoustic_modeling.trainer.TransitionAccArguments`
        Arguments for the function
    """

    def __init__(self, args: TransitionAccArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.model_path = args.model_path

    def _run(self) -> None:
        """Run the function"""

        with self.session() as session:
            job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            transition_model, acoustic_model = read_gmm_model(self.model_path)
            for dict_id in job.dictionary_ids:
                ali_path = job.construct_path(self.working_directory, "ali", "ark", dict_id)
                if not ali_path.exists():
                    continue
                transition_accs = DoubleVector(transition_model.NumTransitionIds() + 1)
                alignment_archive = AlignmentArchive(ali_path)
                for alignment in alignment_archive:
                    post = AlignmentToPosterior(alignment.alignment)
                    for i in range(len(post)):
                        for j in range(len(post[i])):
                            tid = post[i][j][0]
                            transition_accs[tid] += post[i][j][1]
                    self.callback(1)
                self.callback(transition_accs)


class TrainableAligner(TranscriberMixin, TopLevelMfaWorker, ModelExporterMixin):
    """
    Train acoustic model

    Parameters
    ----------
    training_configuration : list[tuple[str, dict[str, Any]]]
        Training identifiers and parameters for training blocks
    phone_set_type: str
        Type of phone set to use for acoustic modeling

    See Also
    --------
    :class:`~montreal_forced_aligner.alignment.base.CorpusAligner`
        For dictionary and corpus parsing parameters and alignment parameters
    :class:`~montreal_forced_aligner.abc.TopLevelMfaWorker`
        For top-level parameters
    :class:`~montreal_forced_aligner.abc.ModelExporterMixin`
        For model export parameters

    Attributes
    ----------
    param_dict: dict[str, Any]
        Parameters to pass to training blocks
    final_identifier: str
        Identifier of the final training block
    current_subset: int
        Current training block's subset
    current_acoustic_model: :class:`~montreal_forced_aligner.models.AcousticModel`
        Acoustic model to use in aligning, based on previous training block
    training_configs: dict[str, :class:`~montreal_forced_aligner.acoustic_modeling.base.AcousticModelTrainingMixin`]
        Training blocks
    """

    def __init__(
        self,
        training_configuration: List[Tuple[str, Dict[str, Any]]] = None,
        phone_set_type: str = None,
        model_version: str = None,
        subset_word_count: int = 3,
        minimum_utterance_length: int = 2,
        **kwargs,
    ):
        self.param_dict = {
            k: v
            for k, v in kwargs.items()
            if not k.endswith("_directory")
            and not k.endswith("_path")
            and k not in ["speaker_characters"]
        }
        self.final_identifier = None
        self.current_subset: int = 0
        self.subset_word_count = subset_word_count
        self.current_aligner: Optional[AcousticModelTrainingMixin] = None
        self.current_trainer: Optional[AcousticModelTrainingMixin] = None
        self.current_acoustic_model: Optional[AcousticModel] = None
        super().__init__(**kwargs)
        if phone_set_type and phone_set_type != "UNKNOWN":
            self.dictionary_model = DictionaryModel(
                self.dictionary_model.path, phone_set_type=phone_set_type
            )
        self.phone_set_type = self.dictionary_model.phone_set_type
        os.makedirs(self.output_directory, exist_ok=True)
        self.training_configs: Dict[
            str, typing.Union[AcousticModelTrainingMixin, PronunciationProbabilityTrainer]
        ] = {}
        if training_configuration is None:
            training_configuration = TrainableAligner.default_training_configurations()
        for k, v in training_configuration:
            self.add_config(k, v)
        self.final_alignment = True
        self.model_version = model_version
        self.boost_silence = 1.5
        self.minimum_utterance_length = minimum_utterance_length

    @classmethod
    def default_training_configurations(cls) -> List[Tuple[str, Dict[str, Any]]]:
        """Default MFA training configuration"""
        training_params = []
        training_params.append(("monophone", {"subset": 10000, "boost_silence": 1.25}))
        training_params.append(
            (
                "triphone",
                {
                    "subset": 20000,
                    "boost_silence": 1.25,
                    "num_leaves": 2000,
                    "max_gaussians": 10000,
                },
            )
        )
        training_params.append(
            ("lda", {"subset": 20000, "num_leaves": 2500, "max_gaussians": 15000})
        )
        training_params.append(
            ("sat", {"subset": 20000, "num_leaves": 2500, "max_gaussians": 15000})
        )
        training_params.append(
            ("sat", {"subset": 50000, "num_leaves": 4200, "max_gaussians": 40000})
        )
        training_params.append(("pronunciation_probabilities", {"subset": 50000}))
        training_params.append(
            ("sat", {"subset": 150000, "num_leaves": 5000, "max_gaussians": 100000})
        )
        training_params.append(
            (
                "pronunciation_probabilities",
                {"subset": 150000, "optional": True},
            )
        )
        training_params.append(
            (
                "sat",
                {
                    "subset": 0,
                    "num_leaves": 7000,
                    "optional": True,
                    "max_gaussians": 150000,
                    "num_iterations": 20,
                    "quick": True,
                },
            )
        )
        return training_params

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[Path] = None,
        args: Optional[Dict[str, Any]] = None,
        unknown_args: Optional[typing.Iterable[str]] = None,
    ) -> MetaDict:
        """
        Parse configuration parameters from a config file and command line arguments

        Parameters
        ----------
        config_path: :class:`~pathlib.Path`, optional
            Path to yaml configuration file
        args: dict[str, Any]
            Parsed arguments
        unknown_args: list[str]
            Optional list of arguments that were not parsed

        Returns
        -------
        dict[str, Any]
            Dictionary of specified configuration parameters
        """
        global_params = {}
        training_params = []
        use_default = True
        if config_path:
            data = load_configuration(config_path)
            training_params = []
            for k, v in data.items():
                if k == "training":
                    for t in v:
                        for k2, v2 in t.items():
                            if "features" in v2:
                                global_params.update(parse_old_features(v2["features"]))
                                del v2["features"]
                            training_params.append((k2, v2))
                elif k == "features":
                    global_params.update(parse_old_features(v))
                else:
                    if v is None and k in cls.nullable_fields:
                        v = []
                    global_params[k] = v
            if training_params:
                use_default = False
        if use_default:  # default training configuration
            training_params = TrainableAligner.default_training_configurations()
        if training_params:
            if training_params[0][0] != "monophone":
                raise ConfigError("The first round of training must be monophone.")
        global_params["training_configuration"] = training_params
        global_params.update(cls.parse_args(args, unknown_args))
        return global_params

    def setup_trainers(self):
        self.write_training_information()
        with self.session() as session:
            workflows: typing.Dict[str, CorpusWorkflow] = {
                x.name: x for x in session.query(CorpusWorkflow)
            }
            for i, (identifier, c) in enumerate(self.training_configs.items()):
                if isinstance(c, str):
                    continue
                c.non_silence_phones = self.non_silence_phones
                ali_identifier = f"{identifier}_ali"
                if identifier not in workflows:
                    self.create_new_current_workflow(
                        WorkflowType.acoustic_training, name=identifier
                    )
                    self.create_new_current_workflow(WorkflowType.alignment, name=ali_identifier)
                else:
                    wf = workflows[identifier]
                    if wf.dirty and not wf.done:
                        shutil.rmtree(wf.working_directory, ignore_errors=True)
                    ali_wf = workflows[ali_identifier]
                    if ali_wf.dirty and not ali_wf.done:
                        shutil.rmtree(ali_wf.working_directory, ignore_errors=True)
                    if i == 0:
                        wf.current = True
            session.commit()

    def filter_training_utterances(self):
        logger.info("Filtering utterances for training...")
        with self.session() as session:
            dictionaries = session.query(Dictionary)
            for d in dictionaries:
                update_mapping = []
                word_mapping = d.word_mapping
                utterances = (
                    session.query(Utterance.id, Utterance.normalized_text)
                    .join(Utterance.speaker)
                    .filter(Utterance.ignored == False)  # noqa
                    .filter(Speaker.dictionary_id == d.id)
                )
                for u_id, text in utterances:
                    if not text:
                        update_mapping.append({"id": u_id, "ignored": True})
                        continue
                    words = text.split()
                    if (
                        self.minimum_utterance_length > 1
                        and len(words) < self.minimum_utterance_length
                    ):
                        update_mapping.append({"id": u_id, "ignored": True})
                        continue
                    if any(x in word_mapping for x in words):
                        continue
                    update_mapping.append({"id": u_id, "ignored": True})
                if update_mapping:
                    bulk_update(session, Utterance, update_mapping)
                    session.commit()

    def setup(self) -> None:
        """Setup for acoustic model training"""
        super().setup()
        self.initialized = self.features_generated
        if self.initialized:
            logger.info("Using previous initialization.")
            return
        try:
            self.load_corpus()
            self.setup_trainers()
            self.filter_training_utterances()
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise
        self.initialized = True

    @property
    def configuration(self) -> MetaDict:
        """Configuration for the worker"""
        config = super().configuration
        config.update(
            {
                "dictionary_path": str(self.dictionary_model.path),
                "corpus_directory": str(self.corpus_directory),
            }
        )
        return config

    @property
    def meta(self) -> MetaDict:
        """Metadata about the final round of training"""
        meta = self.training_configs[self.final_identifier].meta
        if self.model_version is not None:
            meta["version"] = self.model_version
        return meta

    def add_config(self, train_type: str, params: MetaDict) -> None:
        """
        Add a trainer to the pipeline

        Parameters
        ----------
        train_type: str
            Type of trainer to add, one of ``monophone``, ``triphone``, ``lda`` or ``sat``
        params: dict[str, Any]
            Parameters to initialize trainer

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.ConfigError`
            If an invalid train_type is specified
        """
        from montreal_forced_aligner.acoustic_modeling.lda import LdaTrainer
        from montreal_forced_aligner.acoustic_modeling.monophone import MonophoneTrainer
        from montreal_forced_aligner.acoustic_modeling.pronunciation_probabilities import (  # noqa
            PronunciationProbabilityTrainer,
        )
        from montreal_forced_aligner.acoustic_modeling.sat import SatTrainer
        from montreal_forced_aligner.acoustic_modeling.triphone import TriphoneTrainer

        p = {}
        p.update(self.param_dict)
        p.update(params)
        identifier = train_type
        index = 2
        while identifier in self.training_configs:
            identifier = f"{train_type}_{index}"
            index += 1
        self.final_identifier = identifier
        if train_type == "monophone":
            p = {
                k: v for k, v in p.items() if k in MonophoneTrainer.get_configuration_parameters()
            }
            config = MonophoneTrainer(identifier=identifier, worker=self, **p)

        elif train_type == "triphone":
            p = {k: v for k, v in p.items() if k in TriphoneTrainer.get_configuration_parameters()}
            config = TriphoneTrainer(identifier=identifier, worker=self, **p)
        elif train_type == "lda":
            p = {k: v for k, v in p.items() if k in LdaTrainer.get_configuration_parameters()}
            config = LdaTrainer(identifier=identifier, worker=self, **p)
        elif train_type == "sat":
            p = {k: v for k, v in p.items() if k in SatTrainer.get_configuration_parameters()}
            config = SatTrainer(identifier=identifier, worker=self, **p)
        elif train_type == "pronunciation_probabilities":
            p = {
                k: v
                for k, v in p.items()
                if k in PronunciationProbabilityTrainer.get_configuration_parameters()
            }
            previous_trainer = self.training_configs[list(self.training_configs.keys())[-1]]
            config = PronunciationProbabilityTrainer(
                identifier=identifier, previous_trainer=previous_trainer, worker=self, **p
            )
        else:
            raise ConfigError(f"Invalid training type '{train_type}' in config file")

        self.training_configs[identifier] = config

    def export_model(self, output_model_path: Path) -> None:
        """
        Export an acoustic model to the specified path

        Parameters
        ----------
        output_model_path : str
            Path to save acoustic model
        """
        if "pronunciation_probabilities" in self.training_configs:
            export_directory = os.path.dirname(output_model_path)
            if export_directory:
                os.makedirs(export_directory, exist_ok=True)
            self.export_trained_rules(
                self.training_configs[self.final_identifier].working_directory
            )
            with self.session() as session:
                for d in session.query(Dictionary):
                    base_name = self.dictionary_base_names[d.id]
                    if d.use_g2p:
                        shutil.copyfile(
                            self.phone_symbol_table_path,
                            os.path.join(
                                self.training_configs[self.final_identifier].working_directory,
                                "phones.txt",
                            ),
                        )
                        shutil.copyfile(
                            self.grapheme_symbol_table_path,
                            os.path.join(
                                self.training_configs[self.final_identifier].working_directory,
                                "graphemes.txt",
                            ),
                        )
                        shutil.copyfile(
                            d.lexicon_fst_path,
                            os.path.join(
                                self.training_configs[self.final_identifier].working_directory,
                                self.dictionary_base_names[d.id] + ".fst",
                            ),
                        )
                        shutil.copyfile(
                            d.align_lexicon_path,
                            os.path.join(
                                self.training_configs[self.final_identifier].working_directory,
                                self.dictionary_base_names[d.id] + "_align.fst",
                            ),
                        )
                    else:
                        output_dictionary_path = os.path.join(
                            export_directory, base_name + ".dict"
                        )
                        self.export_lexicon(
                            d.id,
                            output_dictionary_path,
                            probability=True,
                        )
        self.training_configs[self.final_identifier].export_model(output_model_path)
        logger.info(f"Saved model to {output_model_path}")

    def quality_check_subset(self):
        from _kalpy.util import Int32VectorWriter
        from kalpy.gmm.data import AlignmentArchive
        from kalpy.utils import generate_write_specifier

        with self.session() as session:
            utterance_ids = set(
                x[0]
                for x in session.query(Utterance.id)
                .filter(Utterance.in_subset == True, Utterance.duration_deviation > 10)  # noqa
                .all()
            )
            logger.debug(
                f"Removing {len(utterance_ids)} utterances from subset due to large duration deviations"
            )
            bulk_update(session, Utterance, [{"id": x, "in_subset": False} for x in utterance_ids])
            session.commit()
            for j in self.jobs:
                ali_paths = j.construct_path_dictionary(self.working_directory, "ali", "ark")
                temp_ali_paths = j.construct_path_dictionary(
                    self.working_directory, "temp_ali", "ark"
                )
                for dict_id, ali_path in ali_paths.items():
                    if not ali_path.exists():
                        continue
                    new_path = temp_ali_paths[dict_id]
                    write_specifier = generate_write_specifier(new_path)
                    writer = Int32VectorWriter(write_specifier)
                    alignment_archive = AlignmentArchive(ali_path)

                    for alignment in alignment_archive:
                        if alignment.utterance_id in utterance_ids:
                            continue
                        writer.Write(str(alignment.utterance_id), alignment.alignment)
                    del alignment_archive
                    writer.Close()
                    ali_path.unlink()
                    new_path.rename(ali_path)
                    feat_path = j.construct_path(
                        j.corpus.current_subset_directory, "feats", "scp", dictionary_id=dict_id
                    )
                    feat_lines = []
                    with mfa_open(feat_path, "r") as feat_file:
                        for line in feat_file:
                            utterance_id = line.split(maxsplit=1)[0]
                            if utterance_id in utterance_ids:
                                continue
                            feat_lines.append(line)

                    with mfa_open(feat_path, "w") as feat_file:
                        for line in feat_lines:
                            feat_file.write(line)

    def train(self) -> None:
        """
        Run through the training configurations to produce a final acoustic model
        """
        self.setup()
        previous = None
        begin = time.time()
        for trainer in self.training_configs.values():
            if self.current_subset is None and trainer.optional:
                logger.info(
                    "Exiting training early to save time as the corpus is below the subset size for later training stages"
                )
                break
            if trainer.subset < self.num_utterances:
                self.current_subset = trainer.subset
            else:
                self.current_subset = None
                trainer.subset = 0
            self.subset_directory(self.current_subset)
            if previous is not None:
                self.set_current_workflow(f"{previous.identifier}_ali")
                self.current_aligner = previous
                os.makedirs(self.working_log_directory, exist_ok=True)
                self.current_acoustic_model = AcousticModel(
                    previous.exported_model_path, self.working_directory
                )
                if (
                    not self.current_workflow.done
                    or not self.current_workflow.working_directory.exists()
                ):
                    self.align()
                    with self.session() as session:
                        session.query(WordInterval).delete()
                        session.query(PhoneInterval).delete()
                        session.commit()
                    self.collect_alignments()
                    self.analyze_alignments()
                    if self.current_subset != 0:
                        self.quality_check_subset()
                else:
                    logger.debug(f"Skipping {self.current_aligner.identifier} alignments")

            self.set_current_workflow(trainer.identifier)
            if trainer.identifier.startswith("pronunciation_probabilities"):
                with self.session() as session:
                    session.query(WordInterval).delete()
                    session.query(PhoneInterval).delete()
                    session.commit()
                trainer.train_pronunciation_probabilities()
            else:
                trainer.train()
            previous = trainer
            self.final_identifier = trainer.identifier
        self.current_subset = None
        self.current_aligner = previous
        self.set_current_workflow(f"{previous.identifier}_ali")

        os.makedirs(self.working_log_directory, exist_ok=True)
        self.current_acoustic_model = AcousticModel(
            previous.exported_model_path, self.working_directory
        )
        self.acoustic_model = AcousticModel(previous.exported_model_path, self.working_directory)
        self.align()
        self.finalize_training()
        counts_path = self.working_directory.joinpath("phone_pdf.counts")
        new_counts_path = os.path.join(previous.working_directory, "phone_pdf.counts")
        if not os.path.exists(new_counts_path):
            shutil.copyfile(counts_path, new_counts_path)

        phone_lm_path = os.path.join(self.phones_dir, "phone_lm.fst")
        new_phone_lm_path = os.path.join(previous.working_directory, "phone_lm.fst")
        if not os.path.exists(new_phone_lm_path) and os.path.exists(phone_lm_path):
            shutil.copyfile(phone_lm_path, new_phone_lm_path)
        logger.info(f"Completed training in {time.time() - begin} seconds!")

    def transition_acc_arguments(self) -> List[TransitionAccArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.acoustic_modeling.trainer.TransitionAccArguments`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.trainer.TransitionAccArguments`]
            Arguments for processing
        """

        return [
            TransitionAccArguments(
                j.id,
                getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                self.working_log_directory.joinpath(f"test_utterances.{j.id}.log"),
                self.working_directory,
                self.model_path,
            )
            for j in self.jobs
        ]

    def compute_phone_pdf_counts(self) -> None:
        """
        Calculate the counts of pdfs corresponding to phones
        """
        phone_pdf_counts_path = self.working_directory.joinpath("phone_pdf.counts")
        if phone_pdf_counts_path.exists():
            return
        logger.info("Accumulating transition stats...")

        begin = time.time()
        log_directory = self.working_log_directory
        os.makedirs(log_directory, exist_ok=True)
        arguments = self.transition_acc_arguments()
        transition_model, acoustic_model = read_gmm_model(self.model_path)
        transition_accs = DoubleVector(transition_model.NumTransitionIds() + 1)
        for result in run_kaldi_function(
            TransitionAccFunction, arguments, total_count=self.num_utterances
        ):
            if not isinstance(result, int):
                transition_accs.AddVec(1.0, result)
        smoothing = 1
        phone_pdf_mapping = collections.defaultdict(collections.Counter)
        for tid in range(1, transition_model.NumTransitionIds() + 1):
            pdf_id = transition_model.TransitionIdToPdf(tid)
            phone_id = transition_model.TransitionIdToPhone(tid)
            phone = self.reversed_phone_mapping[phone_id]
            t_count = smoothing + float(transition_accs[tid])
            phone_pdf_mapping[phone][pdf_id] += t_count
        with mfa_open(phone_pdf_counts_path, "w") as f:
            json.dump(phone_pdf_mapping, f, ensure_ascii=False)
        logger.debug(f"Accumulating transition stats took {time.time() - begin:.3f} seconds")
        logger.info("Finished accumulating transition stats!")

    def finalize_training(self):
        with self.session() as session:
            session.query(WordInterval).delete()
            session.query(PhoneInterval).delete()
            session.commit()
        self.compute_phone_pdf_counts()
        self.collect_alignments()
        self.analyze_alignments()
        self.train_phone_lm()

    def export_files(
        self,
        output_directory: Path,
        output_format: Optional[str] = None,
        include_original_text: bool = False,
    ) -> None:
        """
        Export a TextGrid file for every sound file in the dataset

        Parameters
        ----------
        output_directory: :class:`~pathlib.Path`
            Directory to save to
        output_format: str, optional
            Format to save alignments, one of 'long_textgrids' (the default), 'short_textgrids', or 'json', passed to praatio
        include_original_text: bool
            Flag for including the original text of the corpus files as a tier
        """
        self.align()
        super(TrainableAligner, self).export_files(
            output_directory, output_format, include_original_text
        )

    @property
    def num_current_utterances(self) -> int:
        """Number of utterances in the current subset"""
        if self.current_subset and self.current_subset < self.num_utterances:
            return self.current_subset
        return self.num_utterances

    @property
    def align_options(self) -> MetaDict:
        """Alignment options"""
        if self.current_aligner is not None:
            options = self.current_aligner.align_options
        else:
            options = super().align_options
        return options

    def align(self) -> None:
        """
        Multiprocessing function that aligns based on the current model.

        See Also
        --------
        :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignFunction`
            Multiprocessing helper function for each job
        :meth:`.AlignMixin.align_arguments`
            Job method for generating arguments for the helper function
        :kaldi_steps:`align_si`
            Reference Kaldi script
        :kaldi_steps:`align_fmllr`
            Reference Kaldi script
        """
        wf = self.current_workflow
        if wf.done and wf.working_directory.exists():
            logger.debug(f"Skipping {self.current_aligner.identifier} alignments")
            return
        try:
            self.current_acoustic_model.export_model(self.working_directory)
            self.uses_speaker_adaptation = False
            if self.current_acoustic_model.meta["features"]["uses_speaker_adaptation"]:
                self.uses_speaker_adaptation = False
                for j in self.jobs:
                    for path in j.construct_path_dictionary(
                        j.corpus.current_subset_directory, "trans", "scp"
                    ).values():
                        path.unlink(missing_ok=True)
            self.compile_train_graphs()
            self.align_utterances()
            if self.current_acoustic_model.meta["features"]["uses_speaker_adaptation"]:
                assert self.alignment_model_path.suffix == ".alimdl"
                self.calc_fmllr()
                self.uses_speaker_adaptation = True
                assert self.alignment_model_path.suffix == ".mdl"

                self.align_utterances()
            with self.session() as session:
                session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update(
                    {"done": True}
                )
                session.commit()
        except Exception as e:
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
    def alignment_model_path(self) -> Path:
        """Current alignment model path"""
        path = self.working_directory.joinpath("final.alimdl")
        if os.path.exists(path) and not self.uses_speaker_adaptation:
            return path
        return self.model_path

    @property
    def model_path(self) -> Path:
        """Current model path"""
        if self.current_trainer is not None:
            return self.current_trainer.model_path
        return self.working_directory.joinpath("final.mdl")

    @property
    def data_directory(self) -> str:
        """Current data directory based on the trainer's subset"""
        return self.subset_directory(self.current_subset)

    @property
    def working_directory(self) -> Optional[Path]:
        """Working directory"""
        if self.current_trainer is not None and not self.current_trainer.training_complete:
            return self.current_trainer.working_directory
        if self.current_aligner is None:
            return None
        return self.output_directory.joinpath(f"{self.current_aligner.identifier}_ali")

    @property
    def working_log_directory(self) -> Optional[str]:
        """Current log directory"""
        return self.working_directory.joinpath("log")
