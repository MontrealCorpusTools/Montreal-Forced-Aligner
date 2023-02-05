"""Class definitions for trainable aligners"""
from __future__ import annotations

import collections
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import time
import typing
from queue import Empty
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import tqdm
from sqlalchemy.orm import Session, joinedload, subqueryload

from montreal_forced_aligner.abc import KaldiFunction, ModelExporterMixin, TopLevelMfaWorker
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.data import MfaArguments, WorkflowType
from montreal_forced_aligner.db import CorpusWorkflow, Dictionary, Job
from montreal_forced_aligner.exceptions import ConfigError, KaldiProcessingError
from montreal_forced_aligner.helper import load_configuration, mfa_open, parse_old_features
from montreal_forced_aligner.models import AcousticModel, DictionaryModel
from montreal_forced_aligner.transcription.transcriber import TranscriberMixin
from montreal_forced_aligner.utils import (
    KaldiProcessWorker,
    Stopped,
    log_kaldi_errors,
    thirdparty_binary,
)

if TYPE_CHECKING:
    from dataclasses import dataclass

    from montreal_forced_aligner.abc import MetaDict
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

    model_path: str


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

    done_pattern = re.compile(
        r"^LOG \(post-to-tacc.*Done computing transition stats over (?P<utterances>\d+) utterances.*$"
    )

    def __init__(self, args: TransitionAccArguments):
        super().__init__(args)
        self.model_path = args.model_path

    def _run(self) -> typing.Generator[typing.Tuple[int, str]]:
        """Run the function"""

        with mfa_open(self.log_path, "w") as log_file, Session(self.db_engine) as session:
            job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            workflow: CorpusWorkflow = (
                session.query(CorpusWorkflow)
                .filter(CorpusWorkflow.current == True)  # noqa
                .first()
            )
            for dict_id in job.dictionary_ids:
                ali_path = job.construct_path(workflow.working_directory, "ali", "ark", dict_id)

                tacc_path = job.construct_path(workflow.working_directory, "t", "acc", dict_id)

                ali_post_proc = subprocess.Popen(
                    [
                        thirdparty_binary("ali-to-post"),
                        f"ark:{ali_path}",
                        "ark:-",
                    ],
                    stdout=subprocess.PIPE,
                    env=os.environ,
                    stderr=log_file,
                )

                tacc_proc = subprocess.Popen(
                    [
                        thirdparty_binary("post-to-tacc"),
                        self.model_path,
                        "ark:-",
                        tacc_path,
                    ],
                    stdin=ali_post_proc.stdout,
                    env=os.environ,
                    stderr=subprocess.PIPE,
                    encoding="utf8",
                )
                for line in tacc_proc.stderr:
                    log_file.write(line)
                    m = self.done_pattern.match(line.strip())
                    if m:
                        progress_update = int(m.group("utterances"))
                        yield progress_update
                self.check_call(tacc_proc)


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
                    "num_leaves": 2500,
                    "max_gaussians": 10000,
                },
            )
        )
        training_params.append(
            ("lda", {"subset": 20000, "num_leaves": 3000, "max_gaussians": 15000})
        )
        training_params.append(
            ("sat", {"subset": 20000, "num_leaves": 4000, "max_gaussians": 15000})
        )
        training_params.append(
            ("sat", {"subset": 50000, "num_leaves": 5000, "max_gaussians": 40000})
        )
        training_params.append(("pronunciation_probabilities", {"subset": 50000}))
        training_params.append(
            ("sat", {"subset": 150000, "num_leaves": 6000, "max_gaussians": 100000})
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
        config_path: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        unknown_args: Optional[typing.Iterable[str]] = None,
    ) -> MetaDict:
        """
        Parse configuration parameters from a config file and command line arguments

        Parameters
        ----------
        config_path: str, optional
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
            for i, (identifier, config) in enumerate(self.training_configs.items()):
                if isinstance(config, str):
                    continue
                config.non_silence_phones = self.non_silence_phones
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

    def setup(self) -> None:
        """Setup for acoustic model training"""
        super().setup()
        self.ignore_empty_utterances = True
        if self.initialized:
            return
        try:
            self.load_corpus()
            self.setup_trainers()
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
                "dictionary_path": self.dictionary_model.path,
                "corpus_directory": self.corpus_directory,
            }
        )
        return config

    @property
    def meta(self) -> MetaDict:
        """Metadata about the final round of training"""
        return self.training_configs[self.final_identifier].meta

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

    def export_model(self, output_model_path: str) -> None:
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
            # self.export_trained_rules(
            #    self.training_configs[self.final_identifier].working_directory
            # )
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

    @property
    def tree_path(self) -> str:
        """Tree path of the final model"""
        return self.training_configs[self.final_identifier].tree_path

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
                os.makedirs(self.working_directory, exist_ok=True)
                self.current_acoustic_model = AcousticModel(
                    previous.exported_model_path, self.working_directory
                )
                self.align()

            self.set_current_workflow(trainer.identifier)
            if trainer.identifier.startswith("pronunciation_probabilities"):
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
        counts_path = os.path.join(self.working_directory, "phone_pdf.counts")
        new_counts_path = os.path.join(previous.working_directory, "phone_pdf.counts")
        if not os.path.exists(new_counts_path):
            shutil.copyfile(counts_path, new_counts_path)

        phone_lm_path = os.path.join(self.phones_dir, "phone_lm.fst")
        new_phone_lm_path = os.path.join(previous.working_directory, "phone_lm.fst")
        if not os.path.exists(new_phone_lm_path) and os.path.exists(phone_lm_path):
            shutil.copyfile(phone_lm_path, new_phone_lm_path)
        logger.info(f"Completed training in {time.time()-begin} seconds!")

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
                getattr(self, "db_string", ""),
                os.path.join(self.working_log_directory, f"test_utterances.{j.id}.log"),
                self.model_path,
            )
            for j in self.jobs
        ]

    def compute_phone_pdf_counts(self) -> None:
        """
        Calculate the counts of pdfs corresponding to phones
        """
        try:

            logger.info("Accumulating transition stats...")

            begin = time.time()
            log_directory = self.working_log_directory
            os.makedirs(log_directory, exist_ok=True)
            arguments = self.transition_acc_arguments()
            with tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
                if GLOBAL_CONFIG.use_mp:
                    error_dict = {}
                    return_queue = mp.Queue()
                    stopped = Stopped()
                    procs = []
                    for i, args in enumerate(arguments):
                        function = TransitionAccFunction(args)
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
                        pbar.update(result)
                    for p in procs:
                        p.join()
                    if error_dict:
                        for v in error_dict.values():
                            raise v
                else:
                    logger.debug("Not using multiprocessing...")
                    for args in arguments:
                        function = TransitionAccFunction(args)
                        for result in function.run():
                            pbar.update(result)
            t_accs = []
            for j in self.jobs:
                for dict_id in j.dictionary_ids:
                    t_accs.append(j.construct_path(self.working_directory, "t", "acc", dict_id))
            subprocess.check_call(
                [
                    thirdparty_binary("vector-sum"),
                    "--binary=false",
                    *t_accs,
                    os.path.join(self.working_directory, "final.tacc"),
                ],
                stderr=subprocess.DEVNULL,
            )
            for f in t_accs:
                os.remove(f)
            smoothing = 1
            show_proc = subprocess.Popen(
                [
                    thirdparty_binary("show-transitions"),
                    self.phone_symbol_table_path,
                    self.model_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                encoding="utf8",
                env=os.environ,
            )
            phone_pdfs = {}
            phone, pdf = None, None
            max_pdf = 0
            max_phone = 0
            for line in show_proc.stdout:
                line = line.strip()
                m = re.match(
                    r"^Transition-state.*phone = (?P<phone>[^ ]+) .*pdf = (?P<pdf>\d+)$", line
                )
                if m:
                    phone = m.group("phone")
                    pdf = int(m.group("pdf"))
                    if pdf > max_pdf:
                        max_pdf = pdf
                    if self.phone_mapping[phone] > max_phone:
                        max_phone = self.phone_mapping[phone]
                else:
                    m = re.search(r"Transition-id = (?P<transition_id>\d+)", line)
                    if m:
                        transition_id = int(m.group("transition_id"))
                        phone_pdfs[transition_id] = (phone, pdf)
            with mfa_open(os.path.join(self.working_directory, "final.tacc"), "r") as f:
                data = f.read().strip().split()[1:-1]

                transition_counts = {
                    i: smoothing + int(float(x)) for i, x in enumerate(data) if i != 0
                }
            assert len(transition_counts) == len(phone_pdfs)
            pdf_counts = collections.Counter()
            pdf_phone_counts = collections.Counter()
            phone_pdf_mapping = collections.defaultdict(collections.Counter)
            for transition_id, (phone, pdf) in phone_pdfs.items():
                pdf_counts[pdf] += transition_counts[transition_id]
                pdf_phone_counts[(phone, pdf)] += transition_counts[transition_id]
                phone_pdf_mapping[phone][pdf] += transition_counts[transition_id]
            with mfa_open(os.path.join(self.working_directory, "phone_pdf.counts"), "w") as f:
                json.dump(phone_pdf_mapping, f, ensure_ascii=False)
            logger.debug(f"Accumulating transition stats took {time.time() - begin:.3f} seconds")
            logger.info("Finished accumulating transition stats!")

        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise

    def finalize_training(self):
        self.compute_phone_pdf_counts()
        self.collect_alignments()
        self.train_phone_lm()

    def export_files(
        self,
        output_directory: str,
        output_format: Optional[str] = None,
        include_original_text: bool = False,
    ) -> None:
        """
        Export a TextGrid file for every sound file in the dataset

        Parameters
        ----------
        output_directory: str
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
            return self.current_aligner.align_options
        return super().align_options

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
        if wf.done:
            logger.debug(f"Skipping {self.current_aligner.identifier} alignments")
            return
        try:
            self.current_acoustic_model.export_model(self.working_directory)
            self.uses_speaker_adaptation = False
            self.compile_train_graphs()
            self.align_utterances()
            if self.current_acoustic_model.meta["features"]["uses_speaker_adaptation"]:

                arguments = self.calc_fmllr_arguments()
                missing_transforms = False
                for arg in arguments:
                    for path in arg.trans_paths.values():
                        if not os.path.exists(path):
                            missing_transforms = True
                if missing_transforms:
                    assert self.alignment_model_path.endswith(".alimdl")
                    self.calc_fmllr()
                self.uses_speaker_adaptation = True
                assert self.alignment_model_path.endswith(".mdl")
                self.align_utterances()
            if self.current_subset:
                logger.debug(
                    f"Analyzing alignment diagnostics for {self.current_aligner.identifier} on {self.current_subset} utterances"
                )
            else:
                logger.debug(
                    f"Analyzing alignment diagnostics for {self.current_aligner.identifier} on the full corpus"
                )
            self.compile_information()
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
    def alignment_model_path(self) -> str:
        """Current alignment model path"""
        path = os.path.join(self.working_directory, "final.alimdl")
        if os.path.exists(path) and not self.uses_speaker_adaptation:
            return path
        return self.model_path

    @property
    def model_path(self) -> str:
        """Current model path"""
        if self.current_trainer is not None:
            return self.current_trainer.model_path
        return os.path.join(self.working_directory, "final.mdl")

    @property
    def data_directory(self) -> str:
        """Current data directory based on the trainer's subset"""
        return self.subset_directory(self.current_subset)

    @property
    def working_directory(self) -> Optional[str]:
        """Working directory"""
        if self.current_trainer is not None and not self.current_trainer.training_complete:
            return self.current_trainer.working_directory
        if self.current_aligner is None:
            return None
        return os.path.join(self.output_directory, f"{self.current_aligner.identifier}_ali")

    @property
    def working_log_directory(self) -> Optional[str]:
        """Current log directory"""
        return os.path.join(self.working_directory, "log")
