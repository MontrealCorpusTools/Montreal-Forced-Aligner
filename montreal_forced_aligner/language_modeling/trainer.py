"""Classes for training language models"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import typing
from pathlib import Path
from queue import Empty, Queue

import sqlalchemy
from tqdm.rich import tqdm

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import DatabaseMixin, MfaWorker, TopLevelMfaWorker, TrainerMixin
from montreal_forced_aligner.corpus.text_corpus import TextCorpusMixin
from montreal_forced_aligner.data import WordType, WorkflowType
from montreal_forced_aligner.db import Dictionary, Utterance, Word
from montreal_forced_aligner.dictionary.mixins import DictionaryMixin
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionaryMixin
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.language_modeling.multiprocessing import (
    TrainLmArguments,
    TrainLmFunction,
)
from montreal_forced_aligner.models import LanguageModel
from montreal_forced_aligner.utils import KaldiProcessWorker, thirdparty_binary

if typing.TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict

__all__ = [
    "LmCorpusTrainerMixin",
    "LmTrainerMixin",
    "MfaLmArpaTrainer",
    "LmDictionaryCorpusTrainerMixin",
    "MfaLmCorpusTrainer",
    "MfaLmDictionaryCorpusTrainer",
]

logger = logging.getLogger("mfa")


class LmTrainerMixin(DictionaryMixin, TrainerMixin, MfaWorker):
    """
    Abstract mixin class for training language models

    Parameters
    ----------
    prune_method: str
        Pruning method for pruning the ngram model, defaults to "relative_entropy"
    prune_thresh_small: float
        Pruning threshold for the small language model, defaults to 0.0000003
    prune_thresh_medium: float
        Pruning threshold for the medium language model, defaults to 0.0000001

    See Also
    --------
    :class:`~montreal_forced_aligner.dictionary.mixins.DictionaryMixin`
        For dictionary parsing parameters
    :class:`~montreal_forced_aligner.abc.TrainerMixin`
        For training parameters
    :class:`~montreal_forced_aligner.abc.MfaWorker`
        For worker parameters
    """

    def __init__(
        self,
        prune_method="relative_entropy",
        order: int = 3,
        method: str = "kneser_ney",
        prune_thresh_small=0.0000003,
        prune_thresh_medium=0.0000001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prune_method = prune_method
        self.order = order
        self.method = method
        self.prune_thresh_small = prune_thresh_small
        self.prune_thresh_medium = prune_thresh_medium

    @property
    def mod_path(self) -> Path:
        """Internal temporary path to the model file"""
        return self.working_directory.joinpath(f"{self.data_source_identifier}.mod")

    @property
    def far_path(self) -> Path:
        """Internal temporary path to the FAR file"""
        return self.working_directory.joinpath(f"{self.data_source_identifier}.far")

    @property
    def large_arpa_path(self) -> Path:
        """Internal temporary path to the large arpa file"""
        return self.working_directory.joinpath(f"{self.data_source_identifier}.arpa")

    @property
    def medium_arpa_path(self) -> Path:
        """Internal temporary path to the medium arpa file"""
        return self.working_directory.joinpath(f"{self.data_source_identifier}_medium.arpa")

    @property
    def small_arpa_path(self) -> Path:
        """Internal temporary path to the small arpa file"""
        return self.working_directory.joinpath(f"{self.data_source_identifier}_small.arpa")

    def initialize_training(self) -> None:
        """Initialize training"""
        pass

    def train_iteration(self) -> None:
        """Run one training iteration"""
        pass

    def finalize_training(self) -> None:
        """Run one training iteration"""
        pass

    def prune_large_language_model(self) -> None:
        """Prune the large language model into small and medium versions"""
        logger.info("Pruning large ngram model to medium and small versions...")
        small_mod_path = self.mod_path.with_stem(self.mod_path.stem + "_small")
        med_mod_path = self.mod_path.with_stem(self.mod_path.stem + "_med")
        with mfa_open(self.working_log_directory.joinpath("prune.log"), "w") as log_file:
            subprocess.check_call(
                [
                    "ngramshrink",
                    f"--method={self.prune_method}",
                    f"--theta={self.prune_thresh_medium}",
                    self.mod_path,
                    med_mod_path,
                ],
                stderr=log_file,
            )
            assert med_mod_path.exists()
            if getattr(self, "sym_path", None):
                subprocess.check_call(
                    [
                        "ngramprint",
                        "--ARPA",
                        f"--symbols={self.sym_path}",
                        med_mod_path,
                        self.medium_arpa_path,
                    ],
                    stderr=log_file,
                )
            else:
                subprocess.check_call(
                    ["ngramprint", "--ARPA", med_mod_path, self.medium_arpa_path],
                    stderr=log_file,
                )
            assert self.medium_arpa_path.exists()

            logger.debug("Finished pruning medium arpa!")
            subprocess.check_call(
                [
                    "ngramshrink",
                    f"--method={self.prune_method}",
                    f"--theta={self.prune_thresh_small}",
                    self.mod_path,
                    small_mod_path,
                ],
                stderr=log_file,
            )
            assert small_mod_path.exists()
            if getattr(self, "sym_path", None):
                subprocess.check_call(
                    [
                        "ngramprint",
                        "--ARPA",
                        f"--symbols={self.sym_path}",
                        small_mod_path,
                        self.small_arpa_path,
                    ],
                    stderr=log_file,
                )
            else:
                subprocess.check_call(
                    ["ngramprint", "--ARPA", small_mod_path, self.small_arpa_path],
                    stderr=log_file,
                )
            assert self.small_arpa_path.exists()

        logger.debug("Finished pruning small arpa!")
        logger.info("Done pruning!")

    def export_model(self, output_model_path: Path) -> None:
        """
        Export language model to specified path

        Parameters
        ----------
        output_model_path: :class:`~pathlib.Path`
            Path to export model
        """
        directory = output_model_path.parent
        directory.mkdir(parents=True, exist_ok=True)

        model_temp_dir = self.working_directory.joinpath("model_archiving")
        os.makedirs(model_temp_dir, exist_ok=True)
        model = LanguageModel.empty(output_model_path.stem, root_directory=model_temp_dir)
        model.add_meta_file(self)
        model.add_arpa_file(self.large_arpa_path)
        model.add_arpa_file(self.medium_arpa_path)
        model.add_arpa_file(self.small_arpa_path)
        model.dump(output_model_path)


class LmCorpusTrainerMixin(LmTrainerMixin, TextCorpusMixin):
    """
    Top-level worker to train a language model from a text corpus

    Parameters
    ----------
    order: int
        Ngram order, defaults to 3
    method:str
        Smoothing method for the ngram model, defaults to "kneser_ney"
    count_threshold:int
        Minimum count needed to not be treated as an OOV item, defaults to 1

    See Also
    --------
    :class:`~montreal_forced_aligner.language_modeling.trainer.LmTrainerMixin`
        For  language model training parsing parameters
    :class:`~montreal_forced_aligner.corpus.text_corpus.TextCorpusMixin`
        For corpus parsing parameters
    :class:`~montreal_forced_aligner.abc.TopLevelMfaWorker`
        For top-level parameters
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.large_perplexity = None
        self.medium_perplexity = None
        self.small_perplexity = None

    @property
    def sym_path(self) -> str:
        """Internal path to symbols file"""
        return self.working_directory.joinpath("lm.sym")

    @property
    def far_path(self) -> str:
        """Internal path to FAR file"""
        return self.working_directory.joinpath("lm.far")

    @property
    def cnts_path(self) -> str:
        """Internal path to counts file"""
        return self.working_directory.joinpath("lm.cnts")

    @property
    def training_path(self) -> str:
        """Internal path to training data"""
        return self.working_directory.joinpath("training.txt")

    @property
    def meta(self) -> MetaDict:
        """Metadata information for the language model"""
        from datetime import datetime

        from ..utils import get_mfa_version

        with self.session() as session:
            word_count = (
                session.query(sqlalchemy.func.sum(Word.count))
                .filter(Word.word_type.in_(WordType.speech_types()))
                .scalar()
            )
            oov_count = (
                session.query(sqlalchemy.func.sum(Word.count))
                .filter(Word.word_type == WordType.oov)
                .scalar()
            )
            if not oov_count:
                oov_count = 0
        meta = {
            "architecture": "ngram",
            "order": self.order,
            "method": self.method,
            "train_date": str(datetime.now()),
            "version": get_mfa_version(),
            "training": {
                "num_words": word_count,
                "num_oovs": oov_count,
            },
            "evaluation_training": {
                "large_perplexity": self.large_perplexity,
                "medium_perplexity": self.medium_perplexity,
                "small_perplexity": self.small_perplexity,
            },
        }
        if self.model_version is not None:
            meta["version"] = self.model_version
        return meta

    def evaluate(self) -> None:
        """
        Run an evaluation over the training data to generate perplexity score
        """
        log_path = self.working_log_directory.joinpath("evaluate.log")

        small_mod_path = self.mod_path.with_stem(self.mod_path.stem + "_small")
        med_mod_path = self.mod_path.with_stem(self.mod_path.stem + "_med")
        with self.session() as session, mfa_open(log_path, "w") as log_file:
            word_query = session.query(Word.word).filter(
                Word.word_type.in_(WordType.speech_types())
            )
            included_words = set(x[0] for x in word_query)
            utterance_query = session.query(Utterance.normalized_text, Utterance.text)

            with open(self.far_path, "wb") as f:
                farcompile_proc = subprocess.Popen(
                    [
                        thirdparty_binary("farcompilestrings"),
                        "--fst_type=compact",
                        "--token_type=symbol",
                        "--generate_keys=16",
                        f"--symbols={self.sym_path}",
                        "--keep_symbols",
                    ],
                    stderr=log_file,
                    stdin=subprocess.PIPE,
                    stdout=f,
                    env=os.environ,
                )
                for normalized_text, text in utterance_query:
                    if not normalized_text:
                        normalized_text = text
                    text = " ".join(
                        x if x in included_words else self.oov_word
                        for x in normalized_text.split()
                    )
                    farcompile_proc.stdin.write(f"{text}\n".encode("utf8"))
                    farcompile_proc.stdin.flush()
                farcompile_proc.stdin.close()
                farcompile_proc.wait()
            perplexity_proc = subprocess.Popen(
                [
                    thirdparty_binary("ngramperplexity"),
                    f"--OOV_symbol={self.oov_word}",
                    self.mod_path,
                    self.far_path,
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                encoding="utf8",
            )
            stdout, stderr = perplexity_proc.communicate()
            num_sentences = None
            num_words = None
            num_oovs = None
            perplexity = None
            for line in stdout.splitlines():
                m = re.search(r"\d+ sentences", line)
                if m:
                    num_sentences = m.group(0)
                m = re.search(r"\d+ words", line)
                if m:
                    num_words = m.group(0)
                m = re.search(r"\d+ OOVs", line)
                if m:
                    num_oovs = m.group(0)
                m = re.search(r"perplexity = (?P<perplexity>[\d.]+)", line)
                if m:
                    perplexity = float(m.group("perplexity"))
            self.large_perplexity = perplexity
            logger.info(f"{num_sentences}, {num_words}, {num_oovs}")
            logger.info(f"Perplexity of large model: {perplexity}")

            perplexity_proc = subprocess.Popen(
                [
                    thirdparty_binary("ngramperplexity"),
                    f"--OOV_symbol={self.oov_word}",
                    med_mod_path,
                    self.far_path,
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                encoding="utf8",
            )
            stdout, stderr = perplexity_proc.communicate()

            perplexity = None
            for line in stdout.splitlines():
                m = re.search(r"perplexity = (?P<perplexity>[\d.]+)", line)
                if m:
                    perplexity = float(m.group("perplexity"))
            self.medium_perplexity = perplexity
            logger.info(f"Perplexity of medium model: {perplexity}")
            perplexity_proc = subprocess.Popen(
                [
                    thirdparty_binary("ngramperplexity"),
                    f"--OOV_symbol={self.oov_word}",
                    small_mod_path,
                    self.far_path,
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                encoding="utf8",
            )
            stdout, stderr = perplexity_proc.communicate()

            perplexity = None
            for line in stdout.splitlines():
                m = re.search(r"perplexity = (?P<perplexity>[\d.]+)", line)
                if m:
                    perplexity = float(m.group("perplexity"))
            self.small_perplexity = perplexity
            logger.info(f"Perplexity of small model: {perplexity}")

    def train_large_lm(self) -> None:
        """Train a large language model"""
        logger.info("Beginning training large ngram model...")
        log_path = self.working_log_directory.joinpath("lm_training.log")
        return_queue = Queue()
        stopped = threading.Event()
        error_dict = {}
        procs = []
        count_paths = []

        for j in self.jobs:
            args = TrainLmArguments(
                j.id,
                getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                self.working_log_directory.joinpath(f"ngram_count.{j.id}.log"),
                self.working_directory,
                self.sym_path,
                self.order,
                self.oov_word,
            )
            function = TrainLmFunction(args)
            p = KaldiProcessWorker(j.id, return_queue, function, stopped)
            procs.append(p)
            p.start()
            count_paths.append(self.working_directory.joinpath(f"{j.id}.cnts"))
        with tqdm(total=self.num_utterances, disable=config.QUIET) as pbar:
            while True:
                try:
                    result = return_queue.get(timeout=1)
                    if isinstance(result, Exception):
                        error_dict[getattr(result, "job_name", 0)] = result
                        continue
                    if stopped.is_set():
                        continue
                    return_queue.task_done()
                except Empty:
                    for proc in procs:
                        if not proc.finished.is_set():
                            break
                    else:
                        break
                    continue
                pbar.update(1)
        logger.info("Training model...")
        with mfa_open(log_path, "w") as log_file:
            merged_file = self.working_directory.joinpath("merged.cnts")
            if len(count_paths) > 1:
                ngrammerge_proc = subprocess.Popen(
                    [
                        thirdparty_binary("ngrammerge"),
                        f"--ofile={merged_file}",
                        *count_paths,
                    ],
                    stderr=log_file,
                    env=os.environ,
                )
                ngrammerge_proc.communicate()
            else:
                os.rename(count_paths[0], merged_file)
            ngrammake_proc = subprocess.Popen(
                [
                    thirdparty_binary("ngrammake"),
                    "--v=2",
                    "--method=kneser_ney",
                    merged_file,
                    self.mod_path,
                ],
                stderr=log_file,
                env=os.environ,
            )
            ngrammake_proc.communicate()
            subprocess.check_call(
                [
                    "ngramprint",
                    "--ARPA",
                    f"--symbols={self.sym_path}",
                    str(self.mod_path),
                    str(self.large_arpa_path),
                ],
                stderr=log_file,
                stdout=log_file,
            )
            assert os.path.exists(self.large_arpa_path)

        logger.info("Large ngram model created!")

    def train(self) -> None:
        """
        Train a language model
        """
        self.train_large_lm()
        self.prune_large_language_model()
        self.evaluate()


class LmDictionaryCorpusTrainerMixin(MultispeakerDictionaryMixin, LmCorpusTrainerMixin):
    """
    Mixin class for training a language model and incorporate a pronunciation dictionary for marking words as OOV

    See Also
    --------
    :class:`~montreal_forced_aligner.language_modeling.trainer.LmTrainerMixin`
        For language model training parsing parameters
    :class:`~montreal_forced_aligner.dictionary.multispeaker.MultispeakerDictionaryMixin`
        For dictionary parsing parameters
    """

    @property
    def sym_path(self) -> str:
        """Internal path to symbols file"""
        with self.session() as session:
            default_dictionary = session.get(Dictionary, self._default_dictionary_id)
            words_path = default_dictionary.words_symbol_path
        return words_path


class MfaLmArpaTrainer(LmTrainerMixin, TopLevelMfaWorker, DatabaseMixin):
    """
    Top-level worker to convert an existing ARPA-format language model to MFA format

    See Also
    --------
    :class:`~montreal_forced_aligner.language_modeling.trainer.LmTrainerMixin`
        For language model training parsing parameters
    :class:`~montreal_forced_aligner.abc.TopLevelMfaWorker`
        For top-level parsing parameters
    """

    def __init__(self, arpa_path: Path, keep_case: bool = False, **kwargs):
        self.arpa_path = arpa_path
        self.keep_case = keep_case
        super().__init__(**kwargs)

    @property
    def working_directory(self) -> Path:
        return self.output_directory.joinpath(self.data_source_identifier)

    def setup(self) -> None:
        """Set up language model training"""
        super().setup()
        os.makedirs(self.working_log_directory, exist_ok=True)
        with mfa_open(self.arpa_path, "r") as inf, mfa_open(
            self.large_arpa_path, "w", newline=""
        ) as outf:
            for line in inf:
                if not self.keep_case:
                    line = line.lower()
                outf.write(line.rstrip() + "\n")
        self.initialized = True

    @property
    def data_directory(self) -> str:
        """Data directory"""
        return ""

    @property
    def data_source_identifier(self) -> str:
        """Data source identifier"""
        return os.path.splitext(os.path.basename(self.arpa_path))[0]

    @property
    def meta(self) -> MetaDict:
        """Metadata information for the trainer"""
        return {}

    def train(self) -> None:
        """Convert the arpa model to MFA format"""
        logger.info("Parsing large ngram model...")

        with mfa_open(self.working_log_directory.joinpath("read.log"), "w") as log_file:
            subprocess.check_call(
                ["ngramread", "--ARPA", self.large_arpa_path, self.mod_path], stderr=log_file
            )
        assert os.path.exists(self.mod_path)

        logger.info("Large ngram model parsed!")

        self.prune_large_language_model()


class MfaLmDictionaryCorpusTrainer(LmDictionaryCorpusTrainerMixin, TopLevelMfaWorker):
    """
    Top-level worker to train a language model and incorporate a pronunciation dictionary for marking words as OOV

    See Also
    --------
    :class:`~montreal_forced_aligner.language_modeling.trainer.LmTrainerMixin`
        For language model training parsing parameters
    :class:`~montreal_forced_aligner.dictionary.multispeaker.MultispeakerDictionaryMixin`
        For dictionary parsing parameters
    """

    def setup(self) -> None:
        """Set up language model training"""
        super().setup()
        if self.initialized:
            return
        self.create_new_current_workflow(WorkflowType.language_model_training)
        os.makedirs(self.working_log_directory, exist_ok=True)
        self.dictionary_setup()
        self._load_corpus()
        self.initialize_jobs()
        self.normalize_text()
        self.write_lexicon_information()

        self.save_oovs_found(self.working_directory)

        self.initialized = True


class MfaLmCorpusTrainer(LmCorpusTrainerMixin, TopLevelMfaWorker):
    """
    Trainer class for generating a language model from a corpus
    """

    def setup(self) -> None:
        """Set up language model training"""
        super().setup()
        if self.initialized:
            return
        self.create_new_current_workflow(WorkflowType.language_model_training)
        os.makedirs(self.working_log_directory, exist_ok=True)
        self._load_corpus()
        self._create_dummy_dictionary()
        self.initialize_jobs()
        self.normalize_text()
        with mfa_open(self.sym_path, "w") as f, self.session() as session:
            words = session.query(Word.mapping_id, Word.word)
            f.write(f"{self.silence_word} 0\n")
            for m_id, w in words:
                f.write(f"{w} {m_id}\n")

        self.initialized = True
