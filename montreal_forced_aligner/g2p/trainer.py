"""Class definitions for training G2P models"""
from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
import operator
import os
import queue
import random
import re
import shutil
import subprocess
import time
from typing import Any, List, NamedTuple, Set

import pynini
import pywrapfst
import tqdm
from pynini import Fst

from montreal_forced_aligner.abc import MetaDict, MfaWorker, TopLevelMfaWorker, TrainerMixin
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.data import WordType, WorkflowType
from montreal_forced_aligner.db import Pronunciation, Word
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionaryMixin
from montreal_forced_aligner.exceptions import KaldiProcessingError, PyniniAlignmentError
from montreal_forced_aligner.g2p.generator import PyniniValidator
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.models import G2PModel
from montreal_forced_aligner.utils import Stopped, thirdparty_binary

Labels = List[Any]

TOKEN_TYPES = ["byte", "utf8"]
INF = float("inf")
RAND_MAX = 32767

__all__ = ["RandomStartWorker", "PyniniTrainer", "G2PTrainer"]

logger = logging.getLogger("mfa")


class RandomStart(NamedTuple):
    """Parameters for random starts"""

    idx: int
    seed: int
    input_far_path: str
    output_far_path: str
    cg_path: str
    tempdir: str
    train_opts: List[str]


def _get_far_labels(far_path: str) -> Set[int]:
    """Extracts label set from acceptors in a FAR.
    Args:
      far_path: path to FAR file.
    Returns:
      A set of integer labels found in the FAR.
    """
    labels: Set[int] = set()
    reader = pywrapfst.FarReader.open(far_path)
    while not reader.done():
        fst = reader.get_fst()
        assert fst.properties(pywrapfst.ACCEPTOR, True) == pywrapfst.ACCEPTOR
        for state in fst.states():
            labels.update(arc.ilabel for arc in fst.arcs(state))
        next(reader)
    assert not reader.error()
    return labels


class RandomStartWorker(mp.Process):
    """
    Random start worker
    """

    def __init__(
        self,
        job_name: int,
        job_q: mp.Queue,
        return_queue: mp.Queue,
        log_file: str,
        stopped: Stopped,
    ):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.job_q = job_q
        self.return_queue = return_queue
        self.log_file = log_file
        self.stopped = stopped
        self.finished = Stopped()

    def run(self) -> None:
        """Run the random start worker"""
        with mfa_open(self.log_file, "w") as log_file:
            while True:
                try:
                    args = self.job_q.get(timeout=1)
                except queue.Empty:
                    break
                if self.stopped.stop_check():
                    continue
                try:
                    start = time.time()
                    # Randomize channel model.
                    rfst_path = os.path.join(args.tempdir, f"random-{args.seed:05d}.fst")
                    afst_path = os.path.join(args.tempdir, f"aligner-{args.seed:05d}.fst")
                    likelihood_path = afst_path.replace(".fst", ".like")
                    if not os.path.exists(afst_path):
                        cmd = [
                            thirdparty_binary("baumwelchrandomize"),
                            f"--seed={args.seed}",
                            args.cg_path,
                            rfst_path,
                        ]
                        subprocess.check_call(cmd, stderr=log_file, env=os.environ)
                        random_end = time.time()
                        log_file.write(
                            f"{args.seed} randomization took {random_end - start} seconds\n"
                        )
                        # Train on randomized channel model.

                        likelihood = INF
                        cmd = [
                            thirdparty_binary("baumwelchtrain"),
                            *args.train_opts,
                            args.input_far_path,
                            args.output_far_path,
                            rfst_path,
                            afst_path,
                        ]
                        log_file.write(f"{args.seed} train command: {' '.join(cmd)}\n")
                        log_file.flush()
                        with subprocess.Popen(
                            cmd, stderr=subprocess.PIPE, text=True, env=os.environ
                        ) as proc:
                            # Parses STDERR to capture the likelihood.
                            for line in proc.stderr:  # type: ignore
                                log_file.write(line)
                                log_file.flush()
                                line = line.rstrip()
                                match = re.match(r"INFO: Iteration \d+: (-?\d*(\.\d*)?)", line)
                                assert match, line
                                likelihood = float(match.group(1))
                                self.return_queue.put(1)
                            with mfa_open(likelihood_path, "w") as f:
                                f.write(str(likelihood))
                        log_file.write(
                            f"{args.seed} training took {time.time() - random_end:.3f} seconds\n"
                        )
                    else:
                        with mfa_open(likelihood_path, "r") as f:
                            likelihood = f.read().strip()
                    self.return_queue.put((afst_path, likelihood))
                except Exception:
                    self.stopped.stop()
                    e = KaldiProcessingError([self.log_file])
                    e.job_name = self.job_name
                    self.return_queue.put(e)
        self.finished.stop()
        return


class G2PTrainer(MfaWorker, TrainerMixin):
    """
    Abstract mixin class for G2P training

    Parameters
    ----------
    validation_proportion: float
        Proportion of words to use as the validation set, defaults to 0.1, only used if ``evaluate`` is True
    num_pronunciations: int
        Number of pronunciations to generate
    evaluation_mode: bool
        Flag for whether to evaluate the model performance on an validation set

    See Also
    --------
    :class:`~montreal_forced_aligner.abc.MfaWorker`
        For base MFA parameters
    :class:`~montreal_forced_aligner.abc.TrainerMixin`
        For base trainer parameters

    Attributes
    ----------
    g2p_training_dictionary: dict[str, list[str]]
        Dictionary of words to pronunciations to train from
    g2p_validation_dictionary: dict[str, list[str]]
        Dictionary of words to pronunciations to validate performance against
    g2p_graphemes: set[str]
        Set of graphemes in the training set
    """

    def __init__(
        self,
        validation_proportion: float = 0.1,
        num_pronunciations: int = 0,
        evaluation_mode: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.evaluation_mode = evaluation_mode
        self.validation_proportion = validation_proportion
        self.num_pronunciations = num_pronunciations
        self.g2p_training_dictionary = {}
        self.g2p_validation_dictionary = None
        self.g2p_training_graphemes = set()
        self.g2p_validation_graphemes = set()
        self.g2p_training_phones = set()
        self.g2p_validation_phones = set()


class PyniniTrainerMixin:
    """
    Mixin for training Pynini G2P models

    Parameters
    ----------
    order: int
        Order of the ngram model, defaults to 7
    random_starts: int
        Number of random starts to use in initialization, defaults to 25
    seed: int
        Seed for randomization, defaults to 1917
    delta: float
        Comparison/quantization delta for Baum-Welch training, defaults to 1/1024
    alpha: float
        Step size reduction power parameter for Baum-Welch training;
        full standard batch EM is run (not stepwise) if set to 0, defaults to 1.0
    batch_size:int
        Batch size for Baum-Welch training, defaults to 200
    num_iterations:int
        Maximum number of iterations to use in Baum-Welch training, defaults to 10
    smoothing_method:str
        Smoothing method for the ngram model, defaults to "kneser_ney"
    pruning_method:str
        Pruning method for pruning the ngram model, defaults to "relative_entropy"
    model_size: int
        Target number of ngrams for pruning, defaults to 1000000
    insertions: bool
        Flag for whether to allow for insertions, default True
    deletions: bool
        Flag for whether to allow for deletions, default True
    fst_default_cache_gc: str
        String to pass to OpenFst binaries for GC behavior
    fst_default_cache_gc_limit: str
        String to pass to OpenFst binaries for GC behavior
    """

    def __init__(
        self,
        order: int = 8,
        random_starts: int = 25,
        seed: int = 1917,
        delta: float = 1 / 1024,
        alpha: float = 1.0,
        batch_size: int = 800,
        num_iterations: int = 10,
        smoothing_method: str = "kneser_ney",
        pruning_method: str = "relative_entropy",
        model_size: int = 1000000,
        insertions: bool = True,
        deletions: bool = True,
        fst_default_cache_gc="",
        fst_default_cache_gc_limit="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not hasattr(self, "_data_source"):
            self._data_source = None
        self.order = order
        self.random_starts = random_starts
        self.seed = seed
        self.delta = delta
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.smoothing_method = smoothing_method
        self.pruning_method = pruning_method
        self.model_size = model_size
        self.insertions = insertions
        self.deletions = deletions
        self.fst_default_cache_gc = fst_default_cache_gc
        self.fst_default_cache_gc_limit = fst_default_cache_gc_limit
        self._sym_path = None
        self._fst_path = None
        self.input_sym_path = None
        self.input_token_type = "utf8"
        self.output_token_type = "utf8"

    @property
    def data_source_identifier(self) -> str:
        """Dictionary name"""
        return self._data_source

    def train_iteration(self) -> None:
        """Train iteration, not used"""
        pass

    @property
    def architecture(self) -> str:
        """Pynini"""
        return "pynini"

    @property
    def input_far_path(self) -> str:
        """Path to store grapheme archive"""
        return os.path.join(self.working_directory, f"{self.data_source_identifier}.g.far")

    @property
    def output_far_path(self) -> str:
        """Path to store phone archive"""
        return os.path.join(self.working_directory, f"{self.data_source_identifier}.p.far")

    @property
    def cg_path(self) -> str:
        """Path to covering grammar FST"""
        return os.path.join(self.working_directory, f"{self.data_source_identifier}.cg.fst")

    @property
    def align_path(self) -> str:
        """Path to store alignment models"""
        return os.path.join(self.working_directory, f"{self.data_source_identifier}.align.fst")

    @property
    def afst_path(self) -> str:
        """Path to store aligned FSTs"""
        return os.path.join(self.working_directory, f"{self.data_source_identifier}.afst.far")

    @property
    def input_path(self) -> str:
        """Path to temporary file to store grapheme training data"""
        return os.path.join(self.working_directory, f"input_{self.data_source_identifier}.txt")

    @property
    def output_path(self) -> str:
        """Path to temporary file to store phone training data"""
        return os.path.join(self.working_directory, f"output_{self.data_source_identifier}.txt")

    def generate_model(self) -> None:
        """
        Generate an ngram G2P model from FAR strings
        """
        assert os.path.exists(self.far_path)
        if os.path.exists(self.fst_path):
            logger.info("Model building already done, skipping!")
            return
        with mfa_open(os.path.join(self.working_log_directory, "model.log"), "w") as logf:
            ngramcount_proc = subprocess.Popen(
                [
                    thirdparty_binary("ngramcount"),
                    "--require_symbols=false",
                    f"--order={self.order}",
                    self.far_path,
                ],
                stderr=logf,
                stdout=subprocess.PIPE,
                env=os.environ,
            )

            ngrammake_proc = subprocess.Popen(
                [thirdparty_binary("ngrammake"), f"--method={self.smoothing_method}"],
                stdin=ngramcount_proc.stdout,
                stderr=logf,
                stdout=subprocess.PIPE,
                env=os.environ,
            )

            ngramshrink_proc = subprocess.Popen(
                [
                    thirdparty_binary("ngramshrink"),
                    f"--method={self.pruning_method}",
                    f"--target_number_of_ngrams={self.model_size}",
                ],
                stdin=ngrammake_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=logf,
                env=os.environ,
            )

            fstencode_proc = subprocess.Popen(
                [thirdparty_binary("fstencode"), "--decode", "-", self.encoder_path, "-"],
                stdin=ngramshrink_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=logf,
                env=os.environ,
            )
            sort_proc = subprocess.Popen(
                [
                    thirdparty_binary("fstarcsort"),
                    "-",
                    self.fst_path,
                ],
                stdin=fstencode_proc.stdout,
                stderr=logf,
                env=os.environ,
            )
            sort_proc.communicate()

    @property
    def fst_path(self) -> str:
        """Internal temporary FST file"""
        if self._fst_path is not None:
            return self._fst_path
        return os.path.join(self.working_directory, f"{self.data_source_identifier}.fst")

    @property
    def far_path(self) -> str:
        """Internal temporary FAR file"""
        return os.path.join(self.working_directory, f"{self.data_source_identifier}.far")

    @property
    def encoder_path(self) -> str:
        """Internal temporary encoder file"""
        return os.path.join(self.working_directory, f"{self.data_source_identifier}.enc")

    @property
    def sym_path(self) -> str:
        """Internal temporary symbol file"""
        if self._sym_path is not None:
            return self._sym_path
        return os.path.join(self.working_directory, "phones.txt")

    def align_g2p(self) -> None:
        """Runs the entire alignment regimen."""
        self._lexicon_covering()
        self._alignments()
        self._encode()

    @staticmethod
    def _narcs(f: Fst) -> int:
        """Computes the number of arcs in an FST."""
        return sum(f.num_arcs(state) for state in f.states())

    def _lexicon_covering(self, input_path=None, output_path=None) -> None:
        """Builds covering grammar and lexicon FARs."""
        # Sets of labels for the covering grammar.
        with mfa_open(
            os.path.join(self.working_log_directory, "covering_grammar.log"), "w"
        ) as log_file:
            if input_path is None:
                input_path = self.input_path
            if output_path is None:
                output_path = self.output_path
            com = [
                thirdparty_binary("farcompilestrings"),
                "--fst_type=compact",
            ]
            if self.input_token_type != "utf8":
                com.append("--token_type=symbol")
                com.append(
                    f"--symbols={self.input_token_type}",
                )
                com.append("--unknown_symbol=<unk>")
            else:
                com.append("--token_type=utf8")
            com.extend([input_path, self.input_far_path])
            print(" ".join(com), file=log_file)
            subprocess.check_call(com, env=os.environ, stderr=log_file, stdout=log_file)
            com = [
                thirdparty_binary("farcompilestrings"),
                "--fst_type=compact",
                "--token_type=symbol",
                f"--symbols={self.phone_symbol_table_path}",
                output_path,
                self.output_far_path,
            ]
            print(" ".join(com), file=log_file)
            subprocess.check_call(com, env=os.environ, stderr=log_file, stdout=log_file)
            ilabels = _get_far_labels(self.input_far_path)
            print(ilabels, file=log_file)
            olabels = _get_far_labels(self.output_far_path)
            print(olabels, file=log_file)
            cg = pywrapfst.VectorFst()
            state = cg.add_state()
            cg.set_start(state)
            one = pywrapfst.Weight.one(cg.weight_type())
            for ilabel, olabel in itertools.product(ilabels, olabels):
                cg.add_arc(state, pywrapfst.Arc(ilabel, olabel, one, state))
            # Handles epsilons, carefully avoiding adding a useless 0:0 label.
            if self.insertions:
                for olabel in olabels:
                    cg.add_arc(state, pywrapfst.Arc(0, olabel, one, state))
            if self.deletions:
                for ilabel in ilabels:
                    cg.add_arc(state, pywrapfst.Arc(ilabel, 0, one, state))
            cg.set_final(state)
            assert cg.verify(), "Label acceptor is ill-formed"
            cg.write(self.cg_path)

    def _alignments(self) -> None:
        """Trains the aligner and constructs the alignments FAR."""
        if not os.path.exists(self.align_path):
            logger.info("Training aligner")
            train_opts = []
            if self.batch_size:
                train_opts.append(f"--batch_size={self.batch_size}")
            if self.delta:
                train_opts.append(f"--delta={self.delta}")
            if self.fst_default_cache_gc:
                train_opts.append(f"--fst_default_cache_gc={self.fst_default_cache_gc}")
            if self.fst_default_cache_gc_limit:
                train_opts.append(
                    f"--fst_default_cache_gc_limit={self.fst_default_cache_gc_limit}"
                )
            if self.alpha:
                train_opts.append(f"--alpha={self.alpha}")
            if self.num_iterations:
                train_opts.append(f"--max_iters={self.num_iterations}")
            # Constructs the actual command vectors (plus an index for logging
            # purposes).
            random.seed(self.seed)
            starts = [
                (
                    RandomStart(
                        idx,
                        seed,
                        self.input_far_path,
                        self.output_far_path,
                        self.cg_path,
                        self.working_directory,
                        train_opts,
                    )
                )
                for (idx, seed) in enumerate(
                    random.sample(range(1, RAND_MAX), self.random_starts), 1
                )
            ]
            stopped = Stopped()
            num_commands = len(starts)
            job_queue = mp.JoinableQueue()
            fst_likelihoods = {}
            # Actually runs starts.
            logger.info("Calculating alignments...")
            begin = time.time()
            with tqdm.tqdm(
                total=num_commands * self.num_iterations, disable=GLOBAL_CONFIG.quiet
            ) as pbar:
                for start in starts:
                    job_queue.put(start)
                error_dict = {}
                return_queue = mp.Queue()
                procs = []
                for i in range(GLOBAL_CONFIG.num_jobs):
                    log_path = os.path.join(self.working_log_directory, f"baumwelch.{i}.log")
                    p = RandomStartWorker(
                        i,
                        job_queue,
                        return_queue,
                        log_path,
                        stopped,
                    )
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
                    except queue.Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    if isinstance(result, int):
                        pbar.update(result)
                    else:
                        fst_likelihoods[result[0]] = result[1]
                for p in procs:
                    p.join()
            if error_dict:
                raise PyniniAlignmentError(error_dict)
            (best_fst, best_likelihood) = min(fst_likelihoods.items(), key=operator.itemgetter(1))
            logger.info(f"Best likelihood: {best_likelihood}")
            logger.debug(
                f"Ran {self.random_starts} random starts in {time.time() - begin:.3f} seconds"
            )
            # Moves best likelihood solution to the requested location.
            shutil.move(best_fst, self.align_path)
        cmd = [thirdparty_binary("baumwelchdecode")]
        if self.fst_default_cache_gc:
            cmd.append(f"--fst_default_cache_gc={self.fst_default_cache_gc}")
        if self.fst_default_cache_gc_limit:
            cmd.append(f"--fst_default_cache_gc_limit={self.fst_default_cache_gc_limit}")
        cmd.append(self.input_far_path)
        cmd.append(self.output_far_path)
        cmd.append(self.align_path)
        cmd.append(self.afst_path)
        logger.debug(f"Subprocess call: {cmd}")
        subprocess.check_call(cmd, env=os.environ)
        logger.info("Completed computing alignments!")

    def _encode(self) -> None:
        """Encodes the alignments."""
        logger.info("Encoding the alignments as FSAs")
        subprocess.check_call(
            [
                thirdparty_binary("farencode"),
                "--encode_labels",
                self.afst_path,
                self.encoder_path,
                self.far_path,
            ],
            env=os.environ,
        )
        logger.info(f"Success! FAR path: {self.far_path}; encoder path: {self.encoder_path}")


class PyniniTrainer(
    MultispeakerDictionaryMixin, PyniniTrainerMixin, G2PTrainer, TopLevelMfaWorker
):
    """
    Top-level G2P trainer that uses Pynini functionality

    See Also
    --------
    :class:`~montreal_forced_aligner.g2p.trainer.G2PTrainer`
        For base G2P training parameters
    :class:`~montreal_forced_aligner.abc.TopLevelMfaWorker`
        For top-level parameters
    """

    def __init__(
        self,
        **kwargs,
    ):
        self._data_source = os.path.splitext(os.path.basename(kwargs["dictionary_path"]))[0]
        super().__init__(**kwargs)
        self._fst_path = None
        self._sym_path = None
        self.position_dependent_phones = False
        self.wer = None
        self.ler = None

    @property
    def data_directory(self) -> str:
        """Data directory for trainer"""
        return self.working_directory

    @property
    def configuration(self) -> MetaDict:
        """Configuration for G2P trainer"""
        config = super().configuration
        config.update({"dictionary_path": str(self.dictionary_model.path)})
        return config

    def setup(self) -> None:
        """Setup for G2P training"""
        super().setup()
        self.create_new_current_workflow(WorkflowType.train_g2p)
        wf = self.current_workflow
        if wf.done:
            logger.info("G2P training already done, skipping.")
            return
        self.dictionary_setup()
        os.makedirs(self.phones_dir, exist_ok=True)
        self._write_phone_symbol_table()
        self._write_grapheme_symbol_table()
        os.makedirs(self.working_log_directory, exist_ok=True)
        self.initialize_training()
        self.initialized = True

    @property
    def meta(self) -> MetaDict:
        """Metadata for exported G2P model"""
        from datetime import datetime

        from ..utils import get_mfa_version

        m = {
            "version": get_mfa_version(),
            "architecture": self.architecture,
            "train_date": str(datetime.now()),
            "phones": sorted(self.non_silence_phones),
            "graphemes": self.g2p_training_graphemes,
            "evaluation": {},
            "training": {
                "num_words": len(self.g2p_training_dictionary),
                "num_graphemes": len(self.g2p_training_graphemes),
                "num_phones": len(self.non_silence_phones),
            },
        }

        if self.evaluation_mode:
            m["evaluation"]["num_words"] = len(self.g2p_validation_dictionary)
            m["evaluation"]["word_error_rate"] = self.wer
            m["evaluation"]["phone_error_rate"] = self.ler
        return m

    def initialize_training(self) -> None:
        """Initialize training G2P model"""
        random.seed(self.seed)
        self._sym_path = self.phone_symbol_table_path
        self.output_token_type = pynini.SymbolTable.read_text(self.phone_symbol_table_path)
        with self.session() as session:
            self.g2p_training_dictionary = {}
            pronunciations = (
                session.query(Word.word, Pronunciation.pronunciation)
                .join(Pronunciation.word)
                .filter(Word.word_type.in_([WordType.speech, WordType.clitic]))
            )
            for w, p in pronunciations:
                if w not in self.g2p_training_dictionary:
                    self.g2p_training_dictionary[w] = set()
                self.g2p_training_dictionary[w].add(p)
            if self.evaluation_mode:
                word_dict = self.g2p_training_dictionary
                words = sorted(word_dict.keys())
                total_items = len(words)
                validation_items = int(total_items * self.validation_proportion)
                validation_words = random.sample(words, validation_items)
                self.g2p_training_dictionary = {
                    k: v for k, v in word_dict.items() if k not in validation_words
                }
                self.g2p_validation_dictionary = {
                    k: v for k, v in word_dict.items() if k in validation_words
                }
                if GLOBAL_CONFIG.debug:
                    with mfa_open(
                        os.path.join(self.working_directory, "validation_set.txt"),
                        "w",
                        encoding="utf8",
                    ) as f:
                        for word in self.g2p_validation_dictionary:
                            f.write(word + "\n")

            with mfa_open(self.input_path, "w") as inf, mfa_open(self.output_path, "w") as outf:
                for word, pronunciations in self.g2p_training_dictionary.items():
                    if re.match(r"\W", word) is not None:
                        continue
                    self.g2p_training_graphemes.update(word)
                    for p in pronunciations:
                        self.g2p_training_phones.update(p.split())
                        print(word, file=inf)
                        print(p, file=outf)
            logger.debug(f"Graphemes in training data: {sorted(self.g2p_training_graphemes)}")
            logger.debug(f"Phones in training data: {sorted(self.g2p_training_phones)}")
            if self.evaluation_mode:
                for word, pronunciations in self.g2p_validation_dictionary.items():
                    self.g2p_validation_graphemes.update(word)
                    for p in pronunciations:
                        self.g2p_validation_phones.update(p.split())
                logger.debug(
                    f"Graphemes in validation data: {sorted(self.g2p_validation_graphemes)}"
                )
                logger.debug(f"Phones in validation data: {sorted(self.g2p_validation_phones)}")
                grapheme_diff = sorted(self.g2p_validation_graphemes - self.g2p_training_graphemes)
                phone_diff = sorted(self.g2p_validation_phones - self.g2p_training_phones)
                if grapheme_diff:
                    logger.debug(
                        f"The following graphemes appear only in the validation set: {', '.join(grapheme_diff)}"
                    )
                if phone_diff:
                    logger.debug(
                        f"The following phones appear only in the validation set: {', '.join(phone_diff)}"
                    )

    def clean_up(self) -> None:
        """
        Clean up temporary files
        """
        if GLOBAL_CONFIG.debug:
            return
        for name in os.listdir(self.working_directory):
            path = os.path.join(self.working_directory, name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif not name.endswith(".log"):
                os.remove(path)

    def export_model(self, output_model_path: str) -> None:
        """
        Export G2P model to specified path

        Parameters
        ----------
        output_model_path:str
            Path to export model
        """
        directory, filename = os.path.split(output_model_path)
        basename, _ = os.path.splitext(filename)
        models_temp_dir = os.path.join(self.working_directory, "model_archive_temp")
        model = G2PModel.empty(basename, root_directory=models_temp_dir)
        model.add_meta_file(self)
        model.add_fst_model(self.working_directory)
        model.add_sym_path(self.working_directory)
        if directory:
            os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(output_model_path)
        model.dump(basename)
        model.clean_up()
        # self.clean_up()
        logger.info(f"Saved model to {output_model_path}")

    def train(self) -> None:
        """
        Train a G2P model
        """
        os.makedirs(self.working_log_directory, exist_ok=True)
        begin = time.time()
        if os.path.exists(self.far_path) and os.path.exists(self.encoder_path):
            logger.info("Alignment already done, skipping!")
        else:
            self.align_g2p()
            logger.debug(
                f"Aligning {len(self.g2p_training_dictionary)} words took {time.time() - begin:.3f} seconds"
            )
        begin = time.time()
        self.generate_model()
        logger.debug(
            f"Generating model for {len(self.g2p_training_dictionary)} words took {time.time() - begin:.3f} seconds"
        )
        self.finalize_training()

    def finalize_training(self) -> None:
        """Finalize training"""
        shutil.copyfile(self.fst_path, os.path.join(self.working_directory, "model.fst"))
        shutil.copyfile(
            self.phone_symbol_table_path, os.path.join(self.working_directory, "phones.txt")
        )
        if self.evaluation_mode:
            self.evaluate_g2p_model()

    def evaluate_g2p_model(self) -> None:
        """
        Validate the G2P model against held out data
        """
        temp_model_path = os.path.join(self.working_log_directory, "g2p_model.zip")
        self.export_model(temp_model_path)

        gen = PyniniValidator(
            g2p_model_path=temp_model_path,
            word_list=list(self.g2p_validation_dictionary.keys()),
            num_jobs=GLOBAL_CONFIG.num_jobs,
            num_pronunciations=self.num_pronunciations,
        )
        gen.evaluate_g2p_model(self.g2p_training_dictionary)
        self.wer = gen.wer
        self.ler = gen.ler
