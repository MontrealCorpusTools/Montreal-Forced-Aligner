"""Class definitions for training G2P models"""
from __future__ import annotations

import functools
import logging
import multiprocessing as mp
import operator
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, NamedTuple, Optional

import tqdm

from montreal_forced_aligner.abc import MetaDict, MfaWorker, TopLevelMfaWorker, TrainerMixin
from montreal_forced_aligner.dictionary.pronunciation import PronunciationDictionaryMixin
from montreal_forced_aligner.g2p.generator import PyniniGenerator
from montreal_forced_aligner.helper import score
from montreal_forced_aligner.models import G2PModel
from montreal_forced_aligner.utils import Counter, Stopped

try:
    import pynini
    import pywrapfst
    from pynini import Fst, TokenType
    from pywrapfst import convert

    G2P_DISABLED = False

except ImportError:
    pynini = None
    pywrapfst = None
    TokenType = Optional[str]
    Fst = None

    def convert(x):
        """stub function"""
        pass

    G2P_DISABLED = True


Labels = list[Any]

TOKEN_TYPES = ["byte", "utf8"]
INF = float("inf")
RAND_MAX = 32767

__all__ = ["RandomStartWorker", "PairNGramAligner", "PyniniTrainer", "G2PTrainer"]


class RandomStart(NamedTuple):
    """Parameters for random starts"""

    idx: int
    seed: int
    g_path: str
    p_path: str
    c_path: str
    tempdir: str
    train_opts: list[str]


class RandomStartWorker(mp.Process):
    """
    Random start worker
    """

    def __init__(
        self,
        job_q: mp.Queue,
        return_dict: dict,
        function: Callable,
        counter: Counter,
        stopped: Stopped,
    ):
        mp.Process.__init__(self)
        self.job_q = job_q
        self.return_dict = return_dict
        self.function = function
        self.counter = counter
        self.stopped = stopped

    def run(self) -> None:
        """Run the random start worker"""
        while True:
            try:
                args = self.job_q.get(timeout=1)
            except queue.Empty:
                break
            self.job_q.task_done()
            if self.stopped.stop_check():
                continue
            try:
                fst_path, likelihood = self.function(args)
                self.return_dict[fst_path] = likelihood
            except Exception:
                self.stopped.stop()
                self.return_dict["MFA_ERROR"] = args, Exception(
                    traceback.format_exception(*sys.exc_info())
                )
            self.counter.increment()
        return


class PairNGramAligner:
    """Produces FSA alignments for pair n-gram model training."""

    _compactor = functools.partial(convert, fst_type="compact_string")

    def __init__(self, working_directory: str):
        self.working_directory = working_directory
        self.g_path = os.path.join(self.working_directory, "g.far")
        self.p_path = os.path.join(self.working_directory, "p.far")
        self.c_path = os.path.join(self.working_directory, "c.fst")
        self.align_path = os.path.join(self.working_directory, "align.fst")
        self.afst_path = os.path.join(self.working_directory, "afst.far")
        self.align_log_path = os.path.join(self.working_directory, "align.log")
        self.logger = logging.getLogger("g2p_aligner")
        self.logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(self.align_log_path)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def align(
        self,
        # Input TSV path.
        tsv_path: str,
        # Output FAR path.
        far_path: str,
        encoder_path: str,
        # Arguments for constructing the lexicon and covering grammar.
        input_token_type: TokenType,
        input_epsilon: bool,
        output_token_type: TokenType,
        output_epsilon: bool,
        # Arguments used during the alignment phase.
        cores: int,
        random_starts: int,
        seed: int,
        batch_size: int = 0,
        delta: float = 1 / 1024,
        lr: float = 1.0,
        max_iters: int = 50,
        fst_default_cache_gc: str = "",
        fst_default_cache_gc_limit: str = "",
    ):
        """Runs the entire alignment regimen."""
        self._lexicon_covering(
            tsv_path,
            input_token_type,
            input_epsilon,
            output_token_type,
            output_epsilon,
        )
        self._alignments(
            cores,
            random_starts,
            seed,
            batch_size,
            delta,
            lr,
            max_iters,
            fst_default_cache_gc,
            fst_default_cache_gc_limit,
        )
        self._encode(far_path, encoder_path)
        self.logger.info("Success! FAR path: %s; encoder path: %s", far_path, encoder_path)

    @staticmethod
    def _label_union(labels: set[int], epsilon: bool) -> Fst:
        """Creates FSA over a union of the labels."""
        side = pynini.Fst()
        src = side.add_state()
        side.set_start(src)
        dst = side.add_state()
        if epsilon:
            labels.add(0)
        one = pynini.Weight.one(side.weight_type())
        for label in labels:
            side.add_arc(src, pynini.Arc(label, label, one, dst))
        side.set_final(dst)
        assert side.verify(), "FST is ill-formed"
        return side

    @staticmethod
    def _narcs(f: Fst) -> int:
        """Computes the number of arcs in an FST."""
        return sum(f.num_arcs(state) for state in f.states())

    NON_SYMBOL = ("byte", "utf8")

    def _lexicon_covering(
        self,
        tsv_path: str,
        input_token_type: TokenType,
        input_epsilon: bool,
        output_token_type: TokenType,
        output_epsilon: bool,
    ) -> None:
        """Builds covering grammar and lexicon FARs."""
        # Sets of labels for the covering grammar.
        g_labels: set[int] = set()
        p_labels: set[int] = set()
        self.logger.info("Constructing grapheme and phoneme FARs")
        g_writer = pywrapfst.FarWriter.create(self.g_path)
        p_writer = pywrapfst.FarWriter.create(self.p_path)
        with open(tsv_path, "r") as source:
            for (linenum, line) in enumerate(source, 1):
                key = f"{linenum:08x}"
                (g, p) = line.rstrip().split("\t", 1)
                # For both G and P, we compile a FSA, store the labels, and
                # then write the compact version to the FAR.
                g_fst = pynini.accep(g, token_type=input_token_type)
                g_labels.update(g_fst.paths().ilabels())
                g_writer[key] = self._compactor(g_fst)
                p_fst = pynini.accep(p, token_type=output_token_type)
                p_labels.update(p_fst.paths().ilabels())
                p_writer[key] = self._compactor(p_fst)
        self.logger.info(f"Processed{linenum:,d} examples")
        self.logger.info("Constructing covering grammar")
        self.logger.info(f"{len(g_labels)} unique graphemes")
        g_side = self._label_union(g_labels, input_epsilon)
        self.logger.info(f"{len(p_labels)} unique phones")
        p_side = self._label_union(p_labels, output_epsilon)
        # The covering grammar is given by (G job_name P)^*.
        covering = pynini.cross(g_side, p_side).closure().optimize()
        assert covering.num_states() == 1, "Covering grammar FST is ill-formed"
        self.logger.info(
            f"Covering grammar has {PairNGramAligner._narcs(covering):,d} arcs",
        )
        covering.write(self.c_path)

    @staticmethod
    def _random_start(random_start: RandomStart) -> tuple[str, float]:
        """Performs a single random start."""
        start = time.time()
        logger = logging.getLogger("g2p_aligner")
        # Randomize channel model.
        c_path = os.path.join(random_start.tempdir, f"c-{random_start.seed:05d}.fst")
        t_path = os.path.join(random_start.tempdir, f"t-{random_start.seed:05d}.fst")
        likelihood_path = t_path.replace(".fst", ".like")
        if not os.path.exists(t_path):
            cmd = [
                "baumwelchrandomize",
                f"--seed={random_start.seed}",
                random_start.c_path,
                c_path,
            ]
            subprocess.check_call(cmd)
            random_end = time.time()
            logger.debug(f"{random_start.seed} randomization took {random_end - start} seconds")
            # Train on randomized channel model.

            likelihood = INF
            cmd = [
                "baumwelchtrain",
                *random_start.train_opts,
                random_start.g_path,
                random_start.p_path,
                c_path,
                t_path,
            ]
            logger.debug(f"{random_start.seed} train command: {' '.join(cmd)}")
            with subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True) as proc:
                # Parses STDERR to capture the likelihood.
                for line in proc.stderr:  # type: ignore
                    line = line.rstrip()
                    match = re.match(r"INFO: Iteration \d+: (-?\d*(\.\d*)?)", line)
                    assert match, line
                    likelihood = float(match.group(1))
                with open(likelihood_path, "w") as f:
                    f.write(str(likelihood))
            logger.debug(f"{random_start.seed} training took {time.time() - random_end} seconds")
        else:
            with open(likelihood_path, "r") as f:
                likelihood = f.read().strip()
        return t_path, likelihood

    def _alignments(
        self,
        cores: int,
        random_starts: int,
        seed: int,
        batch_size: Optional[int] = None,
        delta: Optional[float] = None,
        lr: Optional[float] = None,
        max_iters: Optional[int] = None,
        fst_default_cache_gc: str = "",
        fst_default_cache_gc_limit: str = "",
    ) -> None:
        """Trains the aligner and constructs the alignments FAR."""
        if not os.path.exists(self.align_path):
            self.logger.info("Training aligner")
            train_opts = []
            if batch_size:
                train_opts.append(f"--batch_size={batch_size}")
            if delta:
                train_opts.append(f"--delta={delta}")
            if fst_default_cache_gc:
                train_opts.append(f"--fst_default_cache_gc={fst_default_cache_gc}")
            if fst_default_cache_gc_limit:
                train_opts.append(f"--fst_default_cache_gc_limit={fst_default_cache_gc_limit}")
            if lr:
                train_opts.append(f"--lr={lr}")
            if max_iters:
                train_opts.append(f"--max_iters={max_iters}")
            # Constructs the actual command vectors (plus an index for logging
            # purposes).
            random.seed(seed)
            starts = [
                (
                    RandomStart(
                        idx,
                        seed,
                        self.g_path,
                        self.p_path,
                        self.c_path,
                        self.working_directory,
                        train_opts,
                    )
                )
                for (idx, seed) in enumerate(random.sample(range(1, RAND_MAX), random_starts), 1)
            ]
            stopped = Stopped()
            num_commands = len(starts)
            if cores > len(starts):
                cores = len(starts)
            job_queue = mp.JoinableQueue(cores + 2)

            # Actually runs starts.
            self.logger.info("Calculating alignments...")
            begin = time.time()
            last_value = 0
            ind = 0
            with tqdm.tqdm(total=num_commands) as pbar:
                while True:
                    if ind == num_commands:
                        break
                    try:
                        job_queue.put(starts[ind], False)
                    except queue.Full:
                        break
                    ind += 1
                manager = mp.Manager()
                return_dict = manager.dict()
                procs = []
                counter = Counter()
                for _ in range(cores):
                    p = RandomStartWorker(
                        job_queue, return_dict, self._random_start, counter, stopped
                    )
                    procs.append(p)
                    p.start()
                while True:
                    if ind == num_commands:
                        break
                    job_queue.put(starts[ind])
                    value = counter.value()
                    pbar.update(value - last_value)
                    last_value = value
                    ind += 1
                while True:
                    time.sleep(30)
                    value = counter.value()
                    if value != last_value:
                        pbar.update(value - last_value)
                        last_value = value
                    if value >= random_starts:
                        break
                job_queue.join()
                for p in procs:
                    p.join()
            if "MFA_ERROR" in return_dict:
                element, exc = return_dict["MFA_ERROR"]
                print(element)
                raise exc
            (best_fst, best_likelihood) = min(return_dict.items(), key=operator.itemgetter(1))
            self.logger.info(f"Best likelihood: {best_likelihood}")
            self.logger.debug(
                f"Ran {random_starts} random starts in {time.time() - begin} seconds"
            )
            # Moves best likelihood solution to the requested location.
            shutil.move(best_fst, self.align_path)
        self.logger.info("Computing alignments")
        cmd = ["baumwelchdecode"]
        if fst_default_cache_gc:
            cmd.append(f"--fst_default_cache_gc={fst_default_cache_gc}")
        if fst_default_cache_gc_limit:
            cmd.append(f"--fst_default_cache_gc_limit={fst_default_cache_gc_limit}")
        cmd.append(self.g_path)
        cmd.append(self.p_path)
        cmd.append(self.align_path)
        cmd.append(self.afst_path)
        self.logger.debug(f"Subprocess call: {cmd}")
        subprocess.check_call(cmd)

    def _encode(self, far_path: str, encoder_path: str) -> None:
        """Encodes the alignments."""
        self.logger.info("Encoding the alignments as FSAs")
        encoder = pywrapfst.EncodeMapper(encode_labels=True)
        a_reader = pywrapfst.FarReader.open(self.afst_path)
        a_writer = pywrapfst.FarWriter.create(far_path)
        # Curries converter function for the FAR.
        converter = functools.partial(pywrapfst.convert, fst_type="vector")
        while not a_reader.done():
            key = a_reader.get_key()
            fst = converter(a_reader.get_fst())
            fst.encode(encoder)
            a_writer[key] = self._compactor(fst)
            try:
                next(a_reader)
            except StopIteration:
                break
        encoder.write(encoder_path)


class PyniniValidator(PyniniGenerator):
    """
    Class for running validation for G2P model training

    Parameters
    ----------
    word_list: list[str]
        List of words to generate pronunciations

    See Also
    --------
    :class:`~montreal_forced_aligner.g2p.generator.PyniniGenerator`
        For parameters to generate pronunciations
    """

    def __init__(self, word_list: list[str], **kwargs):
        super().__init__(**kwargs)
        self.word_list = word_list

    @property
    def words_to_g2p(self) -> list[str]:
        """Words to produce pronunciations"""
        return self.word_list


class G2PTrainer(MfaWorker, TrainerMixin, PronunciationDictionaryMixin):
    """
    Abstract mixin class for G2P training

    Parameters
    ----------
    validation_proportion: float
        Proportion of words to use as the validation set, defaults to 0.1, only used if ``evaluate`` is True
    num_pronunciations: int
        Number of pronunciations to generate
    evaluate: bool
        Flag for whether to evaluate the model performance on an validation set

    See Also
    --------
    :class:`~montreal_forced_aligner.abc.MfaWorker`
        For base MFA parameters
    :class:`~montreal_forced_aligner.abc.TrainerMixin`
        For base trainer parameters
    :class:`~montreal_forced_aligner.dictionary.pronunciation.PronunciationDictionaryMixin`
        For pronunciation dictionary parameters

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
        num_pronunciations: int = 1,
        evaluate: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.evaluate = evaluate
        self.validation_proportion = validation_proportion
        self.num_pronunciations = num_pronunciations
        self.g2p_training_dictionary = {}
        self.g2p_validation_dictionary = None
        self.g2p_graphemes = set()


class PyniniTrainer(G2PTrainer, TopLevelMfaWorker):
    """
    Top-level G2P trainer that uses Pynini functionality

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
    lr: float
        Learning rate for Baum-Welch training, defaults to 1.0
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
    input_epsilon: bool
        Flag for whether to allow for epsilon on input strings, default True
    output_epsilon: bool
        Flag for whether to allow for epsilon on output strings, default True
    fst_default_cache_gc: str
        String to pass to OpenFst binaries for GC behavior
    fst_default_cache_gc_limit: str
        String to pass to OpenFst binaries for GC behavior

    See Also
    --------
    :class:`~montreal_forced_aligner.g2p.trainer.G2PTrainer`
        For base G2P training parameters
    :class:`~montreal_forced_aligner.abc.TopLevelMfaWorker`
        For top-level parameters
    """

    def __init__(
        self,
        order: int = 7,
        random_starts: int = 25,
        seed: int = 1917,
        delta: float = 1 / 1024,
        lr: float = 1.0,
        batch_size: int = 200,
        num_iterations: int = 10,
        smoothing_method: str = "kneser_ney",
        pruning_method: str = "relative_entropy",
        model_size: int = 1000000,
        input_epsilon: bool = True,
        output_epsilon: bool = True,
        fst_default_cache_gc="",
        fst_default_cache_gc_limit="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.order = order
        self.random_starts = random_starts
        self.seed = seed
        self.delta = delta
        self.lr = lr
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.smoothing_method = smoothing_method
        self.pruning_method = pruning_method
        self.model_size = model_size
        self.input_epsilon = input_epsilon
        self.output_epsilon = output_epsilon
        self.fst_default_cache_gc = fst_default_cache_gc
        self.fst_default_cache_gc_limit = fst_default_cache_gc_limit

    @property
    def data_source_identifier(self) -> str:
        """Dictionary name"""
        return self.dictionary_model.name

    @property
    def data_directory(self) -> str:
        """Data directory for trainer"""
        return self.working_directory

    @property
    def workflow_identifier(self) -> str:
        """Identifier for Pynini G2P trainer"""
        return "pynini_train_g2p"

    @property
    def configuration(self) -> MetaDict:
        """Configuration for G2P trainer"""
        config = super().configuration
        config.update({"dictionary_path": self.dictionary_model.path})
        return config

    def train_iteration(self) -> None:
        """Train iteration, not used"""
        pass

    def setup(self) -> None:
        """Setup for G2P training"""
        if self.initialized:
            return
        os.makedirs(self.working_log_directory, exist_ok=True)
        self.g2p_training_dictionary = self.words
        self.initialize_training()
        self.initialized = True

    @property
    def architecture(self) -> str:
        """Pynini"""
        return "pynini"

    @property
    def meta(self) -> MetaDict:
        """Metadata for exported G2P model"""
        from datetime import datetime

        from ..utils import get_mfa_version

        return {
            "version": get_mfa_version(),
            "architecture": self.architecture,
            "train_date": str(datetime.now()),
            "phones": sorted(self.non_silence_phones),
            "graphemes": self.graphemes,
        }

    @property
    def input_path(self) -> str:
        """Path to temporary file to store training data"""
        return os.path.join(self.working_directory, "input.txt")

    def initialize_training(self) -> None:
        """Initialize training G2P model"""
        if self.evaluate:
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
        for k in self.g2p_training_dictionary.keys():
            self.g2p_graphemes.update(k)
        phones_path = os.path.join(self.working_directory, "phones_only.txt")

        with open(self.input_path, "w", encoding="utf8") as f2, open(
            phones_path, "w", encoding="utf8"
        ) as phonef:
            for word, v in self.g2p_training_dictionary.items():
                if re.match(r"\W", word) is not None:
                    continue
                for v2 in v:
                    f2.write(f"{word}\t{' '.join(v2['pronunciation'])}\n")
                for v2 in v:
                    phonef.write(f"{' '.join(v2['pronunciation'])}\n")
        subprocess.call(["ngramsymbols", phones_path, self.sym_path])
        os.remove(phones_path)

    def clean_up(self) -> None:
        """
        Clean up temporary files
        """
        for name in os.listdir(self.working_directory):
            path = os.path.join(self.working_directory, name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif not name.endswith(".log"):
                os.remove(path)

    def generate_model(self) -> None:
        """
        Generate an ngram G2P model from FAR strings
        """
        assert os.path.exists(self.far_path)
        with open(
            os.path.join(self.working_log_directory, "model.log"), "w", encoding="utf8"
        ) as logf:
            ngram_count_path = os.path.join(self.working_directory, "ngram.count")
            ngram_make_path = os.path.join(self.working_directory, "ngram.make")
            ngram_shrink_path = os.path.join(self.working_directory, "ngram.shrink")
            ngramcount_proc = subprocess.Popen(
                [
                    "ngramcount",
                    "--require_symbols=false",
                    f"--order={self.order}",
                    self.far_path,
                    ngram_count_path,
                ],
                stderr=logf,
            )
            ngramcount_proc.communicate()

            ngrammake_proc = subprocess.Popen(
                [
                    "ngrammake",
                    f"--method={self.smoothing_method}",
                    ngram_count_path,
                    ngram_make_path,
                ],
                stderr=logf,
            )
            ngrammake_proc.communicate()

            ngramshrink_proc = subprocess.Popen(
                [
                    "ngramshrink",
                    f"--method={self.pruning_method}",
                    f"--target_number_of_ngrams={self.model_size}",
                    ngram_make_path,
                    ngram_shrink_path,
                ],
                stderr=logf,
            )
            ngramshrink_proc.communicate()

            fstencode_proc = subprocess.Popen(
                ["fstencode", "--decode", ngram_shrink_path, self.encoder_path, self.fst_path],
                stderr=logf,
            )
            fstencode_proc.communicate()

        os.remove(ngram_count_path)
        os.remove(ngram_make_path)
        os.remove(ngram_shrink_path)

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
        models_temp_dir = os.path.join(self.working_directory, "model_archive_tempo")
        model = G2PModel.empty(basename, root_directory=models_temp_dir)
        model.add_meta_file(self)
        model.add_fst_model(self.working_directory)
        model.add_sym_path(self.working_directory)
        if directory:
            os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(output_model_path)
        model.dump(basename)
        model.clean_up()
        self.clean_up()
        self.logger.info(f"Saved model to {output_model_path}")

    @property
    def fst_path(self):
        """Internal temporary FST file"""
        return os.path.join(self.working_directory, "model.fst")

    @property
    def far_path(self):
        """Internal temporary FAR file"""
        return os.path.join(self.working_directory, f"{self.data_source_identifier}.far")

    @property
    def encoder_path(self):
        """Internal temporary encoder file"""
        return os.path.join(self.working_directory, f"{self.data_source_identifier}.enc")

    @property
    def sym_path(self):
        """Internal temporary symbol file"""
        return os.path.join(self.working_directory, "phones.sym")

    def train(self) -> None:
        """
        Train a G2P model
        """
        aligner = PairNGramAligner(self.working_directory)
        input_token_type = "utf8"
        output_token_type = pynini.SymbolTable.read_text(self.sym_path)
        begin = time.time()
        if not os.path.exists(self.far_path) or not os.path.exists(self.encoder_path):
            aligner.align(
                self.input_path,
                self.far_path,
                self.encoder_path,
                input_token_type,
                self.input_epsilon,
                output_token_type,
                self.output_epsilon,
                self.num_jobs,
                self.random_starts,
                self.seed,
                self.batch_size,
                self.delta,
                self.lr,
                self.num_iterations,
                self.fst_default_cache_gc,
                self.fst_default_cache_gc_limit,
            )
        self.logger.debug(
            f"Aligning {len(self.g2p_training_dictionary)} words took {time.time() - begin} seconds"
        )
        begin = time.time()
        self.generate_model()
        self.logger.debug(
            f"Generating model for {len(self.g2p_training_dictionary)} words took {time.time() - begin} seconds"
        )

    def finalize_training(self) -> None:
        """Finalize training"""
        if self.evaluate:
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
            temporary_directory=os.path.join(self.working_directory, "validation"),
            num_jobs=self.num_jobs,
            num_pronunciations=self.num_pronunciations,
        )
        output = gen.generate_pronunciations()
        self.compute_validation_errors(output)

    def compute_validation_errors(
        self,
        hypothesis_values: dict[str, list[str]],
    ):
        """
        Computes validation errors

        Parameters
        ----------
        hypothesis_values: dict[str, list[str]]
            Hypothesis labels
        """
        begin = time.time()
        # Word-level measures.
        correct = 0
        incorrect = 0
        # Label-level measures.
        total_edits = 0
        total_length = 0
        # Since the edit distance algorithm is quadratic, let's do this with
        # multiprocessing.
        with mp.Pool(self.num_jobs) as pool:
            to_comp = []
            for word, hyp in hypothesis_values.items():
                g = self.g2p_validation_dictionary[word][0]["pronunciation"]
                hyp = [h.split(" ") for h in hyp]
                to_comp.append((g, hyp, True))  # Multiple hypotheses to compare
            gen = pool.starmap(score, to_comp)
            for (edits, length) in gen:
                if edits == 0:
                    correct += 1
                else:
                    incorrect += 1
                total_edits += edits
                total_length += length
            for w, gold in self.g2p_validation_dictionary.items():
                if w not in hypothesis_values:
                    incorrect += 1
                    gold = gold[0]["pronunciation"]
                    total_edits += len(gold)
                    total_length += len(gold)
        wer = 100 * incorrect / (correct + incorrect)
        ler = 100 * total_edits / total_length
        self.logger.info(f"WER:\t{wer:.2f}")
        self.logger.info(f"LER:\t{ler:.2f}")
        self.logger.debug(
            f"Computation of errors for {len(self.g2p_validation_dictionary)} words took {time.time() - begin} seconds"
        )
