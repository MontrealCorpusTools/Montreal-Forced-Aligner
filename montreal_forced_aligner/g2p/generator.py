"""Class for generating pronunciations from G2P models"""
from __future__ import annotations

import csv
import functools
import logging
import multiprocessing as mp
import os
import queue
import statistics
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import pynini
import tqdm
from pynini import Fst, TokenType
from pynini.lib import rewrite
from pywrapfst import SymbolTable

from montreal_forced_aligner.abc import DatabaseMixin, TopLevelMfaWorker
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.corpus.text_corpus import TextCorpusMixin
from montreal_forced_aligner.exceptions import PyniniGenerationError
from montreal_forced_aligner.g2p.mixins import G2PTopLevelMixin
from montreal_forced_aligner.helper import comma_join, mfa_open, score_g2p
from montreal_forced_aligner.models import G2PModel
from montreal_forced_aligner.utils import Stopped

if TYPE_CHECKING:
    SpeakerCharacterType = Union[str, int]


__all__ = [
    "Rewriter",
    "RewriterWorker",
    "PyniniGenerator",
    "PyniniCorpusGenerator",
    "PyniniWordListGenerator",
]

logger = logging.getLogger("mfa")


def threshold_lattice_to_dfa(
    lattice: pynini.Fst, threshold: float = 1.0, state_multiplier: int = 2
) -> pynini.Fst:
    """Constructs a (possibly pruned) weighted DFA of output strings.
    Given an epsilon-free lattice of output strings (such as produced by
    rewrite_lattice), attempts to determinize it, pruning non-optimal paths if
    optimal_only is true. This is valid only in a semiring with the path property.
    To prevent unexpected blowup during determinization, a state threshold is
    also used and a warning is logged if this exact threshold is reached. The
    threshold is a multiplier of the size of input lattice (by default, 4), plus
    a small constant factor. This is intended by a sensible default and is not an
    inherently meaningful value in and of itself.

    Parameters
    ----------
    lattice: :class:`~pynini.Fst`
        Epsilon-free non-deterministic finite acceptor.
    threshold: float
        Threshold for weights, 1.0 is optimal only, 0 is for all paths, greater than 1
        prunes the lattice to include paths with costs less than the optimal path's score times the threshold
    state_multiplier: int
        Max ratio for the number of states in the DFA lattice to the NFA lattice; if exceeded, a warning is logged.

    Returns
    -------
    :class:`~pynini.Fst`
        Epsilon-free deterministic finite acceptor.
    """
    weight_type = lattice.weight_type()
    weight_threshold = pynini.Weight(weight_type, threshold)
    state_threshold = 256 + state_multiplier * lattice.num_states()
    lattice = pynini.determinize(lattice, nstate=state_threshold, weight=weight_threshold)
    return lattice


def optimal_rewrites(
    string: pynini.FstLike,
    rule: pynini.Fst,
    input_token_type: Optional[TokenType] = None,
    output_token_type: Optional[TokenType] = None,
    threshold: float = 1,
) -> List[str]:
    """Returns all optimal rewrites.
    Args:
    string: Input string or FST.
    rule: Input rule WFST.
    input_token_type: Optional input token type, or symbol table.
    output_token_type: Optional output token type, or symbol table.
    threshold: Threshold for weights (1 is optimal only, 0 is for all paths)
    Returns:
    A tuple of output strings.
    """
    lattice = rewrite.rewrite_lattice(string, rule, input_token_type)
    lattice = threshold_lattice_to_dfa(lattice, threshold, 4)
    return rewrite.lattice_to_strings(lattice, output_token_type)


class Rewriter:
    """
    Helper object for rewriting

    Parameters
    ----------
    fst: pynini.Fst
        G2P FST model
    input_token_type: pynini.TokenType
        Grapheme symbol table or "utf8"
    output_token_type: pynini.SymbolTable
        Phone symbol table
    num_pronunciations: int
        Number of pronunciations, default to 0.  If this is 0, thresholding is used
    threshold: float
        Threshold to use for pruning rewrite lattice, defaults to 1.5, only used if num_pronunciations is 0
    """

    def __init__(
        self,
        fst: Fst,
        input_token_type: TokenType,
        output_token_type: SymbolTable,
        num_pronunciations: int = 0,
        threshold: float = 1,
    ):
        if num_pronunciations > 0:
            self.rewrite = functools.partial(
                rewrite.top_rewrites,
                nshortest=num_pronunciations,
                rule=fst,
                input_token_type=input_token_type,
                output_token_type=output_token_type,
            )
        else:
            self.rewrite = functools.partial(
                optimal_rewrites,
                threshold=threshold,
                rule=fst,
                input_token_type=input_token_type,
                output_token_type=output_token_type,
            )

    def __call__(self, i: str) -> List[Tuple[str, ...]]:  # pragma: no cover
        """Call the rewrite function"""
        hypotheses = self.rewrite(i)
        return [x for x in hypotheses if x]


class PhonetisaurusRewriter:
    """
    Helper function for rewriting

    Parameters
    ----------
    fst: pynini.Fst
        G2P FST model
    input_token_type: pynini.SymbolTable
        Grapheme symbol table
    output_token_type: pynini.SymbolTable
    num_pronunciations: int
        Number of pronunciations, default to 0.  If this is 0, thresholding is used
    threshold: float
        Threshold to use for pruning rewrite lattice, defaults to 1.5, only used if num_pronunciations is 0
    grapheme_order: int
        Maximum number of graphemes to consider single segment
    seq_sep: str
        Separator to use between grapheme symbols
    """

    def __init__(
        self,
        fst: Fst,
        input_token_type: SymbolTable,
        output_token_type: SymbolTable,
        num_pronunciations: int = 0,
        threshold: float = 1.5,
        grapheme_order: int = 2,
        seq_sep: str = "|",
    ):
        self.fst = fst
        self.seq_sep = seq_sep
        self.input_token_type = input_token_type
        self.output_token_type = output_token_type
        self.grapheme_order = grapheme_order
        if num_pronunciations > 0:
            self.rewrite = functools.partial(
                rewrite.top_rewrites,
                nshortest=num_pronunciations,
                rule=fst,
                input_token_type=None,
                output_token_type=output_token_type,
            )
        else:
            self.rewrite = functools.partial(
                optimal_rewrites,
                threshold=threshold,
                rule=fst,
                input_token_type=None,
                output_token_type=output_token_type,
            )

    def __call__(self, graphemes: str) -> List[Tuple[str, ...]]:  # pragma: no cover
        """Call the rewrite function"""
        fst = pynini.Fst()
        one = pynini.Weight.one(fst.weight_type())
        max_state = 0
        for i in range(len(graphemes)):
            start_state = fst.add_state()
            for j in range(1, self.grapheme_order + 1):
                if i + j <= len(graphemes):
                    substring = self.seq_sep.join(graphemes[i : i + j])
                    ilabel = self.input_token_type.find(substring)
                    if ilabel != pynini.NO_LABEL:
                        fst.add_arc(start_state, pynini.Arc(ilabel, ilabel, one, i + j))
                    if i + j >= max_state:
                        max_state = i + j
        for _ in range(fst.num_states(), max_state + 1):
            fst.add_state()
        fst.set_start(0)
        fst.set_final(len(graphemes), one)
        fst.set_input_symbols(self.input_token_type)
        fst.set_output_symbols(self.input_token_type)
        hypotheses = self.rewrite(fst)
        hypotheses = [x.replace(self.seq_sep, " ") for x in hypotheses if x]
        return hypotheses


class RewriterWorker(mp.Process):
    """
    Rewriter process

    Parameters
    ----------
    job_queue: :class:`~multiprocessing.Queue`
        Queue to pull words from
    return_queue: :class:`~multiprocessing.Queue`
        Queue to put pronunciations
    rewriter: :class:`~montreal_forced_aligner.g2p.generator.Rewriter`
        Function to generate pronunciations of words
    stopped: :class:`~montreal_forced_aligner.utils.Stopped`
        Stop check
    """

    def __init__(
        self,
        job_queue: mp.Queue,
        return_queue: mp.Queue,
        rewriter: Rewriter,
        stopped: Stopped,
    ):
        mp.Process.__init__(self)
        self.job_queue = job_queue
        self.return_queue = return_queue
        self.rewriter = rewriter
        self.stopped = stopped
        self.finished = Stopped()

    def run(self) -> None:
        """Run the rewriting function"""
        while True:
            try:
                word = self.job_queue.get(timeout=1)
            except queue.Empty:
                break
            if self.stopped.stop_check():
                continue
            try:
                rep = self.rewriter(word)
                self.return_queue.put((word, rep))
            except rewrite.Error:
                pass
            except Exception as e:  # noqa
                self.stopped.stop()
                self.return_queue.put(e)
                raise
        self.finished.stop()
        return


def clean_up_word(word: str, graphemes: Set[str]) -> Tuple[str, Set[str]]:
    """
    Clean up word by removing graphemes not in a specified set

    Parameters
    ----------
    word : str
        Input string
    graphemes: set[str]
        Set of allowable graphemes

    Returns
    -------
    str
        Cleaned up word
    Set[str]
        Graphemes excluded
    """
    new_word = []
    missing_graphemes = set()
    for c in word:
        if c not in graphemes:
            missing_graphemes.add(c)
        else:
            new_word.append(c)
    return "".join(new_word), missing_graphemes


class OrthographyGenerator(G2PTopLevelMixin):
    """
    Abstract mixin class for generating "pronunciations" based off the orthographic word

    See Also
    --------
    :class:`~montreal_forced_aligner.g2p.mixins.G2PTopLevelMixin`
        For top level G2P generation parameters
    """

    def generate_pronunciations(self) -> Dict[str, List[str]]:
        """
        Generate pronunciations for the word set

        Returns
        -------
        dict[str, Word]
            Mapping of words to their "pronunciation"
        """
        pronunciations = {}
        for word in self.words_to_g2p:
            pronunciations[word] = [" ".join(word)]
        return pronunciations


class PyniniGenerator(G2PTopLevelMixin):
    """
    Class for generating pronunciations from a Pynini G2P model

    Parameters
    ----------
    g2p_model_path: str
        Path to G2P model
    strict_graphemes: bool
        Flag for whether to be strict with missing graphemes and skip words containing new graphemes

    See Also
    --------
    :class:`~montreal_forced_aligner.g2p.mixins.G2PTopLevelMixin`
        For top level G2P generation parameters

    Attributes
    ----------
    g2p_model: G2PModel
        G2P model
    """

    def __init__(self, g2p_model_path: str, strict_graphemes: bool = False, **kwargs):
        self.strict_graphemes = strict_graphemes
        super().__init__(**kwargs)
        self.g2p_model = G2PModel(
            g2p_model_path, root_directory=getattr(self, "workflow_directory", None)
        )
        self.output_token_type = "utf8"
        self.input_token_type = "utf8"
        self.rewriter = None

    def setup(self):
        self.fst = pynini.Fst.read(self.g2p_model.fst_path)
        if self.g2p_model.meta["architecture"] == "phonetisaurus":
            self.output_token_type = pynini.SymbolTable.read_text(self.g2p_model.sym_path)
            self.input_token_type = pynini.SymbolTable.read_text(self.g2p_model.grapheme_sym_path)
            self.fst.set_input_symbols(self.input_token_type)
            self.fst.set_output_symbols(self.output_token_type)
            self.rewriter = PhonetisaurusRewriter(
                self.fst,
                self.input_token_type,
                self.output_token_type,
                num_pronunciations=self.num_pronunciations,
                threshold=self.g2p_threshold,
                grapheme_order=self.g2p_model.meta["grapheme_order"],
            )
        else:
            if self.g2p_model.sym_path is not None and os.path.exists(self.g2p_model.sym_path):
                self.output_token_type = pynini.SymbolTable.read_text(self.g2p_model.sym_path)

            self.rewriter = Rewriter(
                self.fst,
                self.input_token_type,
                self.output_token_type,
                num_pronunciations=self.num_pronunciations,
                threshold=self.g2p_threshold,
            )

    def generate_pronunciations(self) -> Dict[str, List[str]]:
        """
        Generate pronunciations

        Returns
        -------
        dict[str, list[str]]
            Mappings of keys to their generated pronunciations
        """

        num_words = len(self.words_to_g2p)
        begin = time.time()
        missing_graphemes = set()
        if self.rewriter is None:
            self.setup()
        logger.info("Generating pronunciations...")
        to_return = {}
        skipped_words = 0
        if num_words < 30 or GLOBAL_CONFIG.num_jobs == 1:
            with tqdm.tqdm(total=num_words, disable=GLOBAL_CONFIG.quiet) as pbar:
                for word in self.words_to_g2p:
                    w, m = clean_up_word(word, self.g2p_model.meta["graphemes"])
                    pbar.update(1)
                    missing_graphemes = missing_graphemes | m
                    if self.strict_graphemes and m:
                        skipped_words += 1
                        continue
                    if not w:
                        skipped_words += 1
                        continue
                    try:
                        prons = self.rewriter(w)
                    except rewrite.Error:
                        continue
                    to_return[word] = prons
                logger.debug(
                    f"Skipping {skipped_words} words for containing the following graphemes: "
                    f"{comma_join(sorted(missing_graphemes))}"
                )
        else:
            stopped = Stopped()
            job_queue = mp.Queue()
            for word in self.words_to_g2p:
                w, m = clean_up_word(word, self.g2p_model.meta["graphemes"])
                missing_graphemes = missing_graphemes | m
                if self.strict_graphemes and m:
                    skipped_words += 1
                    continue
                if not w:
                    skipped_words += 1
                    continue
                job_queue.put(w)
            logger.debug(
                f"Skipping {skipped_words} words for containing the following graphemes: "
                f"{comma_join(sorted(missing_graphemes))}"
            )
            error_dict = {}
            return_queue = mp.Queue()
            procs = []
            for _ in range(GLOBAL_CONFIG.num_jobs):
                p = RewriterWorker(
                    job_queue,
                    return_queue,
                    self.rewriter,
                    stopped,
                )
                procs.append(p)
                p.start()
            num_words -= skipped_words
            with tqdm.tqdm(total=num_words, disable=GLOBAL_CONFIG.quiet) as pbar:
                while True:
                    try:
                        word, result = return_queue.get(timeout=1)
                        if stopped.stop_check():
                            continue
                    except queue.Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(1)
                    if isinstance(result, Exception):
                        error_dict[word] = result
                        continue
                    to_return[word] = result

            for p in procs:
                p.join()
            if error_dict:
                raise PyniniGenerationError(error_dict)
        logger.debug(f"Processed {num_words} in {time.time() - begin:.3f} seconds")
        return to_return


class PyniniValidator(PyniniGenerator, TopLevelMfaWorker):
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

    def __init__(self, word_list: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        if word_list is None:
            word_list = []
        self.word_list = word_list

    @property
    def words_to_g2p(self) -> List[str]:
        """Words to produce pronunciations"""
        return self.word_list

    @property
    def data_source_identifier(self) -> str:
        """Dummy "validation" data source"""
        return "validation"

    @property
    def data_directory(self) -> str:
        """Data directory"""
        return self.working_directory

    @property
    def evaluation_csv_path(self) -> str:
        """Path to working directory's CSV file"""
        return os.path.join(self.working_directory, "pronunciation_evaluation.csv")

    def setup(self) -> None:
        """Set up the G2P validator"""
        TopLevelMfaWorker.setup(self)
        if self.initialized:
            return
        self._current_workflow = "validation"
        os.makedirs(self.working_log_directory, exist_ok=True)
        self.g2p_model.validate(self.words_to_g2p)
        PyniniGenerator.setup(self)
        self.initialized = True
        self.wer = None
        self.ler = None

    def compute_validation_errors(
        self,
        gold_values: Dict[str, Set[str]],
        hypothesis_values: Dict[str, List[str]],
    ):
        """
        Computes validation errors

        Parameters
        ----------
        gold_values: dict[str, set[str]]
            Gold pronunciations
        hypothesis_values: dict[str, list[str]]
            Hypothesis pronunciations
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
        logger.debug(f"Processing results for {len(hypothesis_values)} hypotheses")
        to_comp = []
        indices = []
        hyp_pron_count = 0
        gold_pron_count = 0
        output = []
        for word, gold_pronunciations in gold_values.items():
            if word not in hypothesis_values:
                incorrect += 1
                gold_length = statistics.mean(len(x.split()) for x in gold_pronunciations)
                total_edits += gold_length
                total_length += gold_length
                output.append(
                    {
                        "Word": word,
                        "Gold pronunciations": ", ".join(gold_pronunciations),
                        "Hypothesis pronunciations": "",
                        "Accuracy": 0,
                        "Error rate": 1.0,
                        "Length": gold_length,
                    }
                )
                continue
            hyp = hypothesis_values[word]
            for h in hyp:
                if h in gold_pronunciations:
                    correct += 1
                    total_length += len(h)
                    output.append(
                        {
                            "Word": word,
                            "Gold pronunciations": ", ".join(gold_pronunciations),
                            "Hypothesis pronunciations": ", ".join(hyp),
                            "Accuracy": 1,
                            "Error rate": 0.0,
                            "Length": len(h),
                        }
                    )
                    break
            else:
                incorrect += 1
                indices.append(word)
                to_comp.append((gold_pronunciations, hyp))  # Multiple hypotheses to compare
            logger.debug(
                f"For the word {word}: gold is {gold_pronunciations}, hypothesized are: {hyp}"
            )
            hyp_pron_count += len(hyp)
            gold_pron_count += len(gold_pronunciations)
        logger.debug(
            f"Generated an average of {hyp_pron_count /len(hypothesis_values)} variants "
            f"The gold set had an average of {gold_pron_count/len(hypothesis_values)} variants."
        )
        with mp.Pool(GLOBAL_CONFIG.num_jobs) as pool:
            gen = pool.starmap(score_g2p, to_comp)
            for i, (edits, length) in enumerate(gen):
                word = indices[i]
                gold_pronunciations = gold_values[word]
                hyp = hypothesis_values[word]
                output.append(
                    {
                        "Word": word,
                        "Gold pronunciations": ", ".join(gold_pronunciations),
                        "Hypothesis pronunciations": ", ".join(hyp),
                        "Accuracy": 1,
                        "Error rate": edits / length,
                        "Length": length,
                    }
                )
                total_edits += edits
                total_length += length
        with mfa_open(self.evaluation_csv_path, "w") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Word",
                    "Gold pronunciations",
                    "Hypothesis pronunciations",
                    "Accuracy",
                    "Error rate",
                    "Length",
                ],
            )
            writer.writeheader()
            for line in output:
                writer.writerow(line)
        self.wer = 100 * incorrect / (correct + incorrect)
        self.ler = 100 * total_edits / total_length
        logger.info(f"WER:\t{self.wer:.2f}")
        logger.info(f"LER:\t{self.ler:.2f}")
        logger.debug(
            f"Computation of errors for {len(gold_values)} words took {time.time() - begin:.3f} seconds"
        )

    def evaluate_g2p_model(self, gold_pronunciations: Dict[str, Set[str]]) -> None:
        """
        Evaluate a G2P model on the word list

        Parameters
        ----------
        gold_pronunciations: dict[str, set[str]]
            Gold pronunciations
        """
        output = self.generate_pronunciations()
        self.compute_validation_errors(gold_pronunciations, output)


class PyniniWordListGenerator(PyniniValidator, DatabaseMixin):
    """
    Top-level worker for generating pronunciations from a word list and a Pynini G2P model

    Parameters
    ----------
    word_list_path: str
        Path to word list file

    See Also
    --------
    :class:`~montreal_forced_aligner.g2p.generator.PyniniGenerator`
        For Pynini G2P generation parameters
    :class:`~montreal_forced_aligner.abc.TopLevelMfaWorker`
        For top-level parameters

    Attributes
    ----------
    word_list: list[str]
        Word list to generate pronunciations
    """

    def __init__(self, word_list_path: str, **kwargs):
        self.word_list_path = word_list_path
        super().__init__(**kwargs)

    @property
    def data_directory(self) -> str:
        """Data directory"""
        return self.working_directory

    @property
    def data_source_identifier(self) -> str:
        """Name of the word list file"""
        return os.path.splitext(os.path.basename(self.word_list_path))[0]

    def setup(self) -> None:
        """Set up the G2P generator"""
        if self.initialized:
            return
        with mfa_open(self.word_list_path, "r") as f:
            for line in f:
                self.word_list.extend(line.strip().split())
        if not self.include_bracketed:
            self.word_list = [x for x in self.word_list if not self.check_bracketed(x)]
        super().setup()
        self.g2p_model.validate(self.words_to_g2p)
        self.initialized = True


class PyniniCorpusGenerator(PyniniGenerator, TextCorpusMixin, TopLevelMfaWorker):
    """
    Top-level worker for generating pronunciations from a corpus and a Pynini G2P model

    See Also
    --------
    :class:`~montreal_forced_aligner.g2p.generator.PyniniGenerator`
        For Pynini G2P generation parameters
    :class:`~montreal_forced_aligner.corpus.text_corpus.TextCorpusMixin`
        For corpus parsing parameters
    :class:`~montreal_forced_aligner.abc.TopLevelMfaWorker`
        For top-level parameters
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self) -> None:
        """Set up the pronunciation generator"""
        if self.initialized:
            return
        self._load_corpus()
        self.initialize_jobs()
        super().setup()
        self._create_dummy_dictionary()
        self.normalize_text()
        self.g2p_model.validate(self.words_to_g2p)
        self.initialized = True

    @property
    def words_to_g2p(self) -> List[str]:
        """Words to produce pronunciations"""
        word_list = self.corpus_word_set
        if not self.include_bracketed:
            word_list = [x for x in word_list if not self.check_bracketed(x)]
        return word_list
