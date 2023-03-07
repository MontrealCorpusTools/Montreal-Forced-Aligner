"""Class for generating pronunciations from G2P models"""
from __future__ import annotations

import csv
import functools
import itertools
import logging
import multiprocessing as mp
import os
import queue
import statistics
import time
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import pynini
import pywrapfst
from praatio import textgrid
from pynini import Fst, TokenType
from pynini.lib import rewrite
from pywrapfst import SymbolTable
from sqlalchemy.orm import Session, selectinload
from tqdm.rich import tqdm

from montreal_forced_aligner.abc import DatabaseMixin, KaldiFunction, TopLevelMfaWorker
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.corpus.text_corpus import DictionaryTextCorpusMixin, TextCorpusMixin
from montreal_forced_aligner.data import MfaArguments, TextgridFormats, WordType, WorkflowType
from montreal_forced_aligner.db import File, Utterance, Word, bulk_update
from montreal_forced_aligner.exceptions import PyniniGenerationError
from montreal_forced_aligner.g2p.mixins import G2PTopLevelMixin
from montreal_forced_aligner.helper import comma_join, mfa_open, score_g2p
from montreal_forced_aligner.models import G2PModel
from montreal_forced_aligner.textgrid import construct_output_path
from montreal_forced_aligner.utils import Stopped, run_kaldi_function

if TYPE_CHECKING:
    from dataclasses import dataclass

    SpeakerCharacterType = Union[str, int]
else:
    from dataclassy import dataclass


__all__ = [
    "Rewriter",
    "RewriterWorker",
    "PyniniGenerator",
    "PyniniCorpusGenerator",
    "PyniniWordListGenerator",
    "PyniniValidator",
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
        graphemes: Set[str] = None,
    ):
        self.graphemes = graphemes
        self.input_token_type = input_token_type
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

    def create_word_fst(self, word: str) -> pynini.Fst:
        if self.graphemes is not None:
            word = "".join([x for x in word if x in self.graphemes])
        fst = pynini.accep(word, token_type=self.input_token_type)
        return fst

    def __call__(self, graphemes: str) -> List[str]:  # pragma: no cover
        """Call the rewrite function"""
        if " " in graphemes:
            words = graphemes.split()
            hypotheses = []
            for w in words:
                w_fst = self.create_word_fst(w)
                hypotheses.append(self.rewrite(w_fst))
            hypotheses = sorted(set(" ".join(x) for x in itertools.product(*hypotheses)))
        else:
            fst = self.create_word_fst(graphemes)
            hypotheses = self.rewrite(fst)
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
        graphemes: Set[str] = None,
    ):
        self.fst = fst
        self.seq_sep = seq_sep
        self.input_token_type = input_token_type
        self.output_token_type = output_token_type
        self.grapheme_order = grapheme_order
        self.graphemes = graphemes
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

    def create_word_fst(self, word: str) -> pynini.Fst:
        if self.graphemes is not None:
            word = [x for x in word if x in self.graphemes]
        fst = pynini.Fst()
        one = pywrapfst.Weight.one(fst.weight_type())
        max_state = 0
        for i in range(len(word)):
            start_state = fst.add_state()
            for j in range(1, self.grapheme_order + 1):
                if i + j <= len(word):
                    substring = self.seq_sep.join(word[i : i + j])
                    ilabel = self.input_token_type.find(substring)
                    if ilabel != pywrapfst.NO_LABEL:
                        fst.add_arc(start_state, pywrapfst.Arc(ilabel, ilabel, one, i + j))
                    if i + j >= max_state:
                        max_state = i + j
        for _ in range(fst.num_states(), max_state + 1):
            fst.add_state()
        fst.set_start(0)
        fst.set_final(len(word), one)
        fst.set_input_symbols(self.input_token_type)
        fst.set_output_symbols(self.input_token_type)
        return fst

    def __call__(self, graphemes: str) -> List[str]:  # pragma: no cover
        """Call the rewrite function"""
        if " " in graphemes:
            words = graphemes.split()
            hypotheses = []
            for w in words:
                w_fst = self.create_word_fst(w)
                hypotheses.append(self.rewrite(w_fst))
            hypotheses = sorted(set(" ".join(x) for x in itertools.product(*hypotheses)))
        else:
            fst = self.create_word_fst(graphemes)
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


@dataclass
class G2PArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.CompileTrainGraphsFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    tree_path: :class:`~pathlib.Path`
        Path to tree file
    model_path: :class:`~pathlib.Path`
        Path to model file
    use_g2p: bool
        Flag for whether acoustic model uses g2p
    """

    rewriter: Rewriter


class G2PFunction(KaldiFunction):
    def __init__(self, args: G2PArguments):
        super().__init__(args)
        self.rewriter = args.rewriter

    def _run(self) -> typing.Generator[typing.Tuple[int, str]]:
        """Run the function"""

        with mfa_open(self.log_path, "w") as log_file, Session(self.db_engine()) as session:
            query = (
                session.query(Utterance.id, Utterance.normalized_text)
                .filter(Utterance.job_id == self.job_name)
                .filter(Utterance.normalized_text != "")
            )
            for id, text in query:
                try:
                    pronunciation_text = self.rewriter(text)[0]
                    yield id, pronunciation_text
                except pynini.lib.rewrite.Error:
                    log_file.write(f"Error on generating pronunciation for {text}\n")


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

    def __init__(self, g2p_model_path: Path = None, strict_graphemes: bool = False, **kwargs):
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
                graphemes=self.g2p_model.meta["graphemes"],
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
            with tqdm(total=num_words, disable=GLOBAL_CONFIG.quiet) as pbar:
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
            with tqdm(total=num_words, disable=GLOBAL_CONFIG.quiet) as pbar:
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


class PyniniConsoleGenerator(PyniniGenerator):
    @property
    def data_directory(self) -> Path:
        return Path("-")

    @property
    def working_directory(self) -> Path:
        return GLOBAL_CONFIG.current_profile.temporary_directory.joinpath("g2p_stdin")

    def cleanup(self) -> None:
        pass


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
    def data_directory(self) -> Path:
        """Data directory"""
        return self.working_directory

    @property
    def evaluation_csv_path(self) -> Path:
        """Path to working directory's CSV file"""
        return self.working_directory.joinpath("pronunciation_evaluation.csv")

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
            if not isinstance(hyp, list):
                hyp = [hyp]
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
    word_list_path: :class:`~pathlib.Path`
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

    def __init__(self, word_list_path: Path, **kwargs):
        self.word_list_path = word_list_path
        super().__init__(**kwargs)

    @property
    def data_directory(self) -> Path:
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

    def __init__(self, per_utterance: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.per_utterance = per_utterance

    def setup(self) -> None:
        """Set up the pronunciation generator"""
        if self.initialized:
            return
        self._load_corpus()
        self.initialize_jobs()
        super().setup()
        self._create_dummy_dictionary()
        self.normalize_text()
        self.create_new_current_workflow(WorkflowType.g2p)
        self.g2p_model.validate(self.words_to_g2p)
        self.initialized = True

    def g2p_arguments(self) -> List[G2PArguments]:
        return [
            G2PArguments(
                j.id,
                getattr(self, "db_string", ""),
                self.working_log_directory.joinpath(f"g2p_utterances.{j.id}.log"),
                self.rewriter,
            )
            for j in self.jobs
        ]

    def export_file_pronunciations(self, output_file_path: Path):
        """
        Generate and export per-utterance G2P

        Parameters
        ----------
        output_file_path: :class:`~pathlib.Path`
            Output directory to save utterance pronunciations

        """
        output_file_path.mkdir(parents=True, exist_ok=True)
        if self.num_pronunciations != 1:
            logger.warning(
                "Number of pronunciations is hard-coded to 1 for generating per-utterance pronunciations"
            )
            self.num_pronunciations = 1
        begin = time.time()
        if self.rewriter is None:
            self.setup()
        logger.info("Generating pronunciations...")
        with tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            update_mapping = []
            for utt_id, pronunciation in run_kaldi_function(
                G2PFunction, self.g2p_arguments(), pbar.update
            ):
                update_mapping.append({"id": utt_id, "transcription_text": pronunciation})
        with self.session() as session:
            bulk_update(session, Utterance, update_mapping)
            session.commit()

        logger.debug(f"Processed {self.num_utterances} in {time.time() - begin:.3f} seconds")
        logger.info("Exporting files...")
        with self.session() as session:
            files = session.query(File).options(
                selectinload(File.utterances), selectinload(File.speakers)
            )
            for file in files:
                utterance_count = len(file.utterances)
                if file.sound_file is not None:
                    duration = file.sound_file.duration
                else:
                    duration = file.utterances[-1].end

                if utterance_count == 0:
                    logger.debug(f"Could not find any utterances for {file.name}")
                elif (
                    utterance_count == 1
                    and file.utterances[0].begin == 0
                    and file.utterances[0].end == duration
                ):
                    output_format = "lab"
                else:
                    output_format = TextgridFormats.SHORT_TEXTGRID
                output_path = construct_output_path(
                    file.name,
                    file.relative_path,
                    output_file_path,
                    output_format=output_format,
                )
                data = file.construct_transcription_tiers()
                if output_format == "lab":
                    for intervals in data.values():
                        with mfa_open(output_path, "w") as f:
                            f.write(intervals["transcription"][0].label)
                else:
                    tg = textgrid.Textgrid()
                    tg.minTimestamp = 0
                    tg.maxTimestamp = round(duration, 5)
                    for speaker in file.speakers:
                        speaker = speaker.name
                        intervals = data[speaker]["transcription"]
                        tier = textgrid.IntervalTier(
                            speaker,
                            [x.to_tg_interval() for x in intervals],
                            minT=0,
                            maxT=round(duration, 5),
                        )

                        tg.addTier(tier)
                    tg.save(output_path, includeBlankSpaces=True, format=output_format)

    @property
    def words_to_g2p(self) -> List[str]:
        """Words to produce pronunciations"""
        word_list = self.corpus_word_set
        if not self.include_bracketed:
            word_list = [x for x in word_list if not self.check_bracketed(x)]
        return word_list

    def export_pronunciations(self, output_file_path: typing.Union[str, Path]) -> None:
        if self.per_utterance:
            self.export_file_pronunciations(output_file_path)
        else:
            super().export_pronunciations(output_file_path)


class PyniniDictionaryCorpusGenerator(
    PyniniGenerator, DictionaryTextCorpusMixin, TopLevelMfaWorker
):
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
        self._word_list = None

    def setup(self) -> None:
        """Set up the pronunciation generator"""
        if self.initialized:
            return
        self.load_corpus()
        super().setup()
        self.g2p_model.validate(self.words_to_g2p)
        self.initialized = True

    @property
    def words_to_g2p(self) -> List[str]:
        """Words to produce pronunciations"""
        if self._word_list is None:
            with self.session() as session:
                query = (
                    session.query(Word.word)
                    .filter(Word.word_type == WordType.oov, Word.word != self.oov_word)
                    .order_by(Word.word)
                )
            self._word_list = [x for x, in query]
        return self._word_list
