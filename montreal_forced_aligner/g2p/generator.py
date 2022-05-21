"""Class for generating pronunciations from G2P models"""
from __future__ import annotations

import functools
import multiprocessing as mp
import os
import queue
import time
from typing import TYPE_CHECKING, Collection, Dict, List, Optional, Set, Tuple, Union

import tqdm

from montreal_forced_aligner.abc import TopLevelMfaWorker
from montreal_forced_aligner.corpus.text_corpus import TextCorpusMixin
from montreal_forced_aligner.exceptions import G2PError, PyniniGenerationError
from montreal_forced_aligner.g2p.mixins import G2PTopLevelMixin
from montreal_forced_aligner.helper import comma_join
from montreal_forced_aligner.models import G2PModel
from montreal_forced_aligner.utils import Stopped

try:
    import pynini
    from pynini import Fst, TokenType
    from pynini.lib import rewrite

    G2P_DISABLED = False
except ImportError:
    pynini = None
    TokenType = str
    Fst = None
    rewrite = None
    G2P_DISABLED = True

if TYPE_CHECKING:
    SpeakerCharacterType = Union[str, int]


__all__ = [
    "Rewriter",
    "RewriterWorker",
    "PyniniGenerator",
    "PyniniCorpusGenerator",
    "PyniniWordListGenerator",
]


def threshold_lattice_to_dfa(
    lattice: pynini.Fst, threshold: float = 0.99, state_multiplier: int = 2
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
    Args:
    lattice: Epsilon-free non-deterministic finite acceptor.
    threshold: Threshold for weights (1 is optimal only, 0 is for all paths)
    state_multiplier: Max ratio for the number of states in the DFA lattice to
      the NFA lattice; if exceeded, a warning is logged.
    Returns:
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
    input_token_type: Optional[pynini.TokenType] = None,
    output_token_type: Optional[pynini.TokenType] = None,
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
    lattice = threshold_lattice_to_dfa(lattice, threshold)
    return rewrite.lattice_to_strings(lattice, output_token_type)


def scored_match(
    input_string: pynini.FstLike,
    output_string: pynini.FstLike,
    rule: pynini.Fst,
    input_token_type: Optional[pynini.TokenType] = None,
    output_token_type: Optional[pynini.TokenType] = None,
    threshold: float = 6,
    state_multiplier: int = 4,
    lattice=None,
) -> float:
    if lattice is None:
        lattice = rewrite.rewrite_lattice(input_string, rule, input_token_type)
    with pynini.default_token_type(output_token_type):
        matched_lattice = pynini.intersect(lattice, output_string, compose_filter="sequence")
        matched_lattice = rewrite.lattice_to_dfa(matched_lattice, True, state_multiplier)
    if matched_lattice.start() == pynini.NO_STATE_ID:
        return -1
    matched_weight = float(matched_lattice.paths().weight())
    return matched_weight


class Rewriter:
    """Helper object for rewriting."""

    def __init__(
        self,
        fst: Fst,
        input_token_type: TokenType,
        output_token_type: TokenType,
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


class MatchScorer:
    """Helper object for matches input and output strings."""

    def __init__(
        self,
        fst: Fst,
        input_token_type: TokenType,
        output_token_type: TokenType,
        threshold: float = 1.0,
    ):
        self.fst = fst
        self.input_token_type = input_token_type
        self.output_token_type = output_token_type
        self.threshold = threshold
        self.match = functools.partial(
            scored_match,
            threshold=threshold,
            rule=fst,
            input_token_type=input_token_type,
            output_token_type=output_token_type,
        )

    def __call__(self, i: Tuple[str, Collection[str]]) -> Dict[str, float]:  # pragma: no cover
        """Call the rewrite function"""
        best_score = 100000
        word, pronunciations = i
        lattice = rewrite.rewrite_lattice(word, self.fst, self.input_token_type)
        output = {}
        for p in pronunciations:
            score = self.match(word, p, lattice=lattice)
            if score >= 0 and score < best_score:
                best_score = score
            output[p] = score
        for p, score in output.items():
            if score > 0:
                relative_score = best_score / score
            elif score == 0:
                relative_score = 1.0
            else:
                relative_score = 0.0
            output[p] = (score, relative_score)
        return output


class RewriterWorker(mp.Process):
    """
    Rewriter process
    """

    def __init__(
        self,
        job_q: mp.Queue,
        return_queue: mp.Queue,
        rewriter: Rewriter,
        stopped: Stopped,
    ):
        mp.Process.__init__(self)
        self.job_q = job_q
        self.return_queue = return_queue
        self.rewriter = rewriter
        self.stopped = stopped
        self.finished = Stopped()

    def run(self) -> None:
        """Run the rewriting function"""
        while True:
            try:
                word = self.job_q.get(timeout=1)
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
        self.g2p_model = G2PModel(g2p_model_path)
        self.strict_graphemes = strict_graphemes
        super().__init__(**kwargs)

    def generate_pronunciations(self) -> Dict[str, List[str]]:
        """
        Generate pronunciations

        Returns
        -------
        dict[str, list[str]]
            Mappings of keys to their generated pronunciations
        """
        if self.g2p_model.meta["architecture"] == "phonetisaurus":
            raise G2PError(
                "Previously trained Phonetisaurus models from 1.1 and earlier are not currently supported. "
                "Please retrain your model using 2.0+"
            )

        input_token_type = "utf8"
        fst = pynini.Fst.read(self.g2p_model.fst_path)

        output_token_type = "utf8"
        if self.g2p_model.sym_path is not None and os.path.exists(self.g2p_model.sym_path):
            output_token_type = pynini.SymbolTable.read_text(self.g2p_model.sym_path)
        rewriter = Rewriter(
            fst,
            input_token_type,
            output_token_type,
            num_pronunciations=self.num_pronunciations,
            threshold=self.g2p_threshold,
        )

        num_words = len(self.words_to_g2p)
        begin = time.time()
        missing_graphemes = set()
        self.log_info("Generating pronunciations...")
        to_return = {}
        skipped_words = 0
        if num_words < 30 or self.num_jobs == 1:
            with tqdm.tqdm(total=num_words, disable=getattr(self, "quiet", False)) as pbar:
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
                        prons = rewriter(w)
                    except rewrite.Error:
                        continue
                    to_return[word] = prons
                self.log_debug(
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
            self.log_debug(
                f"Skipping {skipped_words} words for containing the following graphemes: "
                f"{comma_join(sorted(missing_graphemes))}"
            )
            error_dict = {}
            return_queue = mp.Queue()
            procs = []
            for _ in range(self.num_jobs):
                p = RewriterWorker(
                    job_queue,
                    return_queue,
                    rewriter,
                    stopped,
                )
                procs.append(p)
                p.start()
            num_words -= skipped_words
            with tqdm.tqdm(total=num_words, disable=getattr(self, "quiet", False)) as pbar:
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
        self.log_debug(f"Processed {num_words} in {time.time() - begin} seconds")
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
        """Data directory"""
        return ""

    @property
    def data_directory(self) -> str:
        """Data directory"""
        return self.working_directory

    def setup(self) -> None:
        """Set up the G2P generator"""
        if self.initialized:
            return
        self.g2p_model.validate(self.words_to_g2p)
        self.initialized = True


class PyniniWordListGenerator(PyniniValidator):
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
        with open(self.word_list_path, "r", encoding="utf8") as f:
            for line in f:
                self.word_list.extend(line.strip().split())
        if not self.include_bracketed:
            self.word_list = [x for x in self.word_list if not self.check_bracketed(x)]
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
        self.calculate_word_counts()
        self.g2p_model.validate(self.words_to_g2p)
        self.initialized = True

    @property
    def words_to_g2p(self) -> List[str]:
        """Words to produce pronunciations"""
        word_list = self.corpus_word_set
        if not self.include_bracketed:
            word_list = [x for x in word_list if not self.check_bracketed(x)]
        return word_list


class OrthographicCorpusGenerator(OrthographyGenerator, TextCorpusMixin, TopLevelMfaWorker):
    """
    Top-level class for generating "pronunciations" from the orthography of a corpus

    See Also
    --------
    :class:`~montreal_forced_aligner.g2p.generator.OrthographyGenerator`
        For orthography-based G2P generation parameters
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
        self.calculate_word_counts()
        self.initialized = True

    @property
    def words_to_g2p(self) -> List[str]:
        """Words to produce pronunciations"""
        word_list = self.corpus_word_set
        if not self.include_bracketed:
            word_list = [x for x in word_list if not self.check_bracketed(x)]
        return word_list


class OrthographicWordListGenerator(OrthographyGenerator, TopLevelMfaWorker):
    """
    Top-level class for generating "pronunciations" from the orthography of a corpus

    Parameters
    ----------
    word_list_path: str
        Path to word list file
    See Also
    --------
    :class:`~montreal_forced_aligner.g2p.generator.OrthographyGenerator`
        For orthography-based G2P generation parameters
    :class:`~montreal_forced_aligner.abc.TopLevelMfaWorker`
        For top-level parameters

    Attributes
    ----------
    word_list: list[str]
        Word list to generate pronunciations
    """

    def __init__(self, word_list_path: str, **kwargs):
        super().__init__(**kwargs)
        self.word_list_path = word_list_path
        self.word_list = []

    def setup(self) -> None:
        """Set up the pronunciation generator"""
        if self.initialized:
            return
        with open(self.word_list_path, "r", encoding="utf8") as f:
            for line in f:
                self.word_list.extend(line.strip().split())
        if not self.include_bracketed:
            self.word_list = [x for x in self.word_list if not self.check_bracketed(x)]
        self.initialized = True

    @property
    def words_to_g2p(self) -> List[str]:
        """Words to produce pronunciations"""
        return self.word_list
