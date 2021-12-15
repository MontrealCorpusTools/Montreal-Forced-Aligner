"""Class for generating pronunciations from G2P models"""
from __future__ import annotations

import functools
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from typing import TYPE_CHECKING, Dict, List, Set, Tuple, Union

import tqdm

from montreal_forced_aligner.abc import TopLevelMfaWorker
from montreal_forced_aligner.corpus.text_corpus import TextCorpusMixin
from montreal_forced_aligner.exceptions import G2PError, PyniniGenerationError
from montreal_forced_aligner.g2p.mixins import G2PTopLevelMixin
from montreal_forced_aligner.helper import comma_join
from montreal_forced_aligner.models import G2PModel
from montreal_forced_aligner.utils import Counter, Stopped

try:
    import pynini
    from pynini import Fst, TokenType
    from pynini.lib import rewrite

    G2P_DISABLED = False
except ImportError:
    pynini = None
    TokenType = None
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


class Rewriter:
    """Helper object for rewriting."""

    def __init__(
        self, fst: Fst, input_token_type: TokenType, output_token_type: TokenType, nshortest=1
    ):
        self.rewrite = functools.partial(
            rewrite.top_rewrites,
            nshortest=nshortest,
            rule=fst,
            input_token_type=input_token_type,
            output_token_type=output_token_type,
        )

    def __call__(self, i: str) -> str:  # pragma: no cover
        """Call the rewrite function"""
        try:
            return self.rewrite(i)
        except rewrite.Error:
            return "<composition failure>"


class RewriterWorker(mp.Process):
    """
    Rewriter process
    """

    def __init__(
        self,
        job_q: mp.Queue,
        return_dict: Dict[str, str],
        error_dict: Dict[str, Exception],
        rewriter: Rewriter,
        counter: Counter,
        stopped: Stopped,
        finished_signal: Stopped,
    ):
        mp.Process.__init__(self)
        self.job_q = job_q
        self.return_dict = return_dict
        self.error_dict = error_dict
        self.rewriter = rewriter
        self.counter = counter
        self.stopped = stopped
        self.finished_signal = finished_signal

    def run(self) -> None:
        """Run the rewriting function"""
        while True:
            try:
                word = self.job_q.get(timeout=1)
            except queue.Empty:
                break
            self.job_q.task_done()
            if self.stopped.stop_check():
                continue
            try:
                rep = self.rewriter.rewrite(word)
                if rep != "<composition failure>":
                    self.return_dict[word] = rep
            except rewrite.Error:
                pass
            except Exception:  # noqa
                self.stopped.stop()
                self.finished_signal.stop()
                self.error_dict[word] = Exception(traceback.format_exception(*sys.exc_info()))
            self.counter.increment()
        self.finished_signal.stop()
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
    list
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
        dict[str, list[str]]
            Mapping of words to their "pronunciation"
        """
        pronunciations = {}
        for word in self.words_to_g2p:
            pronunciation = list(word)
            pronunciations[word] = pronunciation
        return pronunciations


class PyniniGenerator(G2PTopLevelMixin):
    """
    Class for generating pronunciations from a Pynini G2P model

    Parameters
    ----------
    g2p_model_path: str
        Path to G2P model

    See Also
    --------
    :class:`~montreal_forced_aligner.g2p.mixins.G2PTopLevelMixin`
        For top level G2P generation parameters

    Attributes
    ----------
    g2p_model: G2PModel
        G2P model
    """

    def __init__(self, g2p_model_path: str, **kwargs):
        self.g2p_model = G2PModel(g2p_model_path)
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
        rewriter = Rewriter(fst, input_token_type, output_token_type, self.num_pronunciations)

        num_words = len(self.words_to_g2p)
        words = list(self.words_to_g2p)
        begin = time.time()
        last_value = 0
        missing_graphemes = set()
        self.log_info("Generating pronunciations...")
        to_return = {}
        if num_words < 30 or self.num_jobs < 2:
            for word in words:
                w, m = clean_up_word(word, self.g2p_model.meta["graphemes"])
                missing_graphemes = missing_graphemes | m
                if not w:
                    continue
                try:
                    pron = rewriter.rewrite(w)
                except rewrite.Error:
                    continue
                to_return[word] = pron
        else:
            stopped = Stopped()
            job_queue = mp.JoinableQueue()
            offset = 0
            for word in words:
                w, m = clean_up_word(word, self.g2p_model.meta["graphemes"])
                missing_graphemes = missing_graphemes | m
                if not w:
                    offset += 1
                    continue
                job_queue.put(w)
            self.log_debug(
                f"Skipping {offset} words due to only containing the following graphemes: "
                f"{comma_join(sorted(missing_graphemes))}"
            )
            manager = mp.Manager()
            error_dict = manager.dict()
            return_dict = manager.dict()
            procs = []
            counter = Counter()
            finished_signals = [Stopped() for _ in range(self.num_jobs)]
            for i in range(self.num_jobs):
                p = RewriterWorker(
                    job_queue,
                    return_dict,
                    error_dict,
                    rewriter,
                    counter,
                    stopped,
                    finished_signals[i],
                )
                procs.append(p)
                p.start()
            value = 0
            if num_words - offset > 10000:
                sleep_increment = 10
            else:
                sleep_increment = 2
            with tqdm.tqdm(total=num_words - offset) as pbar:
                while value < num_words - offset:
                    time.sleep(sleep_increment)
                    if stopped.stop_check():
                        break
                    value = counter.value()
                    if value != last_value:
                        pbar.update(value - last_value)
                        last_value = value
                        for sig in finished_signals:
                            if not sig.stop_check():
                                break
                        else:
                            break
            job_queue.join()
            for p in procs:
                p.join()
            if error_dict:
                raise PyniniGenerationError(error_dict)
            for w in self.words_to_g2p:
                if w in return_dict:
                    to_return[w] = return_dict[w]
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
    def data_directory(self) -> str:
        """Data directory"""
        return self.working_directory

    @property
    def data_source_identifier(self) -> str:
        """Name of the word list file"""
        return "validation"

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
