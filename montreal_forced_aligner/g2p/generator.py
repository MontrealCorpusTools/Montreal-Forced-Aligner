"""Class for generating pronunciations from G2P models"""
from __future__ import annotations

import functools
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from typing import TYPE_CHECKING, Any, Collection, Dict, List, Optional, Set, Tuple, Union

import tqdm

from ..config import TEMP_DIR
from ..exceptions import G2PError
from ..multiprocessing import Counter, Stopped

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
    from ..models import G2PModel


__all__ = ["Rewriter", "RewriterWorker", "PyniniDictionaryGenerator"]


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
        return_dict: Dict[str, Union[str, Any]],
        rewriter: Rewriter,
        counter: Counter,
        stopped: Stopped,
    ):
        mp.Process.__init__(self)
        self.job_q = job_q
        self.return_dict = return_dict
        self.rewriter = rewriter
        self.counter = counter
        self.stopped = stopped

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
            except Exception:
                self.stopped.stop()
                self.return_dict["MFA_EXCEPTION"] = word, Exception(
                    traceback.format_exception(*sys.exc_info())
                )
            self.counter.increment()
        return


def clean_up_word(word: str, graphemes: Set[str]) -> Tuple[str, List[str]]:
    """
    Clean up word by removing graphemes not in a specified set

    Parameters
    ----------
    word : str
        Input string
    graphemes: Set[str]
        Set of allowable graphemes

    Returns
    -------
    str
        Cleaned up word
    list
        Graphemes excluded
    """
    new_word = []
    missing_graphemes = []
    for c in word:
        if c not in graphemes:
            missing_graphemes.append(c)
        else:
            new_word.append(c)
    return "".join(new_word), missing_graphemes


class PyniniDictionaryGenerator:
    """
    Class for generating pronunciations from a G2P model
    """

    def __init__(
        self,
        g2p_model: G2PModel,
        word_set: Collection[str],
        temp_directory: Optional[str] = None,
        num_jobs: int = 3,
        num_pronunciations: int = 1,
        logger: Optional[logging.Logger] = None,
    ):
        super(PyniniDictionaryGenerator, self).__init__()
        if not temp_directory:
            temp_directory = TEMP_DIR
        temp_directory = os.path.join(temp_directory, "G2P")
        self.model = g2p_model

        self.temp_directory = os.path.join(temp_directory, self.model.name)
        log_dir = os.path.join(self.temp_directory, "logging")
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "g2p.log")
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger("g2p")
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_file, "w", "utf-8")
            handler.setFormatter = logging.Formatter("%(name)s %(message)s")
            self.logger.addHandler(handler)
        self.words = word_set
        self.num_jobs = num_jobs
        self.num_pronunciations = num_pronunciations

    def generate(self) -> Dict[str, List[str]]:
        """
        Generate pronunciations

        Returns
        -------
        Dict
            Mappings of keys to their generated pronunciations
        """
        if self.model.meta["architecture"] == "phonetisaurus":
            raise G2PError(
                "Previously trained Phonetisaurus models from 1.1 and earlier are not currently supported. "
                "Please retrain your model using 2.0+"
            )

        input_token_type = "utf8"
        fst = pynini.Fst.read(self.model.fst_path)

        output_token_type = "utf8"
        if self.model.sym_path is not None and os.path.exists(self.model.sym_path):
            output_token_type = pynini.SymbolTable.read_text(self.model.sym_path)
        rewriter = Rewriter(fst, input_token_type, output_token_type, self.num_pronunciations)

        ind = 0
        num_words = len(self.words)
        words = list(self.words)
        begin = time.time()
        last_value = 0
        missing_graphemes = set()
        print("Generating pronunciations...")
        to_return = {}
        if num_words < 30 or self.num_jobs < 2:
            for word in words:
                w, m = clean_up_word(word, self.model.meta["graphemes"])
                missing_graphemes.update(m)
                if not w:
                    continue
                try:
                    pron = rewriter.rewrite(w)
                except rewrite.Error:
                    continue
                to_return[word] = pron
        else:
            stopped = Stopped()
            job_queue = mp.JoinableQueue(100)
            with tqdm.tqdm(total=num_words) as pbar:
                while True:
                    if ind == num_words:
                        break
                    try:
                        w, m = clean_up_word(words[ind], self.model.meta["graphemes"])
                        missing_graphemes.update(m)
                        if not w:
                            ind += 1
                            continue
                        job_queue.put(w, False)
                    except queue.Full:
                        break
                    ind += 1
                manager = mp.Manager()
                return_dict = manager.dict()
                procs = []
                counter = Counter()
                for _ in range(self.num_jobs):
                    p = RewriterWorker(job_queue, return_dict, rewriter, counter, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    if ind == num_words:
                        break
                    w, m = clean_up_word(words[ind], self.model.meta["graphemes"])
                    missing_graphemes.update(m)
                    if not w:
                        ind += 1
                        continue
                    job_queue.put(w)
                    value = counter.value()
                    pbar.update(value - last_value)
                    last_value = value
                    ind += 1
                job_queue.join()
            for p in procs:
                p.join()
            if "MFA_EXCEPTION" in return_dict:
                element, exc = return_dict["MFA_EXCEPTION"]
                print(element)
                raise exc
            for w in self.words:
                if w in return_dict:
                    to_return[w] = return_dict[w]
        self.logger.debug(f"Processed {num_words} in {time.time() - begin} seconds")
        return to_return

    def output(self, outfile: str) -> None:
        """
        Output pronunciations to text file

        Parameters
        ----------
        outfile: str
            Path to save
        """
        results = self.generate()
        with open(outfile, "w", encoding="utf8") as f:
            for (word, pronunciation) in results.items():
                if not pronunciation:
                    continue
                if isinstance(pronunciation, list):
                    for p in pronunciation:
                        if not p:
                            continue
                        f.write(f"{word}\t{p}\n")
                else:
                    f.write(f"{word}\t{pronunciation}\n")
