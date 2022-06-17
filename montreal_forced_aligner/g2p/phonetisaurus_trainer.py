from __future__ import annotations

import multiprocessing as mp
import os
import queue
import random
import subprocess
import time
from typing import Dict

import dataclassy
import numpy
import tqdm

from montreal_forced_aligner.abc import MetaDict, TopLevelMfaWorker
from montreal_forced_aligner.data import WordType
from montreal_forced_aligner.db import Pronunciation, Word
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionaryMixin
from montreal_forced_aligner.g2p.generator import PyniniValidator
from montreal_forced_aligner.g2p.trainer import G2PTrainer
from montreal_forced_aligner.models import G2PModel
from montreal_forced_aligner.utils import Stopped, thirdparty_binary

try:
    import pynini
    import pywrapfst
    from pynini.lib import rewrite

    G2P_DISABLED = False

except ImportError:
    pynini = None
    pywrapfst = None
    rewrite = None

    G2P_DISABLED = True


__all__ = ["PhonetisaurusTrainerMixin", "PhonetisaurusTrainer"]


@dataclassy.dataclass(slots=True)
class LabelData:
    """Label data class for penalizing alignments based on the size of their left/right side"""

    max: int
    tot: int
    lhs: int
    rhs: int
    lhsE: bool
    rhsE: bool


@dataclassy.dataclass(slots=True)
class MaximizationArguments:
    """Arguments for the maximization worker"""

    far_path: str
    alignment_model: Dict[int, pynini.Weight]
    penalize_em: bool
    penalties: Dict[int, LabelData]


@dataclassy.dataclass(slots=True)
class AlignmentInitArguments:
    """Arguments for the alignment initialization worker"""

    far_path: str
    deletions: bool
    insertions: bool
    restrict: bool
    phone_order: int
    grapheme_order: int
    s1s2_sep: str
    seq_sep: str
    skip: str


class AlignmentInitWorker(mp.Process):
    """
    Multiprocessing worker that initializes alignment FSTs for a subset of the data

    Parameters
    ----------
    job_q: mp.Queue
        Queue of grapheme-phoneme transcriptions to process
    return_queue: mp.Queue
        Queue to return data
    stopped: Stopped
        Stop check
    finished_adding: Stopped
        Check for whether the job queue is done
    symbol_dict: dict
        Symbol to integer ID mapping dictionary
    next_symbol: mp.Value
        Integer value to use for the next symbol
    lock: mp.Lock
        Lock to use for data shared across processes
    args: :class:`~montreal_forced_aligner.g2p.phonetisaurus_trainer.AlignmentInitArguments`
        Arguments for initialization
    """

    def __init__(
        self,
        job_q: mp.Queue,
        return_queue: mp.Queue,
        stopped: Stopped,
        finished_adding: Stopped,
        symbol_dict: dict,
        next_symbol: mp.Value,
        lock: mp.Lock,
        args: AlignmentInitArguments,
    ):
        mp.Process.__init__(self)
        self.job_q = job_q
        self.return_queue = return_queue
        self.stopped = stopped
        self.symbol_cache = {}
        self.symbol_dict = symbol_dict
        self.next_symbol: mp.Value = next_symbol
        self.lock = lock
        self.finished = Stopped()
        self.finished_adding = finished_adding
        self.deletions = args.deletions
        self.insertions = args.insertions
        self.restrict = args.restrict
        self.phone_order = args.phone_order
        self.grapheme_order = args.grapheme_order
        self.s1s2_sep = args.s1s2_sep
        self.seq_sep = args.seq_sep
        self.skip = args.skip
        self.far_path = args.far_path

    def look_up_symbol(self, symbol: str) -> int:
        """
        Look up a symbol based on the process's symbol cache and symbol table

        Parameters
        ----------
        symbol: str
            Symbol to look up

        Returns
        -------
        int
            Symbol ID in table
        """
        if symbol not in self.symbol_cache:
            with self.lock:
                if symbol not in self.symbol_dict:
                    self.symbol_dict[symbol] = self.next_symbol.value
                    self.next_symbol.value += 1
                self.symbol_cache[symbol] = self.symbol_dict[symbol]
        return self.symbol_cache[symbol]

    def run(self) -> None:
        """Run the function"""
        current_index = 0
        far_writer = pywrapfst.FarWriter.create(self.far_path, arc_type="log")
        while True:
            try:
                graphemes, phones = self.job_q.get(timeout=1)
            except queue.Empty:
                if self.finished_adding.stop_check():
                    break
                continue
            if self.stopped.stop_check():
                continue
            try:
                key = f"{current_index:08x}"
                fst = pynini.Fst(arc_type="log")
                initial_weight = pywrapfst.Weight(fst.weight_type(), 99)
                final_state = ((len(graphemes) + 1) * (len(phones) + 1)) - 1
                for _ in range(final_state + 1):
                    fst.add_state()
                for i in range(len(graphemes) + 1):
                    for j in range(len(phones) + 1):
                        istate = i * (len(phones) + 1) + j
                        if self.deletions:
                            for phone_range in range(1, self.phone_order + 1):
                                if j + phone_range <= len(phones):
                                    subseq_phones = phones[j : j + phone_range]
                                    symbol = self.look_up_symbol(
                                        self.s1s2_sep.join(
                                            [self.skip, self.seq_sep.join(subseq_phones)]
                                        )
                                    )
                                    ostate = i * (len(phones) + 1) + (j + phone_range)
                                    fst.add_arc(
                                        istate,
                                        pywrapfst.Arc(symbol, symbol, initial_weight, ostate),
                                    )
                        if self.insertions:
                            for k in range(1, self.grapheme_order + 1):
                                if i + k <= len(graphemes):
                                    subseq_graphemes = graphemes[i : i + k]
                                    symbol = self.look_up_symbol(
                                        self.s1s2_sep.join(
                                            [self.seq_sep.join(subseq_graphemes), self.skip]
                                        )
                                    )
                                    ostate = (i + k) * (len(phones) + 1) + j
                                    fst.add_arc(
                                        istate,
                                        pywrapfst.Arc(symbol, symbol, initial_weight, ostate),
                                    )

                        for grapheme_range in range(1, self.grapheme_order + 1):
                            for phone_range in range(1, self.phone_order + 1):
                                if i + grapheme_range <= len(graphemes) and j + phone_range <= len(
                                    phones
                                ):
                                    if self.restrict and grapheme_range > 1 and phone_range > 1:
                                        continue
                                    subseq_phones = phones[j : j + phone_range]
                                    phone_string = self.seq_sep.join(subseq_phones)
                                    subseq_graphemes = graphemes[i : i + grapheme_range]
                                    grapheme_string = self.seq_sep.join(subseq_graphemes)
                                    symbol = self.look_up_symbol(
                                        self.s1s2_sep.join([grapheme_string, phone_string])
                                    )
                                    ostate = (i + grapheme_range) * (len(phones) + 1) + (
                                        j + phone_range
                                    )
                                    weight = pywrapfst.Weight(
                                        fst.weight_type(), float(grapheme_range * phone_range)
                                    )
                                    fst.add_arc(
                                        istate, pywrapfst.Arc(symbol, symbol, weight, ostate)
                                    )
                fst.set_start(0)
                fst.set_final(final_state, pywrapfst.Weight.one(fst.weight_type()))

                if not self.insertions or not self.deletions:
                    fst = pynini.connect(fst)
                far_writer[key] = fst
                self.return_queue.put(fst)
                current_index += 1
            except Exception as e:  # noqa
                self.stopped.stop()
                self.return_queue.put(e)
                raise
        self.finished.stop()
        return


class MaximizationWorker(mp.Process):
    """
    Multiprocessing worker that runs the maximization step of training for a subset of the data

    Parameters
    ----------
    return_queue: mp.Queue
        Queue to return data
    stopped: Stopped
        Stop check
    args: :class:`~montreal_forced_aligner.g2p.phonetisaurus_trainer.MaximizationArguments`
        Arguments for maximization
    """

    def __init__(self, return_queue: mp.Queue, stopped: Stopped, args: MaximizationArguments):
        mp.Process.__init__(self)
        self.return_queue = return_queue
        self.stopped = stopped
        self.finished = Stopped()
        self.penalize_em = args.penalize_em
        self.alignment_model = args.alignment_model
        self.penalties = args.penalties
        self.far_path = args.far_path

    def run(self) -> None:
        """Run the function"""
        zero = pynini.Weight.zero("log")
        far_reader = pywrapfst.FarReader.open(self.far_path)
        far_writer = pywrapfst.FarWriter.create(self.far_path + ".temp", arc_type="log")
        while not far_reader.done():
            if self.stopped.stop_check():
                break
            key = far_reader.get_key()
            fst = far_reader.get_fst()
            for state_id in fst.states():
                maiter = fst.mutable_arcs(state_id)
                while not maiter.done():
                    arc = maiter.value()
                    if not self.penalize_em:
                        arc.weight = self.alignment_model[arc.ilabel]
                    else:
                        label_data = self.penalties[arc.ilabel]
                        if label_data.lhs > 1 and label_data.rhs > 1:
                            arc.weight = pynini.Weight(fst.weight_type(), 99)
                        elif not label_data.lhsE and not label_data.rhsE:
                            arc.weight = pynini.Weight(
                                fst.weight_type(), float(arc.weight) * label_data.tot
                            )
                    if arc.weight == zero:
                        arc.weight = pynini.Weight(fst.weight_type(), 99)
                    arc = pywrapfst.Arc(arc.ilabel, arc.olabel, arc.weight, arc.nextstate)
                    maiter.set_value(arc)
                    next(maiter)
            far_writer[key] = fst
            next(far_reader)
            self.return_queue.put(1)
        os.remove(self.far_path)
        os.rename(self.far_path + ".temp", self.far_path)
        self.finished.stop()


class ExpectationWorker(mp.Process):
    """
    Multiprocessing worker that runs the expectation step of training for a subset of the data

    Parameters
    ----------
    far_path: str
        Path to FST archive file
    return_queue: mp.Queue
        Queue to return data
    stopped: Stopped
        Stop check
    """

    def __init__(
        self,
        far_path: str,
        return_queue: mp.Queue,
        stopped: Stopped,
    ):
        mp.Process.__init__(self)
        self.far_path = far_path
        self.return_queue = return_queue
        self.stopped = stopped
        self.finished = Stopped()

    def run(self) -> None:
        """Run the function"""
        far_reader = pywrapfst.FarReader.open(self.far_path)
        while not far_reader.done():
            if self.stopped.stop_check():
                break
            key = far_reader.get_key()
            fst = far_reader.get_fst()
            data = {}
            try:
                fst = pynini.Fst.read_from_string(fst.write_to_string())
                alpha = pynini.shortestdistance(fst)
                beta = pynini.shortestdistance(fst, reverse=True)
                for state_id in fst.states():
                    for arc in fst.arcs(state_id):
                        gamma = pynini.divide(
                            pynini.times(
                                pynini.times(alpha[state_id], arc.weight), beta[arc.nextstate]
                            ),
                            beta[0],
                        )
                        if float(gamma) != numpy.inf:
                            if arc.ilabel not in data:
                                data[arc.ilabel] = 0
                            data[arc.ilabel] += float(gamma)
                self.return_queue.put((key, data))
                next(far_reader)
            except Exception as e:  # noqa
                self.stopped.stop()
                self.return_queue.put(e)
                raise
        self.finished.stop()
        return


class PhonetisaurusTrainerMixin:
    """
    Mixin class for training Phonetisaurus style models

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
    grapheme_order: int
        Maximum number of graphemes to map to single phones
    phone_order: int
        Maximum number of phones to map to single graphemes
    fst_default_cache_gc: str
        String to pass to OpenFst binaries for GC behavior
    fst_default_cache_gc_limit: str
        String to pass to OpenFst binaries for GC behavior
    """

    def __init__(
        self,
        order: int = 8,
        num_iterations: int = 10,
        smoothing_method: str = "kneser_ney",
        pruning_method: str = "relative_entropy",
        model_size: int = 1000000,
        insertions: bool = True,
        deletions: bool = True,
        grapheme_order: int = 2,
        phone_order: int = 2,
        fst_default_cache_gc="",
        fst_default_cache_gc_limit="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not hasattr(self, "_data_source"):
            self._data_source = None
        self.order = order
        self.num_iterations = num_iterations
        self.smoothing_method = smoothing_method
        self.pruning_method = pruning_method
        self.model_size = model_size
        self.insertions = insertions
        self.deletions = deletions
        self.fst_default_cache_gc = fst_default_cache_gc
        self.fst_default_cache_gc_limit = fst_default_cache_gc_limit
        self.grapheme_order = grapheme_order
        self.phone_order = phone_order
        self.seq_sep = "|"
        self.s1s2_sep = "}"
        self.skip = "_"
        self.eps = "<eps>"
        self.restrict = True
        self.penalize_em = True
        self.penalize = True
        self.g2p_num_training_pronunciations = 0

    @property
    def architecture(self) -> str:
        """Phonetisaurus"""
        return "phonetisaurus"

    def initialize_alignments(self) -> None:
        """
        Initialize alignment FSTs for training

        """
        self.symbol_table = pynini.SymbolTable()
        self.symbol_table.add_symbol(self.eps)
        self.symbol_table.add_symbol(self.skip)
        self.symbol_table.add_symbol(f"{self.seq_sep}_{self.seq_sep}")
        self.symbol_table.add_symbol(self.s1s2_sep)
        model_params = [
            "true" if self.deletions else "false",
            "true" if self.insertions else "false",
            str(self.grapheme_order),
            str(self.phone_order),
        ]
        self.symbol_table.add_symbol("_".join(model_params))
        self.alignment_model: Dict[int, pynini.Weight] = {}
        self.prev_alignment_model: Dict[int, pynini.Weight] = {}
        self.penalties: Dict[int, LabelData] = {}
        self.total = pynini.Weight.zero("log")
        self.prev_total = pynini.Weight.zero("log")

        self.fsas = []
        self.log_info("Creating alignment FSTs...")
        with mp.Manager() as manager:
            mp_symbol_dict = manager.dict()
            lock = mp.Lock()
            next_symbol = mp.Value("i", self.symbol_table.num_symbols())
            for i in range(self.symbol_table.num_symbols()):
                sym = self.symbol_table.find(i)
                mp_symbol_dict[sym] = i
            job_queue = mp.Queue()
            return_queue = mp.Queue()
            stopped = Stopped()
            finished_adding = Stopped()
            procs = []
            for i in range(self.num_jobs):
                args = AlignmentInitArguments(
                    os.path.join(self.working_directory, f"{i}.far"),
                    self.deletions,
                    self.insertions,
                    self.restrict,
                    self.phone_order,
                    self.grapheme_order,
                    self.s1s2_sep,
                    self.seq_sep,
                    self.skip,
                )
                procs.append(
                    AlignmentInitWorker(
                        job_queue,
                        return_queue,
                        stopped,
                        finished_adding,
                        mp_symbol_dict,
                        next_symbol,
                        lock,
                        args,
                    )
                )
                procs[i].start()
            self.g2p_num_training_pronunciations = 0
            for word, pronunciations in self.g2p_training_dictionary.items():
                graphemes = list(word)
                for p in pronunciations:
                    phones = p.split()
                    job_queue.put((graphemes, phones))
                    self.g2p_num_training_pronunciations += 1
            finished_adding.stop()
            error_list = []
            with tqdm.tqdm(
                total=self.g2p_num_training_pronunciations, disable=getattr(self, "quiet", False)
            ) as pbar:
                while True:
                    try:
                        fst = return_queue.get(timeout=1)
                        if isinstance(fst, Exception):
                            error_list.append(fst)
                            continue
                        if stopped.stop_check():
                            continue
                    except queue.Empty:
                        for p in procs:
                            if not p.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    for state_id in fst.states():
                        for arc in fst.arcs(state_id):
                            if arc.ilabel not in self.prev_alignment_model:
                                self.prev_alignment_model[arc.ilabel] = arc.weight
                                sym = mp_symbol_dict[arc.ilabel]
                                d = sym.find("}")
                                c = sym.find("|")
                                left_side_count = 1
                                right_side_count = 1
                                if c != -1:
                                    if c < d:
                                        left_side_count += 1
                                    else:
                                        right_side_count += 1
                                max_count = max(left_side_count, right_side_count)
                                self.penalties[arc.ilabel] = LabelData(
                                    tot=left_side_count + right_side_count,
                                    max=max_count,
                                    lhs=left_side_count,
                                    rhs=right_side_count,
                                    lhsE=False,
                                    rhsE=False,
                                )
                            else:
                                self.prev_alignment_model[arc.ilabel] = pynini.plus(
                                    self.prev_alignment_model[arc.ilabel], arc.weight
                                )
                            self.total = pynini.plus(self.total, arc.weight)
                    pbar.update(1)
            for p in procs:
                p.join()
            for k in sorted(mp_symbol_dict.keys(), key=lambda x: mp_symbol_dict[x]):
                if self.symbol_table.find(k) == pynini.NO_SYMBOL:
                    self.symbol_table.add_symbol(k)
        if error_list:
            for v in error_list:
                raise v

    def maximization(self) -> float:
        """
        Run the maximization step for training

        Returns
        -------
        float
            Current iteration's score
        """
        self.log_info("Performing maximization step...")
        cond = False
        change = abs(float(self.total) - float(self.prev_total))
        zero = pynini.Weight.zero("log")
        if not cond:
            self.prev_total = self.total
            for ilabel, weight in self.prev_alignment_model.items():
                self.alignment_model[ilabel] = pynini.divide(weight, self.total)
                self.prev_alignment_model[ilabel] = zero
        return_queue = mp.Queue()
        stopped = Stopped()
        procs = []
        for i in range(self.num_jobs):
            args = MaximizationArguments(
                os.path.join(self.working_directory, f"{i}.far"),
                self.alignment_model,
                self.penalize_em,
                self.penalties,
            )
            procs.append(MaximizationWorker(return_queue, stopped, args))
            procs[i].start()

        error_list = []
        with tqdm.tqdm(
            total=self.g2p_num_training_pronunciations, disable=getattr(self, "quiet", False)
        ) as pbar:
            while True:
                try:
                    result = return_queue.get(timeout=1)
                    if isinstance(result, Exception):
                        error_list.append(result)
                        continue
                    if stopped.stop_check():
                        continue
                except queue.Empty:
                    for p in procs:
                        if not p.finished.stop_check():
                            break
                    else:
                        break
                    continue
                pbar.update(1)
        for p in procs:
            p.join()

        if error_list:
            for v in error_list:
                raise v
        self.total = zero
        self.log_info(f"Maximization done! Change from last iteration was {change:.3f}")
        return change

    def expectation(self) -> None:
        """
        Run the expectation step for training
        """
        self.log_info("Performing expectation step...")
        return_queue = mp.Queue()
        stopped = Stopped()
        error_list = []
        procs = []
        for i in range(self.num_jobs):
            procs.append(
                ExpectationWorker(
                    os.path.join(self.working_directory, f"{i}.far"), return_queue, stopped
                )
            )
            procs[i].start()

        with tqdm.tqdm(
            total=self.g2p_num_training_pronunciations, disable=getattr(self, "quiet", False)
        ) as pbar:
            while True:
                try:
                    result = return_queue.get(timeout=1)
                    if isinstance(result, Exception):
                        error_list.append(result)
                        continue
                    if stopped.stop_check():
                        continue
                    index, data = result
                except queue.Empty:
                    for p in procs:
                        if not p.finished.stop_check():
                            break
                    else:
                        break
                    continue
                for ilabel, gamma in data.items():
                    gamma = pynini.Weight("log", gamma)
                    self.prev_alignment_model[ilabel] = pynini.plus(
                        self.prev_alignment_model[ilabel], gamma
                    )
                    self.total = pynini.plus(self.total, gamma)
                pbar.update(1)
        for p in procs:
            p.join()

        if error_list:
            for v in error_list:
                raise v
        self.log_info("Expectation done!")

    def train_ngram_model(self) -> None:
        """
        Train an ngram model on the aligned FSTs
        """
        if os.path.exists(self.fst_path):
            self.log_info("Ngram model already exists.")
            return
        self.log_info("Training ngram model...")
        with open(
            os.path.join(self.working_log_directory, "model.log"), "w", encoding="utf8"
        ) as logf:
            ngramcount_proc = subprocess.Popen(
                [
                    thirdparty_binary("ngramcount"),
                    "--require_symbols=false",
                    "--round_to_int",
                    f"--order={self.order}",
                    self.far_path,
                ],
                stderr=logf,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            ngramcount_proc.communicate()
            ngrammake_proc = subprocess.Popen(
                [
                    thirdparty_binary("ngrammake"),
                    f"--method={self.smoothing_method}",
                ],
                stdin=ngramcount_proc.stdout,
                stderr=logf,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            ngrammake_proc.communicate()

            ngramshrink_proc = subprocess.Popen(
                [
                    thirdparty_binary("ngramshrink"),
                    f"--method={self.pruning_method}",
                    f"--target_number_of_ngrams={self.model_size}",
                    "-",
                    self.ngram_path,
                ],
                stdin=ngrammake_proc.stdout,
                stderr=logf,
                env=os.environ,
            )
            ngramshrink_proc.communicate()

            ngram_fst = pynini.Fst.read(self.ngram_path)
            grapheme_symbols = pynini.SymbolTable()
            grapheme_symbols.add_symbol(self.eps)
            grapheme_symbols.add_symbol(self.seq_sep)
            grapheme_symbols.add_symbol(self.skip)
            phone_symbols = pynini.SymbolTable()
            phone_symbols.add_symbol(self.eps)
            phone_symbols.add_symbol(self.seq_sep)
            phone_symbols.add_symbol(self.skip)
            single_phone_symbols = pynini.SymbolTable()
            single_phone_symbols.add_symbol(self.eps)
            single_phone_fst = pynini.Fst()
            start_state = single_phone_fst.add_state()
            single_phone_fst.set_start(start_state)
            one = pynini.Weight.one(single_phone_fst.weight_type())
            single_phone_fst.set_final(start_state, one)
            current_ind = 1
            for state in ngram_fst.states():
                maiter = ngram_fst.mutable_arcs(state)
                while not maiter.done():
                    arc = maiter.value()
                    symbol = self.symbol_table.find(arc.ilabel)
                    try:
                        grapheme, phone = symbol.split(self.s1s2_sep)
                        g_symbol = grapheme_symbols.find(grapheme)
                        if g_symbol == pynini.NO_SYMBOL:
                            g_symbol = grapheme_symbols.add_symbol(grapheme)
                        p_symbol = phone_symbols.find(phone)
                        if p_symbol == pynini.NO_SYMBOL:
                            p_symbol = phone_symbols.add_symbol(phone)
                            singles = phone.split(self.seq_sep)
                            for i, s in enumerate(singles):
                                s_symbol = single_phone_symbols.find(s)

                                if s_symbol == pynini.NO_SYMBOL:
                                    s_symbol = single_phone_symbols.add_symbol(s)
                                if i == 0:
                                    single_start = start_state
                                else:
                                    single_start = current_ind
                                if i < len(singles) - 1:
                                    current_ind = single_phone_fst.add_state()
                                    end_state = current_ind
                                else:
                                    end_state = start_state
                                single_phone_fst.add_arc(
                                    single_start,
                                    pywrapfst.Arc(
                                        p_symbol if i == 0 else 0, s_symbol, one, end_state
                                    ),
                                )

                        arc = pywrapfst.Arc(g_symbol, p_symbol, arc.weight, arc.nextstate)
                        maiter.set_value(arc)
                    except ValueError:
                        if symbol == "<eps>":
                            arc = pywrapfst.Arc(0, 0, arc.weight, arc.nextstate)
                            maiter.set_value(arc)
                        else:
                            raise
                        pass
                    next(maiter)
            for i in range(grapheme_symbols.num_symbols()):
                sym = grapheme_symbols.find(i)
                if sym in {self.eps, self.seq_sep, self.skip}:
                    continue
                parts = sym.split(self.seq_sep)
                if len(parts) > 1:
                    for s in parts:
                        if grapheme_symbols.find(s) == pynini.NO_SYMBOL:
                            k = grapheme_symbols.add_symbol(s)
                            ngram_fst.add_arc(1, pywrapfst.Arc(k, 2, 99, 1))

            for i in range(phone_symbols.num_symbols()):
                sym = phone_symbols.find(i)
                if sym in {self.eps, self.seq_sep, self.skip}:
                    continue
                parts = sym.split(self.seq_sep)
                if len(parts) > 1:
                    for s in parts:
                        if phone_symbols.find(s) == pynini.NO_SYMBOL:
                            k = phone_symbols.add_symbol(s)
                            ngram_fst.add_arc(1, pywrapfst.Arc(2, k, 99, 1))
            single_phone_fst.set_input_symbols(phone_symbols)
            single_phone_fst.set_output_symbols(single_phone_symbols)
            ngram_fst.set_input_symbols(grapheme_symbols)
            ngram_fst.set_output_symbols(phone_symbols)
            single_ngram_fst = pynini.compose(ngram_fst, single_phone_fst)
            single_ngram_fst.set_input_symbols(grapheme_symbols)
            single_ngram_fst.set_output_symbols(single_phone_symbols)
            grapheme_symbols.write_text(self.grapheme_symbols_path)
            single_phone_symbols.write_text(self.phone_symbols_path)
            single_ngram_fst.write(self.fst_path)

    def train_alignments(self) -> None:
        """
        Run an Expectation-Maximization (EM) training on alignment FSTs to generate well-aligned FSTs for ngram modeling
        """
        if os.path.exists(self.far_path):
            self.log_info("Using existing alignments.")
            self.symbol_table = pynini.SymbolTable.read_text(self.alignment_symbols_path)
            return
        self.initialize_alignments()

        self.maximization()
        self.log_info("Training alignments...")
        for i in range(self.num_iterations):
            self.log_info(f"Iteration {i}")
            self.expectation()
            change = self.maximization()
            if change < 1e-10:
                break
        self.export_alignments()

    @property
    def data_directory(self) -> str:
        """Data directory for trainer"""
        return self.working_directory

    def train_iteration(self) -> None:
        """Train iteration, not used"""
        pass

    @property
    def workflow_identifier(self) -> str:
        """Identifier for Phonetisaurus G2P trainer"""
        return "phonetisaurus_train_g2p"

    @property
    def data_source_identifier(self) -> str:
        """Dictionary name"""
        return self._data_source

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
        self.log_info(f"Saved model to {output_model_path}")

    @property
    def alignment_model_path(self) -> str:
        """Path to store alignment model FST"""
        return os.path.join(self.working_directory, "align.fst")

    @property
    def ngram_path(self) -> str:
        """Path to store ngram model"""
        return os.path.join(self.working_directory, "ngram.fst")

    @property
    def fst_path(self) -> str:
        """Path to store final trained model"""
        return os.path.join(self.working_directory, "model.fst")

    @property
    def alignment_symbols_path(self) -> str:
        """Path to alignment symbol table"""
        return os.path.join(self.working_directory, "alignment.syms")

    @property
    def grapheme_symbols_path(self) -> str:
        """Path to final model's grapheme symbol table"""
        return os.path.join(self.working_directory, "graphemes.txt")

    @property
    def phone_symbols_path(self) -> str:
        """Path to final model's phone symbol table"""
        return os.path.join(self.working_directory, "phones.txt")

    @property
    def far_path(self) -> str:
        """Path to store final aligned FSTs"""
        return os.path.join(self.working_directory, "aligned.far")

    def export_alignments(self) -> None:
        """
        Combine alignment training archives to a final combined FST archive to train the ngram model
        """
        self.log_info("Exporting final alignments...")
        model = pynini.Fst(arc_type="log")
        model.add_state()
        model.set_start(0)
        model.set_final(0, pynini.Weight.one(model.arc_type()))
        for ilabel, weight in self.alignment_model.items():
            model.add_arc(0, pywrapfst.Arc(ilabel, ilabel, weight, 0))
        model.set_input_symbols(self.symbol_table)
        model.write(self.alignment_model_path)
        set_symbols = False
        self.symbol_table.write_text(self.alignment_symbols_path)
        far_writer = pywrapfst.FarWriter.create(self.far_path)
        zero = pynini.Weight.zero("log")
        one = pynini.Weight.one("log")
        index = 0
        with tqdm.tqdm(
            total=self.g2p_num_training_pronunciations, disable=getattr(self, "quiet", False)
        ) as pbar:
            for i in range(self.num_jobs):
                far_reader = pywrapfst.FarReader.open(
                    os.path.join(self.working_directory, f"{i}.far")
                )
                while not far_reader.done():
                    fst = far_reader.get_fst()
                    tfst = pynini.arcmap(
                        pynini.Fst.read_from_string(fst.write_to_string()), map_type="to_std"
                    )
                    if self.penalize:
                        for state in tfst.states():
                            maiter = tfst.mutable_arcs(state)
                            while not maiter.done():
                                arc = maiter.value()
                                ld = self.penalties[arc.ilabel]
                                if ld.lhs > 1 and ld.rhs > 1:
                                    arc.weight = pynini.Weight(tfst.weight_type(), 999)
                                else:
                                    arc.weight = pynini.Weight(
                                        tfst.weight_type(), float(arc.weight) * ld.max
                                    )
                                maiter.set_value(arc)
                                next(maiter)
                    tfst = rewrite.lattice_to_dfa(tfst, True, 4).project("output").rmepsilon()
                    lfst = pynini.arcmap(tfst, map_type="to_log")
                    pfst = pynini.push(lfst, reweight_type="to_final", push_weights=True)
                    for state in pfst.states():
                        if pfst.final(state) != zero:
                            pfst.set_final(state, one)
                    lattice = pynini.arcmap(pfst, map_type="to_std")

                    if not set_symbols:
                        lattice.set_input_symbols(self.symbol_table)
                        lattice.set_output_symbols(self.symbol_table)
                        set_symbols = True
                    key = f"{index:08x}"
                    far_writer[key] = lattice
                    pbar.update(1)
                    index += 1
                    next(far_reader)


class PhonetisaurusTrainer(
    MultispeakerDictionaryMixin, PhonetisaurusTrainerMixin, G2PTrainer, TopLevelMfaWorker
):
    """
    Top level trainer class for Phonetisaurus-style models
    """

    def __init__(
        self,
        **kwargs,
    ):
        self._data_source = os.path.splitext(os.path.basename(kwargs["dictionary_path"]))[0]
        super().__init__(**kwargs)
        self.ler = None
        self.wer = None

    @property
    def data_directory(self) -> str:
        """Data directory for trainer"""
        return self.working_directory

    @property
    def workflow_identifier(self) -> str:
        """Identifier for Phonetisaurus G2P trainer"""
        return "phonetisaurus_train_g2p"

    @property
    def configuration(self) -> MetaDict:
        """Configuration for G2P trainer"""
        config = super().configuration
        config.update({"dictionary_path": self.dictionary_model.path})
        return config

    def setup(self) -> None:
        """Setup for G2P training"""
        if self.initialized:
            return
        self.initialize_database()
        self.dictionary_setup()
        os.makedirs(self.phones_dir, exist_ok=True)
        self.initialize_training()
        self.initialized = True

    def train(self) -> None:
        """
        Train a G2P model
        """
        os.makedirs(self.working_log_directory, exist_ok=True)
        begin = time.time()
        self.train_alignments()
        self.log_debug(
            f"Aligning {len(self.g2p_training_dictionary)} words took {time.time() - begin} seconds"
        )
        begin = time.time()
        self.train_ngram_model()
        self.log_debug(
            f"Generating model for {len(self.g2p_training_dictionary)} words took {time.time() - begin} seconds"
        )
        self.finalize_training()

    def finalize_training(self) -> None:
        """Finalize training and run evaluation if specified"""
        if self.evaluation_mode:
            self.evaluate_g2p_model()

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
            "grapheme_order": self.grapheme_order,
            "phone_order": self.phone_order,
            "seq_sep": self.seq_sep,
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

    def evaluate_g2p_model(self) -> None:
        """
        Validate the G2P model against held out data
        """
        temp_model_path = os.path.join(self.working_log_directory, "g2p_model.zip")
        self.export_model(temp_model_path)
        temp_dir = os.path.join(self.working_directory, "validation")
        gen = PyniniValidator(
            g2p_model_path=temp_model_path,
            word_list=list(self.g2p_validation_dictionary.keys()),
            temporary_directory=temp_dir,
            num_jobs=self.num_jobs,
            num_pronunciations=self.num_pronunciations,
        )
        output = gen.generate_pronunciations()
        with open(os.path.join(temp_dir, "validation_output.txt"), "w", encoding="utf8") as f:
            for (orthography, pronunciations) in output.items():
                if not pronunciations:
                    continue
                for p in pronunciations:
                    if not p:
                        continue
                    f.write(f"{orthography}\t{p}\n")
        self.compute_validation_errors(output)

    def initialize_training(self) -> None:
        """Initialize training G2P model"""
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
                validation_words = set(random.sample(words, validation_items))
                self.g2p_training_dictionary = {
                    k: v for k, v in word_dict.items() if k not in validation_words
                }
                self.g2p_validation_dictionary = {
                    k: v for k, v in word_dict.items() if k in validation_words
                }
                if self.debug:
                    with open(
                        os.path.join(self.working_directory, "validation_set.txt"),
                        "w",
                        encoding="utf8",
                    ) as f:
                        for word in self.g2p_validation_dictionary:
                            f.write(word + "\n")
            grapheme_count = 0
            phone_count = 0
            self.character_sets = set()
            for word, pronunciations in self.g2p_training_dictionary.items():
                # if re.match(r"\W", word) is not None:
                #    continue
                word = list(word)
                grapheme_count += len(word)
                self.g2p_training_graphemes.update(word)
                for p in pronunciations:
                    self.g2p_training_phones.update(p.split())
                    phone_count += len(p.split())
            self.log_debug(f"Graphemes in training data: {sorted(self.g2p_training_graphemes)}")
            self.log_debug(f"Phones in training data: {sorted(self.g2p_training_phones)}")
            self.log_debug(f"Averages phones per grapheme: {phone_count / grapheme_count}")

            if self.evaluation_mode:
                for word, pronunciations in self.g2p_validation_dictionary.items():
                    self.g2p_validation_graphemes.update(word)
                    for p in pronunciations:
                        self.g2p_validation_phones.update(p.split())
                self.log_debug(
                    f"Graphemes in validation data: {sorted(self.g2p_validation_graphemes)}"
                )
                self.log_debug(f"Phones in validation data: {sorted(self.g2p_validation_phones)}")
                grapheme_diff = sorted(self.g2p_validation_graphemes - self.g2p_training_graphemes)
                phone_diff = sorted(self.g2p_validation_phones - self.g2p_training_phones)
                if grapheme_diff:
                    self.log_warning(
                        f"The following graphemes appear only in the validation set: {', '.join(grapheme_diff)}"
                    )
                if phone_diff:
                    self.log_warning(
                        f"The following phones appear only in the validation set: {', '.join(phone_diff)}"
                    )
