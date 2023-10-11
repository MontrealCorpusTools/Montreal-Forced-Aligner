from __future__ import annotations

import collections
import logging
import multiprocessing as mp
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from queue import Queue

import dataclassy
import numpy
import pynini
import pywrapfst
import sqlalchemy
from pynini.lib import rewrite
from tqdm.rich import tqdm

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import MetaDict, TopLevelMfaWorker
from montreal_forced_aligner.data import WordType, WorkflowType
from montreal_forced_aligner.db import (
    Job,
    M2M2Job,
    M2MSymbol,
    Pronunciation,
    Word,
    Word2Job,
    bulk_update,
)
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionaryMixin
from montreal_forced_aligner.exceptions import PhonetisaurusSymbolError
from montreal_forced_aligner.g2p.generator import PyniniValidator
from montreal_forced_aligner.g2p.trainer import G2PTrainer
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.models import G2PModel
from montreal_forced_aligner.utils import thirdparty_binary

__all__ = ["PhonetisaurusTrainerMixin", "PhonetisaurusTrainer"]

logger = logging.getLogger("mfa")


@dataclassy.dataclass(slots=True)
class MaximizationArguments:
    """Arguments for the MaximizationWorker"""

    db_string: str
    far_path: Path
    penalize_em: bool
    batch_size: int


@dataclassy.dataclass(slots=True)
class ExpectationArguments:
    """Arguments for the ExpectationWorker"""

    db_string: str
    far_path: Path
    batch_size: int


@dataclassy.dataclass(slots=True)
class AlignmentExportArguments:
    """Arguments for the AlignmentExportWorker"""

    log_path: Path
    far_path: Path
    penalize: bool


@dataclassy.dataclass(slots=True)
class NgramCountArguments:
    """Arguments for the NgramCountWorker"""

    log_path: Path
    far_path: Path
    alignment_symbols_path: Path
    order: int


@dataclassy.dataclass(slots=True)
class AlignmentInitArguments:
    """Arguments for the alignment initialization worker"""

    db_string: str
    log_path: Path
    far_path: Path
    deletions: bool
    insertions: bool
    restrict: bool
    output_order: int
    input_order: int
    eps: str
    s1s2_sep: str
    seq_sep: str
    skip: str
    batch_size: int


class AlignmentInitWorker(mp.Process):
    """
    Multiprocessing worker that initializes alignment FSTs for a subset of the data

    Parameters
    ----------
    job_name: int
        Integer ID for the job
    return_queue: :class:`multiprocessing.Queue`
        Queue to return data
    stopped: :class:`~mp.Event`
        Stop check
    finished_adding: :class:`~mp.Event`
        Check for whether the job queue is done
    args: :class:`~montreal_forced_aligner.g2p.phonetisaurus_trainer.AlignmentInitArguments`
        Arguments for initialization
    """

    def __init__(
        self,
        job_name: int,
        return_queue: mp.Queue,
        stopped: mp.Event,
        finished_adding: mp.Event,
        args: AlignmentInitArguments,
    ):
        super().__init__()
        self.job_name = job_name
        self.return_queue = return_queue
        self.stopped = stopped
        self.finished = mp.Event()
        self.finished_adding = finished_adding
        self.deletions = args.deletions
        self.insertions = args.insertions
        self.restrict = args.restrict
        self.output_order = args.output_order
        self.input_order = args.input_order
        self.eps = args.eps
        self.s1s2_sep = args.s1s2_sep
        self.seq_sep = args.seq_sep
        self.skip = args.skip
        self.far_path = args.far_path
        self.sym_path = self.far_path.with_suffix(".syms")
        self.log_path = args.log_path
        self.db_string = args.db_string
        self.batch_size = args.batch_size

    def data_generator(self, session):
        query = (
            session.query(Word.word, Pronunciation.pronunciation)
            .join(Pronunciation.word)
            .join(Word.job)
            .filter(Word2Job.training == True)  # noqa
            .filter(Word2Job.job_id == self.job_name)
        )
        for w, p in query:
            w = list(w)
            p = p.split()
            yield w, p

    def run(self) -> None:
        """Run the function"""

        engine = sqlalchemy.create_engine(
            self.db_string,
            poolclass=sqlalchemy.NullPool,
            pool_reset_on_return=None,
            isolation_level="AUTOCOMMIT",
            logging_name=f"{type(self).__name__}_engine",
        ).execution_options(logging_token=f"{type(self).__name__}_engine")
        try:
            symbol_table = pywrapfst.SymbolTable()
            symbol_table.add_symbol(self.eps)
            valid_output_ngrams = set()
            base_dir = os.path.dirname(self.far_path)
            with mfa_open(os.path.join(base_dir, "output_ngram.ngrams"), "r") as f:
                for line in f:
                    line = line.strip()
                    valid_output_ngrams.add(line)
            valid_input_ngrams = set()
            with mfa_open(os.path.join(base_dir, "input_ngram.ngrams"), "r") as f:
                for line in f:
                    line = line.strip()
                    valid_input_ngrams.add(line)
            count = 0
            data = {}
            with mfa_open(self.log_path, "w") as log_file, sqlalchemy.orm.Session(
                engine
            ) as session:
                far_writer = pywrapfst.FarWriter.create(self.far_path, arc_type="log")
                for current_index, (input, output) in enumerate(self.data_generator(session)):
                    if self.stopped.is_set():
                        continue
                    try:
                        key = f"{current_index:08x}"
                        fst = pynini.Fst(arc_type="log")
                        final_state = ((len(input) + 1) * (len(output) + 1)) - 1

                        for _ in range(final_state + 1):
                            fst.add_state()
                        for i in range(len(input) + 1):
                            for j in range(len(output) + 1):
                                istate = i * (len(output) + 1) + j
                                if self.deletions:
                                    for output_range in range(1, self.output_order + 1):
                                        if j + output_range <= len(output):
                                            subseq_output = output[j : j + output_range]
                                            output_string = self.seq_sep.join(subseq_output)
                                            if (
                                                output_range > 1
                                                and output_string not in valid_output_ngrams
                                            ):
                                                continue
                                            symbol = self.s1s2_sep.join([self.skip, output_string])
                                            ilabel = symbol_table.find(symbol)
                                            if ilabel == pywrapfst.NO_LABEL:
                                                ilabel = symbol_table.add_symbol(symbol)
                                            ostate = i * (len(output) + 1) + (j + output_range)
                                            fst.add_arc(
                                                istate,
                                                pywrapfst.Arc(
                                                    ilabel,
                                                    ilabel,
                                                    pywrapfst.Weight("log", 99.0),
                                                    ostate,
                                                ),
                                            )
                                if self.insertions:
                                    for input_range in range(1, self.input_order + 1):
                                        if i + input_range <= len(input):
                                            subseq_input = input[i : i + input_range]
                                            input_string = self.seq_sep.join(subseq_input)
                                            if (
                                                input_range > 1
                                                and input_string not in valid_input_ngrams
                                            ):
                                                continue
                                            symbol = self.s1s2_sep.join([input_string, self.skip])
                                            ilabel = symbol_table.find(symbol)
                                            if ilabel == pywrapfst.NO_LABEL:
                                                ilabel = symbol_table.add_symbol(symbol)
                                            ostate = (i + input_range) * (len(output) + 1) + j
                                            fst.add_arc(
                                                istate,
                                                pywrapfst.Arc(
                                                    ilabel,
                                                    ilabel,
                                                    pywrapfst.Weight("log", 99.0),
                                                    ostate,
                                                ),
                                            )

                                for input_range in range(1, self.input_order + 1):
                                    for output_range in range(1, self.output_order + 1):
                                        if i + input_range <= len(
                                            input
                                        ) and j + output_range <= len(output):
                                            if (
                                                self.restrict
                                                and input_range > 1
                                                and output_range > 1
                                            ):
                                                continue
                                            subseq_output = output[j : j + output_range]
                                            output_string = self.seq_sep.join(subseq_output)
                                            if (
                                                output_range > 1
                                                and output_string not in valid_output_ngrams
                                            ):
                                                continue
                                            subseq_input = input[i : i + input_range]
                                            input_string = self.seq_sep.join(subseq_input)
                                            if (
                                                input_range > 1
                                                and input_string not in valid_input_ngrams
                                            ):
                                                continue
                                            symbol = self.s1s2_sep.join(
                                                [input_string, output_string]
                                            )
                                            ilabel = symbol_table.find(symbol)
                                            if ilabel == pywrapfst.NO_LABEL:
                                                ilabel = symbol_table.add_symbol(symbol)
                                            ostate = (i + input_range) * (len(output) + 1) + (
                                                j + output_range
                                            )
                                            fst.add_arc(
                                                istate,
                                                pywrapfst.Arc(
                                                    ilabel,
                                                    ilabel,
                                                    pywrapfst.Weight(
                                                        "log", float(input_range * output_range)
                                                    ),
                                                    ostate,
                                                ),
                                            )
                        fst.set_start(0)
                        fst.set_final(final_state, pywrapfst.Weight.one(fst.weight_type()))
                        fst = pynini.connect(fst)
                        for state in fst.states():
                            for arc in fst.arcs(state):
                                sym = symbol_table.find(arc.ilabel)
                                if sym not in data:
                                    data[sym] = arc.weight
                                else:
                                    data[sym] = pywrapfst.plus(data[sym], arc.weight)
                        if count >= self.batch_size:
                            data = {k: float(v) for k, v in data.items()}
                            self.return_queue.put((self.job_name, data, count))
                            data = {}
                            count = 0
                        log_file.flush()
                        far_writer[key] = fst
                        del fst
                        count += 1
                    except Exception as e:  # noqa
                        self.stopped.set()
                        self.return_queue.put(e)
            if data:
                data = {k: float(v) for k, v in data.items()}
                self.return_queue.put((self.job_name, data, count))
            symbol_table.write_text(self.far_path.with_suffix(".syms"))
            return
        except Exception as e:
            self.stopped.set()
            self.return_queue.put(e)
        finally:
            self.finished.set()
            del far_writer


class ExpectationWorker(mp.Process):
    """
    Multiprocessing worker that runs the expectation step of training for a subset of the data

    Parameters
    ----------
    job_name: int
        Integer ID for the job
    return_queue: :class:`multiprocessing.Queue`
        Queue to return data
    stopped: :class:`~mp.Event`
        Stop check
    args: :class:`~montreal_forced_aligner.g2p.phonetisaurus_trainer.ExpectationArguments`
        Arguments for the function
    """

    def __init__(
        self,
        job_name: int,
        return_queue: mp.Queue,
        stopped: mp.Event,
        args: ExpectationArguments,
    ):
        super().__init__()
        self.job_name = job_name
        self.db_string = args.db_string
        self.far_path = args.far_path
        self.batch_size = args.batch_size
        self.return_queue = return_queue
        self.stopped = stopped
        self.finished = mp.Event()

    def run(self) -> None:
        """Run the function"""
        engine = sqlalchemy.create_engine(
            self.db_string,
            poolclass=sqlalchemy.NullPool,
            pool_reset_on_return=None,
            isolation_level="AUTOCOMMIT",
            logging_name=f"{type(self).__name__}_engine",
        ).execution_options(logging_token=f"{type(self).__name__}_engine")
        far_reader = pywrapfst.FarReader.open(self.far_path)
        symbol_table = pywrapfst.SymbolTable.read_text(self.far_path.with_suffix(".syms"))
        symbol_mapper = {}
        data = {}
        count = 0
        with sqlalchemy.orm.Session(engine) as session:
            query = (
                session.query(M2MSymbol.symbol, M2MSymbol.id)
                .join(M2MSymbol.jobs)
                .filter(M2M2Job.job_id == self.job_name)
            )
            for symbol, sym_id in query:
                symbol_mapper[symbol_table.find(symbol)] = sym_id
        while not far_reader.done():
            if self.stopped.is_set():
                break
            fst = far_reader.get_fst()
            zero = pywrapfst.Weight.zero("log")
            try:
                fst = pynini.Fst.read_from_string(fst.write_to_string())
                alpha = pynini.shortestdistance(fst)
                beta = pynini.shortestdistance(fst, reverse=True)
                for state_id in fst.states():
                    for arc in fst.arcs(state_id):
                        gamma = pywrapfst.divide(
                            pywrapfst.times(
                                pywrapfst.times(alpha[state_id], arc.weight), beta[arc.nextstate]
                            ),
                            beta[0],
                        )
                        if float(gamma) != numpy.inf:
                            sym_id = symbol_mapper[arc.ilabel]
                            if sym_id not in data:
                                data[sym_id] = zero
                            data[sym_id] = pywrapfst.plus(data[sym_id], gamma)
                if count >= self.batch_size:
                    data = {k: float(v) for k, v in data.items()}
                    self.return_queue.put((data, count))
                    data = {}
                    count = 0
                next(far_reader)
                del alpha
                del beta
                del fst
                count += 1
            except Exception as e:  # noqa
                self.stopped.set()
                self.return_queue.put(e)
                raise
        if data:
            data = {k: float(v) for k, v in data.items()}
            self.return_queue.put((data, count))
        self.finished.set()
        del far_reader
        return


class MaximizationWorker(mp.Process):
    """
    Multiprocessing worker that runs the maximization step of training for a subset of the data

    Parameters
    ----------
    job_name: int
        Integer ID for the job
    return_queue: :class:`multiprocessing.Queue`
        Queue to return data
    stopped: :class:`~multiprocessing.Event`
        Stop check
    args: :class:`~montreal_forced_aligner.g2p.phonetisaurus_trainer.MaximizationArguments`
        Arguments for maximization
    """

    def __init__(
        self,
        job_name: int,
        return_queue: mp.Queue,
        stopped: mp.Event,
        args: MaximizationArguments,
    ):
        super().__init__()
        self.job_name = job_name
        self.return_queue = return_queue
        self.stopped = stopped
        self.finished = mp.Event()
        self.db_string = args.db_string
        self.penalize_em = args.penalize_em
        self.far_path = args.far_path
        self.batch_size = args.batch_size

    def run(self) -> None:
        """Run the function"""
        symbol_table = pywrapfst.SymbolTable.read_text(self.far_path.with_suffix(".syms"))
        count = 0
        engine = sqlalchemy.create_engine(
            self.db_string,
            poolclass=sqlalchemy.NullPool,
            pool_reset_on_return=None,
            isolation_level="AUTOCOMMIT",
            logging_name=f"{type(self).__name__}_engine",
        ).execution_options(logging_token=f"{type(self).__name__}_engine")
        try:
            alignment_model = {}
            with sqlalchemy.orm.Session(engine) as session:
                query = (
                    session.query(M2MSymbol)
                    .join(M2MSymbol.jobs)
                    .filter(M2M2Job.job_id == self.job_name)
                )
                for m2m in query:
                    weight = pywrapfst.Weight("log", m2m.weight)
                    if self.penalize_em:
                        if m2m.grapheme_order > 1 or m2m.phone_order > 1:
                            weight = pywrapfst.Weight("log", float(weight) * m2m.total_order)
                        if weight == pywrapfst.Weight.zero("log") or float(weight) == numpy.inf:
                            weight = pywrapfst.Weight("log", 99)
                    alignment_model[symbol_table.find(m2m.symbol)] = weight
            far_reader = pywrapfst.FarReader.open(self.far_path)
            far_writer = pywrapfst.FarWriter.create(
                self.far_path.with_suffix(self.far_path.suffix + ".temp"), arc_type="log"
            )
            while not far_reader.done():
                if self.stopped.is_set():
                    break
                key = far_reader.get_key()
                fst = far_reader.get_fst()
                for state_id in fst.states():
                    maiter = fst.mutable_arcs(state_id)
                    while not maiter.done():
                        arc = maiter.value()
                        arc.weight = alignment_model[arc.ilabel]
                        arc = pywrapfst.Arc(arc.ilabel, arc.olabel, arc.weight, arc.nextstate)
                        maiter.set_value(arc)
                        next(maiter)
                    del maiter
                far_writer[key] = fst
                next(far_reader)
                if count >= self.batch_size:
                    self.return_queue.put(count)
                    count = 0
                del fst
                count += 1
            del far_reader
            del far_writer
            os.remove(self.far_path)
            os.rename(self.far_path.with_suffix(self.far_path.suffix + ".temp"), self.far_path)
        except Exception as e:
            self.stopped.set()
            self.return_queue.put(e)
            raise
        finally:
            if count >= 1:
                self.return_queue.put(count)
            self.finished.set()


class AlignmentExporter(mp.Process):
    """
    Multiprocessing worker to generate Ngram counts for aligned FST archives

    Parameters
    ----------
    return_queue: :class:`multiprocessing.Queue`
        Queue to return data
    stopped: :class:`~multiprocessing.Event`
        Stop check
    args: :class:`~montreal_forced_aligner.g2p.phonetisaurus_trainer.AlignmentExportArguments`
        Arguments for maximization
    """

    def __init__(self, return_queue: mp.Queue, stopped: mp.Event, args: AlignmentExportArguments):
        super().__init__()
        self.return_queue = return_queue
        self.stopped = stopped
        self.finished = mp.Event()
        self.penalize = args.penalize
        self.far_path = args.far_path
        self.log_path = args.log_path

    def run(self) -> None:
        """Run the function"""
        symbol_table = pywrapfst.SymbolTable.read_text(self.far_path.with_suffix(".syms"))
        with mfa_open(self.log_path, "w") as log_file:
            far_reader = pywrapfst.FarReader.open(self.far_path)
            one_best_path = self.far_path.with_suffix(".strings")
            no_alignment_count = 0
            total = 0
            with mfa_open(one_best_path, "w") as f:
                while not far_reader.done():
                    fst = far_reader.get_fst()
                    total += 1
                    if fst.num_states() == 0:
                        next(far_reader)
                        no_alignment_count += 1
                        self.return_queue.put(1)
                        continue
                    tfst = pynini.arcmap(
                        pynini.Fst.read_from_string(fst.write_to_string()), map_type="to_std"
                    )
                    if self.penalize:
                        for state in tfst.states():
                            maiter = tfst.mutable_arcs(state)
                            while not maiter.done():
                                arc = maiter.value()
                                sym = symbol_table.find(arc.ilabel)
                                ld = self.penalties[sym]
                                if ld.lhs > 1 and ld.rhs > 1:
                                    arc.weight = pywrapfst.Weight(tfst.weight_type(), 999)
                                else:
                                    arc.weight = pywrapfst.Weight(
                                        tfst.weight_type(), float(arc.weight) * ld.max
                                    )
                                maiter.set_value(arc)
                                next(maiter)
                            del maiter
                    pfst = rewrite.lattice_to_dfa(tfst, True, 8).project("output").rmepsilon()

                    if pfst.start() != pywrapfst.NO_SYMBOL:
                        path = pynini.shortestpath(pfst)
                    else:
                        pfst = rewrite.lattice_to_dfa(tfst, False, 8).project("output").rmepsilon()
                        path = pynini.shortestpath(pfst)
                    string = path.string(symbol_table)
                    f.write(f"{string}\n")
                    log_file.flush()
                    next(far_reader)
                    self.return_queue.put(1)
                    del fst
                    del pfst
                    del path
                    del tfst
            log_file.write(
                f"Done {total - no_alignment_count}, no alignment for {no_alignment_count}"
            )
            log_file.flush()
            self.finished.set()
            del far_reader


class NgramCountWorker(threading.Thread):
    """
    Multiprocessing worker to generate Ngram counts for aligned FST archives

    Parameters
    ----------
    return_queue: :class:`multiprocessing.Queue`
        Queue to return data
    stopped: :class:`~threading.Event`
        Stop check
    args: :class:`~montreal_forced_aligner.g2p.phonetisaurus_trainer.NgramCountArguments`
        Arguments for maximization
    """

    def __init__(self, return_queue: Queue, stopped: threading.Event, args: NgramCountArguments):
        super().__init__()
        self.return_queue = return_queue
        self.stopped = stopped
        self.finished = threading.Event()
        self.order = args.order
        self.far_path = args.far_path
        self.log_path = args.log_path
        self.alignment_symbols_path = args.alignment_symbols_path

    def run(self) -> None:
        """Run the function"""
        with mfa_open(self.log_path, "w") as log_file:
            one_best_path = self.far_path.with_suffix(".strings")
            ngram_count_path = self.far_path.with_suffix(".cnts")
            farcompile_proc = subprocess.Popen(
                [
                    thirdparty_binary("farcompilestrings"),
                    "--token_type=symbol",
                    f"--symbols={self.alignment_symbols_path}",
                    one_best_path,
                ],
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            ngramcount_proc = subprocess.Popen(
                [
                    thirdparty_binary("ngramcount"),
                    "--require_symbols=false",
                    "--round_to_int",
                    f"--order={self.order}",
                    "-",
                    ngram_count_path,
                ],
                stderr=log_file,
                stdin=farcompile_proc.stdout,
                env=os.environ,
            )
            ngramcount_proc.communicate()
            self.finished.set()


class PhonetisaurusTrainerMixin:
    """
    Mixin class for training Phonetisaurus style models

    Parameters
    ----------
    order: int
        Order of the ngram model, defaults to 8
    batch_size:int
        Batch size for training, defaults to 1000
    num_iterations:int
        Maximum number of iterations to use in Baum-Welch training, defaults to 10
    smoothing_method:str
        Smoothing method for the ngram model, defaults to "kneser_ney"
    pruning_method:str
        Pruning method for pruning the ngram model, defaults to "relative_entropy"
    model_size: int
        Target number of ngrams for pruning, defaults to 1000000
    initial_prune_threshold: float
        Pruning threshold for calculating the multiple phone/grapheme strings that are to be allowed, defaults to 0.0001
    insertions: bool
        Flag for whether to allow for insertions, default True
    deletions: bool
        Flag for whether to allow for deletions, default True
    restrict_m2m: bool
        Flag for whether to restrict possible alignments to one-to-many and disable many-to-many alignments, default False
    penalize_em: bool
        Flag for whether to many-to-many and one-to-many are penalized over one-to-one mappings during training, default False
    penalize: bool
        Flag for whether to many-to-many and one-to-many are penalized over one-to-one mappings during export, default False
    sequence_separator: str
        Character to use for concatenating and aligning multiple phones or graphemes, defaults to "|"
    skip: str
        Character to use to represent deletions or insertions, defaults to "_"
    alignment_separator: str
        Character to use for concatenating grapheme strings and phone strings, defaults to ";"
    grapheme_order: int
        Maximum number of graphemes to map to single phones
    phone_order: int
        Maximum number of phones to map to single graphemes
    em_threshold: float
        Threshold of minimum change for early stopping of EM training
    """

    alignment_init_function = AlignmentInitWorker

    def __init__(
        self,
        order: int = 8,
        batch_size: int = 1000,
        num_iterations: int = 10,
        smoothing_method: str = "kneser_ney",
        pruning_method: str = "relative_entropy",
        model_size: int = 1000000,
        initial_prune_threshold: float = 0.0001,
        insertions: bool = True,
        deletions: bool = True,
        restrict_m2m: bool = False,
        penalize_em: bool = False,
        penalize: bool = False,
        sequence_separator: str = "|",
        skip: str = "_",
        alignment_separator: str = ";",
        grapheme_order: int = 2,
        phone_order: int = 2,
        em_threshold: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not hasattr(self, "_data_source"):
            self._data_source = None
        self.order = order
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.smoothing_method = smoothing_method
        self.pruning_method = pruning_method
        self.model_size = model_size
        self.initial_prune_threshold = initial_prune_threshold
        self.insertions = insertions
        self.deletions = deletions
        self.grapheme_order = grapheme_order
        self.phone_order = phone_order
        self.input_order = self.grapheme_order
        self.output_order = self.phone_order
        self.sequence_separator = sequence_separator
        self.alignment_separator = alignment_separator
        self.skip = skip
        self.eps = "<eps>"
        self.restrict_m2m = restrict_m2m
        self.penalize_em = penalize_em
        self.penalize = penalize
        self.em_threshold = em_threshold
        self.g2p_num_training_pronunciations = 0

        self.symbol_table = pywrapfst.SymbolTable()
        self.symbol_table.add_symbol(self.eps)
        self.total = pywrapfst.Weight.zero("log")
        self.prev_total = pywrapfst.Weight.zero("log")

    @property
    def architecture(self) -> str:
        """Phonetisaurus"""
        return "phonetisaurus"

    def initialize_alignments(self) -> None:
        """
        Initialize alignment FSTs for training

        """

        logger.info("Creating alignment FSTs...")

        return_queue = mp.Queue()
        stopped = mp.Event()
        finished_adding = mp.Event()
        procs = []
        for i in range(1, config.NUM_JOBS + 1):
            args = AlignmentInitArguments(
                self.db_string,
                self.working_log_directory.joinpath(f"alignment_init.{i}.log"),
                self.working_directory.joinpath(f"{i}.far"),
                self.deletions,
                self.insertions,
                self.restrict_m2m,
                self.phone_order,
                self.grapheme_order,
                self.eps,
                self.alignment_separator,
                self.sequence_separator,
                self.skip,
                self.batch_size,
            )
            procs.append(
                self.alignment_init_function(
                    i,
                    return_queue,
                    stopped,
                    finished_adding,
                    args,
                )
            )
            procs[-1].start()

        finished_adding.set()
        error_list = []
        symbols = {}
        job_symbols = {}
        symbol_id = 1
        with tqdm(
            total=self.g2p_num_training_pronunciations, disable=config.QUIET
        ) as pbar, self.session() as session:
            while True:
                try:
                    result = return_queue.get(timeout=2)
                    if isinstance(result, Exception):
                        error_list.append(result)
                        continue
                    if stopped.is_set():
                        continue
                except queue.Empty:
                    for p in procs:
                        if not p.finished.is_set():
                            break
                    else:
                        break
                    continue
                job_name, weights, count = result
                for symbol, weight in weights.items():
                    weight = pynini.Weight("log", weight)
                    if symbol not in symbols:
                        left_side, right_side = symbol.split(self.alignment_separator)
                        if left_side == self.skip:
                            left_side_order = 0
                        else:
                            left_side_order = 1 + left_side.count(self.sequence_separator)
                        if right_side == self.skip:
                            right_side_order = 0
                        else:
                            right_side_order = 1 + right_side.count(self.sequence_separator)
                        max_order = max(left_side_order, right_side_order)
                        total_order = left_side_order + right_side_order
                        symbols[symbol] = {
                            "symbol": symbol,
                            "id": symbol_id,
                            "total_order": total_order,
                            "max_order": max_order,
                            "grapheme_order": left_side_order,
                            "phone_order": right_side_order,
                            "weight": weight,
                        }
                        symbol_id += 1
                    else:
                        symbols[symbol]["weight"] = pywrapfst.plus(
                            symbols[symbol]["weight"], weight
                        )
                    self.total = pywrapfst.plus(self.total, weight)
                    if job_name not in job_symbols:
                        job_symbols[job_name] = set()
                    job_symbols[job_name].add(symbols[symbol]["id"])
                pbar.update(count)
            for p in procs:
                p.join()
            if error_list:
                for v in error_list:
                    raise v
            logger.debug(f"Total of {len(symbols)} symbols, initial total: {self.total}")
            symbols = [x for x in symbols.values()]
            for data in symbols:
                data["weight"] = float(data["weight"])
            session.bulk_insert_mappings(
                M2MSymbol, symbols, return_defaults=False, render_nulls=True
            )
            session.flush()
            del symbols
            mappings = []
            for j, sym_ids in job_symbols.items():
                mappings.extend({"m2m_id": x, "job_id": j} for x in sym_ids)
            session.bulk_insert_mappings(
                M2M2Job, mappings, return_defaults=False, render_nulls=True
            )

            session.commit()

    def maximization(self, last_iteration=False) -> float:
        """
        Run the maximization step for training

        Returns
        -------
        float
            Current iteration's score
        """
        logger.info("Performing maximization step...")
        change = abs(float(self.total) - float(self.prev_total))
        logger.debug(f"Previous total: {float(self.prev_total)}")
        logger.debug(f"Current total: {float(self.total)}")
        logger.debug(f"Change: {change}")

        self.prev_total = self.total
        with self.session() as session:
            session.query(M2MSymbol).update(
                {"weight": M2MSymbol.weight - float(self.total)}, synchronize_session=False
            )
            session.commit()
        return_queue = mp.Queue()
        stopped = mp.Event()
        procs = []
        for i in range(1, config.NUM_JOBS + 1):
            args = MaximizationArguments(
                self.db_string,
                self.working_directory.joinpath(f"{i}.far"),
                self.penalize_em,
                self.batch_size,
            )
            procs.append(MaximizationWorker(i, return_queue, stopped, args))
            procs[-1].start()

        error_list = []
        with tqdm(total=self.g2p_num_training_pronunciations, disable=config.QUIET) as pbar:
            while True:
                try:
                    result = return_queue.get(timeout=1)
                    if isinstance(result, Exception):
                        error_list.append(result)
                        continue
                    if stopped.is_set():
                        continue
                except queue.Empty:
                    for p in procs:
                        if not p.finished.is_set():
                            break
                    else:
                        break
                    continue

                pbar.update(result)
        for p in procs:
            p.join()

        if error_list:
            for v in error_list:
                raise v
        if not last_iteration and change >= self.em_threshold:  # we're still converging
            self.total = pywrapfst.Weight.zero("log")
            with self.session() as session:
                session.query(M2MSymbol).update({"weight": 0.0})
                session.commit()
        logger.info(f"Maximization done! Change from last iteration was {change:.3f}")
        return change

    def expectation(self) -> None:
        """
        Run the expectation step for training
        """
        logger.info("Performing expectation step...")
        return_queue = mp.Queue()
        stopped = mp.Event()
        error_list = []
        procs = []
        for i in range(1, config.NUM_JOBS + 1):
            args = ExpectationArguments(
                self.db_string,
                self.working_directory.joinpath(f"{i}.far"),
                self.batch_size,
            )
            procs.append(ExpectationWorker(i, return_queue, stopped, args))
            procs[-1].start()
        mappings = {}
        zero = pywrapfst.Weight.zero("log")
        with tqdm(total=self.g2p_num_training_pronunciations, disable=config.QUIET) as pbar:
            while True:
                try:
                    result = return_queue.get(timeout=1)
                    if isinstance(result, Exception):
                        error_list.append(result)
                        continue
                    if stopped.is_set():
                        continue
                except queue.Empty:
                    for p in procs:
                        if not p.finished.is_set():
                            break
                    else:
                        break
                    continue
                result, count = result
                for sym_id, gamma in result.items():
                    gamma = pywrapfst.Weight("log", gamma)
                    if sym_id not in mappings:
                        mappings[sym_id] = zero
                    mappings[sym_id] = pywrapfst.plus(mappings[sym_id], gamma)
                    self.total = pywrapfst.plus(self.total, gamma)
                pbar.update(count)
        for p in procs:
            p.join()

        if error_list:
            for v in error_list:
                raise v
        with self.session() as session:
            bulk_update(
                session, M2MSymbol, [{"id": k, "weight": float(v)} for k, v in mappings.items()]
            )
            session.commit()
        logger.info("Expectation done!")

    def train_ngram_model(self) -> None:
        """
        Train an ngram model on the aligned FSTs
        """
        if os.path.exists(self.fst_path):
            logger.info("Ngram model already exists.")
            return
        logger.info("Generating ngram counts...")
        return_queue = Queue()
        stopped = threading.Event()
        error_list = []
        procs = []
        count_paths = []
        for i in range(1, config.NUM_JOBS + 1):
            args = NgramCountArguments(
                self.working_log_directory.joinpath(f"ngram_count.{i}.log"),
                self.working_directory.joinpath(f"{i}.far"),
                self.alignment_symbols_path,
                self.order,
            )
            procs.append(NgramCountWorker(return_queue, stopped, args))
            count_paths.append(args.far_path.with_suffix(".cnts"))
            procs[-1].start()

        with tqdm(total=self.g2p_num_training_pronunciations, disable=config.QUIET) as pbar:
            while True:
                try:
                    result = return_queue.get(timeout=1)
                    if isinstance(result, Exception):
                        error_list.append(result)
                        continue
                    if stopped.is_set():
                        continue
                    return_queue.task_done()
                except queue.Empty:
                    for p in procs:
                        if not p.finished.is_set():
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
        logger.info("Done counting ngrams!")

        logger.info("Training ngram model...")
        with mfa_open(self.working_log_directory.joinpath("model.log"), "w") as logf:
            if len(count_paths) > 1:
                ngrammerge_proc = subprocess.Popen(
                    [
                        thirdparty_binary("ngrammerge"),
                        f'--ofile={self.ngram_path.with_suffix(".cnts")}',
                        *count_paths,
                    ],
                    stderr=logf,
                    # stdout=subprocess.PIPE,
                    env=os.environ,
                )
                ngrammerge_proc.communicate()
            else:
                os.rename(count_paths[0], self.ngram_path.with_suffix(".cnts"))
            ngrammake_proc = subprocess.Popen(
                [
                    thirdparty_binary("ngrammake"),
                    f"--method={self.smoothing_method}",
                    self.ngram_path.with_suffix(".cnts"),
                ],
                stderr=logf,
                stdout=subprocess.PIPE,
                env=os.environ,
            )

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
            grapheme_symbols = pywrapfst.SymbolTable()
            grapheme_symbols.add_symbol(self.eps)
            grapheme_symbols.add_symbol(self.sequence_separator)
            grapheme_symbols.add_symbol(self.skip)
            phone_symbols = pywrapfst.SymbolTable()
            phone_symbols.add_symbol(self.eps)
            phone_symbols.add_symbol(self.sequence_separator)
            phone_symbols.add_symbol(self.skip)
            single_phone_symbols = pywrapfst.SymbolTable()
            single_phone_symbols.add_symbol(self.eps)
            single_phone_fst = pynini.Fst()
            start_state = single_phone_fst.add_state()
            single_phone_fst.set_start(start_state)
            one = pywrapfst.Weight.one(single_phone_fst.weight_type())
            single_phone_fst.set_final(start_state, one)
            current_ind = 1
            for state in ngram_fst.states():
                maiter = ngram_fst.mutable_arcs(state)
                while not maiter.done():
                    arc = maiter.value()
                    symbol = self.symbol_table.find(arc.ilabel)
                    try:
                        grapheme, phone = symbol.split(self.alignment_separator)
                        if grapheme == self.skip:
                            g_symbol = grapheme_symbols.find(self.eps)
                        else:
                            g_symbol = grapheme_symbols.find(grapheme)
                            if g_symbol == pywrapfst.NO_SYMBOL:
                                g_symbol = grapheme_symbols.add_symbol(grapheme)
                        if phone == self.skip:
                            p_symbol = phone_symbols.find(self.eps)
                        else:
                            p_symbol = phone_symbols.find(phone)
                            if p_symbol == pywrapfst.NO_SYMBOL:
                                p_symbol = phone_symbols.add_symbol(phone)
                                singles = phone.split(self.sequence_separator)
                                for i, s in enumerate(singles):
                                    s_symbol = single_phone_symbols.find(s)

                                    if s_symbol == pywrapfst.NO_SYMBOL:
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
                        if symbol in {"<eps>", "<unk>", "<epsilon>"}:
                            arc = pywrapfst.Arc(0, 0, arc.weight, arc.nextstate)
                            maiter.set_value(arc)
                        else:
                            raise
                    finally:
                        next(maiter)
            for i in range(grapheme_symbols.num_symbols()):
                sym = grapheme_symbols.find(i)
                if sym in {self.eps, self.sequence_separator, self.skip}:
                    continue
                parts = sym.split(self.sequence_separator)
                if len(parts) > 1:
                    for s in parts:
                        if grapheme_symbols.find(s) == pywrapfst.NO_SYMBOL:
                            k = grapheme_symbols.add_symbol(s)
                            ngram_fst.add_arc(1, pywrapfst.Arc(k, 2, 99, 1))

            for i in range(phone_symbols.num_symbols()):
                sym = phone_symbols.find(i)
                if sym in {self.eps, self.sequence_separator, self.skip}:
                    continue
                parts = sym.split(self.sequence_separator)
                if len(parts) > 1:
                    for s in parts:
                        if phone_symbols.find(s) == pywrapfst.NO_SYMBOL:
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
        if os.path.exists(self.alignment_model_path):
            logger.info("Using existing alignments.")
            self.symbol_table = pywrapfst.SymbolTable.read_text(self.alignment_symbols_path)
            return
        self.initialize_alignments()

        self.maximization()
        logger.info("Training alignments...")
        for i in range(self.num_iterations):
            logger.info(f"Iteration {i}")
            self.expectation()
            change = self.maximization(last_iteration=i == self.num_iterations - 1)
            if change < self.em_threshold:
                break

    @property
    def data_directory(self) -> Path:
        """Data directory for trainer"""
        return self.working_directory

    def train_iteration(self) -> None:
        """Train iteration, not used"""
        pass

    @property
    def data_source_identifier(self) -> str:
        """Dictionary name"""
        return self._data_source

    def export_model(self, output_model_path: Path) -> None:
        """
        Export G2P model to specified path

        Parameters
        ----------
        output_model_path: :class:`~pathlib.Path`
            Path to export model
        """
        directory = output_model_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        models_temp_dir = self.working_directory.joinpath("model_archive_temp")
        model = G2PModel.empty(output_model_path.stem, root_directory=models_temp_dir)
        model.add_meta_file(self)
        model.add_fst_model(self.working_directory)
        model.add_sym_path(self.working_directory)
        if directory:
            os.makedirs(directory, exist_ok=True)
        model.dump(output_model_path)
        model.clean_up()
        logger.info(f"Saved model to {output_model_path}")

    @property
    def alignment_model_path(self) -> Path:
        """Path to store alignment model FST"""
        return self.working_directory.joinpath("align.fst")

    @property
    def ngram_path(self) -> Path:
        """Path to store ngram model"""
        return self.working_directory.joinpath("ngram.fst")

    @property
    def fst_path(self) -> Path:
        """Path to store final trained model"""
        return self.working_directory.joinpath("model.fst")

    @property
    def alignment_symbols_path(self) -> Path:
        """Path to alignment symbol table"""
        return self.working_directory.joinpath("alignment.syms")

    @property
    def grapheme_symbols_path(self) -> Path:
        """Path to final model's grapheme symbol table"""
        return self.working_directory.joinpath("graphemes.txt")

    @property
    def phone_symbols_path(self) -> Path:
        """Path to final model's phone symbol table"""
        return self.working_directory.joinpath("phones.txt")

    @property
    def far_path(self) -> Path:
        """Path to store final aligned FSTs"""
        return self.working_directory.joinpath("aligned.far")

    def export_alignments(self) -> None:
        """
        Combine alignment training archives to a final combined FST archive to train the ngram model
        """
        logger.info("Exporting final alignments...")

        return_queue = mp.Queue()
        stopped = mp.Event()
        error_list = []
        procs = []
        count_paths = []
        for i in range(1, config.NUM_JOBS + 1):
            args = AlignmentExportArguments(
                self.working_log_directory.joinpath(f"ngram_count.{i}.log"),
                self.working_directory.joinpath(f"{i}.far"),
                self.penalize,
            )
            procs.append(AlignmentExporter(return_queue, stopped, args))
            count_paths.append(args.far_path.with_suffix(".cnts"))
            procs[-1].start()

        with tqdm(total=self.g2p_num_training_pronunciations, disable=config.QUIET) as pbar:
            while True:
                try:
                    result = return_queue.get(timeout=1)
                    if isinstance(result, Exception):
                        error_list.append(result)
                        continue
                    if stopped.is_set():
                        continue
                except queue.Empty:
                    for p in procs:
                        if not p.finished.is_set():
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

        symbols_proc = subprocess.Popen(
            [
                thirdparty_binary("ngramsymbols"),
                "--OOV_symbol=<unk>",
                "--epsilon_symbol=<eps>",
                "-",
                self.alignment_symbols_path,
            ],
            encoding="utf8",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        for j in range(1, config.NUM_JOBS + 1):
            text_path = self.working_directory.joinpath(f"{j}.strings")
            with mfa_open(text_path, "r") as f:
                for line in f:
                    symbols_proc.stdin.write(line)
                    symbols_proc.stdin.flush()
        symbols_proc.stdin.close()
        symbols_proc.wait()
        self.symbol_table = pynini.SymbolTable.read_text(self.alignment_symbols_path)
        logger.info("Done exporting alignments!")

    def compute_initial_ngrams(self) -> None:
        input_path = self.working_directory.joinpath("input.txt")
        input_ngram_path = self.working_directory.joinpath("input_ngram.fst")
        if input_ngram_path.with_suffix(".ngrams").exists():
            logger.info("Initial ngrams already computed")
            return
        logger.info("Computing initial ngrams...")
        input_symbols_path = self.working_directory.joinpath("input_ngram.syms")
        symbols_proc = subprocess.Popen(
            [
                thirdparty_binary("ngramsymbols"),
                "--OOV_symbol=<unk>",
                "--epsilon_symbol=<eps>",
                input_path,
                input_symbols_path,
            ],
            encoding="utf8",
        )
        symbols_proc.communicate()
        farcompile_proc = subprocess.Popen(
            [
                thirdparty_binary("farcompilestrings"),
                "--token_type=symbol",
                f"--symbols={input_symbols_path}",
                input_path,
            ],
            stdout=subprocess.PIPE,
            env=os.environ,
        )
        ngramcount_proc = subprocess.Popen(
            [
                thirdparty_binary("ngramcount"),
                "--require_symbols=false",
                "--round_to_int",
                f"--order={self.input_order}",
            ],
            stdin=farcompile_proc.stdout,
            stdout=subprocess.PIPE,
            env=os.environ,
        )
        ngrammake_proc = subprocess.Popen(
            [
                thirdparty_binary("ngrammake"),
                f"--method={self.smoothing_method}",
            ],
            stdin=ngramcount_proc.stdout,
            stdout=subprocess.PIPE,
            env=os.environ,
        )

        ngramshrink_proc = subprocess.Popen(
            [
                thirdparty_binary("ngramshrink"),
                f"--method={self.pruning_method}",
                f"--theta={self.initial_prune_threshold}",
            ],
            stdin=ngrammake_proc.stdout,
            stdout=subprocess.PIPE,
            env=os.environ,
        )
        print_proc = subprocess.Popen(
            [
                thirdparty_binary("ngramprint"),
                f"--symbols={input_symbols_path}",
            ],
            env=os.environ,
            stdin=ngramshrink_proc.stdout,
            stdout=subprocess.PIPE,
            encoding="utf8",
        )
        ngrams = set()
        for line in print_proc.stdout:
            line = line.strip().split()[:-1]
            ngram = self.sequence_separator.join(x for x in line if x not in {"<s>", "</s>"})
            if self.sequence_separator not in ngram:
                continue
            ngrams.add(ngram)

        print_proc.wait()
        with mfa_open(input_ngram_path.with_suffix(".ngrams"), "w") as f:
            for ngram in sorted(ngrams):
                f.write(f"{ngram}\n")

        output_path = self.working_directory.joinpath("output.txt")
        output_ngram_path = self.working_directory.joinpath("output_ngram.fst")
        output_symbols_path = self.working_directory.joinpath("output_ngram.syms")
        symbols_proc = subprocess.Popen(
            [
                thirdparty_binary("ngramsymbols"),
                "--OOV_symbol=<unk>",
                "--epsilon_symbol=<eps>",
                output_path,
                output_symbols_path,
            ],
            encoding="utf8",
        )
        symbols_proc.communicate()
        farcompile_proc = subprocess.Popen(
            [
                thirdparty_binary("farcompilestrings"),
                "--token_type=symbol",
                f"--symbols={output_symbols_path}",
                output_path,
            ],
            stdout=subprocess.PIPE,
            env=os.environ,
        )
        ngramcount_proc = subprocess.Popen(
            [
                thirdparty_binary("ngramcount"),
                "--require_symbols=false",
                "--round_to_int",
                f"--order={self.output_order}",
            ],
            stdin=farcompile_proc.stdout,
            stdout=subprocess.PIPE,
            env=os.environ,
        )
        ngrammake_proc = subprocess.Popen(
            [
                thirdparty_binary("ngrammake"),
                f"--method={self.smoothing_method}",
            ],
            stdin=ngramcount_proc.stdout,
            stdout=subprocess.PIPE,
            env=os.environ,
        )

        ngramshrink_proc = subprocess.Popen(
            [
                thirdparty_binary("ngramshrink"),
                f"--method={self.pruning_method}",
                f"--theta={self.initial_prune_threshold}",
            ],
            stdin=ngrammake_proc.stdout,
            stdout=subprocess.PIPE,
            env=os.environ,
        )
        print_proc = subprocess.Popen(
            [thirdparty_binary("ngramprint"), f"--symbols={output_symbols_path}"],
            env=os.environ,
            stdin=ngramshrink_proc.stdout,
            stdout=subprocess.PIPE,
            encoding="utf8",
        )
        ngrams = set()
        for line in print_proc.stdout:
            line = line.strip().split()[:-1]
            ngram = self.sequence_separator.join(x for x in line if x not in {"<s>", "</s>"})
            if self.sequence_separator not in ngram:
                continue
            ngrams.add(ngram)

        print_proc.wait()
        with mfa_open(output_ngram_path.with_suffix(".ngrams"), "w") as f:
            for ngram in sorted(ngrams):
                f.write(f"{ngram}\n")

    def train(self) -> None:
        """
        Train a G2P model
        """
        if os.path.exists(self.fst_path):
            return
        os.makedirs(self.working_log_directory, exist_ok=True)
        begin = time.time()
        self.train_alignments()
        logger.debug(
            f"Aligning {len(self.g2p_training_dictionary)} words took {time.time() - begin:.3f} seconds"
        )
        self.export_alignments()
        begin = time.time()
        self.train_ngram_model()
        logger.debug(
            f"Generating model for {len(self.g2p_training_dictionary)} words took {time.time() - begin:.3f} seconds"
        )
        self.finalize_training()


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
        self._data_source = kwargs["dictionary_path"].stem
        super().__init__(**kwargs)
        self.ler = None
        self.wer = None

    @property
    def data_directory(self) -> Path:
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
        self.initialize_training()
        self.initialized = True

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
            "sequence_separator": self.sequence_separator,
            "evaluation": {},
            "training": {
                "num_words": self.g2p_num_training_words,
                "num_graphemes": len(self.g2p_training_graphemes),
                "num_phones": len(self.non_silence_phones),
            },
        }

        if self.evaluation_mode:
            m["evaluation"]["num_words"] = self.g2p_num_validation_words
            m["evaluation"]["word_error_rate"] = self.wer
            m["evaluation"]["phone_error_rate"] = self.ler
        return m

    def evaluate_g2p_model(self) -> None:
        """
        Validate the G2P model against held out data
        """
        temp_model_path = self.working_log_directory.joinpath("g2p_model.zip")
        self.export_model(temp_model_path)
        temp_dir = self.working_directory.joinpath("validation")
        os.makedirs(temp_dir, exist_ok=True)
        with self.session() as session:
            validation_set = collections.defaultdict(set)
            query = (
                session.query(Word.word, Pronunciation.pronunciation)
                .join(Pronunciation.word)
                .join(Word.job)
                .filter(Word2Job.training == False)  # noqa
            )
            for w, pron in query:
                validation_set[w].add(pron)
        gen = PyniniValidator(
            g2p_model_path=temp_model_path,
            word_list=list(validation_set.keys()),
            num_pronunciations=self.num_pronunciations,
        )
        output = gen.generate_pronunciations()
        with mfa_open(temp_dir.joinpath("validation_output.txt"), "w") as f:
            for (orthography, pronunciations) in output.items():
                if not pronunciations:
                    continue
                for p in pronunciations:
                    if not p:
                        continue
                    f.write(f"{orthography}\t{p}\n")
        gen.compute_validation_errors(validation_set, output)

    def initialize_training(self) -> None:
        """Initialize training G2P model"""
        with self.session() as session:
            session.query(Word2Job).delete()
            session.query(M2M2Job).delete()
            session.query(M2MSymbol).delete()
            session.query(Job).delete()
            session.commit()

            job_objs = [{"id": j} for j in range(1, config.NUM_JOBS + 1)]
            self.g2p_num_training_pronunciations = 0
            self.g2p_num_validation_pronunciations = 0
            self.g2p_num_training_words = 0
            self.g2p_num_validation_words = 0
            # Below we partition sorted list of words to try to have each process handling different symbol tables
            # so they're not completely overlapping and using more memory
            num_words = session.query(Word.id).count()
            words_per_job = int(num_words / config.NUM_JOBS) + 1
            current_job = 1
            words = session.query(Word.id).filter(Word.word_type.in_(WordType.speech_types()))
            mappings = []
            for i, (w,) in enumerate(words):
                if i >= (current_job) * words_per_job and current_job != config.NUM_JOBS + 1:
                    current_job += 1
                mappings.append({"word_id": w, "job_id": current_job, "training": 1})
            session.bulk_insert_mappings(Job, job_objs)
            session.flush()
            session.execute(sqlalchemy.insert(Word2Job.__table__), mappings)
            session.commit()

            if self.evaluation_mode:
                validation_items = int(num_words * self.validation_proportion)
                validation_words = (
                    sqlalchemy.select(Word.id)
                    .order_by(sqlalchemy.func.random())
                    .limit(validation_items)
                    .scalar_subquery()
                )
                query = (
                    sqlalchemy.update(Word2Job)
                    .execution_options(synchronize_session="fetch")
                    .values(training=False)
                    .where(Word2Job.word_id.in_(validation_words))
                )
                with session.begin_nested():
                    session.execute(query)
                    session.flush()
                session.commit()
                query = (
                    session.query(Word.word, Pronunciation.pronunciation)
                    .join(Pronunciation.word)
                    .join(Word.job)
                    .filter(Word2Job.training == False)  # noqa
                )
                for word, pronunciation in query:
                    self.g2p_validation_graphemes.update(word)
                    self.g2p_validation_phones.update(pronunciation.split())
                    self.g2p_num_validation_pronunciations += 1
                self.g2p_num_validation_words = (
                    session.query(Word2Job.word_id)
                    .filter(Word2Job.training == False)  # noqa
                    .count()
                )

            grapheme_count = 0
            phone_count = 0
            self.character_sets = set()
            query = (
                session.query(Pronunciation.pronunciation, Word.word)
                .join(Pronunciation.word)
                .join(Word.job)
                .filter(Word2Job.training == True)  # noqa
            )
            with mfa_open(self.working_directory.joinpath("input.txt"), "w") as word_f, mfa_open(
                self.working_directory.joinpath("output.txt"), "w"
            ) as phone_f:
                for pronunciation, word in query:
                    word = list(word)
                    grapheme_count += len(word)
                    self.g2p_training_graphemes.update(word)
                    self.g2p_num_training_pronunciations += 1
                    self.g2p_training_phones.update(pronunciation.split())
                    phone_count += len(pronunciation.split())
                    word_f.write(" ".join(word) + "\n")
                    phone_f.write(pronunciation + "\n")
            self.g2p_num_training_words = (
                session.query(Word2Job.word_id).filter(Word2Job.training == True).count()  # noqa
            )
            logger.debug(f"Graphemes in training data: {sorted(self.g2p_training_graphemes)}")
            logger.debug(f"Phones in training data: {sorted(self.g2p_training_phones)}")
            logger.debug(f"Averages phones per grapheme: {phone_count / grapheme_count}")

            if self.sequence_separator in self.g2p_training_phones | self.g2p_training_graphemes:
                raise PhonetisaurusSymbolError(self.sequence_separator, "sequence_separator")
            if self.skip in self.g2p_training_phones | self.g2p_training_graphemes:
                raise PhonetisaurusSymbolError(self.skip, "skip")
            if self.alignment_separator in self.g2p_training_phones | self.g2p_training_graphemes:
                raise PhonetisaurusSymbolError(self.alignment_separator, "alignment_separator")

            if self.evaluation_mode:
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
            self.compute_initial_ngrams()
