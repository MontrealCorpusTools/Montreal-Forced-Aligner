"""
Transcription
=============

"""
from __future__ import annotations

import collections
import csv
import logging
import os
import shutil
import subprocess
import threading
import time
import typing
from multiprocessing.pool import ThreadPool
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pywrapfst
from _kalpy.fstext import VectorFst
from kalpy.decoder.decode_graph import DecodeGraphCompiler
from kalpy.utils import kalpy_logger
from praatio import textgrid
from sqlalchemy.orm import joinedload, selectinload
from tqdm.rich import tqdm

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import TopLevelMfaWorker
from montreal_forced_aligner.alignment.base import CorpusAligner
from montreal_forced_aligner.data import (
    ArpaNgramModel,
    TextFileType,
    TextgridFormats,
    WorkflowType,
)
from montreal_forced_aligner.db import (
    CorpusWorkflow,
    Dictionary,
    File,
    Phone,
    SoundFile,
    Speaker,
    Utterance,
    bulk_update,
)
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.helper import (
    load_configuration,
    mfa_open,
    parse_old_features,
    score_wer,
)
from montreal_forced_aligner.language_modeling.multiprocessing import (
    TrainLmArguments,
    TrainPhoneLmFunction,
    TrainSpeakerLmArguments,
    TrainSpeakerLmFunction,
)
from montreal_forced_aligner.models import AcousticModel, LanguageModel
from montreal_forced_aligner.textgrid import construct_output_path
from montreal_forced_aligner.transcription.multiprocessing import (
    CarpaLmRescoreArguments,
    CarpaLmRescoreFunction,
    CreateHclgArguments,
    CreateHclgFunction,
    DecodeArguments,
    DecodeFunction,
    DecodePhoneArguments,
    DecodePhoneFunction,
    FinalFmllrArguments,
    FinalFmllrFunction,
    FmllrRescoreArguments,
    FmllrRescoreFunction,
    InitialFmllrArguments,
    InitialFmllrFunction,
    LmRescoreArguments,
    LmRescoreFunction,
    PerSpeakerDecodeArguments,
    PerSpeakerDecodeFunction,
)
from montreal_forced_aligner.utils import (
    KaldiProcessWorker,
    log_kaldi_errors,
    run_kaldi_function,
    thirdparty_binary,
)

if TYPE_CHECKING:

    from montreal_forced_aligner.abc import MetaDict

__all__ = ["Transcriber", "TranscriberMixin"]

logger = logging.getLogger("mfa")


class TranscriberMixin(CorpusAligner):
    """Abstract class for MFA transcribers

    Parameters
    ----------
    transition_scale: float
        Transition scale, defaults to 1.0
    acoustic_scale: float
        Acoustic scale, defaults to 0.1
    self_loop_scale: float
        Self-loop scale, defaults to 0.1
    beam: int
        Size of the beam to use in decoding, defaults to 10
    silence_weight: float
        Weight on silence in fMLLR estimation
    max_active: int
        Max active for decoding
    lattice_beam: int
        Beam width for decoding lattices
    first_beam: int
        Beam for decoding in initial speaker-independent pass, only used if ``uses_speaker_adaptation`` is true
    first_max_active: int
        Max active for decoding in initial speaker-independent pass, only used if ``uses_speaker_adaptation`` is true
    language_model_weight: float
        Weight of language model
    word_insertion_penalty: float
        Penalty for inserting words
    """

    def __init__(
        self,
        transition_scale: float = 1.0,
        acoustic_scale: float = 0.083333,
        self_loop_scale: float = 0.1,
        beam: int = 10,
        silence_weight: float = 0.0,
        first_beam: int = 10,
        first_max_active: int = 2000,
        language_model_weight: int = 10,
        word_insertion_penalty: float = 0.5,
        evaluation_mode: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.beam = beam
        self.acoustic_scale = acoustic_scale
        self.self_loop_scale = self_loop_scale
        self.transition_scale = transition_scale
        self.silence_weight = silence_weight
        self.first_beam = first_beam
        self.first_max_active = first_max_active
        self.language_model_weight = language_model_weight
        self.word_insertion_penalty = word_insertion_penalty
        self.evaluation_mode = evaluation_mode
        self.alignment_mode = False

    def train_speaker_lm_arguments(
        self,
    ) -> List[TrainSpeakerLmArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.language_modeling.multiprocessing.TrainSpeakerLmFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.language_modeling.multiprocessing.TrainSpeakerLmArguments`]
            Arguments for processing
        """
        arguments = []

        for j in self.jobs:
            arguments.append(
                TrainSpeakerLmArguments(
                    j.id,
                    getattr(self, "session", ""),
                    self.working_log_directory.joinpath(f"train_lm.{j.id}.log"),
                    self.model_path,
                    self.tree_path,
                    self.lexicon_compilers,
                    self.order,
                    self.method,
                    self.target_num_ngrams,
                    self.hclg_options,
                )
            )
        return arguments

    def train_speaker_lms(self) -> None:
        """Train language models for each speaker based on their utterances"""
        begin = time.time()
        log_directory = self.model_log_directory
        os.makedirs(log_directory, exist_ok=True)
        logger.info("Compiling per speaker biased language models...")
        arguments = self.train_speaker_lm_arguments()
        with tqdm(total=self.num_speakers, disable=config.QUIET) as pbar:
            for _ in run_kaldi_function(TrainSpeakerLmFunction, arguments, pbar.update):
                pass
        logger.debug(f"Compiling speaker language models took {time.time() - begin:.3f} seconds")

    @property
    def model_directory(self) -> Path:
        """Model directory for the transcriber"""
        return self.output_directory.joinpath("models")

    @property
    def model_log_directory(self) -> Path:
        """Model directory for the transcriber"""
        return self.model_directory.joinpath("log")

    def lm_rescore(self) -> None:
        """
        Rescore lattices with bigger language model

        See Also
        -------
        :class:`~montreal_forced_aligner.transcription.multiprocessing.LmRescoreFunction`
            Multiprocessing function
        :meth:`.TranscriberMixin.lm_rescore_arguments`
            Arguments for function
        """
        arguments = self.lm_rescore_arguments()
        logger.info("Rescoring lattices with medium G.fst...")
        with tqdm(total=self.num_current_utterances, disable=config.QUIET) as pbar:
            for _ in run_kaldi_function(LmRescoreFunction, arguments, pbar.update):
                pass

    def carpa_lm_rescore(self) -> None:
        """
        Rescore lattices with CARPA language model

        See Also
        -------
        :class:`~montreal_forced_aligner.transcription.multiprocessing.CarpaLmRescoreFunction`
            Multiprocessing function
        :meth:`.TranscriberMixin.carpa_lm_rescore_arguments`
            Arguments for function
        """
        logger.info("Rescoring lattices with large G.carpa...")
        arguments = self.carpa_lm_rescore_arguments()
        with tqdm(total=self.num_utterances, disable=config.QUIET) as pbar:
            for _ in run_kaldi_function(CarpaLmRescoreFunction, arguments, pbar.update):
                pass

    def train_phone_lm(self):
        """Train a phone-based language model (i.e., not using words)."""
        if not self.has_alignments(self.current_workflow.id):
            logger.error("Cannot train phone LM without alignments")
            return
        if self.use_g2p:
            return
        logger.info("Beginning phone LM training...")
        logger.info("Collecting training data...")

        ngram_order = 4
        num_ngrams = 20000
        phone_lm_path = self.phones_dir.joinpath("phone_lm.fst")
        log_path = self.phones_dir.joinpath("phone_lm_training.log")
        unigram_phones = set()
        return_queue = Queue()
        stopped = threading.Event()
        error_dict = {}
        procs = []
        count_paths = []
        allowed_bigrams = collections.defaultdict(set)
        with self.session() as session, tqdm(
            total=self.num_current_utterances, disable=config.QUIET
        ) as pbar:

            with mfa_open(self.phones_dir.joinpath("phone_boundaries.int"), "w") as f:
                for p in session.query(Phone):
                    f.write(f"{p.mapping_id} singleton\n")
            for j in self.jobs:
                args = TrainLmArguments(
                    j.id,
                    getattr(self, "session", ""),
                    self.working_log_directory.joinpath(f"ngram_count.{j.id}.log"),
                    self.phones_dir,
                    self.phone_symbol_table_path,
                    ngram_order,
                    self.oov_word,
                )
                function = TrainPhoneLmFunction(args)
                p = KaldiProcessWorker(j.id, return_queue, function, stopped)
                procs.append(p)
                p.start()
                count_paths.append(self.phones_dir.joinpath(f"{j.id}.cnts"))
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
                _, phones = result
                phones = phones.split()
                unigram_phones.update(phones)
                phones = ["<s>"] + phones + ["</s>"]
                for i in range(len(phones) - 1):
                    allowed_bigrams[phones[i]].add(phones[i + 1])

                pbar.update(1)
        for p in procs:
            p.join()
        if error_dict:
            for v in error_dict.values():
                raise v
        logger.info("Training model...")
        with mfa_open(log_path, "w") as log_file:
            merged_file = self.phones_dir.joinpath("merged.cnts")
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
                [thirdparty_binary("ngrammake"), "--v=2", "--method=kneser_ney", merged_file],
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            ngramshrink_proc = subprocess.Popen(
                [
                    thirdparty_binary("ngramshrink"),
                    "--v=2",
                    "--method=relative_entropy",
                    f"--target_number_of_ngrams={num_ngrams}",
                ],
                stderr=log_file,
                stdin=ngrammake_proc.stdout,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            print_proc = subprocess.Popen(
                [
                    thirdparty_binary("ngramprint"),
                    "--ARPA",
                    f"--symbols={self.phone_symbol_table_path}",
                ],
                stdin=ngramshrink_proc.stdout,
                stderr=log_file,
                encoding="utf8",
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            model = ArpaNgramModel.read(print_proc.stdout)
            phone_symbols = pywrapfst.SymbolTable()
            for _, phone in sorted(self.reversed_phone_mapping.items()):
                phone_symbols.add_symbol(phone)
            log_file.write("Done training initial ngram model\n")
            log_file.flush()
            bigram_fst = model.construct_bigram_fst("#1", allowed_bigrams, phone_symbols)

            bigram_fst.write(self.phones_dir.joinpath("bigram.fst"))
            bigram_fst.project("output")
            push_special_proc = subprocess.Popen(
                [thirdparty_binary("fstpushspecial")],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            minimize_proc = subprocess.Popen(
                [thirdparty_binary("fstminimizeencoded")],
                stdin=push_special_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            rm_syms_proc = subprocess.Popen(
                [
                    thirdparty_binary("fstrmsymbols"),
                    "--remove-from-output=true",
                    self.disambiguation_symbols_int_path,
                    "-",
                    phone_lm_path,
                ],
                stdin=minimize_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            push_special_proc.stdin.write(bigram_fst.write_to_string())
            push_special_proc.stdin.flush()
            push_special_proc.stdin.close()
            rm_syms_proc.communicate()

    def setup_phone_lm(self) -> None:
        """Setup phone language model for phone-based transcription"""

        self.train_phone_lm()
        log_path = self.working_log_directory.joinpath("hclg.log")
        with kalpy_logger("kalpy.decode_graph", log_path):
            compiler = DecodeGraphCompiler(
                self.model_path, self.tree_path, None, **self.hclg_options
            )
            compiler.lg_fst = VectorFst.Read(str(self.phones_dir.joinpath("phone_lm.fst")))
            hclg_path = self.working_directory.joinpath("HCLG_phone.fst")
            compiler.export_hclg(None, hclg_path)

    def transcribe(self, workflow_type: WorkflowType = WorkflowType.transcription):
        self.initialize_database()
        self.create_new_current_workflow(workflow_type)
        if workflow_type is WorkflowType.phone_transcription:
            self.setup_phone_lm()
        self.acoustic_model.export_model(self.working_directory)
        self.transcribe_utterances()

    def transcribe_utterances(self) -> None:
        """
        Transcribe the corpus

        See Also
        --------
        :func:`~montreal_forced_aligner.transcription.multiprocessing.DecodeFunction`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.LmRescoreFunction`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.CarpaLmRescoreFunction`
            Multiprocessing helper function for each job

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        logger.info("Beginning transcription...")
        workflow = self.current_workflow
        if workflow.done:
            logger.info("Transcription already done, skipping!")
            return
        try:
            if workflow.workflow_type is WorkflowType.transcription:
                self.uses_speaker_adaptation = False

            self.decode()
            if workflow.workflow_type is WorkflowType.transcription:
                logger.info("Performing speaker adjusted transcription...")
                self.transcribe_fmllr()
                self.lm_rescore()
                self.carpa_lm_rescore()
            self.collect_alignments()
            if self.fine_tune:
                self.fine_tune_alignments()
            if self.evaluation_mode:
                os.makedirs(self.working_log_directory, exist_ok=True)
                self.evaluate_transcriptions()
            with self.session() as session:
                session.query(CorpusWorkflow).filter(CorpusWorkflow.id == workflow.id).update(
                    {"done": True}
                )
                session.commit()
        except Exception as e:
            with self.session() as session:
                session.query(CorpusWorkflow).filter(CorpusWorkflow.id == workflow.id).update(
                    {"dirty": True}
                )
                session.commit()
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise

    def evaluate_transcriptions(self) -> Tuple[float, float]:
        """
        Evaluates the transcripts if there are reference transcripts

        Returns
        -------
        float, float
            Sentence error rate and word error rate

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        logger.info("Evaluating transcripts...")
        ser, wer, cer = self.compute_wer()
        logger.info(f"SER: {100 * ser:.2f}%, WER: {100 * wer:.2f}%, CER: {100 * cer:.2f}%")

    def save_transcription_evaluation(self, output_directory: Path) -> None:
        """
        Save transcription evaluation to an output directory

        Parameters
        ----------
        output_directory: str
            Directory to save evaluation
        """
        output_path = output_directory.joinpath("transcription_evaluation.csv")
        with mfa_open(output_path, "w") as f, self.session() as session:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "file",
                    "speaker",
                    "begin",
                    "end",
                    "duration",
                    "word_count",
                    "oov_count",
                    "gold_transcript",
                    "hypothesis",
                    "WER",
                    "CER",
                ]
            )
            utterances = (
                session.query(
                    Speaker.name,
                    File.name,
                    Utterance.begin,
                    Utterance.end,
                    Utterance.duration,
                    Utterance.normalized_text,
                    Utterance.transcription_text,
                    Utterance.oovs,
                    Utterance.word_error_rate,
                    Utterance.character_error_rate,
                )
                .join(Utterance.speaker)
                .join(Utterance.file)
                .filter(Utterance.normalized_text != None)  # noqa
                .filter(Utterance.normalized_text != "")
            )

            for (
                speaker,
                file,
                begin,
                end,
                duration,
                text,
                transcription_text,
                oovs,
                word_error_rate,
                character_error_rate,
            ) in utterances:
                word_count = text.count(" ") + 1
                oov_count = oovs.count(" ") + 1
                writer.writerow(
                    [
                        file,
                        speaker,
                        begin,
                        end,
                        duration,
                        word_count,
                        oov_count,
                        text,
                        transcription_text,
                        word_error_rate,
                        character_error_rate,
                    ]
                )

    def compute_wer(self) -> typing.Tuple[float, float, float]:
        """
        Evaluates the transcripts if there are reference transcripts

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        if not hasattr(self, "db_engine"):
            raise Exception("Must be used as part of a class with a database engine")
        logger.info("Evaluating transcripts...")
        # Sentence-level measures
        incorrect = 0
        total_count = 0
        # Word-level measures
        total_word_edits = 0
        total_word_length = 0

        # Character-level measures
        total_character_edits = 0
        total_character_length = 0

        indices = []
        to_comp = []

        update_mappings = []
        with self.session() as session:
            utterances = session.query(Utterance)
            utterances = utterances.filter(Utterance.normalized_text != None)  # noqa
            utterances = utterances.filter(Utterance.normalized_text != "")
            for utt in utterances:
                g = utt.normalized_text.split()
                total_count += 1
                total_word_length += len(g)
                character_length = len("".join(g))
                total_character_length += character_length

                if not utt.transcription_text:
                    incorrect += 1
                    total_word_edits += len(g)
                    total_character_edits += character_length
                    update_mappings.append(
                        {"id": utt.id, "word_error_rate": 1.0, "character_error_rate": 1.0}
                    )
                    continue

                h = utt.transcription_text.split()
                if g != h:
                    indices.append(utt.id)
                    to_comp.append((g, h))
                    incorrect += 1
                else:
                    update_mappings.append(
                        {"id": utt.id, "word_error_rate": 0.0, "character_error_rate": 0.0}
                    )

            with ThreadPool(config.NUM_JOBS) as pool:
                gen = pool.starmap(score_wer, to_comp)
                for i, (word_edits, word_length, character_edits, character_length) in enumerate(
                    gen
                ):
                    utt_id = indices[i]
                    update_mappings.append(
                        {
                            "id": utt_id,
                            "word_error_rate": word_edits / word_length,
                            "character_error_rate": character_edits / character_length,
                        }
                    )
                    total_word_edits += word_edits
                    total_character_edits += character_edits

            bulk_update(session, Utterance, update_mappings)
            session.commit()
        ser = incorrect / total_count
        wer = total_word_edits / total_word_length
        cer = total_character_edits / total_character_length
        return ser, wer, cer

    @property
    def transcribe_fmllr_options(self) -> MetaDict:
        """Options needed for calculating fMLLR transformations"""
        return {
            "silence_weight": self.silence_weight,
        }

    @property
    def lm_rescore_options(self) -> MetaDict:
        """Options needed for rescoring the language model"""
        return {
            "acoustic_scale": self.acoustic_scale,
        }

    def decode(self) -> None:
        """
        Generate lattices

        See Also
        -------
        :class:`~montreal_forced_aligner.transcription.multiprocessing.DecodeFunction`
            Multiprocessing function
        :meth:`.TranscriberMixin.decode_arguments`
            Arguments for function
        """
        logger.info("Generating lattices...")
        with tqdm(total=self.num_utterances, disable=config.QUIET) as pbar:
            workflow = self.current_workflow
            arguments = self.decode_arguments(workflow.workflow_type)
            log_likelihood_sum = 0
            log_likelihood_count = 0
            if workflow.workflow_type is WorkflowType.per_speaker_transcription:
                decode_function = PerSpeakerDecodeFunction
            elif workflow.workflow_type is WorkflowType.phone_transcription:
                decode_function = DecodePhoneFunction
            else:
                decode_function = DecodeFunction
            for _, log_likelihood in run_kaldi_function(decode_function, arguments, pbar.update):
                log_likelihood_sum += log_likelihood
                log_likelihood_count += 1
            if log_likelihood_count:
                with self.session() as session:
                    workflow.score = log_likelihood_sum / log_likelihood_count
                    session.commit()

    def calc_initial_fmllr(self) -> None:
        """
        Calculate initial fMLLR transforms

        See Also
        -------
        :class:`~montreal_forced_aligner.transcription.multiprocessing.InitialFmllrFunction`
            Multiprocessing function
        :meth:`.TranscriberMixin.initial_fmllr_arguments`
            Arguments for function
        """
        logger.info("Calculating initial fMLLR transforms...")
        arguments = self.initial_fmllr_arguments()
        with tqdm(total=self.num_speakers, disable=config.QUIET) as pbar:
            for _ in run_kaldi_function(InitialFmllrFunction, arguments, pbar.update):
                pass

    def calc_final_fmllr(self) -> None:
        """
        Calculate final fMLLR transforms

        See Also
        -------
        :class:`~montreal_forced_aligner.transcription.multiprocessing.FinalFmllrFunction`
            Multiprocessing function
        :meth:`.TranscriberMixin.final_fmllr_arguments`
            Arguments for function
        """
        logger.info("Calculating final fMLLR transforms...")
        arguments = self.final_fmllr_arguments()
        with tqdm(total=self.num_speakers, disable=config.QUIET) as pbar:
            for _ in run_kaldi_function(FinalFmllrFunction, arguments, pbar.update):
                pass

    def fmllr_rescore(self) -> None:
        """
        Rescore lattices with final fMLLR transforms

        See Also
        -------
        :class:`~montreal_forced_aligner.transcription.multiprocessing.FmllrRescoreFunction`
            Multiprocessing function
        :meth:`.TranscriberMixin.fmllr_rescore_arguments`
            Arguments for function
        """
        logger.info("Rescoring fMLLR lattices with final transform...")
        arguments = self.fmllr_rescore_arguments()
        with tqdm(total=self.num_utterances, disable=config.QUIET) as pbar:
            for _ in run_kaldi_function(FmllrRescoreFunction, arguments, pbar.update):
                pass

    def transcribe_fmllr(self) -> None:
        """
        Run fMLLR estimation over initial decoding lattices and rescore

        See Also
        --------
        :func:`~montreal_forced_aligner.transcription.multiprocessing.InitialFmllrFunction`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.LatGenFmllrFunction`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.FinalFmllrFunction`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.FmllrRescoreFunction`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.LmRescoreFunction`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.CarpaLmRescoreFunction`
            Multiprocessing helper function for each job

        """
        self.calc_initial_fmllr()
        self.uses_speaker_adaptation = True
        self.decode()
        self.calc_final_fmllr()

        self.fmllr_rescore()

    def decode_arguments(
        self, workflow: WorkflowType = WorkflowType.transcription
    ) -> List[typing.Union[DecodeArguments, PerSpeakerDecodeArguments]]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.DecodeFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.DecodeArguments`]
            Arguments for processing
        """
        arguments = []
        decode_options = self.decode_options
        if not self.uses_speaker_adaptation:
            decode_options["max_active"] = self.first_max_active
            decode_options["beam"] = self.first_beam
        for j in self.jobs:
            if workflow is WorkflowType.per_speaker_transcription:
                arguments.append(
                    PerSpeakerDecodeArguments(
                        j.id,
                        getattr(self, "session", ""),
                        self.working_log_directory.joinpath(f"per_speaker_decode.{j.id}.log"),
                        self.working_directory,
                        self.model_path,
                        self.tree_path,
                        decode_options,
                        self.order,
                        self.method,
                    )
                )
            elif workflow is WorkflowType.phone_transcription:
                arguments.append(
                    DecodePhoneArguments(
                        j.id,
                        getattr(self, "session", ""),
                        self.working_log_directory.joinpath(f"decode.{j.id}.log"),
                        self.working_directory,
                        self.alignment_model_path,
                        self.working_directory.joinpath("HCLG_phone.fst"),
                        decode_options,
                    )
                )
            else:
                arguments.append(
                    DecodeArguments(
                        j.id,
                        getattr(self, "session", ""),
                        self.working_log_directory.joinpath(f"decode.{j.id}.log"),
                        self.working_directory,
                        self.alignment_model_path,
                        decode_options,
                        j.construct_dictionary_dependent_paths(
                            self.model_directory, "HCLG", "fst"
                        ),
                    )
                )
        return arguments

    def lm_rescore_arguments(self) -> List[LmRescoreArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.LmRescoreFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.LmRescoreArguments`]
            Arguments for processing
        """
        return [
            LmRescoreArguments(
                j.id,
                getattr(self, "session", ""),
                self.working_log_directory.joinpath(f"lm_rescore.{j.id}.log"),
                self.working_directory,
                self.lm_rescore_options,
                j.construct_dictionary_dependent_paths(self.model_directory, "G_small", "fst"),
                j.construct_dictionary_dependent_paths(self.model_directory, "G_med", "fst"),
            )
            for j in self.jobs
        ]

    def carpa_lm_rescore_arguments(self) -> List[CarpaLmRescoreArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.CarpaLmRescoreFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.CarpaLmRescoreArguments`]
            Arguments for processing
        """
        return [
            CarpaLmRescoreArguments(
                j.id,
                getattr(self, "session", ""),
                self.working_log_directory.joinpath(f"carpa_lm_rescore.{j.id}.log"),
                self.working_directory,
                self.lm_rescore_options,
                j.construct_dictionary_dependent_paths(self.model_directory, "G_med", "fst"),
                j.construct_dictionary_dependent_paths(self.model_directory, "G", "carpa"),
            )
            for j in self.jobs
        ]

    def initial_fmllr_arguments(self) -> List[InitialFmllrArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.InitialFmllrFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.InitialFmllrArguments`]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            arguments.append(
                InitialFmllrArguments(
                    j.id,
                    getattr(self, "session", ""),
                    self.working_log_directory.joinpath(f"initial_fmllr.{j.id}.log"),
                    self.working_directory,
                    self.model_path,
                    self.fmllr_options,
                )
            )
        return arguments

    def final_fmllr_arguments(self) -> List[FinalFmllrArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.FinalFmllrFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.FinalFmllrArguments`]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            arguments.append(
                FinalFmllrArguments(
                    j.id,
                    getattr(self, "session", ""),
                    self.working_log_directory.joinpath(f"final_fmllr.{j.id}.log"),
                    self.working_directory,
                    self.model_path,
                    self.fmllr_options,
                )
            )
        return arguments

    def fmllr_rescore_arguments(self) -> List[FmllrRescoreArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.FmllrRescoreFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.FmllrRescoreArguments`]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            rescore_options = {
                "lattice_beam": self.lattice_beam,
                "acoustic_scale": self.acoustic_scale,
            }
            arguments.append(
                FmllrRescoreArguments(
                    j.id,
                    getattr(self, "session", ""),
                    self.working_log_directory.joinpath(f"fmllr_rescore.{j.id}.log"),
                    self.working_directory,
                    self.model_path,
                    rescore_options,
                )
            )
        return arguments


class Transcriber(TranscriberMixin, TopLevelMfaWorker):
    """
    Class for performing transcription.

    Parameters
    ----------
    acoustic_model_path : str
        Path to acoustic model
    language_model_path : str
        Path to language model model
    evaluation_mode: bool
        Flag for evaluating generated transcripts against the actual transcripts, defaults to False

    See Also
    --------
    :class:`~montreal_forced_aligner.transcription.transcriber.TranscriberMixin`
        For transcription parameters
    :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusPronunciationMixin`
        For corpus and dictionary parsing parameters
    :class:`~montreal_forced_aligner.abc.FileExporterMixin`
        For file exporting parameters
    :class:`~montreal_forced_aligner.abc.TopLevelMfaWorker`
        For top-level parameters

    Attributes
    ----------
    acoustic_model: AcousticModel
        Acoustic model
    language_model: LanguageModel
        Language model
    """

    def __init__(
        self,
        acoustic_model_path: Path,
        language_model_path: Path,
        output_type: str = "transcription",
        **kwargs,
    ):
        self.acoustic_model = AcousticModel(acoustic_model_path)
        kwargs.update(self.acoustic_model.parameters)
        super(Transcriber, self).__init__(**kwargs)
        self.language_model = LanguageModel(language_model_path)
        self.output_type = output_type
        self.ignore_empty_utterances = False

    def create_hclgs_arguments(self) -> typing.List[CreateHclgArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.CreateHclgFunction`

        Returns
        -------
        dict[str, :class:`~montreal_forced_aligner.transcription.multiprocessing.CreateHclgArguments`]
            Per dictionary arguments for HCLG
        """
        args = []
        with self.session() as session:
            for d in session.query(Dictionary):
                args.append(
                    CreateHclgArguments(
                        d.id,
                        getattr(self, "session", ""),
                        self.model_directory.joinpath("log", f"hclg.{d.id}.log"),
                        self.lexicon_compilers[d.id],
                        self.model_directory,
                        self.language_model.small_arpa_path,
                        self.language_model.medium_arpa_path,
                        self.language_model.carpa_path,
                        self.model_path,
                        self.tree_path,
                        self.hclg_options,
                    )
                )
        return args

    def create_hclgs(self) -> None:
        """
        Create HCLG.fst files for every dictionary being used by a
        :class:`~montreal_forced_aligner.transcription.transcriber.Transcriber`
        """

        arguments = self.create_hclgs_arguments()
        logger.info("Generating HCLG.fst...")
        with tqdm(total=len(arguments), disable=config.QUIET) as pbar:
            for _ in run_kaldi_function(CreateHclgFunction, arguments, pbar.update):
                pass

    def create_decoding_graph(self) -> None:
        """
        Create decoding graph for use in transcription

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        done_path = os.path.join(self.model_directory, "done")
        if os.path.exists(done_path):
            logger.info("Graph construction already done, skipping!")
        log_dir = os.path.join(self.model_directory, "log")
        os.makedirs(log_dir, exist_ok=True)
        self.write_lexicon_information(write_disambiguation=True)
        with self.session() as session:
            for d in session.query(Dictionary):
                words_path = os.path.join(self.model_directory, f"words.{d.id}.txt")
                shutil.copyfile(d.words_symbol_path, words_path)

        big_arpa_path = self.language_model.carpa_path
        small_arpa_path = self.language_model.small_arpa_path
        medium_arpa_path = self.language_model.medium_arpa_path
        if not os.path.exists(small_arpa_path) or not os.path.exists(medium_arpa_path):
            logger.warning(
                "Creating small and medium language models from scratch, this may take some time. "
                "Running `mfa train_lm` on the ARPA file will remove this warning."
            )
            logger.info("Parsing large ngram model...")
            mod_path = self.model_directory.joinpath("base_lm.mod")
            new_carpa_path = os.path.join(self.model_directory, "base_lm.arpa")
            with mfa_open(big_arpa_path, "r") as inf, mfa_open(new_carpa_path, "w") as outf:
                for line in inf:
                    outf.write(line.lower())
            big_arpa_path = new_carpa_path
            subprocess.call(["ngramread", "--ARPA", big_arpa_path, mod_path])

            if not os.path.exists(small_arpa_path):
                logger.info(
                    "Generating small model from the large ARPA with a pruning threshold of 3e-7"
                )
                prune_thresh_small = 0.0000003
                small_mod_path = mod_path.with_stem(mod_path.stem + "_small")
                subprocess.call(
                    [
                        "ngramshrink",
                        "--method=relative_entropy",
                        f"--theta={prune_thresh_small}",
                        mod_path,
                        small_mod_path,
                    ]
                )
                subprocess.call(["ngramprint", "--ARPA", small_mod_path, small_arpa_path])

            if not os.path.exists(medium_arpa_path):
                logger.info(
                    "Generating medium model from the large ARPA with a pruning threshold of 1e-7"
                )
                prune_thresh_medium = 0.0000001
                med_mod_path = mod_path.with_stem(mod_path.stem + "_med")
                subprocess.call(
                    [
                        "ngramshrink",
                        "--method=relative_entropy",
                        f"--theta={prune_thresh_medium}",
                        mod_path,
                        med_mod_path,
                    ]
                )
                subprocess.call(["ngramprint", "--ARPA", med_mod_path, medium_arpa_path])
        try:
            self.create_hclgs()
        except Exception as e:
            dirty_path = os.path.join(self.model_directory, "dirty")
            with mfa_open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[Path] = None,
        args: Optional[Dict[str, typing.Any]] = None,
        unknown_args: Optional[typing.Iterable[str]] = None,
    ) -> MetaDict:
        """
        Parse configuration parameters from a config file and command line arguments

        Parameters
        ----------
        config_path: :class:`~pathlib.Path`, optional
            Path to yaml configuration file
        args: dict[str, Any]
            Parsed arguments
        unknown_args: list[str]
            Optional list of arguments that were not parsed

        Returns
        -------
        dict[str, Any]
            Dictionary of specified configuration parameters
        """
        global_params = {}
        if config_path and os.path.exists(config_path):
            data = load_configuration(config_path)
            data = parse_old_features(data)
            for k, v in data.items():
                if k == "features":
                    global_params.update(v)
                else:
                    if v is None and k in cls.nullable_fields:
                        v = []
                    global_params[k] = v

        global_params.update(cls.parse_args(args, unknown_args))
        if args.get("language_model_weight", None) is not None:
            global_params["min_language_model_weight"] = args["language_model_weight"]
            global_params["max_language_model_weight"] = args["language_model_weight"] + 1
        if args.get("word_insertion_penalty", None) is not None:
            global_params["word_insertion_penalties"] = [args["word_insertion_penalty"]]
        return global_params

    def setup_acoustic_model(self):
        self.acoustic_model.validate(self)
        self.acoustic_model.export_model(self.model_directory)
        self.acoustic_model.export_model(self.working_directory)
        self.acoustic_model.log_details()

    def setup(self) -> None:
        """Set up transcription"""
        self.alignment_mode = False
        TopLevelMfaWorker.setup(self)
        if self.initialized:
            return
        self.create_new_current_workflow(WorkflowType.transcription)
        begin = time.time()
        os.makedirs(self.working_log_directory, exist_ok=True)
        self.load_corpus()
        dirty_path = self.working_directory.joinpath("dirty")
        if os.path.exists(dirty_path):
            shutil.rmtree(self.working_directory, ignore_errors=True)
        os.makedirs(self.working_log_directory, exist_ok=True)
        dirty_path = os.path.join(self.model_directory, "dirty")

        if os.path.exists(dirty_path):  # if there was an error, let's redo from scratch
            shutil.rmtree(self.model_directory)
        log_dir = os.path.join(self.model_directory, "log")
        os.makedirs(log_dir, exist_ok=True)
        self.setup_acoustic_model()
        self.create_decoding_graph()
        self.initialized = True
        logger.debug(f"Setup for transcription in {time.time() - begin:.3f} seconds")

    def export_transcriptions(self) -> None:
        """Export transcriptions"""
        with self.session() as session:
            files = session.query(File).options(
                selectinload(File.utterances),
                selectinload(File.speakers),
                joinedload(File.sound_file, innerjoin=True).load_only(SoundFile.duration),
            )
            for file in files:
                utterance_count = len(file.utterances)
                duration = file.sound_file.duration

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
                    self.export_output_directory,
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
        if self.evaluation_mode:
            self.save_transcription_evaluation(self.export_output_directory)

    def export_files(
        self,
        output_directory: Path,
        output_format: Optional[str] = None,
        include_original_text: bool = False,
    ) -> None:
        """
        Export transcriptions

        Parameters
        ----------
        output_directory: str
            Directory to save transcriptions
        output_format: str, optional
            Format to save alignments, one of 'long_textgrids' (the default), 'short_textgrids', or 'json', passed to praatio
        """
        if output_format is None:
            output_format = TextFileType.TEXTGRID.value
        self.export_output_directory = output_directory
        os.makedirs(self.export_output_directory, exist_ok=True)
        if self.output_type == "transcription":
            self.export_transcriptions()
        else:
            self.export_textgrids(output_format, include_original_text)
