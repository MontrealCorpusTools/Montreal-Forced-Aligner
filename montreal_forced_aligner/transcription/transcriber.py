"""
Transcription
=============

"""
from __future__ import annotations

import collections
import csv
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import time
import typing
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pywrapfst
import tqdm
from praatio import textgrid
from sqlalchemy.orm import joinedload, selectinload

from montreal_forced_aligner.abc import TopLevelMfaWorker
from montreal_forced_aligner.alignment.base import CorpusAligner
from montreal_forced_aligner.alignment.multiprocessing import construct_output_path
from montreal_forced_aligner.config import GLOBAL_CONFIG
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
    LatGenFmllrArguments,
    LatGenFmllrFunction,
    LmRescoreArguments,
    LmRescoreFunction,
    PerSpeakerDecodeArguments,
    PerSpeakerDecodeFunction,
)
from montreal_forced_aligner.utils import (
    KaldiProcessWorker,
    Stopped,
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
        silence_weight: float = 0.01,
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
        with self.session() as session:
            for j in self.jobs:
                speaker_mapping = {}
                speaker_paths = {}
                words_symbol_paths = {}

                speakers = (
                    session.query(Speaker)
                    .join(Speaker.utterances)
                    .options(joinedload(Speaker.dictionary, innerjoin=True))
                    .filter(Utterance.job_id == j.id)
                    .distinct()
                )
                for s in speakers:
                    dict_id = s.dictionary_id
                    if dict_id not in speaker_mapping:
                        speaker_mapping[dict_id] = []
                        words_symbol_paths[dict_id] = s.dictionary.words_symbol_path
                    speaker_mapping[dict_id].append(s.id)
                    speaker_paths[s.id] = os.path.join(self.data_directory, f"{s.id}.txt")
                arguments.append(
                    TrainSpeakerLmArguments(
                        j.id,
                        getattr(self, "db_string", ""),
                        os.path.join(self.working_log_directory, f"train_lm.{j.id}.log"),
                        self.model_path,
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
        with tqdm.tqdm(total=self.num_speakers, disable=GLOBAL_CONFIG.quiet) as pbar:
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []

                for i, args in enumerate(arguments):
                    function = TrainSpeakerLmFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    if isinstance(result, KaldiProcessingError):
                        error_dict[result.job_name] = result
                        continue
                    pbar.update(1)
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                logger.debug("Not using multiprocessing...")
                for args in arguments:
                    function = TrainSpeakerLmFunction(args)
                    for _ in function.run():
                        pbar.update(1)
        logger.debug(f"Compiling speaker language models took {time.time() - begin:.3f} seconds")

    @property
    def model_directory(self) -> str:
        """Model directory for the transcriber"""
        return os.path.join(self.output_directory, "models")

    @property
    def model_log_directory(self) -> str:
        """Model directory for the transcriber"""
        return os.path.join(self.model_directory, "log")

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
        logger.info("Rescoring lattices with medium G.fst...")
        if GLOBAL_CONFIG.use_mp:
            error_dict = {}
            return_queue = mp.Queue()
            stopped = Stopped()
            procs = []
            for i, args in enumerate(self.lm_rescore_arguments()):
                function = LmRescoreFunction(args)
                p = KaldiProcessWorker(i, return_queue, function, stopped)
                procs.append(p)
                p.start()
            with tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    succeeded, failed = result
                    if failed:
                        logger.warning("Some lattices failed to be rescored")
                    pbar.update(succeeded + failed)
            for p in procs:
                p.join()
            if error_dict:
                for v in error_dict.values():
                    raise v
        else:
            for args in self.lm_rescore_arguments():
                function = LmRescoreFunction(args)
                with tqdm.tqdm(total=GLOBAL_CONFIG.num_jobs, disable=GLOBAL_CONFIG.quiet) as pbar:
                    for succeeded, failed in function.run():
                        if failed:
                            logger.warning("Some lattices failed to be rescored")
                        pbar.update(succeeded + failed)

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
        if GLOBAL_CONFIG.use_mp:
            error_dict = {}
            return_queue = mp.Queue()
            stopped = Stopped()
            procs = []
            for i, args in enumerate(self.carpa_lm_rescore_arguments()):
                function = CarpaLmRescoreFunction(args)
                p = KaldiProcessWorker(i, return_queue, function, stopped)
                procs.append(p)
                p.start()
            with tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    succeeded, failed = result
                    if failed:
                        logger.warning("Some lattices failed to be rescored")
                    pbar.update(succeeded + failed)
            for p in procs:
                p.join()
            if error_dict:
                for v in error_dict.values():
                    raise v
        else:
            for args in self.carpa_lm_rescore_arguments():
                function = CarpaLmRescoreFunction(args)
                with tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
                    for succeeded, failed in function.run():
                        if failed:
                            logger.warning("Some lattices failed to be rescored")
                        pbar.update(succeeded + failed)

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
        phone_lm_path = os.path.join(self.phones_dir, "phone_lm.fst")
        log_path = os.path.join(self.phones_dir, "phone_lm_training.log")
        unigram_phones = set()
        return_queue = mp.Queue()
        stopped = Stopped()
        error_dict = {}
        procs = []
        count_paths = []
        allowed_bigrams = collections.defaultdict(set)
        with self.session() as session, tqdm.tqdm(
            total=self.num_current_utterances, disable=GLOBAL_CONFIG.quiet
        ) as pbar:

            with mfa_open(os.path.join(self.phones_dir, "phone_boundaries.int"), "w") as f:
                for p in session.query(Phone):
                    f.write(f"{p.mapping_id} singleton\n")
            for j in self.jobs:
                args = TrainLmArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(self.working_log_directory, f"ngram_count.{j.id}.log"),
                    self.phones_dir,
                    self.phone_symbol_table_path,
                    ngram_order,
                    self.oov_word,
                )
                function = TrainPhoneLmFunction(args)
                p = KaldiProcessWorker(j.id, return_queue, function, stopped)
                procs.append(p)
                p.start()
                count_paths.append(os.path.join(self.phones_dir, f"{j.id}.cnts"))
            while True:
                try:
                    result = return_queue.get(timeout=1)
                    if isinstance(result, Exception):
                        error_dict[getattr(result, "job_name", 0)] = result
                        continue
                    if stopped.stop_check():
                        continue
                except Empty:
                    for proc in procs:
                        if not proc.finished.stop_check():
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
            merged_file = os.path.join(self.phones_dir, "merged.cnts")
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

            bigram_fst.write(os.path.join(self.phones_dir, "bigram.fst"))
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
        from montreal_forced_aligner.transcription.multiprocessing import compose_clg, compose_hclg

        self.train_phone_lm()
        with mfa_open(os.path.join(self.working_log_directory, "hclg.log"), "w") as log_file:
            context_width = self.hclg_options["context_width"]
            central_pos = self.hclg_options["central_pos"]

            clg_path = os.path.join(
                self.working_directory, f"CLG_{context_width}_{central_pos}.fst"
            )
            hclga_path = os.path.join(self.working_directory, "HCLGa.fst")
            hclg_path = os.path.join(self.working_directory, "HCLG_phone.fst")
            ilabels_temp = os.path.join(
                self.working_directory, f"ilabels_{context_width}_{central_pos}"
            )
            out_disambig = os.path.join(
                self.working_directory, f"disambig_ilabels_{context_width}_{central_pos}.int"
            )

            compose_clg(
                self.disambiguation_symbols_int_path,
                out_disambig,
                context_width,
                central_pos,
                ilabels_temp,
                os.path.join(self.phones_dir, "phone_lm.fst"),
                clg_path,
                log_file,
            )
            log_file.write("Generating HCLGa.fst...")
            compose_hclg(
                self.model_path,
                ilabels_temp,
                self.hclg_options["transition_scale"],
                clg_path,
                hclga_path,
                log_file,
            )
            log_file.write("Generating HCLG.fst...")
            self_loop_proc = subprocess.Popen(
                [
                    thirdparty_binary("add-self-loops"),
                    f"--self-loop-scale={self.hclg_options['self_loop_scale']}",
                    "--reorder=true",
                    self.model_path,
                    hclga_path,
                ],
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            convert_proc = subprocess.Popen(
                [
                    thirdparty_binary("fstconvert"),
                    "--v=100",
                    "--fst_type=const",
                    "-",
                    hclg_path,
                ],
                stdin=self_loop_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            convert_proc.communicate()

    def transcribe(self, workflow_type: WorkflowType = WorkflowType.transcription):
        self.initialize_database()
        previous_working_directory = self.working_directory
        self.create_new_current_workflow(workflow_type)
        if workflow_type is WorkflowType.phone_transcription:
            self.setup_phone_lm()
            for a in self.calc_fmllr_arguments():
                for p in a.trans_paths.values():
                    shutil.copyfile(
                        p.replace(self.working_directory, previous_working_directory), p
                    )
        elif workflow_type is WorkflowType.per_speaker_transcription:
            for a in self.calc_fmllr_arguments():
                for p in a.trans_paths.values():
                    if os.path.exists(p):
                        shutil.copyfile(
                            p.replace(self.working_directory, previous_working_directory), p
                        )
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
                done = True
                for a in self.carpa_lm_rescore_arguments():
                    for p in a.rescored_lat_paths.values():
                        if not os.path.exists(p):
                            done = False
                            break
                if done:
                    logger.info("Rescoring already done.")
                else:
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

    def save_transcription_evaluation(self, output_directory: str) -> None:
        """
        Save transcription evaluation to an output directory

        Parameters
        ----------
        output_directory: str
            Directory to save evaluation
        """
        output_path = os.path.join(output_directory, "transcription_evaluation.csv")
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

            with mp.Pool(GLOBAL_CONFIG.num_jobs) as pool:
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
            "acoustic_scale": self.acoustic_scale,
            "silence_weight": self.silence_weight,
            "lattice_beam": self.lattice_beam,
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
        with tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
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
            for _, log_likelihood, _ in run_kaldi_function(
                decode_function, arguments, pbar.update
            ):
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
        sum_errors = 0
        with tqdm.tqdm(total=self.num_speakers, disable=GLOBAL_CONFIG.quiet) as pbar:
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(self.initial_fmllr_arguments()):
                    function = InitialFmllrFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
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
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(1)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in self.initial_fmllr_arguments():
                    function = InitialFmllrFunction(args)
                    for _ in function.run():
                        pbar.update(1)
            if sum_errors:
                logger.warning(f"{sum_errors} utterances had errors on calculating fMLLR.")

    def lat_gen_fmllr(self) -> None:
        """
        Generate lattice with fMLLR transforms

        See Also
        -------
        :class:`~montreal_forced_aligner.transcription.multiprocessing.LatGenFmllrFunction`
            Multiprocessing function
        :meth:`.TranscriberMixin.lat_gen_fmllr_arguments`
            Arguments for function
        """
        logger.info("Regenerating lattices with fMLLR transforms...")
        workflow = self.current_workflow
        arguments = self.lat_gen_fmllr_arguments(workflow.workflow_type)
        with tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar, mfa_open(
            os.path.join(self.working_log_directory, "lat_gen_fmllr_log_like.csv"),
            "w",
            encoding="utf8",
        ) as log_file:
            log_file.write("utterance,log_likelihood,num_frames\n")
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = LatGenFmllrFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
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
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(1)
                    utterance, log_likelihood, num_frames = result
                    log_file.write(f"{utterance},{log_likelihood},{num_frames}\n")
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in arguments:
                    function = LatGenFmllrFunction(args)
                    for utterance, log_likelihood, num_frames in function.run():
                        log_file.write(f"{utterance},{log_likelihood},{num_frames}\n")
                        pbar.update(1)

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
        sum_errors = 0
        with tqdm.tqdm(total=self.num_speakers, disable=GLOBAL_CONFIG.quiet) as pbar:
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(self.final_fmllr_arguments()):
                    function = FinalFmllrFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
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
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(1)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in self.final_fmllr_arguments():
                    function = FinalFmllrFunction(args)
                    for _ in function.run():
                        pbar.update(1)
            if sum_errors:
                logger.warning(f"{sum_errors} utterances had errors on calculating fMLLR.")

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
        sum_errors = 0
        with tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(self.fmllr_rescore_arguments()):
                    function = FmllrRescoreFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
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
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    done, errors = result
                    sum_errors += errors
                    pbar.update(done + errors)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in self.fmllr_rescore_arguments():
                    function = FmllrRescoreFunction(args)
                    for done, errors in function.run():
                        sum_errors += errors
                        pbar.update(done + errors)
            if sum_errors:
                logger.warning(f"{errors} utterances had errors on calculating fMLLR.")

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
        workflow = self.current_workflow
        self.calc_initial_fmllr()
        self.uses_speaker_adaptation = True
        self.lat_gen_fmllr()
        self.calc_final_fmllr()
        for decode_args in self.decode_arguments(workflow.workflow_type):
            for lat_path in decode_args.lat_paths.values():
                os.remove(lat_path)

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
        for j in self.jobs:
            feat_strings = {}
            for d_id in j.dictionary_ids:
                feat_strings[d_id] = j.construct_feature_proc_string(
                    self.working_directory,
                    d_id,
                    self.feature_options["uses_splices"],
                    self.feature_options["splice_left_context"],
                    self.feature_options["splice_right_context"],
                    self.feature_options["uses_speaker_adaptation"],
                )
            if workflow is WorkflowType.per_speaker_transcription:
                arguments.append(
                    PerSpeakerDecodeArguments(
                        j.id,
                        getattr(self, "db_string", ""),
                        os.path.join(self.working_log_directory, f"per_speaker_decode.{j.id}.log"),
                        self.model_directory,
                        feat_strings,
                        j.construct_path_dictionary(self.working_directory, "lat", "ark"),
                        self.model_path,
                        self.disambiguation_symbols_int_path,
                        self.decode_options,
                        self.tree_path,
                        self.order,
                        self.method,
                    )
                )
            elif workflow is WorkflowType.phone_transcription:
                arguments.append(
                    DecodePhoneArguments(
                        j.id,
                        getattr(self, "db_string", ""),
                        os.path.join(self.working_log_directory, f"decode.{j.id}.log"),
                        j.dictionary_ids,
                        feat_strings,
                        self.decode_options,
                        self.alignment_model_path,
                        j.construct_path_dictionary(self.working_directory, "lat", "ark"),
                        self.phone_symbol_table_path,
                        os.path.join(self.working_directory, "HCLG_phone.fst"),
                    )
                )
            else:
                arguments.append(
                    DecodeArguments(
                        j.id,
                        getattr(self, "db_string", ""),
                        os.path.join(self.working_log_directory, f"decode.{j.id}.log"),
                        j.dictionary_ids,
                        feat_strings,
                        self.decode_options,
                        self.alignment_model_path,
                        j.construct_path_dictionary(self.working_directory, "lat", "ark"),
                        j.construct_dictionary_dependent_paths(
                            self.model_directory, "words", "txt"
                        ),
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
                getattr(self, "db_string", ""),
                os.path.join(self.working_log_directory, f"lm_rescore.{j.id}.log"),
                j.dictionary_ids,
                self.lm_rescore_options,
                j.construct_path_dictionary(self.working_directory, "lat", "ark"),
                j.construct_path_dictionary(self.working_directory, "lat.rescored", "ark"),
                j.construct_dictionary_dependent_paths(self.model_directory, "G.small", "fst"),
                j.construct_dictionary_dependent_paths(self.model_directory, "G.med", "fst"),
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
                getattr(self, "db_string", ""),
                os.path.join(self.working_log_directory, f"carpa_lm_rescore.{j.id}.log"),
                j.dictionary_ids,
                j.construct_path_dictionary(self.working_directory, "lat.rescored", "ark"),
                j.construct_path_dictionary(self.working_directory, "lat.carpa.rescored", "ark"),
                j.construct_dictionary_dependent_paths(self.model_directory, "G.med", "fst"),
                j.construct_dictionary_dependent_paths(self.model_directory, "G", "carpa"),
            )
            for j in self.jobs
        ]

    @property
    def fmllr_options(self) -> MetaDict:
        """Options for calculating fMLLR"""
        options = super().fmllr_options
        options["acoustic_scale"] = self.acoustic_scale
        options["sil_phones"] = self.silence_csl
        options["lattice_beam"] = self.lattice_beam
        return options

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
            feat_strings = {}
            for d_id in j.dictionary_ids:
                feat_strings[d_id] = j.construct_feature_proc_string(
                    self.working_directory,
                    d_id,
                    self.feature_options["uses_splices"],
                    self.feature_options["splice_left_context"],
                    self.feature_options["splice_right_context"],
                    self.feature_options["uses_speaker_adaptation"],
                )
            arguments.append(
                InitialFmllrArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(self.working_log_directory, f"initial_fmllr.{j.id}.log"),
                    j.dictionary_ids,
                    feat_strings,
                    self.model_path,
                    self.fmllr_options,
                    j.construct_path_dictionary(self.working_directory, "trans", "ark"),
                    j.construct_path_dictionary(self.working_directory, "lat", "ark"),
                    j.construct_path_dictionary(self.data_directory, "spk2utt", "scp"),
                )
            )
        return arguments

    def lat_gen_fmllr_arguments(
        self, workflow: WorkflowType = WorkflowType.transcription
    ) -> List[LatGenFmllrArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.LatGenFmllrFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.LatGenFmllrArguments`]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            feat_strings = {}
            word_paths = {}
            hclg_paths = {}
            if workflow is not WorkflowType.phone_transcription:
                for d in j.dictionaries:
                    word_paths[d.id] = d.words_symbol_path
                    hclg_paths[d.id] = os.path.join(self.model_directory, f"HCLG.{d.id}.fst")

                    feat_strings[d.id] = j.construct_feature_proc_string(
                        self.working_directory,
                        d.id,
                        self.feature_options["uses_splices"],
                        self.feature_options["splice_left_context"],
                        self.feature_options["splice_right_context"],
                        self.feature_options["uses_speaker_adaptation"],
                    )
            else:
                hclg_paths = os.path.join(self.working_directory, "HCLG_phone.fst")
                word_paths = self.phone_symbol_table_path

                feat_strings = j.construct_feature_proc_string(
                    self.working_directory,
                    None,
                    self.feature_options["uses_splices"],
                    self.feature_options["splice_left_context"],
                    self.feature_options["splice_right_context"],
                    self.feature_options["uses_speaker_adaptation"],
                )

            arguments.append(
                LatGenFmllrArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(self.working_log_directory, f"lat_gen_fmllr.{j.id}.log"),
                    j.dictionary_ids,
                    feat_strings,
                    self.model_path,
                    self.decode_options,
                    word_paths,
                    hclg_paths,
                    j.construct_path_dictionary(self.working_directory, "lat.tmp", "ark"),
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
            feat_strings = {}
            for d_id in j.dictionary_ids:
                feat_strings[d_id] = j.construct_feature_proc_string(
                    self.working_directory,
                    d_id,
                    self.feature_options["uses_splices"],
                    self.feature_options["splice_left_context"],
                    self.feature_options["splice_right_context"],
                    self.feature_options["uses_speaker_adaptation"],
                )
            arguments.append(
                FinalFmllrArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(self.working_log_directory, f"final_fmllr.{j.id}.log"),
                    j.dictionary_ids,
                    feat_strings,
                    self.model_path,
                    self.fmllr_options,
                    j.construct_path_dictionary(self.working_directory, "trans", "ark"),
                    j.construct_path_dictionary(self.data_directory, "spk2utt", "scp"),
                    j.construct_path_dictionary(self.working_directory, "lat.tmp", "ark"),
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
            feat_strings = {}
            for d_id in j.dictionary_ids:
                feat_strings[d_id] = j.construct_feature_proc_string(
                    self.working_directory,
                    d_id,
                    self.feature_options["uses_splices"],
                    self.feature_options["splice_left_context"],
                    self.feature_options["splice_right_context"],
                    self.feature_options["uses_speaker_adaptation"],
                )
            arguments.append(
                FmllrRescoreArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(self.working_log_directory, f"fmllr_rescore.{j.id}.log"),
                    j.dictionary_ids,
                    feat_strings,
                    self.model_path,
                    self.fmllr_options,
                    j.construct_path_dictionary(self.working_directory, "lat.tmp", "ark"),
                    j.construct_path_dictionary(self.working_directory, "lat", "ark"),
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
        acoustic_model_path: str,
        language_model_path: str,
        output_type: str = "transcription",
        **kwargs,
    ):
        self.acoustic_model = AcousticModel(acoustic_model_path)
        kwargs.update(self.acoustic_model.parameters)
        super(Transcriber, self).__init__(**kwargs)
        self.language_model = LanguageModel(language_model_path)
        self.output_type = output_type
        self.ignore_empty_utterances = False

    def create_hclgs_arguments(self) -> Dict[int, CreateHclgArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.CreateHclgFunction`

        Returns
        -------
        dict[str, :class:`~montreal_forced_aligner.transcription.multiprocessing.CreateHclgArguments`]
            Per dictionary arguments for HCLG
        """
        args = {}
        with self.session() as session:
            for d in session.query(Dictionary):
                args[d.id] = CreateHclgArguments(
                    d.id,
                    getattr(self, "db_string", ""),
                    os.path.join(self.model_directory, "log", f"hclg.{d.id}.log"),
                    self.model_directory,
                    os.path.join(self.model_directory, f"{{file_name}}.{d.id}.fst"),
                    os.path.join(self.model_directory, f"words.{d.id}.txt"),
                    os.path.join(self.model_directory, f"G.{d.id}.carpa"),
                    self.language_model.small_arpa_path,
                    self.language_model.medium_arpa_path,
                    self.language_model.carpa_path,
                    self.model_path,
                    d.lexicon_disambig_fst_path,
                    d.disambiguation_symbols_int_path,
                    self.hclg_options,
                    self.word_mapping(d.id),
                )
        return args

    def create_hclgs(self) -> None:
        """
        Create HCLG.fst files for every dictionary being used by a
        :class:`~montreal_forced_aligner.transcription.transcriber.Transcriber`
        """

        dict_arguments = self.create_hclgs_arguments()
        dict_arguments = list(dict_arguments.values())
        logger.info("Generating HCLG.fst...")
        if GLOBAL_CONFIG.use_mp:
            error_dict = {}
            return_queue = mp.Queue()
            stopped = Stopped()
            procs = []
            for i, args in enumerate(dict_arguments):
                function = CreateHclgFunction(args)
                p = KaldiProcessWorker(i, return_queue, function, stopped)
                procs.append(p)
                p.start()
            with tqdm.tqdm(total=len(dict_arguments) * 7, disable=GLOBAL_CONFIG.quiet) as pbar:
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        elif not isinstance(result, tuple):
                            pbar.update(1)
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    result, hclg_path = result
                    if result:
                        logger.debug(f"Done generating {hclg_path}!")
                    else:
                        logger.warning(f"There was an error in generating {hclg_path}")
                    pbar.update(1)
            for p in procs:
                p.join()
            if error_dict:
                for v in error_dict.values():
                    raise v
        else:
            for args in dict_arguments:
                function = CreateHclgFunction(args)
                with tqdm.tqdm(total=len(dict_arguments), disable=GLOBAL_CONFIG.quiet) as pbar:
                    for result in function.run():
                        if not isinstance(result, tuple):
                            pbar.update(1)
                            continue
                        result, hclg_path = result
                        if result:
                            logger.debug(f"Done generating {hclg_path}!")
                        else:
                            logger.warning(f"There was an error in generating {hclg_path}")
                        pbar.update(1)
        error_logs = []
        for arg in dict_arguments:
            if not os.path.exists(arg.hclg_path):
                error_logs.append(arg.log_path)
        if error_logs:
            raise KaldiProcessingError(error_logs)

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
            mod_path = os.path.join(self.model_directory, "base_lm.mod")
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
                small_mod_path = mod_path.replace(".mod", "_small.mod")
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
                med_mod_path = mod_path.replace(".mod", "_med.mod")
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
        config_path: Optional[str] = None,
        args: Optional[Dict[str, typing.Any]] = None,
        unknown_args: Optional[typing.Iterable[str]] = None,
    ) -> MetaDict:
        """
        Parse configuration parameters from a config file and command line arguments

        Parameters
        ----------
        config_path: str, optional
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

    def setup(self) -> None:
        """Set up transcription"""
        super().setup()
        if self.initialized:
            return
        self.create_new_current_workflow(WorkflowType.transcription)
        begin = time.time()
        os.makedirs(self.working_log_directory, exist_ok=True)
        self.load_corpus()
        dirty_path = os.path.join(self.working_directory, "dirty")
        if os.path.exists(dirty_path):
            shutil.rmtree(self.working_directory, ignore_errors=True)
        os.makedirs(self.working_log_directory, exist_ok=True)
        dirty_path = os.path.join(self.model_directory, "dirty")

        if os.path.exists(dirty_path):  # if there was an error, let's redo from scratch
            shutil.rmtree(self.model_directory)
        log_dir = os.path.join(self.model_directory, "log")
        os.makedirs(log_dir, exist_ok=True)
        if self.acoustic_model.meta["version"] < "2.1":
            logger.warning(
                "The acoustic model was trained in an earlier version of MFA. "
                "There may be incompatibilities in feature generation that cause errors. "
                "Please download the latest version of the model via `mfa model download`, "
                "use a different acoustic model, or use version 2.0.6 of MFA."
            )
        self.acoustic_model.validate(self)
        self.acoustic_model.export_model(self.model_directory)
        self.acoustic_model.export_model(self.working_directory)
        self.acoustic_model.log_details()
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
        output_directory: str,
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
