"""Class definitions for corpora"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
import typing
from abc import ABCMeta
from queue import Empty
from typing import List, Optional

import sqlalchemy
import tqdm

from montreal_forced_aligner.abc import MfaWorker
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.corpus.base import CorpusMixin
from montreal_forced_aligner.corpus.classes import FileData
from montreal_forced_aligner.corpus.features import (
    CalcFmllrArguments,
    CalcFmllrFunction,
    ComputeVadFunction,
    FeatureConfigMixin,
    FinalFeatureArguments,
    FinalFeatureFunction,
    MfccArguments,
    MfccFunction,
    PitchRangeArguments,
    PitchRangeFunction,
    VadArguments,
)
from montreal_forced_aligner.corpus.helper import find_exts
from montreal_forced_aligner.corpus.multiprocessing import (
    AcousticDirectoryParser,
    CorpusProcessWorker,
    ExportKaldiFilesArguments,
    ExportKaldiFilesFunction,
)
from montreal_forced_aligner.data import DatabaseImportData, PhoneType, WorkflowType
from montreal_forced_aligner.db import (
    Corpus,
    CorpusWorkflow,
    File,
    Job,
    Phone,
    PhoneInterval,
    SoundFile,
    Speaker,
    TextFile,
    Utterance,
    bulk_update,
)
from montreal_forced_aligner.dictionary.mixins import DictionaryMixin
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionaryMixin
from montreal_forced_aligner.exceptions import (
    FeatureGenerationError,
    KaldiProcessingError,
    SoundFileError,
    TextGridParseError,
    TextParseError,
)
from montreal_forced_aligner.helper import load_scp, mfa_open
from montreal_forced_aligner.textgrid import parse_aligned_textgrid
from montreal_forced_aligner.utils import (
    Counter,
    KaldiProcessWorker,
    Stopped,
    run_kaldi_function,
    thirdparty_binary,
)

__all__ = [
    "AcousticCorpusMixin",
    "AcousticCorpus",
    "AcousticCorpusWithPronunciations",
    "AcousticCorpusPronunciationMixin",
]

logger = logging.getLogger("mfa")


class AcousticCorpusMixin(CorpusMixin, FeatureConfigMixin, metaclass=ABCMeta):
    """
    Mixin class for acoustic corpora

    Parameters
    ----------
    audio_directory: str
        Extra directory to look for audio files

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.base.CorpusMixin`
        For corpus parsing parameters
    :class:`~montreal_forced_aligner.corpus.features.FeatureConfigMixin`
        For feature generation parameters

    Attributes
    ----------
    sound_file_errors: list[str]
        List of sound files with errors in loading
    stopped: Stopped
        Stop check for loading the corpus
    """

    def __init__(self, audio_directory: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.audio_directory = audio_directory
        self.sound_file_errors = []
        self.stopped = Stopped()
        self.features_generated = False
        self.transcription_done = False
        self.alignment_evaluation_done = False

    def has_alignments(self, workflow_id: typing.Optional[int] = None):
        with self.session() as session:
            if workflow_id is None:
                check = session.query(PhoneInterval).limit(1).first() is not None
            else:
                if isinstance(workflow_id, int):
                    check = (
                        session.query(CorpusWorkflow.alignments_collected)
                        .filter(CorpusWorkflow.id == workflow_id)
                        .scalar()
                    )
                else:
                    check = (
                        session.query(CorpusWorkflow.alignments_collected)
                        .filter(CorpusWorkflow.workflow_type == workflow_id)
                        .scalar()
                    )
        return check

    def has_ivectors(self):
        with self.session() as session:
            check = (
                session.query(Corpus)
                .filter(Corpus.ivectors_calculated == True)  # noqa
                .limit(1)
                .first()
                is not None
            )
        return check

    def has_xvectors(self):
        with self.session() as session:
            check = (
                session.query(Corpus)
                .filter(Corpus.xvectors_loaded == True)  # noqa
                .limit(1)
                .first()
                is not None
            )
        return check

    def has_any_ivectors(self):
        with self.session() as session:
            check = (
                session.query(Corpus)
                .filter(
                    sqlalchemy.or_(
                        Corpus.ivectors_calculated == True, Corpus.xvectors_loaded == True  # noqa
                    )
                )
                .limit(1)
                .first()
                is not None
            )
        return check

    @property
    def no_transcription_files(self) -> List[str]:
        """List of sound files without text files"""
        with self.session() as session:
            files = session.query(SoundFile.sound_file_path).filter(
                ~sqlalchemy.exists().where(SoundFile.file_id == TextFile.file_id)
            )
            return [x[0] for x in files]

    @property
    def transcriptions_without_wavs(self) -> List[str]:
        """List of text files without sound files"""
        with self.session() as session:
            files = session.query(TextFile.text_file_path).filter(
                ~sqlalchemy.exists().where(SoundFile.file_id == TextFile.file_id)
            )
            return [x[0] for x in files]

    def inspect_database(self) -> None:
        """Check if a database file exists and create the necessary metadata"""
        self.initialize_database()
        with self.session() as session:
            corpus = session.query(Corpus).first()
            if corpus:
                self.imported = corpus.imported
                self.features_generated = corpus.features_generated
                self.text_normalized = corpus.text_normalized
            else:
                session.add(
                    Corpus(
                        name=self.data_source_identifier,
                        path=self.corpus_directory,
                        data_directory=self.corpus_output_directory,
                    )
                )
                session.commit()

    def load_reference_alignments(self, reference_directory: str) -> None:
        """
        Load reference alignments to use in alignment evaluation from a directory

        Parameters
        ----------
        reference_directory: str
            Directory containing reference alignments

        """
        self.create_new_current_workflow(WorkflowType.reference)
        workflow = self.current_workflow
        if workflow.alignments_collected:
            logger.info("Reference alignments already loaded!")
            return
        logger.info("Loading reference files...")
        indices = []
        jobs = []
        reference_intervals = []
        with tqdm.tqdm(
            total=self.num_files, disable=GLOBAL_CONFIG.quiet
        ) as pbar, self.session() as session:
            phone_mapping = {}
            max_id = 0
            interval_id = session.query(sqlalchemy.func.max(PhoneInterval.id)).scalar()
            if not interval_id:
                interval_id = 0
            interval_id += 1
            for p, p_id in session.query(Phone.phone, Phone.id):
                phone_mapping[p] = p_id
                if p_id > max_id:
                    max_id = p_id
            new_phones = []
            for root, _, files in os.walk(reference_directory, followlinks=True):
                root_speaker = os.path.basename(root)
                for f in files:
                    if f.endswith(".TextGrid"):
                        file_name = f.replace(".TextGrid", "")
                        file_id = session.query(File.id).filter_by(name=file_name).scalar()
                        if not file_id:
                            continue
                        if GLOBAL_CONFIG.use_mp:
                            indices.append(file_id)
                            jobs.append((os.path.join(root, f), root_speaker))
                        else:
                            intervals = parse_aligned_textgrid(os.path.join(root, f), root_speaker)
                            utterances = (
                                session.query(
                                    Utterance.id, Speaker.name, Utterance.begin, Utterance.end
                                )
                                .join(Utterance.speaker)
                                .filter(Utterance.file_id == file_id)
                                .order_by(Utterance.begin)
                            )
                            for u_id, speaker_name, begin, end in utterances:
                                if speaker_name not in intervals:
                                    continue
                                while intervals[speaker_name]:
                                    interval = intervals[speaker_name].pop(0)
                                    dur = interval.end - interval.begin
                                    mid_point = interval.begin + (dur / 2)
                                    if begin <= mid_point <= end:
                                        if interval.label not in phone_mapping:
                                            max_id += 1
                                            phone_mapping[interval.label] = max_id
                                            new_phones.append(
                                                {
                                                    "id": max_id,
                                                    "mapping_id": max_id - 1,
                                                    "phone": interval.label,
                                                    "kaldi_label": interval.label,
                                                    "phone_type": PhoneType.extra,
                                                }
                                            )
                                        reference_intervals.append(
                                            {
                                                "id": interval_id,
                                                "begin": interval.begin,
                                                "end": interval.end,
                                                "phone_id": phone_mapping[interval.label],
                                                "utterance_id": u_id,
                                                "workflow_id": workflow.id,
                                            }
                                        )
                                        interval_id += 1
                                    if mid_point > end:
                                        intervals[speaker_name].insert(0, interval)
                                        break

                            pbar.update(1)

            if GLOBAL_CONFIG.use_mp:
                with mp.Pool(GLOBAL_CONFIG.num_jobs) as pool:
                    gen = pool.starmap(parse_aligned_textgrid, jobs)
                    for i, intervals in enumerate(gen):
                        pbar.update(1)
                        file_id = indices[i]
                        utterances = (
                            session.query(
                                Utterance.id, Speaker.name, Utterance.begin, Utterance.end
                            )
                            .join(Utterance.speaker)
                            .filter(Utterance.file_id == file_id)
                            .order_by(Utterance.begin)
                        )
                        for u_id, speaker_name, begin, end in utterances:
                            if speaker_name not in intervals:
                                continue
                            while intervals[speaker_name]:
                                interval = intervals[speaker_name].pop(0)
                                dur = interval.end - interval.begin
                                mid_point = interval.begin + (dur / 2)
                                if begin <= mid_point <= end:
                                    if interval.label not in phone_mapping:
                                        max_id += 1
                                        phone_mapping[interval.label] = max_id
                                        new_phones.append(
                                            {
                                                "id": max_id,
                                                "mapping_id": max_id - 1,
                                                "phone": interval.label,
                                                "kaldi_label": interval.label,
                                                "phone_type": PhoneType.extra,
                                            }
                                        )
                                    reference_intervals.append(
                                        {
                                            "id": interval_id,
                                            "begin": interval.begin,
                                            "end": interval.end,
                                            "phone_id": phone_mapping[interval.label],
                                            "utterance_id": u_id,
                                            "workflow_id": workflow.id,
                                        }
                                    )
                                    interval_id += 1
                                if mid_point > end:
                                    intervals[speaker_name].insert(0, interval)
                                    break
            if new_phones:
                session.execute(sqlalchemy.insert(Phone.__table__), new_phones)
                session.commit()
            session.execute(sqlalchemy.insert(PhoneInterval.__table__), reference_intervals)
            session.query(CorpusWorkflow).filter(CorpusWorkflow.id == workflow.id).update(
                {CorpusWorkflow.done: True, CorpusWorkflow.alignments_collected: True}
            )
            session.commit()

    def load_corpus(self) -> None:
        """
        Load the corpus
        """
        self.initialize_database()
        self._load_corpus()
        self._create_dummy_dictionary()
        self.initialize_jobs()
        self.normalize_text()
        self.create_corpus_split()
        self.generate_features()

    def generate_final_features(self) -> None:
        """
        Generate features for the corpus

        Parameters
        ----------
        compute_cmvn: bool
            Flag for whether to compute CMVN, defaults to True
        voiced_only: bool
            Flag for whether to select only voiced frames, defaults to False
        """
        logger.info("Generating final features...")
        time_begin = time.time()
        log_directory = os.path.join(self.split_directory, "log")
        os.makedirs(log_directory, exist_ok=True)
        arguments = self.final_feature_arguments()
        with tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            for _ in run_kaldi_function(FinalFeatureFunction, arguments, pbar.update):
                pass
        with self.session() as session:
            update_mapping = {}
            session.query(Utterance).update({"ignored": True})
            session.commit()
            for j in self.jobs:
                with mfa_open(j.feats_scp_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line == "":
                            continue
                        f = line.split(maxsplit=1)
                        utt_id = int(f[0].split("-")[-1])
                        feats = f[1]
                        update_mapping[utt_id] = {
                            "id": utt_id,
                            "features": feats,
                            "ignored": False,
                        }

            bulk_update(session, Utterance, list(update_mapping.values()))
            session.commit()

        with self.session() as session:
            ignored_utterances = (
                session.query(
                    SoundFile.sound_file_path,
                    Speaker.name,
                    Utterance.begin,
                    Utterance.end,
                    Utterance.text,
                )
                .join(Utterance.speaker)
                .join(Utterance.file)
                .join(File.sound_file)
                .filter(Utterance.ignored == True)  # noqa
            )
            ignored_count = 0
            for sound_file_path, speaker_name, begin, end, text in ignored_utterances:
                logger.debug(f"  - Ignored File: {sound_file_path}")
                logger.debug(f"    - Speaker: {speaker_name}")
                logger.debug(f"    - Begin: {begin}")
                logger.debug(f"    - End: {end}")
                logger.debug(f"    - Text: {text}")
                ignored_count += 1
            if ignored_count:
                logger.warning(
                    f"There were {ignored_count} utterances ignored due to an issue in feature generation, see the log file for full "
                    "details or run `mfa validate` on the corpus."
                )
        logger.debug(f"Generating final features took {time.time() - time_begin:.3f} seconds")

    def generate_features(self) -> None:
        """
        Generate features for the corpus

        Parameters
        ----------
        compute_cmvn: bool
            Flag for whether to compute CMVN, defaults to True
        voiced_only: bool
            Flag for whether to select only voiced frames, defaults to False
        """
        with self.session() as session:
            final_features_check = session.query(Corpus).first().features_generated
            if final_features_check:
                self.features_generated = True
                logger.info("Features already generated.")
                return
            feature_check = (
                session.query(Utterance).filter(Utterance.features != None).first()  # noqa
                is not None
            )
        if self.feature_type == "mfcc" and not feature_check:
            self.mfcc()
        self.combine_feats()
        if self.uses_cmvn:
            logger.info("Calculating CMVN...")
            self.calc_cmvn()
        if self.uses_voiced:
            self.compute_vad()
        self.generate_final_features()
        self._write_feats()
        self.features_generated = True
        with self.session() as session:
            session.query(Corpus).update({"features_generated": True})
            session.commit()
        self.create_corpus_split()

    def create_corpus_split(self) -> None:
        """Create the split directory for the corpus"""
        with self.session() as session:
            c = session.query(Corpus).first()
            c.current_subset = 0
            session.commit()
        if self.features_generated:
            logger.info("Creating corpus split with features...")
            super().create_corpus_split()
        else:
            logger.info("Creating corpus split for feature generation...")
            os.makedirs(os.path.join(self.split_directory, "log"), exist_ok=True)
            with self.session() as session, tqdm.tqdm(
                total=self.num_utterances + self.num_files, disable=GLOBAL_CONFIG.quiet
            ) as pbar:
                jobs = session.query(Job)
                arguments = [
                    ExportKaldiFilesArguments(
                        j.id, self.db_string, None, self.split_directory, True
                    )
                    for j in jobs
                ]

                for _ in run_kaldi_function(ExportKaldiFilesFunction, arguments, pbar.update):
                    pass

    def compute_vad_arguments(self) -> List[VadArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.corpus.features.ComputeVadFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.corpus.features.VadArguments`]
            Arguments for processing
        """
        return [
            VadArguments(
                j.id,
                getattr(self, "db_string", ""),
                os.path.join(self.split_directory, "log", f"compute_vad.{j.id}.log"),
                j.construct_path(self.split_directory, "feats", "scp"),
                j.construct_path(self.split_directory, "vad", "scp"),
                self.vad_options,
            )
            for j in self.jobs
        ]

    def calc_fmllr_arguments(self, iteration: Optional[int] = None) -> List[CalcFmllrArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.corpus.features.CalcFmllrFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.corpus.features.CalcFmllrArguments`]
            Arguments for processing
        """
        base_log = "calc_fmllr"
        if iteration is not None:
            base_log += f".{iteration}"
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
                CalcFmllrArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(self.working_log_directory, f"{base_log}.{j.id}.log"),
                    j.dictionary_ids,
                    feat_strings,
                    j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                    self.alignment_model_path,
                    self.model_path,
                    j.construct_path_dictionary(self.data_directory, "spk2utt", "scp"),
                    j.construct_path_dictionary(self.working_directory, "trans", "ark"),
                    self.fmllr_options,
                )
            )
        return arguments

    def mfcc_arguments(self) -> List[MfccArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.corpus.features.MfccFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.corpus.features.MfccArguments`]
            Arguments for processing
        """
        return [
            MfccArguments(
                j.id,
                self.db_string,
                os.path.join(self.split_directory, "log", f"make_mfcc.{j.id}.log"),
                self.split_directory,
                self.mfcc_options,
                self.pitch_options,
            )
            for j in self.jobs
        ]

    def final_feature_arguments(self) -> List[FinalFeatureArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.corpus.features.MfccFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.corpus.features.MfccArguments`]
            Arguments for processing
        """
        return [
            FinalFeatureArguments(
                j.id,
                self.db_string,
                os.path.join(self.split_directory, "log", f"generate_final_features.{j.id}.log"),
                self.split_directory,
                self.uses_cmvn,
                self.uses_voiced,
                getattr(self, "subsample", None),
            )
            for j in self.jobs
        ]

    def pitch_range_arguments(self) -> List[PitchRangeArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.corpus.features.MfccFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.corpus.features.MfccArguments`]
            Arguments for processing
        """
        return [
            PitchRangeArguments(
                j.id,
                self.db_string,
                os.path.join(self.split_directory, "log", f"compute_pitch_range.{j.id}.log"),
                self.split_directory,
                self.pitch_options,
            )
            for j in self.jobs
        ]

    def compute_speaker_pitch_ranges(self):
        logger.info("Calculating per-speaker f0 ranges...")
        log_directory = os.path.join(self.split_directory, "log")
        os.makedirs(log_directory, exist_ok=True)
        arguments = self.pitch_range_arguments()
        update_mapping = []
        with tqdm.tqdm(total=self.num_speakers, disable=GLOBAL_CONFIG.quiet) as pbar:
            for speaker_id, min_f0, max_f0 in run_kaldi_function(
                PitchRangeFunction, arguments, pbar.update
            ):
                update_mapping.append({"id": speaker_id, "min_f0": min_f0, "max_f0": max_f0})
        with self.session() as session:
            bulk_update(session, Speaker, update_mapping)
            session.commit()

    def mfcc(self) -> None:
        """
        Multiprocessing function that converts sound files into MFCCs.

        See :kaldi_docs:`feat` for an overview on feature generation in Kaldi.

        See Also
        --------
        :class:`~montreal_forced_aligner.corpus.features.MfccFunction`
            Multiprocessing helper function for each job
        :meth:`.AcousticCorpusMixin.mfcc_arguments`
            Job method for generating arguments for helper function
        :kaldi_steps:`make_mfcc`
            Reference Kaldi script
        """
        logger.info("Generating MFCCs...")
        begin = time.time()
        log_directory = os.path.join(self.split_directory, "log")
        os.makedirs(log_directory, exist_ok=True)
        arguments = self.mfcc_arguments()
        with tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            for _ in run_kaldi_function(MfccFunction, arguments, pbar.update):
                pass
        logger.debug(f"Generating MFCCs took {time.time() - begin:.3f} seconds")

    def calc_cmvn(self) -> None:
        """
        Calculate CMVN statistics for speakers

        See Also
        --------
        :kaldi_src:`compute-cmvn-stats`
            Relevant Kaldi binary
        """
        self._write_spk2utt()
        spk2utt = os.path.join(self.corpus_output_directory, "spk2utt.scp")
        feats = os.path.join(self.corpus_output_directory, "feats.scp")
        cmvn_ark = os.path.join(self.corpus_output_directory, "cmvn.ark")
        cmvn_scp = os.path.join(self.corpus_output_directory, "cmvn.scp")
        log_path = os.path.join(self.features_log_directory, "cmvn.log")
        with mfa_open(log_path, "w") as logf:
            subprocess.call(
                [
                    thirdparty_binary("compute-cmvn-stats"),
                    f"--spk2utt=ark:{spk2utt}",
                    f"scp:{feats}",
                    f"ark,scp:{cmvn_ark},{cmvn_scp}",
                ],
                stderr=logf,
                env=os.environ,
            )
        update_mapping = []
        with self.session() as session:
            for s, cmvn in load_scp(cmvn_scp).items():
                if isinstance(cmvn, list):
                    cmvn = " ".join(cmvn)
                update_mapping.append({"id": int(s), "cmvn": cmvn})
            bulk_update(session, Speaker, update_mapping)
            session.commit()

            for j in self.jobs:
                query = (
                    session.query(Speaker.id, Speaker.cmvn)
                    .join(Speaker.utterances)
                    .filter(Speaker.cmvn != None, Utterance.job_id == j.id)  # noqa
                    .distinct()
                )
                with mfa_open(j.construct_path(self.split_directory, "cmvn", ".scp"), "w") as f:
                    for s_id, cmvn in query:
                        f.write(f"{s_id} {cmvn}\n")

    def calc_fmllr(self, iteration: Optional[int] = None) -> None:
        """
        Multiprocessing function that computes speaker adaptation transforms via
        feature-space Maximum Likelihood Linear Regression (fMLLR).

        See Also
        --------
        :class:`~montreal_forced_aligner.corpus.features.CalcFmllrFunction`
            Multiprocessing helper function for each job
        :meth:`.AcousticCorpusMixin.calc_fmllr_arguments`
            Job method for generating arguments for the helper function
        :kaldi_steps:`align_fmllr`
            Reference Kaldi script
        :kaldi_steps:`train_sat`
            Reference Kaldi script
        """
        begin = time.time()
        logger.info("Calculating fMLLR for speaker adaptation...")

        arguments = self.calc_fmllr_arguments(iteration=iteration)
        with tqdm.tqdm(total=self.num_speakers, disable=GLOBAL_CONFIG.quiet) as pbar:
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = CalcFmllrFunction(args)
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
                for args in arguments:
                    function = CalcFmllrFunction(args)
                    for _ in function.run():
                        pbar.update(1)

        self.uses_speaker_adaptation = True
        logger.debug(f"Fmllr calculation took {time.time() - begin:.3f} seconds")

    def compute_vad(self) -> None:
        """
        Compute Voice Activity Detection features over the corpus

        See Also
        --------
        :class:`~montreal_forced_aligner.corpus.features.ComputeVadFunction`
            Multiprocessing helper function for each job
        :meth:`.AcousticCorpusMixin.compute_vad_arguments`
            Job method for generating arguments for helper function
        """
        with self.session() as session:
            c = session.query(Corpus).first()
            if c.vad_calculated:
                logger.info("VAD already computed, skipping!")
                return
        begin = time.time()
        logger.info("Computing VAD...")

        arguments = self.compute_vad_arguments()
        with tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = ComputeVadFunction(args)
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
                    done, no_feats, unvoiced = result
                    pbar.update(done + no_feats + unvoiced)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in arguments:
                    function = ComputeVadFunction(args)
                    for done, no_feats, unvoiced in function.run():
                        pbar.update(done + no_feats + unvoiced)
        vad_lines = []
        utterance_mapping = []
        for args in arguments:
            with mfa_open(args.vad_scp_path) as inf:
                for line in inf:
                    vad_lines.append(line)
                    utt_id, ark = line.strip().split(maxsplit=1)
                    utt_id = int(utt_id.split("-")[-1])
                    utterance_mapping.append({"id": utt_id, "vad_ark": ark})
        with self.session() as session:
            bulk_update(session, Utterance, utterance_mapping)
            session.query(Corpus).update({Corpus.vad_calculated: True})
            session.commit()
        with mfa_open(os.path.join(self.corpus_output_directory, "vad.scp"), "w") as outf:
            for line in sorted(vad_lines, key=lambda x: x.split(maxsplit=1)[0]):
                outf.write(line)
        logger.debug(f"VAD computation took {time.time() - begin:.3f} seconds")

    def combine_feats(self) -> None:
        """
        Combine feature generation results and store relevant information
        """
        lines = []
        for j in self.jobs:
            with mfa_open(j.feats_scp_path) as f:
                for line in f:
                    lines.append(line)
        with open(
            os.path.join(self.corpus_output_directory, "feats.scp"), "w", encoding="utf8"
        ) as f:
            for line in sorted(lines):
                f.write(line)

    def _write_feats(self) -> None:
        """Write feats scp file for Kaldi"""
        with self.session() as session, open(
            os.path.join(self.corpus_output_directory, "feats.scp"), "w", encoding="utf8"
        ) as f:
            utterances = (
                session.query(Utterance.kaldi_id, Utterance.features)
                .filter_by(ignored=False)
                .order_by(Utterance.kaldi_id)
            )
            for u_id, features in utterances:

                f.write(f"{u_id} {features}\n")

    def get_feat_dim(self) -> int:
        """
        Calculate the feature dimension for the corpus

        Returns
        -------
        int
            Dimension of feature vectors
        """
        job = self.jobs[0]
        dict_id = None
        log_path = os.path.join(self.features_log_directory, "feat-to-dim.log")
        if job.dictionary_ids:
            dict_id = self.jobs[0].dictionary_ids[0]
        feature_string = job.construct_feature_proc_string(
            self.working_directory,
            dict_id,
            self.feature_options["uses_splices"],
            self.feature_options["splice_left_context"],
            self.feature_options["splice_right_context"],
            self.feature_options["uses_speaker_adaptation"],
        )
        with mfa_open(log_path, "w") as log_file:
            subset_ark_path = os.path.join(self.split_directory, "temp.ark")
            subset_proc = subprocess.Popen(
                [
                    thirdparty_binary("subset-feats"),
                    "--n=1",
                    feature_string,
                    f"ark:{subset_ark_path}",
                ],
                stderr=log_file,
                env=os.environ,
            )
            subset_proc.wait()
            dim_proc = subprocess.Popen(
                [thirdparty_binary("feat-to-dim"), f"ark:{subset_ark_path}", "-"],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
                encoding="utf8",
            )
            feats = dim_proc.stdout.readline().strip()
            dim_proc.wait()
        if not feats:
            with mfa_open(log_path) as f:
                logged = f.read()
            raise FeatureGenerationError(logged)
        feats = int(feats)
        os.remove(subset_ark_path)
        return feats

    def _load_corpus_from_source_mp(self) -> None:
        """
        Load a corpus using multiprocessing
        """
        begin_time = time.process_time()
        job_queue = mp.Queue()
        return_queue = mp.Queue()
        finished_adding = Stopped()
        stopped = Stopped()
        file_counts = Counter()
        error_dict = {}
        procs = []
        parser = AcousticDirectoryParser(
            self.corpus_directory,
            job_queue,
            self.audio_directory,
            stopped,
            finished_adding,
            file_counts,
        )
        parser.start()
        for i in range(GLOBAL_CONFIG.num_jobs):
            p = CorpusProcessWorker(
                i,
                job_queue,
                return_queue,
                stopped,
                finished_adding,
                self.speaker_characters,
                self.sample_frequency,
            )
            procs.append(p)
            p.start()
        last_poll = time.time() - 30
        try:
            with self.session() as session, tqdm.tqdm(
                total=100, disable=GLOBAL_CONFIG.quiet
            ) as pbar:
                import_data = DatabaseImportData()
                while True:
                    try:
                        file = return_queue.get(timeout=1)
                        if isinstance(file, tuple):
                            error_type = file[0]
                            error = file[1]
                            if error_type == "error":
                                error_dict[error_type] = error
                            else:
                                if error_type not in error_dict:
                                    error_dict[error_type] = []
                                error_dict[error_type].append(error)
                            continue
                        if self.stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished_processing.stop_check():
                                break
                        else:
                            break
                        continue
                    if time.time() - last_poll > 5:
                        pbar.total = file_counts.value()
                        last_poll = time.time()
                    pbar.update(1)
                    import_data.add_objects(self.generate_import_objects(file))

                logger.debug(f"Processing queue: {time.process_time() - begin_time}")

                if "error" in error_dict:
                    session.rollback()
                    raise error_dict["error"]
                self._finalize_load(session, import_data)
            for k in ["sound_file_errors", "decode_error_files", "textgrid_read_errors"]:
                if hasattr(self, k):
                    if k in error_dict:
                        logger.info(
                            "There were some issues with files in the corpus. "
                            "Please look at the log file or run the validator for more information."
                        )
                        logger.debug(f"{k} showed {len(error_dict[k])} errors:")
                        if k in {"textgrid_read_errors", "sound_file_errors"}:
                            getattr(self, k).extend(error_dict[k])
                            for e in error_dict[k]:
                                logger.debug(f"{e.file_name}: {e.error}")
                        else:
                            logger.debug(", ".join(error_dict[k]))
                            setattr(self, k, error_dict[k])

        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                logger.info(
                    "Detected ctrl-c, please wait a moment while we clean everything up..."
                )
                self.stopped.set_sigint_source()
            self.stopped.stop()
            finished_adding.stop()
            while True:
                try:
                    _ = job_queue.get(timeout=1)
                    if self.stopped.stop_check():
                        continue
                except Empty:
                    for proc in procs:
                        if not proc.finished_processing.stop_check():
                            break
                    else:
                        break
                try:
                    _ = return_queue.get(timeout=1)
                    _ = job_queue.get(timeout=1)
                    if self.stopped.stop_check():
                        continue
                except Empty:
                    for proc in procs:
                        if not proc.finished_processing.stop_check():
                            break
                    else:
                        break
            raise
        finally:
            parser.join()
            for p in procs:
                p.join()
            if self.stopped.stop_check():
                logger.info(f"Stopped parsing early ({time.process_time() - begin_time} seconds)")
                if self.stopped.source():
                    sys.exit(0)
            else:
                logger.debug(
                    f"Parsed corpus directory with {GLOBAL_CONFIG.num_jobs} jobs in {time.process_time() - begin_time} seconds"
                )

    def _load_corpus_from_source(self) -> None:
        """
        Load a corpus without using multiprocessing
        """
        begin_time = time.time()

        all_sound_files = {}
        use_audio_directory = False
        if self.audio_directory and os.path.exists(self.audio_directory):
            use_audio_directory = True
            for root, _, files in os.walk(self.audio_directory, followlinks=True):
                if self.stopped.stop_check():
                    return
                exts = find_exts(files)
                exts.wav_files = {k: os.path.join(root, v) for k, v in exts.wav_files.items()}
                exts.other_audio_files = {
                    k: os.path.join(root, v) for k, v in exts.other_audio_files.items()
                }
                all_sound_files.update(exts.other_audio_files)
                all_sound_files.update(exts.wav_files)
        logger.debug(f"Walking through {self.corpus_directory}...")
        with self.session() as session:
            import_data = DatabaseImportData()
            for root, _, files in os.walk(self.corpus_directory, followlinks=True):
                exts = find_exts(files)
                relative_path = (
                    root.replace(str(self.corpus_directory), "").lstrip("/").lstrip("\\")
                )
                if self.stopped.stop_check():
                    return
                if not use_audio_directory:
                    all_sound_files = {}
                    wav_files = {k: os.path.join(root, v) for k, v in exts.wav_files.items()}
                    other_audio_files = {
                        k: os.path.join(root, v) for k, v in exts.other_audio_files.items()
                    }
                    all_sound_files.update(other_audio_files)
                    all_sound_files.update(wav_files)
                for file_name in exts.identifiers:

                    wav_path = None
                    transcription_path = None
                    if file_name in all_sound_files:
                        wav_path = all_sound_files[file_name]
                    if file_name in exts.lab_files:
                        lab_name = exts.lab_files[file_name]
                        transcription_path = os.path.join(root, lab_name)
                    elif file_name in exts.textgrid_files:
                        tg_name = exts.textgrid_files[file_name]
                        transcription_path = os.path.join(root, tg_name)
                    if wav_path is None:  # Not a file for MFA
                        continue
                    try:
                        file = FileData.parse_file(
                            file_name,
                            wav_path,
                            transcription_path,
                            relative_path,
                            self.speaker_characters,
                            self.sample_frequency,
                        )
                        import_data.add_objects(self.generate_import_objects(file))
                    except TextParseError as e:
                        self.decode_error_files.append(e)
                    except TextGridParseError as e:
                        self.textgrid_read_errors.append(e)
                    except SoundFileError as e:
                        self.sound_file_errors.append(e)
            self._finalize_load(session, import_data)
        if self.decode_error_files or self.textgrid_read_errors:
            logger.info(
                "There were some issues with files in the corpus. "
                "Please look at the log file or run the validator for more information."
            )
            if self.decode_error_files:
                logger.debug(
                    f"There were {len(self.decode_error_files)} errors decoding text files:"
                )
                logger.debug(", ".join(self.decode_error_files))
            if self.textgrid_read_errors:
                logger.debug(
                    f"There were {len(self.textgrid_read_errors)} errors decoding reading TextGrid files:"
                )
                for e in self.textgrid_read_errors:
                    logger.debug(f"{e.file_name}: {e.error}")

        logger.debug(f"Parsed corpus directory in {time.time() - begin_time:.3f} seconds")


class AcousticCorpusPronunciationMixin(
    AcousticCorpusMixin, MultispeakerDictionaryMixin, metaclass=ABCMeta
):
    """
    Mixin for acoustic corpora with Pronunciation dictionaries

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusMixin`
        For corpus parsing parameters
    :class:`~montreal_forced_aligner.dictionary.multispeaker.MultispeakerDictionaryMixin`
        For dictionary parsing parameters
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_corpus(self) -> None:
        """
        Load the corpus
        """
        all_begin = time.time()

        if self.dictionary_model is not None:
            logger.debug(f"Using {self.phone_set_type}")
            self.dictionary_setup()
            logger.debug(f"Loaded dictionary in {time.time() - all_begin:.3f} seconds")

        begin = time.time()
        self._load_corpus()
        logger.debug(f"Loaded corpus in {time.time() - begin:.3f} seconds")

        begin = time.time()
        self.initialize_jobs()
        logger.debug(f"Initialized jobs in {time.time() - begin:.3f} seconds")

        self.normalize_text()

        begin = time.time()
        self.write_lexicon_information()
        logger.debug(f"Wrote lexicon information in {time.time() - begin:.3f} seconds")

        begin = time.time()
        self.create_corpus_split()
        logger.debug(f"Created corpus split directory in {time.time() - begin:.3f} seconds")

        begin = time.time()
        self.generate_features()
        logger.debug(f"Generated features in {time.time() - begin:.3f} seconds")

        logger.debug(f"Setting up corpus took {time.time() - all_begin:.3f} seconds")


class AcousticCorpus(AcousticCorpusMixin, DictionaryMixin, MfaWorker):
    """
    Standalone class for working with acoustic corpora and pronunciation dictionaries

    Most functionality in MFA will use the :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusPronunciationMixin` class instead of this class.

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusPronunciationMixin`
        For dictionary and corpus parsing parameters
    :class:`~montreal_forced_aligner.abc.MfaWorker`
        For MFA processing parameters
    :class:`~montreal_forced_aligner.abc.TemporaryDirectoryMixin`
        For temporary directory parameters
    """

    def __init__(self, **kwargs):
        super(AcousticCorpus, self).__init__(**kwargs)

    @property
    def identifier(self) -> str:
        """Identifier for the corpus"""
        return self.data_source_identifier

    @property
    def output_directory(self) -> str:
        """Root temporary directory to store corpus and dictionary files"""
        return os.path.join(GLOBAL_CONFIG.temporary_directory, self.identifier)

    @property
    def working_directory(self) -> str:
        """Working directory to save temporary corpus and dictionary files"""
        return self.corpus_output_directory


class AcousticCorpusWithPronunciations(AcousticCorpusPronunciationMixin, MfaWorker):
    """
    Standalone class for parsing an acoustic corpus with a pronunciation dictionary
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def identifier(self) -> str:
        """Identifier for the corpus"""
        return self.data_source_identifier

    @property
    def output_directory(self) -> str:
        """Root temporary directory to store corpus and dictionary files"""
        return os.path.join(GLOBAL_CONFIG.temporary_directory, self.identifier)

    @property
    def working_directory(self) -> str:
        """Working directory to save temporary corpus and dictionary files"""
        return self.output_directory
