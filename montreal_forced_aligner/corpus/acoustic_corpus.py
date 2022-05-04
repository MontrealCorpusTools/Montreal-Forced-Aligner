"""Class definitions for corpora"""
from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
import time
import typing
from abc import ABCMeta
from queue import Empty
from typing import Dict, List, Optional

import sqlalchemy
import tqdm
from sqlalchemy.orm import Session

from montreal_forced_aligner.abc import MfaWorker, TemporaryDirectoryMixin
from montreal_forced_aligner.corpus.base import CorpusMixin
from montreal_forced_aligner.corpus.classes import FileData
from montreal_forced_aligner.corpus.features import (
    CalcFmllrArguments,
    CalcFmllrFunction,
    ComputeVadFunction,
    FeatureConfigMixin,
    MfccArguments,
    MfccFunction,
    VadArguments,
)
from montreal_forced_aligner.corpus.helper import find_exts
from montreal_forced_aligner.corpus.multiprocessing import (
    AcousticDirectoryParser,
    CorpusProcessWorker,
)
from montreal_forced_aligner.data import DatabaseImportData
from montreal_forced_aligner.db import (
    Corpus,
    File,
    ReferencePhoneInterval,
    SoundFile,
    Speaker,
    Utterance,
)
from montreal_forced_aligner.dictionary.mixins import DictionaryMixin, SanitizeFunction
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionaryMixin
from montreal_forced_aligner.exceptions import (
    KaldiProcessingError,
    SoundFileError,
    TextGridParseError,
    TextParseError,
)
from montreal_forced_aligner.helper import load_scp
from montreal_forced_aligner.textgrid import parse_aligned_textgrid
from montreal_forced_aligner.utils import Counter, KaldiProcessWorker, Stopped, thirdparty_binary

__all__ = [
    "AcousticCorpusMixin",
    "AcousticCorpus",
    "AcousticCorpusWithPronunciations",
    "AcousticCorpusPronunciationMixin",
]


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
    transcriptions_without_wavs: list[str]
        List of text files without sound files
    no_transcription_files: list[str]
        List of sound files without transcription files
    stopped: Stopped
        Stop check for loading the corpus
    """

    def __init__(self, audio_directory: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.audio_directory = audio_directory
        self.sound_file_errors = []
        self.transcriptions_without_wavs = []
        self.no_transcription_files = []
        self.stopped = Stopped()
        self.features_generated = False
        self.alignment_done = False
        self.transcription_done = False
        self.has_reference_alignments = False
        self.alignment_evaluation_done = False

    def inspect_database(self) -> None:
        """Check if a database file exists and create the necessary metadata"""
        exist_check = os.path.exists(self.db_path)
        if not exist_check:
            self.initialize_database()
        with self.session() as session:
            corpus = session.query(Corpus).first()
            if corpus:
                self.imported = corpus.imported
                self.features_generated = corpus.features_generated
                self.alignment_done = corpus.alignment_done
                self.transcription_done = corpus.transcription_done
                self.has_reference_alignments = corpus.has_reference_alignments
                self.alignment_evaluation_done = corpus.alignment_evaluation_done
            else:
                session.add(Corpus(name=self.data_source_identifier))
                session.commit()

    def load_reference_alignments(self, reference_directory: str) -> None:
        """
        Load reference alignments to use in alignment evaluation from a directory

        Parameters
        ----------
        reference_directory: str
            Directory containing reference alignments

        """
        if self.has_reference_alignments:
            self.log_info("Reference alignments already loaded!")
            return
        self.log_info("Loading reference files...")
        indices = []
        jobs = []
        reference_intervals = []
        with tqdm.tqdm(
            total=self.num_files, disable=getattr(self, "quiet", False)
        ) as pbar, Session(self.db_engine, autoflush=False) as session:
            for root, _, files in os.walk(reference_directory, followlinks=True):
                root_speaker = os.path.basename(root)
                for f in files:
                    if f.endswith(".TextGrid"):
                        file_name = f.replace(".TextGrid", "")
                        file_id = session.query(File.id).filter_by(name=file_name).scalar()
                        if not file_id:
                            continue
                        if self.use_mp:
                            indices.append(file_id)
                            jobs.append((os.path.join(root, f), root_speaker))
                        else:
                            intervals = parse_aligned_textgrid(os.path.join(root, f), root_speaker)
                            utterances = (
                                session.query(Utterance.id, Speaker.name, Utterance.end)
                                .join(Utterance.speaker)
                                .join(Utterance.file)
                                .filter(File.id == file_id)
                            )
                            for u_id, speaker_name, end in utterances:
                                if speaker_name not in intervals:
                                    continue
                                while (
                                    intervals[speaker_name]
                                    and intervals[speaker_name][0].end <= end
                                ):
                                    interval = intervals[speaker_name].pop(0)
                                    reference_intervals.append(
                                        {
                                            "begin": interval.begin,
                                            "end": interval.end,
                                            "label": interval.label,
                                            "utterance_id": u_id,
                                        }
                                    )

                            pbar.update(1)
            if self.use_mp:
                with mp.Pool(self.num_jobs) as pool:
                    gen = pool.starmap(parse_aligned_textgrid, jobs)
                    for i, intervals in enumerate(gen):
                        pbar.update(1)
                        file_id = indices[i]
                        utterances = (
                            session.query(Utterance.id, Speaker.name, Utterance.end)
                            .join(Utterance.speaker)
                            .filter(Utterance.file_id == file_id)
                        )
                        for u_id, speaker_name, end in utterances:
                            if speaker_name not in intervals:
                                continue
                            while (
                                intervals[speaker_name] and intervals[speaker_name][0].end <= end
                            ):
                                interval = intervals[speaker_name].pop(0)
                                reference_intervals.append(
                                    {
                                        "begin": interval.begin,
                                        "end": interval.end,
                                        "label": interval.label,
                                        "utterance_id": u_id,
                                    }
                                )
            with session.bind.begin() as conn:
                conn.execute(
                    sqlalchemy.insert(ReferencePhoneInterval.__table__), reference_intervals
                )
                session.commit()
            session.query(Corpus).update({"has_reference_alignments": True})
            session.commit()

    def load_corpus(self) -> None:
        """
        Load the corpus
        """
        self.initialize_database()
        self._load_corpus()

        self.initialize_jobs()
        self.create_corpus_split()
        self.generate_features()

    def generate_features(self, compute_cmvn: bool = True) -> None:
        """
        Generate features for the corpus

        Parameters
        ----------
        compute_cmvn: bool
            Flag for whether to compute CMVN, defaults to True
        """
        if self.features_generated:
            return
        self.log_info(f"Generating base features ({self.feature_type})...")
        if self.feature_type == "mfcc":
            self.mfcc()
        self.combine_feats()
        if compute_cmvn:
            self.log_info("Calculating CMVN...")
            self.calc_cmvn()
        self.features_generated = True
        with self.session() as session:
            session.query(Corpus).update({"features_generated": True})
            session.commit()
        self.create_corpus_split()

    def create_corpus_split(self) -> None:
        """Create the split directory for the corpus"""
        if self.features_generated:
            self.log_info("Creating corpus split with features...")
            super().create_corpus_split()
        else:
            self.log_info("Creating corpus split for feature generation...")
            split_dir = self.split_directory
            os.makedirs(os.path.join(split_dir, "log"), exist_ok=True)
            with self.session() as session:
                for job in self.jobs:
                    job.output_for_features(split_dir, session)

    def construct_base_feature_string(self, all_feats: bool = False) -> str:
        """
        Construct the base feature string independent of job name

        Used in initialization of MonophoneTrainer (to get dimension size) and IvectorTrainer (uses all feats)

        Parameters
        ----------
        all_feats: bool
            Flag for whether all features across all jobs should be taken into account

        Returns
        -------
        str
            Base feature string
        """
        j = self.jobs[0]
        if all_feats:
            feat_path = os.path.join(self.base_data_directory, "feats.scp")
            utt2spk_path = os.path.join(self.base_data_directory, "utt2spk.scp")
            cmvn_path = os.path.join(self.base_data_directory, "cmvn.scp")
            feats = f"ark,s,cs:apply-cmvn --utt2spk=ark:{utt2spk_path} scp:{cmvn_path} scp:{feat_path} ark:- |"
            feats += " add-deltas ark:- ark:- |"
            return feats
        utt2spks = j.construct_path_dictionary(self.data_directory, "utt2spk", "scp")
        cmvns = j.construct_path_dictionary(self.data_directory, "cmvn", "scp")
        features = j.construct_path_dictionary(self.data_directory, "feats", "scp")
        for dict_id in j.dictionary_ids:
            feat_path = features[dict_id]
            cmvn_path = cmvns[dict_id]
            utt2spk_path = utt2spks[dict_id]
            feats = f"ark,s,cs:apply-cmvn --utt2spk=ark:{utt2spk_path} scp:{cmvn_path} scp:{feat_path} ark:- |"
            if self.uses_deltas:
                feats += " add-deltas ark:- ark:- |"

            return feats
        else:
            utt2spk_path = j.construct_path(self.data_directory, "utt2spk", "scp")
            cmvn_path = j.construct_path(self.data_directory, "cmvn", "scp")
            feat_path = j.construct_path(self.data_directory, "feats", "scp")
            feats = f"ark,s,cs:apply-cmvn --utt2spk=ark:{utt2spk_path} scp:{cmvn_path} scp:{feat_path} ark:- |"
            if self.uses_deltas:
                feats += " add-deltas ark:- ark:- |"
            return feats

    def construct_feature_proc_strings(
        self,
        speaker_independent: bool = False,
    ) -> typing.Union[List[Dict[str, str]], List[str]]:
        """
        Constructs a feature processing string to supply to Kaldi binaries, taking into account corpus features and the
        current working directory of the aligner (whether fMLLR or LDA transforms should be used, etc).

        Parameters
        ----------
        speaker_independent: bool
            Flag for whether features should be speaker-independent regardless of the presence of fMLLR transforms

        Returns
        -------
        list[dict[str, str]]
            Feature strings per job
        """
        strings = []
        for j in self.jobs:
            lda_mat_path = None
            fmllrs = {}
            if self.working_directory is not None:
                lda_mat_path = os.path.join(self.working_directory, "lda.mat")
                if not os.path.exists(lda_mat_path):
                    lda_mat_path = None

                fmllrs = j.construct_path_dictionary(self.working_directory, "trans", "ark")
            utt2spks = j.construct_path_dictionary(self.data_directory, "utt2spk", "scp")
            cmvns = j.construct_path_dictionary(self.data_directory, "cmvn", "scp")
            features = j.construct_path_dictionary(self.data_directory, "feats", "scp")
            vads = j.construct_path_dictionary(self.data_directory, "vad", "scp")
            feat_strings = {}
            if not j.dictionary_ids:
                utt2spk_path = j.construct_path(self.data_directory, "utt2spk", "scp")
                cmvn_path = j.construct_path(self.data_directory, "cmvn", "scp")
                feat_path = j.construct_path(self.data_directory, "feats", "scp")
                feats = f"ark,s,cs:apply-cmvn --utt2spk=ark:{utt2spk_path} scp:{cmvn_path} scp:{feat_path} ark:- |"
                if self.uses_deltas:
                    feats += " add-deltas ark:- ark:- |"

                strings.append(feats)
                continue

            for dict_id in j.dictionary_ids:
                feat_path = features[dict_id]
                cmvn_path = cmvns[dict_id]
                utt2spk_path = utt2spks[dict_id]
                fmllr_trans_path = None
                try:
                    fmllr_trans_path = fmllrs[dict_id]
                    if not os.path.exists(fmllr_trans_path):
                        fmllr_trans_path = None
                except KeyError:
                    pass
                vad_path = vads[dict_id]
                if self.uses_voiced:
                    feats = f"ark,s,cs:add-deltas scp:{feat_path} ark:- |"
                    if self.uses_cmvn:
                        feats += " apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |"
                    feats += f" select-voiced-frames ark:- scp,s,cs:{vad_path} ark:- |"
                elif not os.path.exists(cmvn_path) and self.uses_cmvn:
                    feats = f"ark,s,cs:add-deltas scp:{feat_path} ark:- |"
                    if self.uses_cmvn:
                        feats += " apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |"
                else:
                    feats = f"ark,s,cs:apply-cmvn --utt2spk=ark:{utt2spk_path} scp:{cmvn_path} scp:{feat_path} ark:- |"
                    if lda_mat_path is not None:
                        feats += f" splice-feats --left-context={self.splice_left_context} --right-context={self.splice_right_context} ark:- ark:- |"
                        feats += f" transform-feats {lda_mat_path} ark:- ark:- |"
                    elif self.uses_splices:
                        feats += f" splice-feats --left-context={self.splice_left_context} --right-context={self.splice_right_context} ark:- ark:- |"
                    elif self.uses_deltas:
                        feats += " add-deltas ark:- ark:- |"
                    if fmllr_trans_path is not None and not (
                        self.speaker_independent or speaker_independent
                    ):
                        if not os.path.exists(fmllr_trans_path):
                            raise Exception(f"Could not find {fmllr_trans_path}")
                        feats += f" transform-feats --utt2spk=ark:{utt2spk_path} ark:{fmllr_trans_path} ark:- ark:- |"
                feat_strings[dict_id] = feats
            strings.append(feat_strings)
        return strings

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
                j.name,
                getattr(self, "db_engine", ""),
                os.path.join(self.split_directory, "log", f"compute_vad.{j.name}.log"),
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
        feature_strings = self.construct_feature_proc_strings()
        base_log = "calc_fmllr"
        if iteration is not None:
            base_log += f".{iteration}"
        return [
            CalcFmllrArguments(
                j.name,
                getattr(self, "db_path", ""),
                os.path.join(self.working_log_directory, f"{base_log}.{j.name}.log"),
                j.dictionary_ids,
                feature_strings[j.name],
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                self.alignment_model_path,
                self.model_path,
                j.construct_path_dictionary(self.data_directory, "spk2utt", "scp"),
                j.construct_path_dictionary(self.working_directory, "trans", "ark"),
                self.fmllr_options,
            )
            for j in self.jobs
        ]

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
                j.name,
                self.db_path,
                os.path.join(self.split_directory, "log", f"make_mfcc.{j.name}.log"),
                j.construct_path(self.split_directory, "wav", "scp"),
                j.construct_path(self.split_directory, "segments", "scp"),
                j.construct_path(self.split_directory, "feats", "scp"),
                self.mfcc_options,
                self.pitch_options,
            )
            for j in self.jobs
        ]

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
        self.log_info("Generating MFCCs...")
        log_directory = os.path.join(self.split_directory, "log")
        os.makedirs(log_directory, exist_ok=True)
        arguments = self.mfcc_arguments()
        with tqdm.tqdm(total=self.num_utterances, disable=getattr(self, "quiet", False)) as pbar:
            if self.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = MfccFunction(args)
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
                    if isinstance(result, Exception):
                        key = "error"
                        if isinstance(result, KaldiProcessingError):
                            key = result.job_name
                        error_dict[key] = result
                        continue
                    pbar.update(result)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in arguments:
                    function = MfccFunction(args)
                    for num_utterances in function.run():
                        pbar.update(num_utterances)
        with self.session() as session:
            update_mapping = []
            session.query(Utterance).update({"ignored": True})
            for j in arguments:
                with open(j.feats_scp_path, "r", encoding="utf8") as f:
                    for line in f:
                        line = line.strip()
                        if line == "":
                            continue
                        f = line.split(maxsplit=1)
                        utt_id = int(f[0].split("-")[-1])
                        feats = f[1]
                        update_mapping.append({"id": utt_id, "features": feats, "ignored": False})
            session.bulk_update_mappings(Utterance, update_mapping)
            session.commit()

    def calc_cmvn(self) -> None:
        """
        Calculate CMVN statistics for speakers

        See Also
        --------
        :kaldi_src:`compute-cmvn-stats`
            Relevant Kaldi binary
        """
        self._write_feats()
        self._write_spk2utt()
        spk2utt = os.path.join(self.corpus_output_directory, "spk2utt.scp")
        feats = os.path.join(self.corpus_output_directory, "feats.scp")
        cmvn_ark = os.path.join(self.corpus_output_directory, "cmvn.ark")
        cmvn_scp = os.path.join(self.corpus_output_directory, "cmvn.scp")
        log_path = os.path.join(self.features_log_directory, "cmvn.log")
        with open(log_path, "w") as logf:
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
                update_mapping.append({"id": int(s), "cmvn": cmvn})
            session.bulk_update_mappings(Speaker, update_mapping)
            session.commit()

    def calc_fmllr(self, iteration: Optional[int] = None) -> None:
        """
        Multiprocessing function that computes speaker adaptation transforms via
        Feature space Maximum Likelihood Linear Regression (fMLLR).

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
        self.log_info("Calculating fMLLR for speaker adaptation...")

        arguments = self.calc_fmllr_arguments(iteration=iteration)
        with tqdm.tqdm(total=self.num_speakers, disable=getattr(self, "quiet", False)) as pbar:
            if self.use_mp:
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

        self.speaker_independent = False
        self.log_debug(f"Fmllr calculation took {time.time() - begin}")

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
        if os.path.exists(os.path.join(self.split_directory, "vad.0.scp")):
            self.log_info("VAD already computed, skipping!")
            return
        begin = time.time()
        self.log_info("Computing VAD...")

        arguments = self.compute_vad_arguments()
        with tqdm.tqdm(total=self.num_speakers, disable=getattr(self, "quiet", False)) as pbar:
            if self.use_mp:
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
        self.log_debug(f"VAD computation took {time.time() - begin}")

    def combine_feats(self) -> None:
        """
        Combine feature generation results and store relevant information
        """

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
                self.log_debug(f"  - Ignored File: {sound_file_path}")
                self.log_debug(f"    - Speaker: {speaker_name}")
                self.log_debug(f"    - Begin: {begin}")
                self.log_debug(f"    - End: {end}")
                self.log_debug(f"    - Text: {text}")
                ignored_count += 1
            if ignored_count:
                self.log_warning(
                    f"There were {ignored_count} utterances ignored due to an issue in feature generation, see the log file for full "
                    "details or run `mfa validate` on the corpus."
                )

    def _write_feats(self):
        """Write feats scp file for Kaldi"""
        feats_path = os.path.join(self.corpus_output_directory, "feats.scp")
        with self.session() as session, open(feats_path, "w", encoding="utf8") as f:
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
        feature_string = self.construct_base_feature_string()
        with open(os.path.join(self.features_log_directory, "feat-to-dim.log"), "w") as log_file:
            subset_proc = subprocess.Popen(
                [
                    thirdparty_binary("subset-feats"),
                    "--n=1",
                    feature_string,
                    "ark:-",
                ],
                stderr=log_file,
                stdout=subprocess.PIPE,
            )
            dim_proc = subprocess.Popen(
                [thirdparty_binary("feat-to-dim"), "ark:-", "-"],
                stdin=subset_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
            )
            stdout, stderr = dim_proc.communicate()
            feats = stdout.decode("utf8").strip()
        return int(feats)

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
        sanitize_function = getattr(self, "sanitize_function", None)
        error_dict = {}
        procs = []
        self.db_engine.dispose()
        parser = AcousticDirectoryParser(
            self.corpus_directory,
            job_queue,
            self.audio_directory,
            stopped,
            finished_adding,
            file_counts,
        )
        parser.start()
        for i in range(self.num_jobs):
            p = CorpusProcessWorker(
                i,
                job_queue,
                return_queue,
                stopped,
                finished_adding,
                self.speaker_characters,
                sanitize_function,
                self.sample_frequency,
            )
            procs.append(p)
            p.start()
        last_poll = time.time() - 30
        import_data = DatabaseImportData()
        try:
            with self.session() as session:
                with tqdm.tqdm(total=100, disable=getattr(self, "quiet", False)) as pbar:
                    while True:
                        try:
                            file = return_queue.get(timeout=1)
                            if self.stopped.stop_check():
                                continue
                        except Empty:
                            for proc in procs:
                                if not proc.finished_processing.stop_check():
                                    break
                            else:
                                break
                            continue
                        if time.time() - last_poll > 15:
                            pbar.total = file_counts.value()
                            last_poll = time.time()
                        pbar.update(1)
                        if isinstance(file, tuple):
                            error_type = file[0]
                            error = file[1]
                            if error_type == "error":
                                error_dict[error_type] = error
                            else:
                                if error_type not in error_dict:
                                    error_dict[error_type] = []
                                error_dict[error_type].append(error)
                        else:
                            import_data.add_objects(self.generate_import_objects(file))

                    self.log_debug(f"Processing queue: {time.process_time() - begin_time}")

                    if "error" in error_dict:
                        session.rollback()
                        raise error_dict["error"][1]
                    self._finalize_load(session, import_data)
            for k in ["sound_file_errors", "decode_error_files", "textgrid_read_errors"]:
                if hasattr(self, k):
                    if k in error_dict:
                        self.log_info(
                            "There were some issues with files in the corpus. "
                            "Please look at the log file or run the validator for more information."
                        )
                        self.log_debug(f"{k} showed {len(error_dict[k])} errors:")
                        if k in {"textgrid_read_errors", "sound_file_errors"}:
                            getattr(self, k).update(error_dict[k])
                            for e in error_dict[k]:
                                self.log_debug(f"{e.file_name}: {e.error}")
                        else:
                            self.log_debug(", ".join(error_dict[k]))
                            setattr(self, k, error_dict[k])

        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                self.log_info(
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
        finally:
            parser.join()
            for p in procs:
                p.join()
            if self.stopped.stop_check():
                self.log_info(
                    f"Stopped parsing early ({time.process_time() - begin_time} seconds)"
                )
                if self.stopped.source():
                    sys.exit(0)
            else:
                self.log_debug(
                    f"Parsed corpus directory with {self.num_jobs} jobs in {time.process_time() - begin_time} seconds"
                )

    def _load_corpus_from_source(self) -> None:
        """
        Load a corpus without using multiprocessing
        """
        begin_time = time.time()
        sanitize_function = None
        if hasattr(self, "sanitize_function"):
            sanitize_function = self.sanitize_function

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
        self.log_debug(f"Walking through {self.corpus_directory}...")
        import_data = DatabaseImportData()
        with self.session() as session:
            for root, _, files in os.walk(self.corpus_directory, followlinks=True):
                exts = find_exts(files)
                relative_path = root.replace(self.corpus_directory, "").lstrip("/").lstrip("\\")
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
                    if wav_path is None and transcription_path is None:  # Not a file for MFA
                        continue
                    if wav_path is None:
                        self.transcriptions_without_wavs.append(transcription_path)
                        continue
                    if transcription_path is None:
                        self.no_transcription_files.append(wav_path)
                    try:
                        file = FileData.parse_file(
                            file_name,
                            wav_path,
                            transcription_path,
                            relative_path,
                            self.speaker_characters,
                            sanitize_function,
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
            self.log_info(
                "There were some issues with files in the corpus. "
                "Please look at the log file or run the validator for more information."
            )
            if self.decode_error_files:
                self.log_debug(
                    f"There were {len(self.decode_error_files)} errors decoding text files:"
                )
                self.log_debug(", ".join(self.decode_error_files))
            if self.textgrid_read_errors:
                self.log_debug(
                    f"There were {len(self.textgrid_read_errors)} errors decoding reading TextGrid files:"
                )
                for e in self.textgrid_read_errors:
                    self.log_debug(f"{e.file_name}: {e.error}")

        self.log_debug(f"Parsed corpus directory in {time.time() - begin_time} seconds")


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
        self.initialize_database()

        self.log_debug(f"Using {self.phone_set_type}")
        self.dictionary_setup()
        self.log_debug(f"Loaded dictionary in {time.time() - all_begin}")

        begin = time.time()
        self.write_lexicon_information()
        self.log_debug(f"Wrote lexicon information in {time.time() - begin}")

        begin = time.time()
        self._load_corpus()
        self.log_debug(f"Loaded corpus in {time.time() - begin}")

        begin = time.time()
        self.initialize_jobs()
        self.log_debug(f"Initialized jobs in {time.time() - begin}")

        begin = time.time()
        self.create_corpus_split()
        self.log_debug(f"Created corpus split directory in {time.time() - begin}")

        begin = time.time()
        self.generate_features()
        self.log_debug(f"Generated features in {time.time() - begin}")

        begin = time.time()
        self.calculate_oovs_found()
        self.log_debug(f"Calculated oovs found in {time.time() - begin}")
        self.log_debug(f"Setting up corpus took {time.time() - all_begin} seconds")


class AcousticCorpus(AcousticCorpusMixin, DictionaryMixin, MfaWorker, TemporaryDirectoryMixin):
    """
    Standalone class for working with acoustic corpora and pronunciation dictionaries

    Most functionality in MFA will use the :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusPronunciationMixin` class instead of this class.

    Parameters
    ----------
    num_jobs: int
        Number of jobs to use in processing the corpus

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusPronunciationMixin`
        For dictionary and corpus parsing parameters
    :class:`~montreal_forced_aligner.abc.MfaWorker`
        For MFA processing parameters
    :class:`~montreal_forced_aligner.abc.TemporaryDirectoryMixin`
        For temporary directory parameters
    """

    def __init__(self, num_jobs=3, **kwargs):
        super(AcousticCorpus, self).__init__(**kwargs)
        self.num_jobs = num_jobs

    @property
    def sanitize_function(self) -> SanitizeFunction:
        """Text sanitization function"""
        return self.construct_sanitize_function()

    @property
    def identifier(self) -> str:
        """Identifier for the corpus"""
        return self.data_source_identifier

    @property
    def output_directory(self) -> str:
        """Root temporary directory to store corpus and dictionary files"""
        return self.temporary_directory

    @property
    def working_directory(self) -> str:
        """Working directory to save temporary corpus and dictionary files"""
        return self.corpus_output_directory


class AcousticCorpusWithPronunciations(
    AcousticCorpusPronunciationMixin, MfaWorker, TemporaryDirectoryMixin
):
    def __init__(self, num_jobs=3, **kwargs):
        super().__init__(**kwargs)
        self.num_jobs = num_jobs

    @property
    def identifier(self) -> str:
        """Identifier for the corpus"""
        return self.data_source_identifier

    @property
    def output_directory(self) -> str:
        """Root temporary directory to store corpus and dictionary files"""
        return os.path.join(self.temporary_directory, self.identifier)

    @property
    def working_directory(self) -> str:
        """Working directory to save temporary corpus and dictionary files"""
        return self.output_directory
