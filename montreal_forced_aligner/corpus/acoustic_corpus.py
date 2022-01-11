"""Class definitions for corpora"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from abc import ABCMeta
from queue import Empty
from typing import Dict, List, Optional

import tqdm

from montreal_forced_aligner.abc import MfaWorker, TemporaryDirectoryMixin
from montreal_forced_aligner.corpus.base import CorpusMixin
from montreal_forced_aligner.corpus.classes import File
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
from montreal_forced_aligner.corpus.multiprocessing import CorpusProcessWorker
from montreal_forced_aligner.data import TextFileType
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionaryMixin
from montreal_forced_aligner.exceptions import TextGridParseError, TextParseError
from montreal_forced_aligner.helper import load_scp
from montreal_forced_aligner.textgrid import parse_aligned_textgrid
from montreal_forced_aligner.utils import KaldiProcessWorker, Stopped, thirdparty_binary

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

    def _initialize_from_json(self, data):
        self.features_generated = data.get("features_generated", False)
        self.alignment_done = data.get("alignment_done", False)
        self.transcription_done = data.get("transcription_done", False)
        self.has_reference_alignments = data.get("has_reference_alignments", False)
        self.alignment_evaluation_done = data.get("alignment_evaluation_done", False)

    @property
    def corpus_meta(self):
        return {
            "features_generated": self.features_generated,
            "alignment_done": self.alignment_done,
            "transcription_done": self.transcription_done,
            "has_reference_alignments": self.has_reference_alignments,
            "alignment_evaluation_done": self.alignment_evaluation_done,
        }

    def load_reference_alignments(self, reference_directory: str):
        self.log_info("Loading reference files...")
        indices = []
        jobs = []
        with tqdm.tqdm(total=len(self.files)) as pbar:
            for root, _, files in os.walk(reference_directory, followlinks=True):
                root_speaker = os.path.basename(root)
                for f in files:
                    if f.endswith(".TextGrid"):
                        file_name = f.replace(".TextGrid", "")
                        if file_name not in self.files:
                            continue
                        if self.use_mp:
                            indices.append(file_name)
                            jobs.append((os.path.join(root, f), root_speaker))
                        else:
                            file = self.files[file_name]
                            intervals = parse_aligned_textgrid(os.path.join(root, f), root_speaker)
                            for u in file.utterances:
                                if file.text_type == TextFileType.LAB:
                                    for v in intervals.values():
                                        self.utterances[u.name].reference_phone_labels = v
                                else:
                                    if u.speaker_name not in intervals:
                                        continue
                                    for interval in intervals[u.speaker_name]:
                                        if interval.begin >= u.begin and interval.end <= u.end:
                                            self.utterances[u.name].reference_phone_labels.append(
                                                interval
                                            )
                            pbar.update(1)
            if self.use_mp:
                with mp.Pool(self.num_jobs) as pool:
                    gen = pool.starmap(parse_aligned_textgrid, jobs)
                    for i, intervals in enumerate(gen):
                        pbar.update(1)
                        file_name = indices[i]
                        file = self.files[file_name]
                        for u in file.utterances:
                            if file.text_type == TextFileType.LAB:
                                for v in intervals.values():
                                    self.utterances[u.name].reference_phone_labels = v
                            else:
                                if u.speaker_name not in intervals:
                                    continue
                                for interval in intervals[u.speaker_name]:
                                    if interval.begin >= u.begin and interval.end <= u.end:
                                        self.utterances[u.name].reference_phone_labels.append(
                                            interval
                                        )
        self.has_reference_alignments = True

    def load_corpus(self) -> None:
        """
        Load the corpus
        """
        self._load_corpus()

        self.initialize_jobs()
        self.write_corpus_information()
        self.create_corpus_split()
        self.generate_features()

    def generate_features(self, overwrite: bool = False, compute_cmvn: bool = True) -> None:
        """
        Generate features for the corpus

        Parameters
        ----------
        overwrite: bool
            Flag for whether to ignore existing files, defaults to False
        compute_cmvn: bool
            Flag for whether to compute CMVN, defaults to True
        """
        if not overwrite and os.path.exists(
            os.path.join(self.corpus_output_directory, "feats.scp")
        ):
            self.features_generated = True
            return
        self.log_info(f"Generating base features ({self.feature_type})...")
        if self.feature_type == "mfcc":
            self.mfcc()
        self.combine_feats()
        if compute_cmvn:
            self.log_info("Calculating CMVN...")
            self.calc_cmvn()
        self.write_corpus_information()
        self.features_generated = True
        self.create_corpus_split()

    def create_corpus_split(self) -> None:
        """Create the split directory for the corpus"""
        if self.features_generated:
            super().create_corpus_split()
        else:
            split_dir = self.split_directory
            os.makedirs(os.path.join(split_dir, "log"), exist_ok=True)
            for job in self.jobs:
                job.output_for_features(split_dir)

    def write_corpus_information(self) -> None:
        """
        Output information to the temporary directory for later loading
        """
        super().write_corpus_information()
        self._write_feats()

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
        for dict_name in j.current_dictionary_names:
            feat_path = features[dict_name]
            cmvn_path = cmvns[dict_name]
            utt2spk_path = utt2spks[dict_name]
            feats = f"ark,s,cs:apply-cmvn --utt2spk=ark:{utt2spk_path} scp:{cmvn_path} scp:{feat_path} ark:- |"
            if self.uses_deltas:
                feats += " add-deltas ark:- ark:- |"

            return feats

    def construct_feature_proc_strings(
        self,
        speaker_independent: bool = False,
    ) -> List[Dict[str, str]]:
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
            for dict_name in j.current_dictionary_names:
                feat_path = features[dict_name]
                cmvn_path = cmvns[dict_name]
                utt2spk_path = utt2spks[dict_name]
                fmllr_trans_path = None
                try:
                    fmllr_trans_path = fmllrs[dict_name]
                    if not os.path.exists(fmllr_trans_path):
                        fmllr_trans_path = None
                except KeyError:
                    pass
                vad_path = vads[dict_name]
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
                        if not os.path.exists(lda_mat_path):
                            raise Exception(f"Could not find {lda_mat_path}")
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
                feat_strings[dict_name] = feats
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
                os.path.join(self.split_directory, "log", f"compute_vad.{j.name}.log"),
                j.construct_path(self.split_directory, "feats", "scp"),
                j.construct_path(self.split_directory, "vad", "scp"),
                self.vad_options,
            )
            for j in self.jobs
        ]

    def calc_fmllr_arguments(self) -> List[CalcFmllrArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.corpus.features.CalcFmllrFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.corpus.features.CalcFmllrArguments`]
            Arguments for processing
        """
        feature_strings = self.construct_feature_proc_strings()
        return [
            CalcFmllrArguments(
                os.path.join(self.working_log_directory, f"calc_fmllr.{j.name}.log"),
                j.current_dictionary_names,
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
                os.path.join(self.split_directory, "log", f"make_mfcc.{j.name}.log"),
                j.construct_path(self.split_directory, "wav", "scp"),
                j.construct_path(self.split_directory, "segments", "scp"),
                j.construct_path(self.split_directory, "feats", "scp"),
                self.mfcc_options,
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
        with tqdm.tqdm(total=self.num_utterances) as pbar:
            if self.use_mp:
                manager = mp.Manager()
                error_dict = manager.dict()
                return_queue = manager.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(self.mfcc_arguments()):
                    function = MfccFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, error_dict, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        num_utterances = return_queue.get(timeout=1)
                        # print(utterance)
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(num_utterances)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in self.mfcc_arguments():
                    function = MfccFunction(args)
                    for num_utterances in function.run():
                        pbar.update(num_utterances)

    def calc_cmvn(self) -> None:
        """
        Calculate CMVN statistics for speakers

        See Also
        --------
        :kaldi_src:`compute-cmvn-stats`
            Relevant Kaldi binary
        """
        spk2utt = os.path.join(self.corpus_output_directory, "spk2utt.scp")
        feats = os.path.join(self.corpus_output_directory, "feats.scp")
        cmvn_directory = os.path.join(self.features_directory, "cmvn")
        os.makedirs(cmvn_directory, exist_ok=True)
        cmvn_ark = os.path.join(cmvn_directory, "cmvn.ark")
        cmvn_scp = os.path.join(cmvn_directory, "cmvn.scp")
        log_path = os.path.join(cmvn_directory, "cmvn.log")
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
        shutil.copy(cmvn_scp, os.path.join(self.corpus_output_directory, "cmvn.scp"))
        for s, cmvn in load_scp(cmvn_scp).items():
            self.speakers[s].cmvn = cmvn
        self.create_corpus_split()

    def calc_fmllr(self) -> None:
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

        arguments = self.calc_fmllr_arguments()
        with tqdm.tqdm(total=self.num_speakers) as pbar:
            if self.use_mp:
                manager = mp.Manager()
                error_dict = manager.dict()
                return_queue = manager.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = CalcFmllrFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, error_dict, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        _ = return_queue.get(timeout=1)
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
        with tqdm.tqdm(total=self.num_speakers) as pbar:
            if self.use_mp:
                manager = mp.Manager()
                error_dict = manager.dict()
                return_queue = manager.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = ComputeVadFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, error_dict, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        done, no_feats, unvoiced = return_queue.get(timeout=1)
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
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
        split_directory = self.split_directory
        ignore_check = []
        for job in self.jobs:
            feats_path = job.construct_path(split_directory, "feats", "scp")

            with open(feats_path, "r") as inf:
                for line in inf:
                    line = line.strip()
                    if line == "":
                        continue
                    f = line.split(maxsplit=1)
                    if self.utterances[f[0]].ignored:
                        continue
                    self.utterances[f[0]].features = f[1]
        for utterance in self.utterances:
            if utterance.features is None or utterance.duration < 0.13:
                utterance.ignored = True
                ignore_check.append(utterance.name)
        if ignore_check:
            self.log_warning(
                "There were some utterances ignored due to short duration, see the log file for full "
                "details or run `mfa validate` on the corpus."
            )
            self.log_debug(
                f"The following utterances were too short to run alignment: "
                f"{' ,'.join(ignore_check)}"
            )
        self.write_corpus_information()

    def _write_feats(self):
        """Write feats scp file for Kaldi"""
        if any(x.features is not None for x in self.utterances):
            with open(
                os.path.join(self.corpus_output_directory, "feats.scp"), "w", encoding="utf8"
            ) as f:
                for utterance in self.utterances:
                    if not utterance.features:
                        continue
                    f.write(f"{utterance.name} {utterance.features}\n")

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
        manager = mp.Manager()
        job_queue = manager.Queue()
        return_queue = manager.Queue()
        return_dict = manager.dict()
        return_dict["sound_file_errors"] = manager.list()
        return_dict["decode_error_files"] = manager.list()
        return_dict["textgrid_read_errors"] = manager.dict()
        finished_adding = Stopped()
        stopped = Stopped()
        try:
            sanitize_function = self.sanitize_function
        except AttributeError:
            sanitize_function = None
        procs = []
        for i in range(self.num_jobs):
            p = CorpusProcessWorker(
                i,
                job_queue,
                return_dict,
                return_queue,
                stopped,
                finished_adding,
                self.speaker_characters,
                sanitize_function,
            )
            procs.append(p)
            p.start()
        try:

            use_audio_directory = False
            all_sound_files = {}
            if self.audio_directory and os.path.exists(self.audio_directory):
                use_audio_directory = True
                for root, _, files in os.walk(self.audio_directory, followlinks=True):
                    exts = find_exts(files)
                    wav_files = {k: os.path.join(root, v) for k, v in exts.wav_files.items()}
                    other_audio_files = {
                        k: os.path.join(root, v) for k, v in exts.other_audio_files.items()
                    }
                    all_sound_files.update(other_audio_files)
                    all_sound_files.update(wav_files)
            file_counts = 0
            begin = time.process_time()
            self.log_debug(f"Walking through {self.corpus_directory}...")
            for root, _, files in os.walk(self.corpus_directory, followlinks=True):
                exts = find_exts(files)
                relative_path = root.replace(self.corpus_directory, "").lstrip("/").lstrip("\\")

                if self.stopped.stop_check():
                    break
                self.log_debug(f"Inside relative root {relative_path}:")
                self.log_debug(f"    Found {len(exts.identifiers)} identifiers")
                self.log_debug(f"    Found {len(exts.wav_files)} .wav files")
                self.log_debug(f"    Found {len(exts.other_audio_files)} other audio files")
                self.log_debug(f"    Found {len(exts.lab_files)} .lab files")
                self.log_debug(f"    Found {len(exts.textgrid_files)} .TextGrid files")
                if not use_audio_directory:
                    all_sound_files = {}
                    exts.wav_files = {k: os.path.join(root, v) for k, v in exts.wav_files.items()}
                    exts.other_audio_files = {
                        k: os.path.join(root, v) for k, v in exts.other_audio_files.items()
                    }
                    all_sound_files.update(exts.other_audio_files)
                    all_sound_files.update(exts.wav_files)
                for file_name in exts.identifiers:
                    if self.stopped.stop_check():
                        break
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
                    job_queue.put((file_name, wav_path, transcription_path, relative_path))
                    file_counts += 1
            self.log_debug(f"Time to walk directory: {time.process_time() - begin}")

            finished_adding.stop()
            self.log_debug("Finished adding jobs!")

            with tqdm.tqdm(total=file_counts) as pbar:
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
                    self.add_file(File.load_from_mp_data(file))
                    pbar.update(1)
            self.log_debug(f"Processing queue: {time.process_time() - begin_time}")

            if "error" in return_dict:
                raise return_dict["error"][1]

            for k in ["sound_file_errors", "decode_error_files", "textgrid_read_errors"]:
                if hasattr(self, k):
                    if return_dict[k]:
                        self.log_info(
                            "There were some issues with files in the corpus. "
                            "Please look at the log file or run the validator for more information."
                        )
                        self.log_debug(f"{k} showed {len(return_dict[k])} errors:")
                        if k == "textgrid_read_errors":
                            getattr(self, k).update(return_dict[k])
                            for f, e in return_dict[k].items():
                                self.log_debug(f"{f}: {e.error}")
                        else:
                            self.log_debug(", ".join(return_dict[k]))
                            setattr(self, k, return_dict[k])

        except KeyboardInterrupt:
            self.log_info("Detected ctrl-c, please wait a moment while we clean everything up...")
            self.stopped.stop()
            finished_adding.stop()
            job_queue.join()
            self.stopped.set_sigint_source()
            while True:
                try:
                    _ = return_queue.get(timeout=1)
                    if self.stopped.stop_check():
                        continue
                except Empty:
                    for proc in procs:
                        if not proc.finished_processing.stop_check():
                            break
                    else:
                        break
        finally:

            finished_adding.stop()
            job_queue.join()
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
        for root, _, files in os.walk(self.corpus_directory, followlinks=True):
            exts = find_exts(files)
            relative_path = root.replace(self.corpus_directory, "").lstrip("/").lstrip("\\")
            if self.stopped.stop_check():
                return
            self.log_debug(f"Inside relative root {relative_path}:")
            self.log_debug(f"    Found {len(exts.identifiers)} identifiers")
            self.log_debug(f"    Found {len(exts.wav_files)} .wav files")
            self.log_debug(f"    Found {len(exts.other_audio_files)} other audio files")
            self.log_debug(f"    Found {len(exts.lab_files)} .lab files")
            self.log_debug(f"    Found {len(exts.textgrid_files)} .TextGrid files")
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
                    file = File.parse_file(
                        file_name,
                        wav_path,
                        transcription_path,
                        relative_path,
                        self.speaker_characters,
                        sanitize_function,
                    )
                    self.add_file(file)
                except TextParseError as e:
                    self.decode_error_files.append(e)
                except TextGridParseError as e:
                    self.textgrid_read_errors[e.file_name] = e
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
                for f, e in self.textgrid_read_errors.items():
                    self.log_debug(f"{f}: {e.error}")

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
        self.dictionary_setup()
        self.log_debug(f"Using {self.phone_set_type}")
        self.log_debug(f"Loaded dictionary in {time.time() - all_begin}")

        begin = time.time()
        self._load_corpus()
        self.log_debug(f"Loaded corpus in {time.time() - begin}")

        begin = time.time()
        self.set_lexicon_word_set(self.corpus_word_set)
        self.log_debug(f"Set up lexicon word set in {time.time() - begin}")

        begin = time.time()

        self.log_debug("Topology:")
        for k, v in self.kaldi_phones_for_topo.items():
            self.log_debug(f"{k}: {', '.join(v)}")
        self.log_debug("Extra questions:")
        for k, v in self.extra_questions_mapping.items():
            self.log_debug(f"{k}: {', '.join(v)}")
        self.write_lexicon_information()
        self.log_debug(f"Wrote lexicon information in {time.time() - begin}")

        begin = time.time()
        for speaker in self.speakers:
            speaker.set_dictionary(self.get_dictionary(speaker.name))
        self.log_debug(f"Set dictionaries for speakers in {time.time() - begin}")

        begin = time.time()
        self.initialize_jobs()
        self.log_debug(f"Initialized jobs in {time.time() - begin}")
        begin = time.time()
        self.write_corpus_information()
        self.log_debug(f"Wrote corpus information in {time.time() - begin}")

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


class AcousticCorpus(AcousticCorpusMixin, MfaWorker, TemporaryDirectoryMixin):
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
