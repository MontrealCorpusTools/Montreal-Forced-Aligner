"""Class definitions for corpora"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from collections import Counter
from queue import Empty
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Union

import yaml

from ..config import FeatureConfig
from ..config.dictionary_config import DictionaryConfig
from ..exceptions import CorpusError, KaldiProcessingError, TextGridParseError, TextParseError
from ..helper import output_mapping
from ..multiprocessing import Job
from ..multiprocessing.corpus import CorpusProcessWorker
from ..multiprocessing.features import calc_cmvn, compute_vad, mfcc
from ..multiprocessing.helper import Stopped
from ..utils import log_kaldi_errors, thirdparty_binary
from .classes import File, Speaker, Utterance, parse_file
from .helper import find_exts

if TYPE_CHECKING:
    from logging import Logger

    from ..dictionary import MultispeakerDictionary


__all__ = ["Corpus"]


class Corpus:
    """
    Class that stores information about the dataset to align.

    Corpus objects have a number of mappings from either utterances or speakers
    to various properties, and mappings between utterances and speakers.

    See http://kaldi-asr.org/doc/data_prep.html for more information about
    the files that are created by this class.


    Parameters
    ----------
    directory : str
        Directory of the dataset to align
    output_directory : str
        Directory to store generated data for the Kaldi binaries
    speaker_characters : int, optional
        Number of characters in the filenames to count as the speaker ID,
        if not specified, speaker IDs are generated from directory names
    num_jobs : int, optional
        Number of processes to use, defaults to 3
    sample_rate : int, optional
        Default sample rate to use for feature generation, defaults to 16000
    debug : bool
        Flag to enable debug mode, defaults to False
    logger : :class:`~logging.Logger`
        Logger to use
    use_mp : bool
        Flag to enable multiprocessing, defaults to True
    punctuation : str, optional
        Characters to treat as punctuation in parsing text
    clitic_markers : str, optional
        Characters to treat as clitic markers in parsing text
    audio_directory : str, optional
        Additional directory to parse for audio files
    skip_load : bool
        Flag to skip loading when initializing, defaults to False
    parse_text_only_files : bool
        Flag to parse text files that do not have associated sound files, defaults to False
    """

    def __init__(
        self,
        directory: str,
        output_directory: str,
        dictionary_config: Optional[DictionaryConfig] = None,
        speaker_characters: Union[int, str] = 0,
        num_jobs: int = 3,
        sample_rate: int = 16000,
        debug: bool = False,
        logger: Optional[Logger] = None,
        use_mp: bool = True,
        audio_directory: Optional[str] = None,
        skip_load: bool = False,
        parse_text_only_files: bool = False,
        ignore_speakers: bool = False,
    ):
        self.audio_directory = audio_directory
        self.dictionary_config = dictionary_config
        self.debug = debug
        self.use_mp = use_mp
        log_dir = os.path.join(output_directory, "logging")
        os.makedirs(log_dir, exist_ok=True)
        self.name = os.path.basename(directory)
        self.log_file = os.path.join(log_dir, "corpus.log")
        if logger is None:
            self.logger = logging.getLogger("corpus_setup")
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_file, "w", "utf-8")
            handler.setFormatter = logging.Formatter("%(name)s %(message)s")
            self.logger.addHandler(handler)
        else:
            self.logger = logger
        if not os.path.exists(directory):
            raise CorpusError(f"The directory '{directory}' does not exist.")
        if not os.path.isdir(directory):
            raise CorpusError(
                f"The specified path for the corpus ({directory}) is not a directory."
            )

        num_jobs = max(num_jobs, 1)
        if num_jobs == 1:
            self.use_mp = False
        self.original_num_jobs = num_jobs
        self.logger.info("Setting up corpus information...")
        self.directory = directory
        self.output_directory = os.path.join(output_directory, "corpus_data")
        self.temp_directory = os.path.join(self.output_directory, "temp")
        os.makedirs(self.temp_directory, exist_ok=True)
        self.speaker_characters = speaker_characters
        if speaker_characters == 0:
            self.speaker_directories = True
        else:
            self.speaker_directories = False
        self.num_jobs = num_jobs
        self.speakers: Dict[str, Speaker] = {}
        self.files: Dict[str, File] = {}
        self.utterances: Dict[str, Utterance] = {}
        self.sound_file_errors = []
        self.decode_error_files = []
        self.transcriptions_without_wavs = []
        self.no_transcription_files = []
        self.textgrid_read_errors = {}
        self.groups = []
        self.speaker_groups = []
        self.word_counts = Counter()
        self.sample_rate = sample_rate
        if self.use_mp:
            self.stopped = Stopped()
        else:
            self.stopped = False

        self.skip_load = skip_load
        self.utterances_time_sorted = False
        self.parse_text_only_files = parse_text_only_files
        self.feature_config = FeatureConfig()
        self.vad_config = {"energy_threshold": 5.5, "energy_mean_scale": 0.5}
        self.no_speakers = ignore_speakers
        self.vad_segments = {}
        if self.use_mp:
            self.stopped = Stopped()
        else:
            self.stopped = False
        if not self.skip_load:
            self.load()

    def normalized_text_iter(self, min_count: int = 1) -> Generator:
        """
        Construct an iterator over the normalized texts in the corpus

        Parameters
        ----------
        min_count: int
            Minimum word count to include in the output, otherwise will use OOV code, defaults to 1

        Yields
        -------
        str
            Normalized text
        """
        unk_words = {k for k, v in self.word_counts.items() if v <= min_count}
        for u in self.utterances.values():
            text = u.text.split()
            new_text = []
            for t in text:
                if u.speaker.dictionary is not None:
                    u.speaker.dictionary.to_int(t)
                    lookup = u.speaker.dictionary.split_clitics(t)
                    if lookup is None:
                        continue
                else:
                    lookup = [t]
                for item in lookup:
                    if item in unk_words:
                        new_text.append("<unk>")
                    elif (
                        u.speaker.dictionary is not None and item not in u.speaker.dictionary.words
                    ):
                        new_text.append("<unk>")
                    else:
                        new_text.append(item)
            yield " ".join(new_text)

    def subset_directory(self, subset: Optional[int]) -> str:
        """
        Construct a subset directory for the corpus

        Parameters
        ----------
        subset: int, optional
            Number of utterances to include, if larger than the total number of utterance or not specified, the
            split_directory is returned

        Returns
        -------
        str
            Path to subset directory
        """
        if subset is None or subset > self.num_utterances or subset <= 0:
            for j in self.jobs:
                j.set_subset(None)
            return self.split_directory
        directory = os.path.join(self.output_directory, f"subset_{subset}")
        self.create_subset(subset)
        return directory

    def initialize_utt_fsts(self) -> None:
        """
        Construct utterance FSTs
        """
        for j in self.jobs:
            j.output_utt_fsts(self)

    def create_subset(self, subset: Optional[int]) -> None:
        """
        Create a subset of utterances to use for training

        Parameters
        ----------
        subset: int
            Number of utterances to include in subset
        """
        subset_directory = os.path.join(self.output_directory, f"subset_{subset}")

        larger_subset_num = subset * 10
        if larger_subset_num < self.num_utterances:
            # Get all shorter utterances that are not one word long
            utts = sorted(
                (utt for utt in self.utterances.values() if " " in utt.text),
                key=lambda x: x.duration,
            )
            larger_subset = utts[:larger_subset_num]
        else:
            larger_subset = sorted(self.utterances.values())
        random.seed(1234)  # make it deterministic sampling
        subset_utts = set(random.sample(larger_subset, subset))
        log_dir = os.path.join(subset_directory, "log")
        os.makedirs(log_dir, exist_ok=True)

        for j in self.jobs:
            j.set_subset(subset_utts)
            j.output_to_directory(subset_directory)

    def load(self) -> None:
        """
        Load the corpus
        """
        loaded = self._load_from_temp()
        if not loaded:
            if self.use_mp:
                self.logger.debug("Loading from source with multiprocessing")
                self._load_from_source_mp()
            else:
                self.logger.debug("Loading from source without multiprocessing")
                self._load_from_source()
        else:
            self.logger.debug("Successfully loaded from temporary files")

    @property
    def file_speaker_mapping(self) -> Dict[str, List[str]]:
        """Speaker ordering for each file"""
        return {file_name: file.speaker_ordering for file_name, file in self.files.items()}

    def _load_from_temp(self) -> bool:
        """
        Load a corpus from saved data in the temporary directory

        Returns
        -------
        bool
            Whether loading from temporary files was successful
        """
        begin_time = time.time()
        for f in os.listdir(self.output_directory):
            if f.startswith("split"):
                old_num_jobs = int(f.replace("split", ""))
                if old_num_jobs != self.num_jobs:
                    self.logger.info(
                        f"Found old run with {old_num_jobs} rather than the current {self.num_jobs}, "
                        f"setting to {old_num_jobs}.  If you would like to use {self.num_jobs}, re-run the command "
                        f"with --clean."
                    )
                    self.num_jobs = old_num_jobs
        speakers_path = os.path.join(self.output_directory, "speakers.yaml")
        files_path = os.path.join(self.output_directory, "files.yaml")
        utterances_path = os.path.join(self.output_directory, "utterances.yaml")

        if not os.path.exists(speakers_path):
            self.logger.debug(f"Could not find {speakers_path}, cannot load from temp")
            return False
        if not os.path.exists(files_path):
            self.logger.debug(f"Could not find {files_path}, cannot load from temp")
            return False
        if not os.path.exists(utterances_path):
            self.logger.debug(f"Could not find {utterances_path}, cannot load from temp")
            return False
        self.logger.debug("Loading from temporary files...")

        with open(speakers_path, "r", encoding="utf8") as f:
            speaker_data = yaml.safe_load(f)

        for entry in speaker_data:
            self.speakers[entry["name"]] = Speaker(entry["name"])
            self.speakers[entry["name"]].cmvn = entry["cmvn"]

        with open(files_path, "r", encoding="utf8") as f:
            files_data = yaml.safe_load(f)
        for entry in files_data:
            self.files[entry["name"]] = File(
                entry["wav_path"], entry["text_path"], entry["relative_path"]
            )
            self.files[entry["name"]].speaker_ordering = [
                self.speakers[x] for x in entry["speaker_ordering"]
            ]
            self.files[entry["name"]].wav_info = entry["wav_info"]

        with open(utterances_path, "r", encoding="utf8") as f:
            utterances_data = yaml.safe_load(f)
        for entry in utterances_data:
            s = self.speakers[entry["speaker"]]
            f = self.files[entry["file"]]
            u = Utterance(
                s,
                f,
                begin=entry["begin"],
                end=entry["end"],
                channel=entry["channel"],
                text=entry["text"],
            )
            self.utterances[u.name] = u
            if u.text:
                self.word_counts.update(u.text.split())
            self.utterances[u.name].features = entry["features"]
            self.utterances[u.name].ignored = entry["ignored"]

        self.logger.debug(
            f"Loaded from corpus_data temp directory in {time.time()-begin_time} seconds"
        )
        return True

    def _load_from_source_mp(self) -> None:
        """
        Load a corpus using multiprocessing
        """
        if self.stopped is None:
            self.stopped = Stopped()
        begin_time = time.time()
        manager = mp.Manager()
        job_queue = manager.Queue()
        return_queue = manager.Queue()
        return_dict = manager.dict()
        return_dict["sound_file_errors"] = manager.list()
        return_dict["decode_error_files"] = manager.list()
        return_dict["textgrid_read_errors"] = manager.dict()
        finished_adding = Stopped()
        procs = []
        for _ in range(self.num_jobs):
            p = CorpusProcessWorker(
                job_queue, return_dict, return_queue, self.stopped, finished_adding
            )
            procs.append(p)
            p.start()
        try:

            use_audio_directory = False
            all_sound_files = {}
            if self.audio_directory and os.path.exists(self.audio_directory):
                use_audio_directory = True
                for root, _, files in os.walk(self.audio_directory, followlinks=True):
                    (
                        identifiers,
                        wav_files,
                        lab_files,
                        textgrid_files,
                        other_audio_files,
                    ) = find_exts(files)
                    wav_files = {k: os.path.join(root, v) for k, v in wav_files.items()}
                    other_audio_files = {
                        k: os.path.join(root, v) for k, v in other_audio_files.items()
                    }
                    all_sound_files.update(other_audio_files)
                    all_sound_files.update(wav_files)

            for root, _, files in os.walk(self.directory, followlinks=True):
                identifiers, wav_files, lab_files, textgrid_files, other_audio_files = find_exts(
                    files
                )
                relative_path = root.replace(self.directory, "").lstrip("/").lstrip("\\")

                if self.stopped.stop_check():
                    break
                if not use_audio_directory:
                    all_sound_files = {}
                    wav_files = {k: os.path.join(root, v) for k, v in wav_files.items()}
                    other_audio_files = {
                        k: os.path.join(root, v) for k, v in other_audio_files.items()
                    }
                    all_sound_files.update(other_audio_files)
                    all_sound_files.update(wav_files)
                for file_name in identifiers:
                    if self.stopped.stop_check():
                        break
                    wav_path = None
                    transcription_path = None
                    if file_name in all_sound_files:
                        wav_path = all_sound_files[file_name]
                    if file_name in lab_files:
                        lab_name = lab_files[file_name]
                        transcription_path = os.path.join(root, lab_name)

                    elif file_name in textgrid_files:
                        tg_name = textgrid_files[file_name]
                        transcription_path = os.path.join(root, tg_name)
                    if wav_path is None and not self.parse_text_only_files:
                        self.transcriptions_without_wavs.append(transcription_path)
                        continue
                    if transcription_path is None:
                        self.no_transcription_files.append(wav_path)
                    job_queue.put(
                        (
                            file_name,
                            wav_path,
                            transcription_path,
                            relative_path,
                            self.speaker_characters,
                            self.sample_rate,
                            self.dictionary_config,
                        )
                    )

            finished_adding.stop()
            self.logger.debug("Finished adding jobs!")
            job_queue.join()

            self.logger.debug("Waiting for workers to finish...")
            for p in procs:
                p.join()

            while True:
                try:
                    file = return_queue.get(timeout=1)
                    if self.stopped.stop_check():
                        continue
                except Empty:
                    break

                self.add_file(file)

            if "error" in return_dict:
                raise return_dict["error"][1]

            for k in ["sound_file_errors", "decode_error_files", "textgrid_read_errors"]:
                if hasattr(self, k):
                    if return_dict[k]:
                        self.logger.info(
                            "There were some issues with files in the corpus. "
                            "Please look at the log file or run the validator for more information."
                        )
                        self.logger.debug(f"{k} showed {len(return_dict[k])} errors:")
                        if k == "textgrid_read_errors":
                            getattr(self, k).update(return_dict[k])
                            for f, e in return_dict[k].items():
                                self.logger.debug(f"{f}: {e.error}")
                        else:
                            self.logger.debug(", ".join(return_dict[k]))
                            setattr(self, k, return_dict[k])

        except KeyboardInterrupt:
            self.logger.info(
                "Detected ctrl-c, please wait a moment while we clean everything up..."
            )
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
                    break
        finally:

            if self.stopped.stop_check():
                self.logger.info(f"Stopped parsing early ({time.time() - begin_time} seconds)")
                if self.stopped.source():
                    sys.exit(0)
            else:
                self.logger.debug(
                    f"Parsed corpus directory with {self.num_jobs} jobs in {time.time() - begin_time} seconds"
                )

    def _load_from_source(self) -> None:
        """
        Load a corpus without using multiprocessing
        """
        begin_time = time.time()
        self.stopped = False

        all_sound_files = {}
        use_audio_directory = False
        if self.audio_directory and os.path.exists(self.audio_directory):
            use_audio_directory = True
            for root, _, files in os.walk(self.audio_directory, followlinks=True):
                if self.stopped:
                    return
                identifiers, wav_files, lab_files, textgrid_files, other_audio_files = find_exts(
                    files
                )
                wav_files = {k: os.path.join(root, v) for k, v in wav_files.items()}
                other_audio_files = {
                    k: os.path.join(root, v) for k, v in other_audio_files.items()
                }
                all_sound_files.update(other_audio_files)
                all_sound_files.update(wav_files)

        for root, _, files in os.walk(self.directory, followlinks=True):
            identifiers, wav_files, lab_files, textgrid_files, other_audio_files = find_exts(files)
            relative_path = root.replace(self.directory, "").lstrip("/").lstrip("\\")
            if self.stopped:
                return
            if not use_audio_directory:
                all_sound_files = {}
                wav_files = {k: os.path.join(root, v) for k, v in wav_files.items()}
                other_audio_files = {
                    k: os.path.join(root, v) for k, v in other_audio_files.items()
                }
                all_sound_files.update(other_audio_files)
                all_sound_files.update(wav_files)
            for file_name in identifiers:

                wav_path = None
                transcription_path = None
                if file_name in all_sound_files:
                    wav_path = all_sound_files[file_name]
                if file_name in lab_files:
                    lab_name = lab_files[file_name]
                    transcription_path = os.path.join(root, lab_name)
                elif file_name in textgrid_files:
                    tg_name = textgrid_files[file_name]
                    transcription_path = os.path.join(root, tg_name)
                if wav_path is None and not self.parse_text_only_files:
                    self.transcriptions_without_wavs.append(transcription_path)
                    continue
                if transcription_path is None:
                    self.no_transcription_files.append(wav_path)

                try:
                    file = parse_file(
                        file_name,
                        wav_path,
                        transcription_path,
                        relative_path,
                        self.speaker_characters,
                        self.sample_rate,
                        self.dictionary_config,
                    )
                    self.add_file(file)
                except TextParseError as e:
                    self.decode_error_files.append(e)
                except TextGridParseError as e:
                    self.textgrid_read_errors[e.file_name] = e
        if self.decode_error_files or self.textgrid_read_errors:
            self.logger.info(
                "There were some issues with files in the corpus. "
                "Please look at the log file or run the validator for more information."
            )
            if self.decode_error_files:
                self.logger.debug(
                    f"There were {len(self.decode_error_files)} errors decoding text files:"
                )
                self.logger.debug(", ".join(self.decode_error_files))
            if self.textgrid_read_errors:
                self.logger.debug(
                    f"There were {len(self.textgrid_read_errors)} errors decoding reading TextGrid files:"
                )
                for f, e in self.textgrid_read_errors.items():
                    self.logger.debug(f"{f}: {e.error}")

        self.logger.debug(f"Parsed corpus directory in {time.time()-begin_time} seconds")

    def add_file(self, file: File) -> None:
        """
        Add a file to the corpus

        Parameters
        ----------
        file: :class:`~montreal_forced_aligner.corpus.File`
            File to be added
        """
        self.files[file.name] = file
        for speaker in file.speaker_ordering:
            if speaker.name not in self.speakers:
                self.speakers[speaker.name] = speaker
            else:
                self.speakers[speaker.name].merge(speaker)
        for u in file.utterances.values():
            self.utterances[u.name] = u
            if u.text:
                self.word_counts.update(u.text.split())

    def get_word_frequency(self) -> Dict[str, float]:
        """
        Calculate the word frequency across all the texts in the corpus

        Returns
        -------
        Dict[str, float]
            PronunciationDictionary of words and their relative frequencies
        """
        word_counts = Counter()
        for u in self.utterances.values():
            text = u.text
            speaker = u.speaker
            d = speaker.dictionary
            new_text = []
            text = text.split()
            for t in text:

                lookup = d.split_clitics(t)
                if lookup is None:
                    continue
                new_text.extend(x for x in lookup if x != "")
            word_counts.update(new_text)
        return {k: v / sum(word_counts.values()) for k, v in word_counts.items()}

    @property
    def word_set(self) -> List[str]:
        """Set of words used in the corpus"""
        return sorted(self.word_counts)

    def add_utterance(self, utterance: Utterance) -> None:
        """
        Add an utterance to the corpus

        Parameters
        ----------
        utterance: :class:`~montreal_forced_aligner.corpus.Utterance`
            Utterance to add
        """
        self.utterances[utterance.name] = utterance
        if utterance.speaker.name not in self.speakers:
            self.speakers[utterance.speaker.name] = utterance.speaker
        if utterance.file.name not in self.files:
            self.files[utterance.file.name] = utterance.file

    def delete_utterance(self, utterance: Union[str, Utterance]) -> None:
        """
        Delete an utterance from the corpus

        Parameters
        ----------
        utterance: :class:`~montreal_forced_aligner.corpus.Utterance`
            Utterance to delete
        """
        if isinstance(utterance, str):
            utterance = self.utterances[utterance]
        utterance.speaker.delete_utterance(utterance)
        utterance.file.delete_utterance(utterance)
        del self.utterances[utterance.name]

    def initialize_jobs(self) -> None:
        """
        Initialize the corpus's Jobs
        """
        if len(self.speakers) < self.num_jobs:
            self.num_jobs = len(self.speakers)
        self.jobs = [Job(i) for i in range(self.num_jobs)]
        job_ind = 0
        for s in self.speakers.values():
            self.jobs[job_ind].add_speaker(s)
            job_ind += 1
            if job_ind == self.num_jobs:
                job_ind = 0

    def initialize_corpus(
        self,
        dictionary: Optional[MultispeakerDictionary] = None,
        feature_config: Optional[FeatureConfig] = None,
    ) -> None:
        """
        Initialize corpus for use

        Parameters
        ----------
        dictionary: :class:`~montreal_forced_aligner.dictionary.MultispeakerDictionary`, optional
            PronunciationDictionary to use
        feature_config: :class:`~montreal_forced_aligner.config.FeatureConfig`, optional
            Feature configuration to use
        """
        if not self.files:
            raise CorpusError(
                "There were no wav files found for transcribing this corpus. Please validate the corpus. "
                "This error can also be caused if you're trying to find non-wav files without sox available "
                "on the system path."
            )

        if dictionary is not None:
            for speaker in self.speakers.values():
                speaker.set_dictionary(dictionary.get_dictionary(speaker.name))
        self.initialize_jobs()
        for j in self.jobs:
            j.set_feature_config(feature_config)
        self.feature_config = feature_config
        self.write()
        self.split()
        if self.feature_config is not None:
            try:
                self.generate_features()
            except Exception as e:
                if isinstance(e, KaldiProcessingError):
                    log_kaldi_errors(e.error_logs, self.logger)
                    e.update_log_file(self.logger.handlers[0].baseFilename)
                raise

    @property
    def num_utterances(self) -> int:
        """Number of utterances in the corpus"""
        return len(self.utterances)

    @property
    def features_directory(self) -> str:
        """Feature directory of the corpus"""
        return os.path.join(self.output_directory, "features")

    @property
    def features_log_directory(self) -> str:
        """Feature log directory"""
        return os.path.join(self.split_directory, "log")

    def speaker_utterance_info(self) -> str:
        """
        Construct message for analyzing high level detail about speakers and their utterances

        Returns
        -------
        str
            Analysis string
        """
        num_speakers = len(self.speakers)
        if not num_speakers:
            raise CorpusError(
                "There were no sound files found of the appropriate format. Please double check the corpus path "
                "and/or run the validation utility (mfa validate)."
            )
        average_utterances = sum(len(x.utterances) for x in self.speakers.values()) / num_speakers
        msg = (
            f"Number of speakers in corpus: {num_speakers}, "
            f"average number of utterances per speaker: {average_utterances}"
        )
        return msg

    @property
    def split_directory(self) -> str:
        """Directory used to store information split by job"""
        directory = os.path.join(self.output_directory, f"split{self.num_jobs}")
        return directory

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
        if not overwrite and os.path.exists(os.path.join(self.output_directory, "feats.scp")):
            return
        self.logger.info(f"Generating base features ({self.feature_config.type})...")
        if self.feature_config.type == "mfcc":
            mfcc(self)
        self.combine_feats()
        if compute_cmvn:
            self.logger.info("Calculating CMVN...")
            calc_cmvn(self)
        self.write()
        self.split()

    def compute_vad(self) -> None:
        """
        Compute Voice Activity Dectection features over the corpus
        """
        if os.path.exists(os.path.join(self.split_directory, "vad.0.scp")):
            self.logger.info("VAD already computed, skipping!")
            return
        self.logger.info("Computing VAD...")
        compute_vad(self)

    def combine_feats(self) -> None:
        """
        Combine feature generation results and store relevant information
        """
        split_directory = self.split_directory
        ignore_check = []
        for job in self.jobs:
            feats_paths = job.construct_path_dictionary(split_directory, "feats", "scp")
            lengths_paths = job.construct_path_dictionary(
                split_directory, "utterance_lengths", "scp"
            )
            for dict_name in job.current_dictionary_names:
                path = feats_paths[dict_name]
                lengths_path = lengths_paths[dict_name]
                if os.path.exists(lengths_path):
                    with open(lengths_path, "r") as inf:
                        for line in inf:
                            line = line.strip()
                            utt, length = line.split()
                            length = int(length)
                            if length < 13:  # Minimum length to align one phone plus silence
                                self.utterances[utt].ignored = True
                                ignore_check.append(utt)
                            self.utterances[utt].feature_length = length
                with open(path, "r") as inf:
                    for line in inf:
                        line = line.strip()
                        if line == "":
                            continue
                        f = line.split(maxsplit=1)
                        if self.utterances[f[0]].ignored:
                            continue
                        self.utterances[f[0]].features = f[1]
        for u, utterance in self.utterances.items():
            if utterance.features is None:
                utterance.ignored = True
                ignore_check.append(u)
        if ignore_check:
            self.logger.warning(
                "There were some utterances ignored due to short duration, see the log file for full "
                "details or run `mfa validate` on the corpus."
            )
            self.logger.debug(
                f"The following utterances were too short to run alignment: "
                f"{' ,'.join(ignore_check)}"
            )
        self.write()

    def get_feat_dim(self) -> int:
        """
        Calculate the feature dimension for the corpus

        Returns
        -------
        int
            Dimension of feature vectors
        """
        feature_string = self.jobs[0].construct_base_feature_string(self)
        with open(os.path.join(self.features_log_directory, "feat-to-dim.log"), "w") as log_file:
            dim_proc = subprocess.Popen(
                [thirdparty_binary("feat-to-dim"), feature_string, "-"],
                stdout=subprocess.PIPE,
                stderr=log_file,
            )
            stdout, stderr = dim_proc.communicate()
            feats = stdout.decode("utf8").strip()
        return int(feats)

    def write(self) -> None:
        """
        Output information to the temporary directory for later loading
        """
        self._write_speakers()
        self._write_files()
        self._write_utterances()
        self._write_spk2utt()
        self._write_feats()

    def _write_spk2utt(self):
        """Write spk2utt scp file for Kaldi"""
        data = {
            speaker.name: sorted(speaker.utterances.keys()) for speaker in self.speakers.values()
        }
        output_mapping(data, os.path.join(self.output_directory, "spk2utt.scp"))

    def write_utt2spk(self):
        """Write utt2spk scp file for Kaldi"""
        data = {u.name: u.speaker.name for u in self.utterances.values()}
        output_mapping(data, os.path.join(self.output_directory, "utt2spk.scp"))

    def _write_feats(self):
        """Write feats scp file for Kaldi"""
        if any(x.features is not None for x in self.utterances.values()):
            with open(os.path.join(self.output_directory, "feats.scp"), "w", encoding="utf8") as f:
                for utterance in self.utterances.values():
                    if not utterance.features:
                        continue
                    f.write(f"{utterance.name} {utterance.features}\n")

    def _write_speakers(self):
        """Write speaker information for speeding up future runs"""
        to_save = []
        for speaker in self.speakers.values():
            to_save.append(speaker.meta)
        with open(os.path.join(self.output_directory, "speakers.yaml"), "w", encoding="utf8") as f:
            yaml.safe_dump(to_save, f)

    def _write_files(self):
        """Write file information for speeding up future runs"""
        to_save = []
        for file in self.files.values():
            to_save.append(file.meta)
        with open(os.path.join(self.output_directory, "files.yaml"), "w", encoding="utf8") as f:
            yaml.safe_dump(to_save, f)

    def _write_utterances(self):
        """Write utterance information for speeding up future runs"""
        to_save = []
        for utterance in self.utterances.values():
            to_save.append(utterance.meta)
        with open(
            os.path.join(self.output_directory, "utterances.yaml"), "w", encoding="utf8"
        ) as f:
            yaml.safe_dump(to_save, f)

    def split(self) -> None:
        """Create split directory and output information from Jobs"""
        split_dir = self.split_directory
        os.makedirs(os.path.join(split_dir, "log"), exist_ok=True)
        self.logger.info("Setting up training data...")
        for job in self.jobs:
            job.output_to_directory(split_dir)
