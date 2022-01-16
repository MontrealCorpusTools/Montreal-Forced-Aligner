"""Class definitions for corpora"""
from __future__ import annotations

import json
import os
import random
import time
from abc import ABCMeta, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Union

import jsonlines
import yaml

from montreal_forced_aligner.abc import MfaWorker, TemporaryDirectoryMixin
from montreal_forced_aligner.corpus.classes import (
    File,
    FileCollection,
    Speaker,
    SpeakerCollection,
    Utterance,
    UtteranceCollection,
)
from montreal_forced_aligner.corpus.multiprocessing import Job
from montreal_forced_aligner.data import CtmInterval, SoundFileInformation
from montreal_forced_aligner.exceptions import CorpusError
from montreal_forced_aligner.helper import jsonl_encoder, output_mapping
from montreal_forced_aligner.utils import Stopped

__all__ = ["CorpusMixin"]


class CorpusMixin(MfaWorker, TemporaryDirectoryMixin, metaclass=ABCMeta):
    """
    Mixin class for processing corpora

    Notes
    -----
    Using characters in files to specify speakers is generally finicky and leads to errors, so I would not
    recommend using it.  Additionally, consider it deprecated and could be removed in future versions

    Parameters
    ----------
    corpus_directory: str
        Path to corpus
    speaker_characters: int or str, optional
        Number of characters in the file name to specify the speaker
    ignore_speakers: bool
        Flag for whether to discard any parsed speaker information during top-level worker's processing

    See Also
    --------
    :class:`~montreal_forced_aligner.abc.MfaWorker`
        For MFA processing parameters
    :class:`~montreal_forced_aligner.abc.TemporaryDirectoryMixin`
        For temporary directory parameters

    Attributes
    ----------
    speakers: :class:`~montreal_forced_aligner.corpus.classes.SpeakerCollection`
        Dictionary of speakers in the corpus
    files: :class:`~montreal_forced_aligner.corpus.classes.FileCollection`
        Dictionary of files in the corpus
    utterances: :class:`~montreal_forced_aligner.corpus.classes.UtteranceCollection`
        Dictionary of utterances in the corpus
    jobs: list[Job]
        List of jobs for processing the corpus and splitting speakers
    word_counts: Counter
        Counts of words in the corpus
    stopped: Stopped
        Stop check for loading the corpus
    decode_error_files: list[str]
        List of text files that could not be loaded with utf8
    textgrid_read_errors: list[str]
        List of TextGrid files that had an error in loading
    """

    def __init__(
        self,
        corpus_directory: str,
        speaker_characters: Union[int, str] = 0,
        ignore_speakers: bool = False,
        **kwargs,
    ):
        if not os.path.exists(corpus_directory):
            raise CorpusError(f"The directory '{corpus_directory}' does not exist.")
        if not os.path.isdir(corpus_directory):
            raise CorpusError(
                f"The specified path for the corpus ({corpus_directory}) is not a directory."
            )
        self.speakers = SpeakerCollection()
        self.files = FileCollection()
        self.utterances = UtteranceCollection()
        self.corpus_directory = corpus_directory
        self.speaker_characters = speaker_characters
        self.ignore_speakers = ignore_speakers
        self.word_counts = Counter()
        self.stopped = Stopped()
        self.decode_error_files = []
        self.textgrid_read_errors = {}
        self.jobs: List[Job] = []
        super().__init__(**kwargs)

    def _initialize_from_json(self, data):
        pass

    @property
    def corpus_meta(self):
        return {}

    @property
    def features_directory(self) -> str:
        """Feature directory of the corpus"""
        return os.path.join(self.corpus_output_directory, "features")

    @property
    def features_log_directory(self) -> str:
        """Feature log directory"""
        return os.path.join(self.split_directory, "log")

    @property
    def split_directory(self) -> str:
        """Directory used to store information split by job"""
        return os.path.join(self.corpus_output_directory, f"split{self.num_jobs}")

    def write_corpus_information(self) -> None:
        """
        Output information to the temporary directory for later loading
        """
        os.makedirs(self.split_directory, exist_ok=True)
        self._write_speakers()
        self._write_files()
        self._write_utterances()
        self._write_spk2utt()
        self._write_corpus_info()

    def _write_spk2utt(self):
        """Write spk2utt scp file for Kaldi"""
        data = {
            speaker.name: sorted(u.name for u in speaker.utterances) for speaker in self.speakers
        }
        output_mapping(data, os.path.join(self.corpus_output_directory, "spk2utt.scp"))

    def write_utt2spk(self):
        """Write utt2spk scp file for Kaldi"""
        data = {u.name: u.speaker.name for u in self.utterances}
        output_mapping(data, os.path.join(self.corpus_output_directory, "utt2spk.scp"))

    def _write_corpus_info(self):
        """Write speaker information for speeding up future runs"""
        with open(
            os.path.join(self.corpus_output_directory, "corpus.json"), "w", encoding="utf8"
        ) as f:
            json.dump(self.corpus_meta, f)

    def _write_speakers(self):
        """Write speaker information for speeding up future runs"""
        with open(
            os.path.join(self.corpus_output_directory, "speakers.jsonl"), "w", encoding="utf8"
        ) as f:
            writer = jsonlines.Writer(f, dumps=jsonl_encoder)
            for speaker in self.speakers:
                writer.write(speaker.meta)

    def _write_files(self):
        """Write file information for speeding up future runs"""
        with open(
            os.path.join(self.corpus_output_directory, "files.jsonl"), "w", encoding="utf8"
        ) as f:
            writer = jsonlines.Writer(f, dumps=jsonl_encoder)
            for file in self.files:
                writer.write(file.meta)

    def _write_utterances(self):
        """Write utterance information for speeding up future runs"""
        with open(
            os.path.join(self.corpus_output_directory, "utterances.jsonl"), "w", encoding="utf8"
        ) as f:
            writer = jsonlines.Writer(f, dumps=jsonl_encoder)
            for utterance in self.utterances:
                writer.write(utterance.meta)

    def create_corpus_split(self) -> None:
        """Create split directory and output information from Jobs"""
        split_dir = self.split_directory
        os.makedirs(os.path.join(split_dir, "log"), exist_ok=True)
        for job in self.jobs:
            job.output_to_directory(split_dir)

    @property
    def file_speaker_mapping(self) -> Dict[str, List[str]]:
        """Speaker ordering for each file"""
        return {file.name: file.speaker_ordering for file in self.files}

    def get_word_frequency(self) -> Dict[str, float]:
        """
        Calculate the relative word frequency across all the texts in the corpus

        Returns
        -------
        dict[str, float]
            Dictionary of words and their relative frequencies
        """
        word_counts = Counter()
        for u in self.utterances:
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
    def corpus_word_set(self) -> List[str]:
        """Set of words used in the corpus"""
        return sorted(self.word_counts)

    def add_utterance(self, utterance: Utterance) -> None:
        """
        Add an utterance to the corpus

        Parameters
        ----------
        utterance: :class:`~montreal_forced_aligner.corpus.classes.Utterance`
            Utterance to add
        """
        self.utterances.add_utterance(utterance)
        if utterance.speaker not in self.speakers:
            self.speakers.add_speaker(utterance.speaker)
        speaker = self.speakers[utterance.speaker.name]
        speaker.add_utterance(utterance)
        utterance.speaker = speaker
        if utterance.file not in self.files:
            self.files.add_file(utterance.file)
        file = self.files[utterance.file.name]
        utterance.file = file
        file.add_utterance(utterance)
        utterance.file = file

    def delete_utterance(self, utterance: Union[str, Utterance]) -> None:
        """
        Delete an utterance from the corpus

        Parameters
        ----------
        utterance: :class:`~montreal_forced_aligner.corpus.classes.Utterance`
            Utterance to delete
        """
        if isinstance(utterance, str):
            utterance = self.utterances[utterance]
        speaker = self.speakers[utterance.speaker.name]
        file = self.files[utterance.file.name]
        speaker.delete_utterance(utterance)
        file.delete_utterance(utterance)
        del self.utterances[utterance.name]

    def initialize_jobs(self) -> None:
        """
        Initialize the corpus's Jobs
        """
        self.log_info("Setting up training data...")
        if len(self.speakers) < self.num_jobs:
            self.num_jobs = len(self.speakers)
        self.jobs = [Job(i) for i in range(self.num_jobs)]
        job_ind = 0
        for s in sorted(self.speakers):
            self.jobs[job_ind].add_speaker(s)
            job_ind += 1
            if job_ind == self.num_jobs:
                job_ind = 0

    def add_file(self, file: File) -> None:
        """
        Add a file to the corpus

        Parameters
        ----------
        file: :class:`~montreal_forced_aligner.corpus.classes.File`
            File to be added
        """
        self.files.add_file(file)
        for i, speaker in enumerate(file.speaker_ordering):
            if speaker.name not in self.speakers:
                self.speakers.add_speaker(speaker)
            else:
                file.speaker_ordering[i] = self.speakers[speaker.name]
        for u in file.utterances:
            self.add_utterance(u)
            if u.text:
                self.word_counts.update(u.text.split())
            if u.normalized_text:
                self.word_counts.update(u.normalized_text)

    @property
    def data_source_identifier(self) -> str:
        """Corpus name"""
        return os.path.basename(self.corpus_directory)

    def create_subset(self, subset: int) -> None:
        """
        Create a subset of utterances to use for training

        Parameters
        ----------
        subset: int
            Number of utterances to include in subset
        """
        self.log_debug(f"Creating subset_{subset} directory...")
        subset_directory = os.path.join(self.corpus_output_directory, f"subset_{subset}")

        larger_subset_num = subset * 10
        if larger_subset_num < len(self.utterances):
            # Get all shorter utterances that are not one word long
            utts = sorted(
                (utt for utt in self.utterances if " " in utt.text),
                key=lambda x: x.duration,
            )
            larger_subset = utts[:larger_subset_num]
        else:
            larger_subset = sorted(self.utterances)
        random.seed(1234)  # make it deterministic sampling
        subset_utts = UtteranceCollection()
        subset_utts.update(random.sample(larger_subset, subset))
        log_dir = os.path.join(subset_directory, "log")
        os.makedirs(log_dir, exist_ok=True)

        for j in self.jobs:
            j.set_subset(subset_utts)
            j.output_to_directory(subset_directory)

    @property
    def num_utterances(self) -> int:
        """Number of utterances in the corpus"""
        return len(self.utterances)

    @property
    def num_speakers(self) -> int:
        """Number of speakers in the corpus"""
        return len(self.speakers)

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
        if subset is None or subset >= len(self.utterances) or subset <= 0:
            for j in self.jobs:
                j.set_subset(None)
            return self.split_directory
        directory = os.path.join(self.corpus_output_directory, f"subset_{subset}")
        if not os.path.exists(directory):
            self.create_subset(subset)
        return directory

    def _load_corpus(self) -> None:
        """
        Load the corpus
        """
        self.log_info("Setting up corpus information...")
        loaded = self._load_corpus_from_temp()
        if not loaded:
            self.log_debug("Could not load from temp")
            self.log_info("Loading corpus from source files...")
            if self.use_mp:

                self._load_corpus_from_source_mp()
            else:
                self._load_corpus_from_source()
        else:
            self.log_debug("Successfully loaded from temporary files")
        if not self.files:
            raise CorpusError(
                "There were no files found for this corpus. Please validate the corpus."
            )
        num_speakers = len(self.speakers)
        if not num_speakers:
            raise CorpusError(
                "There were no sound files found of the appropriate format. Please double check the corpus path "
                "and/or run the validation utility (mfa validate)."
            )
        average_utterances = sum(len(x.utterances) for x in self.speakers) / num_speakers
        self.log_info(
            f"Number of speakers in corpus: {num_speakers}, "
            f"average number of utterances per speaker: {average_utterances}"
        )

    def _load_corpus_from_temp(self) -> bool:
        """
        Load a corpus from saved data in the temporary directory

        Returns
        -------
        bool
            Whether loading from temporary files was successful
        """
        begin_time = time.time()
        if not os.path.exists(self.corpus_output_directory):
            return False
        for f in os.listdir(self.corpus_output_directory):
            if f.startswith("split"):
                old_num_jobs = int(f.replace("split", ""))
                if old_num_jobs != self.num_jobs:
                    self.log_info(
                        f"Found old run with {old_num_jobs} rather than the current {self.num_jobs}, "
                        f"setting to {old_num_jobs}.  If you would like to use {self.num_jobs}, re-run the command "
                        f"with --clean."
                    )
                    self.num_jobs = old_num_jobs
        format = "jsonl"
        corpus_path = os.path.join(self.corpus_output_directory, "corpus.json")
        speakers_path = os.path.join(self.corpus_output_directory, "speakers.jsonl")
        files_path = os.path.join(self.corpus_output_directory, "files.jsonl")
        utterances_path = os.path.join(self.corpus_output_directory, "utterances.jsonl")
        if not os.path.exists(speakers_path):
            format = "yaml"
            speakers_path = os.path.join(self.corpus_output_directory, "speakers.yaml")
            files_path = os.path.join(self.corpus_output_directory, "files.yaml")
            utterances_path = os.path.join(self.corpus_output_directory, "utterances.yaml")

        if not os.path.exists(speakers_path):
            self.log_debug(f"Could not find {speakers_path}, cannot load from temp")
            return False
        if not os.path.exists(files_path):
            self.log_debug(f"Could not find {files_path}, cannot load from temp")
            return False
        if not os.path.exists(utterances_path):
            self.log_debug(f"Could not find {utterances_path}, cannot load from temp")
            return False
        self.log_debug("Loading from temporary files...")

        if os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf8") as f:
                data = json.load(f)
                self._initialize_from_json(data)

        with open(speakers_path, "r", encoding="utf8") as f:
            if format == "jsonl":
                speaker_data = jsonlines.Reader(f)
            else:
                speaker_data = yaml.safe_load(f)

            for entry in speaker_data:
                self.speakers.add_speaker(Speaker(entry["name"]))
                self.speakers[entry["name"]].cmvn = entry["cmvn"]

        with open(files_path, "r", encoding="utf8") as f:
            if format == "jsonl":
                files_data = jsonlines.Reader(f)
            else:
                files_data = yaml.safe_load(f)
            for entry in files_data:
                file = File(
                    name=entry["name"],
                    wav_path=entry["wav_path"],
                    text_path=entry["text_path"],
                    relative_path=entry["relative_path"],
                )
                self.files.add_file(file)
                self.files[file.name].speaker_ordering = [
                    self.speakers[x] for x in entry["speaker_ordering"]
                ]
                self.files[file.name].wav_info = SoundFileInformation(**entry["wav_info"])

        with open(utterances_path, "r", encoding="utf8") as f:
            if format == "jsonl":
                utterances_data = jsonlines.Reader(f)
            else:
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
                u.oovs = set(entry["oovs"])
                u.normalized_text = entry["normalized_text"]
                if u.text:
                    self.word_counts.update(u.text.split())
                if u.normalized_text:
                    self.word_counts.update(u.normalized_text)
                u.word_error_rate = entry.get("word_error_rate", None)
                u.character_error_rate = entry.get("character_error_rate", None)
                u.transcription_text = entry.get("transcription_text", None)
                u.phone_error_rate = entry.get("phone_error_rate", None)
                u.alignment_score = entry.get("alignment_score", None)
                u.alignment_log_likelihood = entry.get("alignment_log_likelihood", None)
                u.reference_phone_labels = [
                    CtmInterval(**x) for x in entry.get("reference_phone_labels", [])
                ]

                phone_labels = entry.get("phone_labels", None)
                if phone_labels:
                    u.phone_labels = [CtmInterval(**x) for x in phone_labels]
                word_labels = entry.get("word_labels", None)
                if word_labels:
                    u.word_labels = [CtmInterval(**x) for x in word_labels]
                u.features = entry.get("features", None)
                u.ignored = entry.get("ignored", False)
                self.add_utterance(u)

        self.log_debug(
            f"Loaded from corpus_data temp directory in {time.time() - begin_time} seconds"
        )
        return True

    @property
    def base_data_directory(self) -> str:
        """Corpus data directory"""
        return self.corpus_output_directory

    @property
    def data_directory(self) -> str:
        """Corpus data directory"""
        return self.split_directory

    @abstractmethod
    def _load_corpus_from_source_mp(self) -> None:
        """Abstract method for loading a corpus with multiprocessing"""
        ...

    @abstractmethod
    def _load_corpus_from_source(self) -> None:
        """Abstract method for loading a corpus without multiprocessing"""
        ...
