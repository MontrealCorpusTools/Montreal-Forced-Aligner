"""
Corpus loading worker
---------------------
"""
from __future__ import annotations

import multiprocessing as mp
import os
from queue import Empty, Queue
from typing import Dict, Optional, Union

import sqlalchemy
import sqlalchemy.engine
from sqlalchemy.orm import Session

from montreal_forced_aligner.corpus.classes import FileData
from montreal_forced_aligner.corpus.helper import find_exts
from montreal_forced_aligner.db import File, SoundFile, Speaker, SpeakerOrdering, Utterance
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerSanitizationFunction
from montreal_forced_aligner.exceptions import SoundFileError, TextGridParseError, TextParseError
from montreal_forced_aligner.utils import Counter, Stopped

__all__ = ["AcousticDirectoryParser", "CorpusProcessWorker", "Job"]


class AcousticDirectoryParser(mp.Process):
    """
    Worker for processing directories for acoustic sound files

    Parameters
    ----------
    corpus_directory: str
        Directory to parse
    job_queue: Queue
        Queue to add file names to
    audio_directory: str
        Directory with additional audio files
    stopped: :class:`~montreal_forced_aligner.utils.Stopped`
        Check for whether to exit early
    finished_adding: :class:`~montreal_forced_aligner.utils.Stopped`
        Check to set when the parser is done adding files to the queue
    file_counts: :class:`~montreal_forced_aligner.utils.Counter`
        Counter for the number of total files that the parser has found
    """

    def __init__(
        self,
        corpus_directory: str,
        job_queue: Queue,
        audio_directory: str,
        stopped: Stopped,
        finished_adding: Stopped,
        file_counts: Counter,
    ):
        mp.Process.__init__(self)
        self.corpus_directory = corpus_directory
        self.job_queue = job_queue
        self.audio_directory = audio_directory
        self.stopped = stopped
        self.finished_adding = finished_adding
        self.file_counts = file_counts

    def run(self) -> None:
        """
        Run the corpus loading job
        """

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
        for root, _, files in os.walk(self.corpus_directory, followlinks=True):
            exts = find_exts(files)
            relative_path = root.replace(self.corpus_directory, "").lstrip("/").lstrip("\\")

            if self.stopped.stop_check():
                break
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
                    continue
                self.job_queue.put((file_name, wav_path, transcription_path, relative_path))
                self.file_counts.increment()

        self.finished_adding.stop()


class CorpusProcessWorker(mp.Process):
    """
    Multiprocessing corpus loading worker

    Attributes
    ----------
    job_q: :class:`~multiprocessing.Queue`
        Job queue for files to process
    return_dict: dict
        Dictionary to catch errors
    return_q: :class:`~multiprocessing.Queue`
        Return queue for processed Files
    stopped: :class:`~montreal_forced_aligner.utils.Stopped`
        Stop check for whether corpus loading should exit
    finished_adding: :class:`~montreal_forced_aligner.utils.Stopped`
        Signal that the main thread has stopped adding new files to be processed
    """

    def __init__(
        self,
        name: int,
        job_q: mp.Queue,
        return_q: mp.Queue,
        stopped: Stopped,
        finished_adding: Stopped,
        speaker_characters: Union[int, str],
        sanitize_function: Optional[MultispeakerSanitizationFunction],
        sample_rate: Optional[int],
    ):
        mp.Process.__init__(self)
        self.name = str(name)
        self.job_q = job_q
        self.return_q = return_q
        self.stopped = stopped
        self.finished_adding = finished_adding
        self.finished_processing = Stopped()
        self.sanitize_function = sanitize_function
        self.speaker_characters = speaker_characters
        self.sample_rate = sample_rate

    def run(self) -> None:
        """
        Run the corpus loading job
        """
        while True:
            try:
                file_name, wav_path, text_path, relative_path = self.job_q.get(timeout=1)
            except Empty:
                if self.finished_adding.stop_check():
                    break
                continue
            if self.stopped.stop_check():
                continue
            try:
                file = FileData.parse_file(
                    file_name,
                    wav_path,
                    text_path,
                    relative_path,
                    self.speaker_characters,
                    self.sanitize_function,
                    self.sample_rate,
                )
                self.return_q.put(file)
            except TextParseError as e:
                self.return_q.put(("decode_error_files", e))
            except TextGridParseError as e:
                self.return_q.put(("textgrid_read_errors", e))
            except SoundFileError as e:
                self.return_q.put(("sound_file_errors", e))
            except Exception as e:
                self.stopped.stop()
                self.return_q.put(("error", e))
        self.finished_processing.stop()
        return


class Job:
    """
    Class representing information about corpus jobs that will be run in parallel.
    Jobs have a set of speakers that they will process, along with all files and utterances associated with that speaker.
    As such, Jobs also have a set of dictionaries that the speakers use, and argument outputs are largely dependent on
    the pronunciation dictionaries in use.

    Parameters
    ----------
    name: int
        Job number is the job's identifier
    db_engine: sqlalchemy.engine.Engine
        Database engine to use in looking up relevant information

    Attributes
    ----------
    dictionary_ids: list[int]
        List of dictionary ids that the job's speakers use
    """

    name: int

    def __init__(self, name: int, db_engine: sqlalchemy.engine.Engine):
        self.name = name
        self.db_engine = db_engine
        self.dictionary_ids = []
        with Session(self.db_engine) as session:
            self.refresh_dictionaries(session)

    def refresh_dictionaries(self, session: Session) -> None:
        """
        Refresh the dictionaries that will be processed by this job

        Parameters
        ----------
        session: :class:`~sqlalchemy.orm.session.Session`
            Session to use for refreshing
        """
        job_dict_query = (
            session.query(Speaker.dictionary_id).filter(Speaker.job_id == self.name).distinct()
        )
        self.dictionary_ids = [x[0] for x in job_dict_query]

    def construct_path_dictionary(
        self, directory: str, identifier: str, extension: str
    ) -> Dict[str, str]:
        """
        Helper function for constructing dictionary-dependent paths for the Job

        Parameters
        ----------
        directory: str
            Directory to use as the root
        identifier: str
            Identifier for the path name, like ali or acc
        extension: str
            Extension of the path, like .scp or .ark

        Returns
        -------
        dict[str, str]
            Path for each dictionary
        """
        output = {}
        for dict_id in self.dictionary_ids:
            if dict_id is None:
                output[dict_id] = os.path.join(directory, f"{identifier}.{self.name}.{extension}")
            else:
                output[dict_id] = os.path.join(
                    directory, f"{identifier}.{dict_id}.{self.name}.{extension}"
                )
        return output

    def construct_path(self, directory: str, identifier: str, extension: str) -> str:
        """
        Helper function for constructing dictionary-dependent paths for the Job

        Parameters
        ----------
        directory: str
            Directory to use as the root
        identifier: str
            Identifier for the path name, like ali or acc
        extension: str
            Extension of the path, like .scp or .ark

        Returns
        -------
        str
            Path
        """
        return os.path.join(directory, f"{identifier}.{self.name}.{extension}")

    def construct_dictionary_dependent_paths(
        self, directory: str, identifier: str, extension: str
    ) -> Dict[str, str]:
        """
        Helper function for constructing paths that depend only on the dictionaries of the job, and not the job name itself.
        These paths should be merged with all other jobs to get a full set of dictionary paths.

        Parameters
        ----------
        directory: str
            Directory to use as the root
        identifier: str
            Identifier for the path name, like ali or acc
        extension: str
            Extension of the path, like .scp or .ark

        Returns
        -------
        dict[str, str]
            Path for each dictionary
        """
        output = {}
        for dict_id in self.dictionary_ids:
            output[dict_id] = os.path.join(directory, f"{identifier}.{dict_id}.{extension}")
        return output

    @property
    def dictionary_count(self):
        """Number of dictionaries currently used"""
        return len(self.dictionary_ids)

    def output_for_features(self, split_directory: str, session) -> None:
        """
        Output the necessary files for Kaldi to generate features

        Parameters
        ----------
        split_directory: str
            Split directory for the corpus
        """
        wav_scp_path = self.construct_path(split_directory, "wav", "scp")
        segments_scp_path = self.construct_path(split_directory, "segments", "scp")
        if os.path.exists(segments_scp_path):
            return
        with open(wav_scp_path, "w", encoding="utf8") as wav_file:
            files = (
                session.query(File.id, SoundFile.sox_string, SoundFile.sound_file_path)
                .join(File.speakers)
                .join(SpeakerOrdering.speaker)
                .join(File.sound_file)
                .distinct()
                .filter(Speaker.job_id == self.name)
                .order_by(File.id.cast(sqlalchemy.String))
            )
            for f_id, sox_string, sound_file_path in files:
                if not sox_string:
                    sox_string = sound_file_path
                wav_file.write(f"{f_id} {sox_string}\n")

        with open(segments_scp_path, "w", encoding="utf8") as segments_file:
            utterances = (
                session.query(
                    Utterance.kaldi_id,
                    Utterance.file_id,
                    Utterance.begin,
                    Utterance.end,
                    Utterance.channel,
                )
                .join(Utterance.speaker)
                .filter(Speaker.job_id == self.name)
                .order_by(Utterance.kaldi_id)
            )
            for u_id, f_id, begin, end, channel in utterances:
                segments_file.write(f"{u_id} {f_id} {begin} {end} {channel}\n")

    def output_to_directory(self, split_directory: str, session, subset=False) -> None:
        """
        Output job information to a directory

        Parameters
        ----------
        split_directory: str
            Directory to output to
        """
        if self.dictionary_ids:
            for dict_id in self.dictionary_ids:
                dict_pattern = f"{self.name}"
                if dict_id is not None:
                    dict_pattern = f"{dict_id}.{self.name}"
                scp_path = os.path.join(split_directory, f"utt2spk.{dict_pattern}.scp")
                if not os.path.exists(scp_path):
                    break
            else:
                return
        no_data = []

        def _write_current() -> None:
            """Write the current data to disk"""
            if not utt2spk:
                if _current_dict_id is not None:
                    no_data.append(_current_dict_id)
                return
            dict_pattern = f"{self.name}"
            if _current_dict_id is not None:
                dict_pattern = f"{_current_dict_id}.{self.name}"
            scp_path = os.path.join(split_directory, f"spk2utt.{dict_pattern}.scp")
            with open(scp_path, "w", encoding="utf8") as f:
                for speaker in sorted(spk2utt.keys()):
                    utts = " ".join(sorted(spk2utt[speaker]))
                    f.write(f"{speaker} {utts}\n")
            scp_path = os.path.join(split_directory, f"cmvn.{dict_pattern}.scp")
            with open(scp_path, "w", encoding="utf8") as f:
                for speaker in sorted(cmvns.keys()):
                    f.write(f"{speaker} {cmvns[speaker]}\n")

            scp_path = os.path.join(split_directory, f"utt2spk.{dict_pattern}.scp")
            with open(scp_path, "w", encoding="utf8") as f:
                for utt in sorted(utt2spk.keys()):
                    f.write(f"{utt} {utt2spk[utt]}\n")
            scp_path = os.path.join(split_directory, f"feats.{dict_pattern}.scp")
            with open(scp_path, "w", encoding="utf8") as f:
                for utt in sorted(feats.keys()):
                    f.write(f"{utt} {feats[utt]}\n")
            scp_path = os.path.join(split_directory, f"text.{dict_pattern}.int.scp")
            with open(scp_path, "w", encoding="utf8") as f:
                for utt in sorted(text_ints.keys()):
                    f.write(f"{utt} {text_ints[utt]}\n")
            scp_path = os.path.join(split_directory, f"text.{dict_pattern}.scp")
            with open(scp_path, "w", encoding="utf8") as f:
                for utt in sorted(texts.keys()):
                    f.write(f"{utt} {texts[utt]}\n")

        spk2utt = {}
        feats = {}
        cmvns = {}
        utt2spk = {}
        text_ints = {}
        texts = {}
        _current_dict_id = None
        if not self.dictionary_ids:
            utterances = (
                session.query(
                    Utterance.id,
                    Utterance.speaker_id,
                    Utterance.features,
                    Utterance.normalized_text,
                    Utterance.normalized_text_int,
                    Speaker.cmvn,
                )
                .join(Utterance.speaker)
                .filter(Speaker.job_id == self.name)
                .filter(Utterance.ignored == False)  # noqa
                .order_by(Utterance.kaldi_id)
            )
            if subset:
                utterances = utterances.filter(Utterance.in_subset == True)  # noqa
            for u_id, s_id, features, normalized_text, normalized_text_int, cmvn in utterances:
                utterance = str(u_id)
                speaker = str(s_id)
                utterance = f"{speaker}-{utterance}"
                if speaker not in spk2utt:
                    spk2utt[speaker] = []
                spk2utt[speaker].append(utterance)
                utt2spk[utterance] = speaker
                feats[utterance] = features
                cmvns[speaker] = cmvn
                text_ints[utterance] = normalized_text_int
                texts[utterance] = normalized_text
            _write_current()
            return
        for _current_dict_id in self.dictionary_ids:
            spk2utt = {}
            feats = {}
            cmvns = {}
            utt2spk = {}
            text_ints = {}
            utterances = (
                session.query(
                    Utterance.kaldi_id,
                    Utterance.speaker_id,
                    Utterance.features,
                    Utterance.normalized_text,
                    Utterance.normalized_text_int,
                    Speaker.cmvn,
                )
                .join(Utterance.speaker)
                .filter(Speaker.job_id == self.name)
                .filter(Speaker.dictionary_id == _current_dict_id)
                .filter(Utterance.ignored == False)  # noqa
                .order_by(Utterance.kaldi_id)
            )
            if subset:
                utterances = utterances.filter(Utterance.in_subset == True)  # noqa
            for (
                utterance,
                s_id,
                features,
                normalized_text,
                normalized_text_int,
                cmvn,
            ) in utterances:
                speaker = str(s_id)
                if speaker not in spk2utt:
                    spk2utt[speaker] = []
                spk2utt[speaker].append(utterance)
                utt2spk[utterance] = speaker
                feats[utterance] = features
                cmvns[speaker] = cmvn
                text_ints[utterance] = normalized_text_int
                texts[utterance] = normalized_text
            _write_current()
        for d in no_data:
            ind = self.dictionary_ids.index(d)
            self.dictionary_ids.pop(ind)
