"""
Corpus loading worker
---------------------
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import traceback
from queue import Empty
from typing import TYPE_CHECKING, Any, Collection, Dict, Generator, List, Optional, Set, Union

from montreal_forced_aligner.corpus.classes import (
    File,
    FileCollection,
    SpeakerCollection,
    Utterance,
    UtteranceCollection,
)
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerSanitizationFunction
from montreal_forced_aligner.exceptions import TextGridParseError, TextParseError
from montreal_forced_aligner.helper import output_mapping
from montreal_forced_aligner.utils import Stopped

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import OneToManyMappingType, OneToOneMappingType
    from montreal_forced_aligner.corpus.helper import SoundFileInfoDict

    FileInfoDict = Dict[
        str, Union[str, SoundFileInfoDict, OneToOneMappingType, OneToManyMappingType]
    ]
    from montreal_forced_aligner.corpus.classes import Speaker
    from montreal_forced_aligner.dictionary import PronunciationDictionaryMixin


__all__ = ["CorpusProcessWorker", "Job"]


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
        return_dict: dict,
        return_q: mp.Queue,
        stopped: Stopped,
        finished_adding: Stopped,
        speaker_characters: Union[int, str],
        sanitize_function: Optional[MultispeakerSanitizationFunction],
    ):
        mp.Process.__init__(self)
        self.name = str(name)
        self.job_q = job_q
        self.return_dict = return_dict
        self.return_q = return_q
        self.stopped = stopped
        self.finished_adding = finished_adding
        self.finished_processing = Stopped()
        self.sanitize_function = sanitize_function
        self.speaker_characters = speaker_characters

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
            self.job_q.task_done()
            if self.stopped.stop_check():
                continue
            try:
                file = File.parse_file(
                    file_name,
                    wav_path,
                    text_path,
                    relative_path,
                    self.speaker_characters,
                    self.sanitize_function,
                )

                self.return_q.put(file.multiprocessing_data)
            except TextParseError as e:
                self.return_dict["decode_error_files"].append(e)
            except TextGridParseError as e:
                self.return_dict["textgrid_read_errors"][e.file_name] = e
            except Exception:
                self.stopped.stop()
                self.return_dict["error"] = file_name, Exception(
                    traceback.format_exception(*sys.exc_info())
                )
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

    Attributes
    ----------
    speakers: list[:class:`~montreal_forced_aligner.corpus.classes.Speaker`]
        List of speakers associated with this job
    dictionaries: set[:class:`~montreal_forced_aligner.dictionary.PronunciationDictionary`]
        Set of dictionaries that the job's speakers use
    subset_utts: set[:class:`~montreal_forced_aligner.corpus.classes.Utterance`]
        When trainers are just using a subset of the corpus, the subset of utterances on each job will be set and used to
        filter the job's utterances
    subset_speakers: set[:class:`~montreal_forced_aligner.corpus.classes.Speaker`]
        When subset_utts is set, this property will be calculated as the subset of speakers that the utterances correspond to
    subset_dictionaries: set[:class:`~montreal_forced_aligner.dictionary.PronunciationDictionary`]
        Subset of dictionaries that the subset of speakers use

    """

    name: int
    speakers: SpeakerCollection
    dictionaries: Set[PronunciationDictionaryMixin]

    def __init__(self, name: int):
        self.name = name
        self.speakers = SpeakerCollection()
        self.files = FileCollection()
        self.dictionaries = set()

        self.subset_utts = set()
        self.subset_files = set()
        self.subset_speakers = set()
        self.subset_dictionaries = set()

    def add_speaker(self, speaker: Speaker) -> None:
        """
        Add a speaker to a job

        Parameters
        ----------
        speaker: :class:`~montreal_forced_aligner.corpus.classes.Speaker`
            Speaker to add
        """
        self.speakers.add_speaker(speaker)
        for u in speaker.utterances:
            self.files.add_file(u.file)
        self.dictionaries.add(speaker.dictionary)

    def set_subset(self, subset_utts: Optional[UtteranceCollection]) -> None:
        """
        Set the current subset for the trainer

        Parameters
        ----------
        subset_utts: Collection[:class:`~montreal_forced_aligner.corpus.classes.Utterance`], optional
            Subset of utterances for this job to use
        """
        self.subset_utts = set()
        self.subset_files = set()
        self.subset_speakers = set()
        self.subset_dictionaries = set()
        if subset_utts:
            for u in subset_utts:
                if u.speaker not in self.speakers:
                    continue
                self.subset_utts.add(u.name)
                self.subset_speakers.add(u.speaker_name)
                self.subset_files.add(u.file_name)
                dict_name = self.speakers[u.speaker_name].dictionary_name
                if dict_name is not None:
                    self.subset_dictionaries.add(dict_name)

    def text_scp_data(self) -> Dict[str, Dict[str, str]]:
        """
        Generate the job's data for Kaldi's text scp files

        Returns
        -------
        dict[str, dict[str, str]]
            Text for each utterance, per dictionary name
        """
        data = {x: {} for x in self.current_dictionary_names}
        for utt in self.current_utterances:
            dict_name = utt.speaker.dictionary_name
            if not utt.text:
                continue
            data[dict_name][utt.name] = utt.text
        return data

    def text_int_scp_data(self) -> Dict[str, Dict[str, str]]:
        """
        Generate the job's data for Kaldi's text int scp files

        Returns
        -------
        dict[str, dict[str, str]]
            Text converted to integer IDs for each utterance, per dictionary name
        """
        data = {x: {} for x in self.current_dictionary_names}
        for utt in self.current_utterances:
            dict_name = utt.speaker.dictionary_name
            if dict_name is None:
                continue
            if not utt.text:
                continue
            data[dict_name][utt.name] = " ".join(map(str, utt.text_int_for_scp()))
            utt.speaker.dictionary.oovs_found.update(utt.oovs)
        return data

    def wav_scp_data(self) -> Dict[str, str]:
        """
        Generate the job's data for Kaldi's wav scp files

        Returns
        -------
        dict[str, str]
            Wav scp strings for each file, per dictionary name
        """
        data = {}
        for file in self.files:
            if any(u.is_segment for u in file.utterances):  # Segmented file
                data[file.name] = file.for_wav_scp()
            else:
                for utt in file.utterances:
                    data[utt.name] = file.for_wav_scp()
        return data

    def utt2spk_scp_data(self) -> Dict[str, Dict[str, str]]:
        """
        Generate the job's data for Kaldi's utt2spk scp files

        Returns
        -------
        dict[str, dict[str, str]]
            Utterance to speaker mapping, per dictionary name
        """
        data = {x: {} for x in self.current_dictionary_names}
        for utt in self.current_utterances:
            dict_name = utt.speaker.dictionary_name
            data[dict_name][utt.name] = utt.speaker_name
        return data

    def feat_scp_data(self) -> Dict[str, Dict[str, str]]:
        """
        Generate the job's data for Kaldi's feature scp files

        Returns
        -------
        dict[str, dict[str, str]]
            Utterance to feature archive ID mapping, per dictionary name
        """
        data = {x: {} for x in self.current_dictionary_names}
        for utt in self.current_utterances:
            if utt.features is None:
                continue
            dict_name = utt.speaker.dictionary_name
            data[dict_name][utt.name] = utt.features
        return data

    def spk2utt_scp_data(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Generate the job's data for Kaldi's spk2utt scp files

        Returns
        -------
        dict[str, dict[str, list[str]]]
            Speaker to utterance mapping, per dictionary name
        """
        data = {x: {} for x in self.current_dictionary_names}
        for utt in self.current_utterances:
            dict_name = utt.speaker.dictionary_name
            if utt.speaker_name not in data[dict_name]:
                data[dict_name][utt.speaker_name] = []
            data[dict_name][utt.speaker_name].append(str(utt))
        for k, v in data.items():
            for s, utts in v.items():
                data[k][s] = sorted(utts)
        return data

    def cmvn_scp_data(self) -> Dict[str, Dict[str, str]]:
        """
        Generate the job's data for Kaldi's CMVN scp files

        Returns
        -------
        dict[str, dict[str, str]]
            Speaker to CMVN mapping, per dictionary name
        """
        data = {x: {} for x in self.current_dictionary_names}
        for s in self.current_speakers:
            if s.dictionary is None:
                key = None
            else:
                key = s.dictionary_name
            if s.cmvn:
                data[key][s.name] = s.cmvn
        return data

    def segments_scp_data(self) -> Dict[str, List[Any]]:
        """
        Generate the job's data for Kaldi's segments scp files

        Returns
        -------
        dict[str, list[Any]]
            Utterance to segment mapping, per dictionary name
        """
        data = {}
        for utt in self.current_utterances:
            if not utt.is_segment:
                continue
            data[utt.name] = utt.segment_for_scp()
        return data

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
        for dict_name in self.current_dictionary_names:
            output[dict_name] = os.path.join(
                directory, f"{identifier}.{dict_name}.{self.name}.{extension}"
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
        for dict_name in self.current_dictionary_names:
            output[dict_name] = os.path.join(directory, f"{identifier}.{dict_name}.{extension}")
        return output

    @property
    def dictionary_count(self):
        """Number of dictionaries currently used"""
        if self.subset_dictionaries:
            return len(self.subset_dictionaries)
        return len(self.dictionaries)

    @property
    def current_dictionaries(self) -> Collection[PronunciationDictionaryMixin]:
        """Current dictionaries depending on whether a subset is being used"""
        if self.subset_dictionaries:
            return [x for x in self.dictionaries if x.name in self.subset_dictionaries]
        return self.dictionaries

    @property
    def current_dictionary_names(self) -> List[Optional[str]]:
        """Current dictionary names depending on whether a subset is being used"""
        if self.subset_dictionaries:
            return sorted(self.subset_dictionaries)
        if self.dictionaries == {None}:
            return [None]
        return sorted(x.name for x in self.dictionaries)

    @property
    def current_speakers(self) -> Generator[Speaker]:
        """Current dictionary names depending on whether a subset is being used"""

        return self.speakers.subset(self.subset_speakers)

    @property
    def current_utterances(self) -> Generator[Utterance]:
        """Current dictionary names depending on whether a subset is being used"""
        for s in self.current_speakers:
            for u in s.utterances.subset(self.subset_utts):
                yield u

    def word_boundary_int_files(self) -> Dict[str, str]:
        """
        Generate mapping for dictionaries to word boundary int files

        Returns
        -------
        dict[str, str]
            Per dictionary word boundary int files
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = os.path.join(dictionary.phones_dir, "word_boundary.int")
        return data

    def output_for_features(self, split_directory: str) -> None:
        """
        Output the necessary files for Kaldi to generate features

        Parameters
        ----------
        split_directory: str
            Split directory for the corpus
        """
        wav_scp_path = self.construct_path(split_directory, "wav", "scp")
        if not os.path.exists(wav_scp_path):
            wav_scp = self.wav_scp_data()
            output_mapping(wav_scp, wav_scp_path, skip_safe=True)

        segments_scp_path = self.construct_path(split_directory, "segments", "scp")
        if not os.path.exists(segments_scp_path):
            segments = self.segments_scp_data()
            output_mapping(segments, segments_scp_path)

    def output_to_directory(self, split_directory: str) -> None:
        """
        Output job information to a directory

        Parameters
        ----------
        split_directory: str
            Directory to output to
        """

        spk2utt = self.spk2utt_scp_data()
        for dict_name, scp in spk2utt.items():
            spk2utt_scp_path = os.path.join(
                split_directory, f"spk2utt.{dict_name}.{self.name}.scp"
            )

            output_mapping(scp, spk2utt_scp_path)

        feats = self.feat_scp_data()
        for dict_name, scp in feats.items():
            feats_scp_path = os.path.join(split_directory, f"feats.{dict_name}.{self.name}.scp")
            output_mapping(scp, feats_scp_path)

        cmvn = self.cmvn_scp_data()
        for dict_name, scp in cmvn.items():
            cmvn_scp_path = os.path.join(split_directory, f"cmvn.{dict_name}.{self.name}.scp")
            output_mapping(scp, cmvn_scp_path)

        utt2spk = self.utt2spk_scp_data()
        for dict_name, scp in utt2spk.items():
            utt2spk_scp_path = os.path.join(
                split_directory, f"utt2spk.{dict_name}.{self.name}.scp"
            )
            output_mapping(scp, utt2spk_scp_path)

        text_scp = self.text_scp_data()
        for dict_name, scp in text_scp.items():
            if not scp:
                continue
            text_scp_path = os.path.join(split_directory, f"text.{dict_name}.{self.name}.scp")
            output_mapping(scp, text_scp_path, skip_safe=True)

        text_int = self.text_int_scp_data()
        for dict_name, scp in text_int.items():
            if dict_name is None:
                continue
            if not scp:
                continue
            text_int_scp_path = os.path.join(
                split_directory, f"text.{dict_name}.{self.name}.int.scp"
            )
            output_mapping(scp, text_int_scp_path, skip_safe=True)
