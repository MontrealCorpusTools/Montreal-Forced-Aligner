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
from typing import TYPE_CHECKING, Collection, Optional, Union

from montreal_forced_aligner.exceptions import TextGridParseError, TextParseError
from montreal_forced_aligner.helper import output_mapping

if TYPE_CHECKING:

    from montreal_forced_aligner.abc import OneToManyMappingType, OneToOneMappingType
    from montreal_forced_aligner.corpus.helper import SoundFileInfoDict

    FileInfoDict = dict[
        str, Union[str, SoundFileInfoDict, OneToOneMappingType, OneToManyMappingType]
    ]
    from montreal_forced_aligner.abc import MappingType, ReversedMappingType, WordsType
    from montreal_forced_aligner.corpus.classes import File, Speaker, Utterance
    from montreal_forced_aligner.dictionary import DictionaryData, PronunciationDictionaryMixin
    from montreal_forced_aligner.utils import Stopped


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
    stopped: :func:`~montreal_forced_aligner.utils.Stopped`
        Stop check for whether corpus loading should exit
    finished_adding: :class:`~montreal_forced_aligner.utils.Stopped`
        Signal that the main thread has stopped adding new files to be processed
    """

    def __init__(
        self,
        job_q: mp.Queue,
        return_dict: dict,
        return_q: mp.Queue,
        stopped: Stopped,
        finished_adding: Stopped,
    ):
        mp.Process.__init__(self)
        self.job_q = job_q
        self.return_dict = return_dict
        self.return_q = return_q
        self.stopped = stopped
        self.finished_adding = finished_adding

    def run(self) -> None:
        """
        Run the corpus loading job
        """
        from ..corpus.classes import parse_file

        while True:
            try:
                arguments = self.job_q.get(timeout=1)
            except Empty:
                if self.finished_adding.stop_check():
                    break
                continue
            self.job_q.task_done()
            if self.stopped.stop_check():
                continue
            try:
                file = parse_file(*arguments, stop_check=self.stopped.stop_check)
                self.return_q.put(file)
            except TextParseError as e:
                self.return_dict["decode_error_files"].append(e)
            except TextGridParseError as e:
                self.return_dict["textgrid_read_errors"][e.file_name] = e
            except Exception:
                self.stopped.stop()
                self.return_dict["error"] = arguments, Exception(
                    traceback.format_exception(*sys.exc_info())
                )
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
    speakers: list[Speaker]
    subset_utts: set[Utterance]
    subset_speakers: set[Speaker]
    dictionaries: set[PronunciationDictionaryMixin]
    subset_dictionaries: set[PronunciationDictionaryMixin]

    def __init__(self, name: int):
        self.name = name
        self.speakers = []
        self.dictionaries = set()

        self.subset_utts = set()
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
        self.speakers.append(speaker)
        self.dictionaries.add(speaker.dictionary)

    def set_subset(self, subset_utts: Optional[Collection[Utterance]]) -> None:
        """
        Set the current subset for the trainer

        Parameters
        ----------
        subset_utts: Collection[:class:`~montreal_forced_aligner.corpus.classes.Utterance`], optional
            Subset of utterances for this job to use
        """
        if subset_utts is None:
            self.subset_utts = set()
            self.subset_speakers = set()
            self.subset_dictionaries = set()
        else:
            self.subset_utts = set(u for u in subset_utts if u.speaker in self.speakers)
            self.subset_speakers = {u.speaker for u in subset_utts if u.speaker in self.speakers}
            self.subset_dictionaries = {s.dictionary for s in self.subset_speakers}

    def text_scp_data(self) -> dict[str, dict[str, list[str]]]:
        """
        Generate the job's data for Kaldi's text scp files

        Returns
        -------
        dict[str, dict[str, list[str]]]
            Text for each utterance, per dictionary name
        """
        data = {}
        utts = self.job_utts()
        for dict_name, utt_data in utts.items():
            data[dict_name] = {}
            for utt in utt_data.values():
                if not utt.text:
                    continue
                data[dict_name][utt.name] = " ".join(map(str, utt.text_for_scp()))
        return data

    def text_int_scp_data(self) -> dict[str, dict[str, str]]:
        """
        Generate the job's data for Kaldi's text int scp files

        Returns
        -------
        dict[str, dict[str, str]]
            Text converted to integer IDs for each utterance, per dictionary name
        """
        data = {}
        utts = self.job_utts()
        for dict_name, utt_data in utts.items():
            data[dict_name] = {}
            for utt in utt_data.values():
                if utt.speaker.dictionary is None:
                    continue
                if not utt.text:
                    continue
                data[dict_name][utt.name] = " ".join(map(str, utt.text_int_for_scp()))
        return data

    def wav_scp_data(self) -> dict[str, dict[str, str]]:
        """
        Generate the job's data for Kaldi's wav scp files

        Returns
        -------
        dict[str, dict[str, str]]
            Wav scp strings for each file, per dictionary name
        """
        data = {}
        done = {}
        utts = self.job_utts()
        for dict_name, utt_data in utts.items():
            data[dict_name] = {}
            done[dict_name] = set()
            for utt in utt_data.values():
                if not utt.is_segment:
                    data[dict_name][utt.name] = utt.file.for_wav_scp()
                elif utt.file.name not in done:
                    data[dict_name][utt.file.name] = utt.file.for_wav_scp()
                    done[dict_name].add(utt.file.name)
        return data

    def utt2spk_scp_data(self) -> dict[str, dict[str, str]]:
        """
        Generate the job's data for Kaldi's utt2spk scp files

        Returns
        -------
        dict[str, dict[str, str]]
            Utterance to speaker mapping, per dictionary name
        """
        data = {}
        utts = self.job_utts()
        for dict_name, utt_data in utts.items():
            data[dict_name] = {}
            for utt in utt_data.values():
                data[dict_name][utt.name] = utt.speaker_name
        return data

    def feat_scp_data(self) -> dict[str, dict[str, str]]:
        """
        Generate the job's data for Kaldi's feature scp files

        Returns
        -------
        dict[str, dict[str, str]]
            Utterance to feature archive ID mapping, per dictionary name
        """
        data = {}
        utts = self.job_utts()
        for dict_name, utt_data in utts.items():
            data[dict_name] = {}
            for utt in utt_data.values():
                if not utt.features:
                    continue
                data[dict_name][utt.name] = utt.features
        return data

    def spk2utt_scp_data(self) -> dict[str, dict[str, list[str]]]:
        """
        Generate the job's data for Kaldi's spk2utt scp files

        Returns
        -------
        dict[str, dict[str, list[str]]]
            Speaker to utterance mapping, per dictionary name
        """
        data = {}
        utts = self.job_utts()
        for dict_name, utt_data in utts.items():
            data[dict_name] = {}
            for utt in utt_data.values():
                if utt.speaker.name not in data[dict_name]:
                    data[dict_name][utt.speaker.name] = []
                data[dict_name][utt.speaker.name].append(str(utt))
        for k, v in data.items():
            for s, utts in v.items():
                data[k][s] = sorted(utts)
        return data

    def cmvn_scp_data(self) -> dict[str, dict[str, str]]:
        """
        Generate the job's data for Kaldi's CMVN scp files

        Returns
        -------
        dict[str, dict[str, str]]
            Speaker to CMVN mapping, per dictionary name
        """
        data = {}
        for s in self.speakers:
            if s.dictionary is None:
                key = None
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            if s.cmvn:
                data[key][s.name] = s.cmvn
        return data

    def segments_scp_data(self) -> dict[str, dict[str, str]]:
        """
        Generate the job's data for Kaldi's segments scp files

        Returns
        -------
        dict[str, dict[str, str]]
            Utterance to segment mapping, per dictionary name
        """
        data = {}
        utts = self.job_utts()
        for dict_name, utt_data in utts.items():
            data[dict_name] = {}
            for utt in utt_data.values():
                if not utt.is_segment:
                    continue
                data[dict_name][utt.name] = utt.segment_for_scp()
        return data

    def construct_path_dictionary(
        self, directory: str, identifier: str, extension: str
    ) -> dict[str, str]:
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

    def construct_dictionary_dependent_paths(
        self, directory: str, identifier: str, extension: str
    ) -> dict[str, str]:
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
            return self.subset_dictionaries
        return self.dictionaries

    @property
    def current_dictionary_names(self) -> list[Optional[str]]:
        """Current dictionary names depending on whether a subset is being used"""
        if self.subset_dictionaries:
            return sorted(x.name for x in self.subset_dictionaries)
        if self.dictionaries == {None}:
            return [None]
        return sorted(x.name for x in self.dictionaries)

    def word_boundary_int_files(self) -> dict[str, str]:
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

    def reversed_phone_mappings(self) -> dict[str, ReversedMappingType]:
        """
        Generate mapping for dictionaries to reversed phone mapping

        Returns
        -------
        dict[str, ReversedMappingType]
            Per dictionary reversed phone mapping
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.reversed_phone_mapping
        return data

    def reversed_word_mappings(self) -> dict[str, ReversedMappingType]:
        """
        Generate mapping for dictionaries to reversed word mapping

        Returns
        -------
        dict[str, ReversedMappingType]
            Per dictionary reversed word mapping
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.reversed_word_mapping
        return data

    def words_mappings(self) -> dict[str, MappingType]:
        """
        Generate mapping for dictionaries to word mapping

        Returns
        -------
        dict[str, MappingType]
            Per dictionary word mapping
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.words_mapping
        return data

    def words(self) -> dict[str, WordsType]:
        """
        Generate mapping for dictionaries to words

        Returns
        -------
        dict[str, WordsType]
            Per dictionary words
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.words
        return data

    def punctuation(self):
        """
        Generate mapping for dictionaries to punctuation

        Returns
        -------
        dict[str, str]
            Per dictionary punctuation
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.punctuation
        return data

    def clitic_set(self) -> dict[str, set[str]]:
        """
        Generate mapping for dictionaries to clitic sets

        Returns
        -------
        dict[str, str]
            Per dictionary clitic sets
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.clitic_set
        return data

    def clitic_markers(self) -> dict[str, list[str]]:
        """
        Generate mapping for dictionaries to clitic markers

        Returns
        -------
        dict[str, str]
            Per dictionary clitic markers
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.clitic_markers
        return data

    def compound_markers(self) -> dict[str, list[str]]:
        """
        Generate mapping for dictionaries to compound markers

        Returns
        -------
        dict[str, str]
            Per dictionary compound markers
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.compound_markers
        return data

    def strip_diacritics(self) -> dict[str, list[str]]:
        """
        Generate mapping for dictionaries to diacritics to strip

        Returns
        -------
        dict[str, list[str]]
            Per dictionary strip diacritics
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.strip_diacritics
        return data

    def oov_codes(self) -> dict[str, str]:
        """
        Generate mapping for dictionaries to oov symbols

        Returns
        -------
        dict[str, str]
            Per dictionary oov symbols
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.oov_word
        return data

    def oov_ints(self) -> dict[str, int]:
        """
        Generate mapping for dictionaries to oov ints

        Returns
        -------
        dict[str, int]
            Per dictionary oov ints
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.oov_int
        return data

    def positions(self) -> dict[str, list[str]]:
        """
        Generate mapping for dictionaries to positions

        Returns
        -------
        dict[str, list[str]]
            Per dictionary positions
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.positions
        return data

    def silences(self) -> dict[str, set[str]]:
        """
        Generate mapping for dictionaries to silence symbols

        Returns
        -------
        dict[str, set[str]]
            Per dictionary silence symbols
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.silences
        return data

    def multilingual_ipa(self) -> dict[str, bool]:
        """
        Generate mapping for dictionaries to multilingual IPA flags

        Returns
        -------
        dict[str, bool]
            Per dictionary multilingual IPA flags
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.multilingual_ipa
        return data

    def job_utts(self) -> dict[str, dict[str, Utterance]]:
        """
        Generate utterances by dictionary name for the Job

        Returns
        -------
        dict[str, dict[str, :class:`~montreal_forced_aligner.corpus.classes.Utterance`]]
            Mapping of dictionary name to Utterance mappings
        """
        data = {}
        if self.subset_utts:
            utterances = self.subset_utts
        else:
            utterances = set()
            for s in self.speakers:
                utterances.update(s.utterances.values())
        for u in utterances:
            if u.ignored:
                continue
            if u.speaker.dictionary is None:
                dict_name = None
            else:
                dict_name = u.speaker.dictionary.name
            if dict_name not in data:
                data[dict_name] = {}
            data[dict_name][u.name] = u

        return data

    def job_files(self) -> dict[str, File]:
        """
        Generate files for the Job

        Returns
        -------
        dict[str, :class:`~montreal_forced_aligner.corpus.classes.File`]
            Mapping of file name to File objects
        """
        data = {}
        if self.subset_utts:
            utterances = self.subset_utts
        else:
            utterances = set()
            for s in self.speakers:
                utterances.update(s.utterances.values())
        for u in utterances:
            if u.ignored:
                continue
            data[u.file_name] = u.file
        return data

    def job_speakers(self) -> dict[str, Speaker]:
        """
        Generate files for the Job

        Returns
        -------
        dict[str, :class:`~montreal_forced_aligner.corpus.classes.Speaker`]
            Mapping of file name to File objects
        """
        data = {}
        if self.subset_speakers:
            speakers = self.subset_speakers
        else:
            speakers = self.speakers
        for s in speakers:
            data[s.name] = s
        return data

    def dictionary_data(self) -> dict[str, DictionaryData]:
        """
        Generate dictionary data for the job

        Returns
        -------
        dict[str, DictionaryData]
            Mapping of dictionary name to dictionary data
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.data()
        return data

    def output_to_directory(self, split_directory: str) -> None:
        """
        Output job information to a directory

        Parameters
        ----------
        split_directory: str
            Directory to output to
        """
        wav = self.wav_scp_data()
        for dict_name, scp in wav.items():
            wav_scp_path = os.path.join(split_directory, f"wav.{dict_name}.{self.name}.scp")
            output_mapping(scp, wav_scp_path, skip_safe=True)

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

        segments = self.segments_scp_data()
        for dict_name, scp in segments.items():
            segments_scp_path = os.path.join(
                split_directory, f"segments.{dict_name}.{self.name}.scp"
            )
            output_mapping(scp, segments_scp_path)

        text_scp = self.text_scp_data()
        for dict_name, scp in text_scp.items():
            if not scp:
                continue
            text_scp_path = os.path.join(split_directory, f"text.{dict_name}.{self.name}.scp")
            output_mapping(scp, text_scp_path)

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
