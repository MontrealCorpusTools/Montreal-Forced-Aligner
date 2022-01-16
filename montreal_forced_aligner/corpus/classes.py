"""Class definitions for Speakers, Files, Utterances and Jobs"""
from __future__ import annotations

import os
import re
import sys
import traceback
from collections import Counter
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import librosa
import numpy as np
from praatio import textgrid
from praatio.utilities.constants import Interval

from montreal_forced_aligner.abc import MfaCorpusClass
from montreal_forced_aligner.corpus.helper import get_wav_info, load_text
from montreal_forced_aligner.data import (
    CtmInterval,
    FileData,
    SoundFileInformation,
    SoundFileType,
    TextFileType,
    UtteranceData,
)
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerSanitizationFunction
from montreal_forced_aligner.exceptions import CorpusError, TextGridParseError, TextParseError

if TYPE_CHECKING:
    from montreal_forced_aligner.dictionary.pronunciation import PronunciationDictionaryMixin


__all__ = ["File", "Speaker", "Utterance"]


class Speaker(MfaCorpusClass):
    """
    Class representing information about a speaker

    Parameters
    ----------
    name: str
        Identifier of the speaker

    Attributes
    ----------
    utterances: :class:`~montreal_forced_aligner.corpus.classes.UtteranceCollection`
        Utterances that the speaker is associated with
    cmvn: str, optional
        String pointing to any CMVN that has been calculated for this speaker
    dictionary: :class:`~montreal_forced_aligner.dictionary.PronunciationDictionary`, optional
        Pronunciation dictionary that the speaker is associated with
    """

    def __init__(self, name):
        self._name = name
        self.utterances = UtteranceCollection()
        self.cmvn = None
        self.dictionary: Optional[PronunciationDictionaryMixin] = None
        self.dictionary_name: Optional[str] = None
        self.word_counts = Counter()

    @property
    def name(self) -> str:
        """Speaker name"""
        return self._name

    def __str__(self) -> str:
        """Return Speaker's name"""
        return self.name

    def __eq__(self, other: Union[Speaker, str]) -> bool:
        """Check if a Speaker is equal to another Speaker"""
        if isinstance(other, Speaker):
            return other.name == self.name
        if isinstance(other, str):
            return self.name == other
        raise TypeError("Speakers can only be compared to other speakers and strings.")

    def __lt__(self, other: Union[Speaker, str]) -> bool:
        """Check if a Speaker is less than another Speaker"""
        if isinstance(other, Speaker):
            return other.name < self.name
        if isinstance(other, str):
            return self.name < other
        raise TypeError("Speakers can only be compared to other speakers and strings.")

    def __gt__(self, other: Union[Speaker, str]) -> bool:
        """Check if a Speaker is greater than another Speaker"""
        if isinstance(other, Speaker):
            return other.name > self.name
        if isinstance(other, str):
            return self.name > other
        raise TypeError("Speakers can only be compared to other speakers and strings.")

    def __hash__(self) -> hash:
        """Get the hash of the speaker"""
        return hash(self.name)

    @property
    def num_utterances(self) -> int:
        """Get the number of utterances for the speaker"""
        return len(self.utterances)

    def add_utterance(self, utterance: Utterance) -> None:
        """
        Associate an utterance with a speaker

        Parameters
        ----------
        utterance: :class:`~montreal_forced_aligner.corpus.classes.Utterance`
            Utterance to be added
        """
        self.utterances.add_utterance(utterance)

    def delete_utterance(self, utterance: Utterance) -> None:
        """
        Delete an utterance associated with a speaker

        Parameters
        ----------
        utterance: :class:`~montreal_forced_aligner.corpus.classes.Utterance`
            Utterance to be deleted
        """
        identifier = utterance.name
        del self.utterances[identifier]

    def set_dictionary(self, dictionary: PronunciationDictionaryMixin) -> None:
        """
        Set the dictionary for the speaker

        Parameters
        ----------
        dictionary: :class:`~montreal_forced_aligner.dictionary.PronunciationDictionary`
            Pronunciation dictionary to associate with the speaker
        """
        self.dictionary = dictionary
        self.dictionary_name = dictionary.name

    @property
    def meta(self) -> Dict[str, str]:
        """Metadata for the speaker"""
        data = {
            "name": self.name,
            "cmvn": self.cmvn,
        }
        if self.dictionary is not None:
            data["dictionary"] = self.dictionary.name
        return data


class File(MfaCorpusClass):
    """
    File class for representing metadata and associations of Files

    Parameters
    ----------
    wav_path: str, optional
        Sound file path
    text_path: str, optional
        Transcription file path
    relative_path: str, optional
        Relative path to the corpus root

    Attributes
    ----------
    utterances: :class:`~montreal_forced_aligner.corpus.classes.UtteranceCollection`
        Utterances in the file
    speaker_ordering: list[Speaker]
        Ordering of speakers in the transcription file
    wav_info: :class:`~montreal_forced_aligner.data.SoundFileInformation`
        Information about sound file
    waveform: numpy.array
        Audio samples
    aligned: bool
        Flag for whether a file has alignments

    Raises
    ------
    :class:`~montreal_forced_aligner.exceptions.CorpusError`
        If both wav_path and text_path are None
    """

    textgrid_regex = re.compile(r"\.textgrid$", flags=re.IGNORECASE)
    wav_regex = re.compile(r"\.wav$", flags=re.IGNORECASE)

    def __init__(
        self,
        wav_path: Optional[str] = None,
        text_path: Optional[str] = None,
        relative_path: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self.wav_path = wav_path
        self.text_path = text_path
        wav_check = self.wav_path is not None
        text_check = self.text_path is not None
        self._name = name
        if not self._name:
            if wav_check:
                self._name = os.path.splitext(os.path.basename(self.wav_path))[0]
            elif text_check:
                self._name = os.path.splitext(os.path.basename(self.text_path))[0]
            else:
                raise CorpusError("File objects must have either a wav_path or text_path")
        self.relative_path = relative_path
        self.wav_info: Optional[SoundFileInformation] = None
        self.waveform = None
        self.speaker_ordering: List[Speaker] = []
        self.utterances = UtteranceCollection()
        self.aligned = False
        if text_check:
            if self.text_path.lower().endswith(".textgrid"):
                self.text_type = TextFileType.TEXTGRID
            else:
                self.text_type = TextFileType.LAB
        else:
            self.text_type = TextFileType.NONE
        if wav_check:

            if self.wav_path.lower().endswith(".wav"):
                self.sound_type = SoundFileType.WAV
            else:
                self.sound_type = SoundFileType.SOX
        else:
            self.sound_type = SoundFileType.NONE

    @property
    def multiprocessing_data(self) -> FileData:
        """
        Data object for the file
        """
        return FileData(
            self._name,
            self.wav_path,
            self.text_path,
            self.relative_path,
            self.wav_info,
            [s.name for s in self.speaker_ordering],
            [u.multiprocessing_data for u in self.utterances],
        )

    @classmethod
    def load_from_mp_data(cls, file_data: FileData) -> File:
        """
        Construct a File from a multiprocessing file data class

        Parameters
        ----------
        file_data: :class:`~montreal_forced_aligner.data.FileData`
            Data for the loaded file

        Returns
        -------
        :class:`~montreal_forced_aligner.corpus.classes.File`
            Loaded file
        """
        file = File(
            file_data.wav_path,
            file_data.text_path,
            relative_path=file_data.relative_path,
            name=file_data.name,
        )
        file.wav_info = file_data.wav_info
        for s in file_data.speaker_ordering:
            file.add_speaker(Speaker(s))
        for u in file_data.utterances:
            u = Utterance.load_from_mp_data(u, file)
            file.utterances.add_utterance(u)
        return file

    @classmethod
    def parse_file(
        cls,
        file_name: str,
        wav_path: Optional[str],
        text_path: Optional[str],
        relative_path: str,
        speaker_characters: Union[int, str],
        sanitize_function: Optional[MultispeakerSanitizationFunction] = None,
    ):
        """
        Parse a collection of sound file and transcription file into a File

        Parameters
        ----------
        file_name: str
            File identifier
        wav_path: str
            Full sound file path
        text_path: str
            Full transcription path
        relative_path: str
            Relative path from the corpus directory root
        speaker_characters: int, optional
            Number of characters in the file name to specify the speaker
        sanitize_function: Callable, optional
            Function to sanitize words and strip punctuation

        Returns
        -------
        :class:`~montreal_forced_aligner.corpus.classes.File`
            Parsed file
        """
        file = File(wav_path, text_path, relative_path=relative_path, name=file_name)
        if file.has_sound_file:
            root = os.path.dirname(wav_path)
            file.wav_info = get_wav_info(wav_path)
        else:
            root = os.path.dirname(text_path)
        if not speaker_characters:
            speaker_name = os.path.basename(root)
        elif isinstance(speaker_characters, int):
            speaker_name = file_name[:speaker_characters]
        elif speaker_characters == "prosodylab":
            speaker_name = file_name.split("_")[1]
        else:
            speaker_name = file_name
        root_speaker = None
        if speaker_characters or file.text_type != TextFileType.TEXTGRID:
            root_speaker = Speaker(speaker_name)
        file.load_text(
            root_speaker=root_speaker,
            sanitize_function=sanitize_function,
        )
        return file

    def __eq__(self, other: Union[File, str]) -> bool:
        """Check if a File is equal to another File"""
        if isinstance(other, File):
            return other.name == self.name
        if isinstance(other, str):
            return self.name == other
        raise TypeError("Files can only be compared to other files and strings.")

    def __lt__(self, other: Union[File, str]) -> bool:
        """Check if a File is less than another File"""
        if isinstance(other, File):
            return other.name < self.name
        if isinstance(other, str):
            return self.name < other
        raise TypeError("Files can only be compared to other files and strings.")

    def __gt__(self, other: Union[File, str]) -> bool:
        """Check if a File is greater than another File"""
        if isinstance(other, File):
            return other.name > self.name
        if isinstance(other, str):
            return self.name > other
        raise TypeError("Files can only be compared to other files and strings.")

    def __hash__(self) -> hash:
        """Get the hash of the file"""
        return hash(self.name)

    @property
    def name(self) -> str:
        """Name of the file"""
        name = self._name
        if self.relative_path:
            prefix = self.relative_path.replace("/", "").replace("\\", "")
            if not name.startswith(prefix):
                name = f"{prefix}_{name}"
        return name

    @property
    def is_fully_aligned(self) -> bool:
        """
        Check if all utterances have been aligned
        """
        for u in self.utterances:
            if u.ignored:
                continue
            if u.word_labels is None:
                return False
            if u.phone_labels is None:
                return False
        return True

    def __repr__(self) -> str:
        """Representation of File objects"""
        return f'<File {self.name} Sound path="{self.wav_path}" Text path="{self.text_path}">'

    def save(
        self,
        output_directory: Optional[str] = None,
        backup_output_directory: Optional[str] = None,
        text_type: Optional[TextFileType] = None,
        save_transcription: bool = False,
    ) -> None:
        """
        Output File to TextGrid or lab.  If ``text_type`` is not specified, the original file type will be used,
        but if there was no text file for the file, it will guess lab format if there is only one utterance, otherwise
        it will output a TextGrid file.

        Parameters
        ----------
        output_directory: str, optional
            Directory to output file, if None, then it will overwrite the original file
        backup_output_directory: str, optional
            If specified, then it will check whether it would overwrite an existing file and
            instead use this directory
        text_type: TextFileType, optional
            Text type to save as, if not provided, it will use either the original file type or guess the file type
        save_transcription: bool
            Flag for whether the hypothesized transcription text should be saved instead of the default text
        """
        utterance_count = len(self.utterances)
        if text_type is None:
            text_type = self.text_type
            if text_type == TextFileType.NONE:
                if utterance_count == 1:
                    text_type = TextFileType.LAB
                else:
                    text_type = TextFileType.TEXTGRID
        if text_type == TextFileType.LAB:
            if utterance_count == 0 and os.path.exists(self.text_path) and not save_transcription:
                os.remove(self.text_path)
                return
            elif utterance_count == 0:
                return
            output_path = self.construct_output_path(
                output_directory, backup_output_directory, enforce_lab=True
            )
            with open(output_path, "w", encoding="utf8") as f:
                for u in self.utterances:
                    if save_transcription:
                        f.write(u.transcription_text if u.transcription_text else "")
                    else:
                        f.write(u.text)
            return
        elif text_type == TextFileType.TEXTGRID:
            output_path = self.construct_output_path(output_directory, backup_output_directory)
            max_time = self.duration
            tiers = {}
            for speaker in self.speaker_ordering:
                if speaker is None:
                    tiers["speech"] = textgrid.IntervalTier("speech", [], minT=0, maxT=max_time)
                else:
                    tiers[speaker] = textgrid.IntervalTier(speaker.name, [], minT=0, maxT=max_time)

            tg = textgrid.Textgrid()
            tg.maxTimestamp = max_time
            for utterance in self.utterances:

                if utterance.speaker is None:
                    speaker = "speech"
                else:
                    speaker = utterance.speaker
                if not self.aligned:

                    if save_transcription:
                        tiers[speaker].entryList.append(
                            Interval(
                                start=utterance.begin,
                                end=utterance.end,
                                label=utterance.transcription_text
                                if utterance.transcription_text
                                else "",
                            )
                        )
                    else:
                        tiers[speaker].entryList.append(
                            Interval(
                                start=utterance.begin, end=utterance.end, label=utterance.text
                            )
                        )
            for t in tiers.values():
                tg.addTier(t)
            tg.save(output_path, includeBlankSpaces=True, format="long_textgrid")

    @property
    def meta(self) -> Dict[str, Any]:
        """Metadata for the File"""
        return {
            "wav_path": self.wav_path,
            "text_path": self.text_path,
            "name": self._name,
            "relative_path": self.relative_path,
            "wav_info": self.wav_info.meta,
            "speaker_ordering": [x.name for x in self.speaker_ordering],
        }

    @property
    def has_sound_file(self) -> bool:
        """Flag for whether the File has a sound file"""
        return self.sound_type != SoundFileType.NONE

    @property
    def has_text_file(self) -> bool:
        """Flag for whether the File has a text file"""
        return self.text_type != TextFileType.NONE

    @property
    def aligned_data(self) -> Dict[str, Dict[str, List[CtmInterval]]]:
        """
        Word and phone alignments for the file

        Returns
        -------
        dict[str, dict[str, list[CtmInterval]]]
            Dictionary of word and phone intervals for each speaker in the file
        """
        data = {}
        for s in self.speaker_ordering:
            if s.name not in data:
                data[s.name] = {"words": [], "phones": []}
        for u in self.utterances:
            if u.word_labels is None:
                continue
            data[u.speaker_name]["words"].extend(u.word_labels)
            data[u.speaker_name]["phones"].extend(u.phone_labels)
        return data

    def clean_up(self) -> None:
        """
        Recombine words that were split up as part of initial text processing
        """
        for u in self.utterances:
            if not u.word_labels:
                continue
            cur_ind = 0
            actual_labels = []
            dictionary = u.speaker.dictionary
            for word in u.text.split():
                splits = dictionary.lookup(word)
                b = 1000000
                e = -1
                for w in splits:
                    cur = u.word_labels[cur_ind]
                    if w == cur.label or cur.label == dictionary.oov_word:
                        if cur.begin < b:
                            b = cur.begin
                        if cur.end > e:
                            e = cur.end
                    cur_ind += 1
                lab = CtmInterval(b, e, word, u.name)
                actual_labels.append(lab)
            u.word_labels = actual_labels
            u.phone_labels = [
                x for x in u.phone_labels if x.label != dictionary.optional_silence_phone
            ]

    def construct_output_path(
        self,
        output_directory: Optional[str] = None,
        backup_output_directory: Optional[str] = None,
        enforce_lab: bool = False,
    ) -> str:
        """
        Construct the output path for the File

        Parameters
        ----------
        output_directory: str, optional
            Directory to output to, if None, it will overwrite the original file
        backup_output_directory: str, optional
            Backup directory to write to in order to avoid overwriting an existing file
        enforce_lab: bool
            Flag for whether to enforce generating a lab file over a TextGrid

        Returns
        -------
        str
            Output path
        """
        if enforce_lab:
            extension = ".lab"
        else:
            extension = ".TextGrid"
        if output_directory is None:
            if self.text_path is None:
                return os.path.splitext(self.wav_path)[0] + extension
            return self.text_path
        if self.relative_path:
            relative = os.path.join(output_directory, self.relative_path)
        else:
            relative = output_directory
        tg_path = os.path.join(relative, self._name + extension)
        if backup_output_directory is not None and os.path.exists(tg_path):
            tg_path = tg_path.replace(output_directory, backup_output_directory)
        os.makedirs(os.path.dirname(tg_path), exist_ok=True)
        return tg_path

    def load_text(
        self,
        root_speaker: Optional[Speaker] = None,
        sanitize_function: Optional[MultispeakerSanitizationFunction] = None,
    ) -> None:
        """
        Load the transcription text from the text_file of the object

        Parameters
        ----------
        root_speaker: :class:`~montreal_forced_aligner.corpus.classes.Speaker`, optional
            Speaker derived from the root directory, ignored for TextGrids
        sanitize_function: :class:`~montreal_forced_aligner.dictionary.mixins.SanitizeFunction`, optional
            Function to sanitize words and strip punctuation
        """
        if self.text_type == TextFileType.LAB:
            try:
                text = load_text(self.text_path)
            except UnicodeDecodeError:
                raise TextParseError(self.text_path)
            utterance = Utterance(speaker=root_speaker, file=self, text=text)
            utterance.parse_transcription(sanitize_function)
            self.add_utterance(utterance)
        elif self.text_type == TextFileType.TEXTGRID:
            try:
                tg = textgrid.openTextgrid(self.text_path, includeEmptyIntervals=False)
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                raise TextGridParseError(
                    self.text_path,
                    "\n".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
                )

            num_tiers = len(tg.tierNameList)
            if num_tiers == 0:
                raise TextGridParseError(self.text_path, "Number of tiers parsed was zero")
            if self.num_channels > 2:
                raise (Exception("More than two channels"))
            for tier_name in tg.tierNameList:
                ti = tg.tierDict[tier_name]
                if tier_name.lower() == "notes":
                    continue
                if not isinstance(ti, textgrid.IntervalTier):
                    continue
                if not root_speaker:
                    speaker_name = tier_name.strip()
                    speaker = Speaker(speaker_name)
                    self.add_speaker(speaker)
                else:
                    speaker = root_speaker
                for begin, end, text in ti.entryList:
                    text = text.lower().strip()
                    if not text:
                        continue
                    begin, end = round(begin, 4), round(end, 4)
                    end = min(end, self.duration)
                    utt = Utterance(speaker=speaker, file=self, begin=begin, end=end, text=text)
                    utt.parse_transcription(sanitize_function)
                    if not utt.text:
                        continue
                    self.add_utterance(utt)
        else:
            utterance = Utterance(speaker=root_speaker, file=self)
            self.add_utterance(utterance)

    def add_speaker(self, speaker: Speaker) -> None:
        """
        Add a speaker to a file

        Parameters
        ----------
        speaker: :class:`~montreal_forced_aligner.corpus.classes.Speaker`
            Speaker to add
        """
        if speaker not in self.speaker_ordering:
            self.speaker_ordering.append(speaker)

    def add_utterance(self, utterance: Utterance) -> None:
        """
        Add an utterance to a file

        Parameters
        ----------
        utterance: :class:`~montreal_forced_aligner.corpus.classes.Utterance`
            Utterance to add
        """
        self.utterances.add_utterance(utterance)
        self.add_speaker(utterance.speaker)

    def delete_utterance(self, utterance: Utterance) -> None:
        """
        Delete an utterance from the file

        Parameters
        ----------
        utterance: :class:`~montreal_forced_aligner.corpus.classes.Utterance`
            Utterance to remove
        """
        identifier = utterance.name
        del self.utterances[identifier]

    def load_info(self) -> None:
        """
        Load sound file info if it hasn't been already
        """
        if self.wav_path is not None:
            self.wav_info = get_wav_info(self.wav_path)

    @property
    def duration(self) -> float:
        """Get the duration of the sound file"""
        if self.wav_path is None:
            return 0
        if not self.wav_info:
            self.load_info()
        return self.wav_info.duration

    @property
    def num_channels(self) -> int:
        """Get the number of channels of the sound file"""
        if self.wav_path is None:
            return 0
        if not self.wav_info:
            self.load_info()
        return self.wav_info.num_channels

    @property
    def num_utterances(self) -> int:
        """Get the number of utterances for the sound file"""
        return len(self.utterances)

    @property
    def num_speakers(self) -> int:
        """Get the number of speakers in the sound file"""
        return len(self.speaker_ordering)

    @property
    def sample_rate(self) -> int:
        """Get the sample rate of the sound file"""
        if self.wav_path is None:
            return 0
        if not self.wav_info:
            self.load_info()
        return self.wav_info.sample_rate

    @property
    def format(self) -> str:
        """Get the sound file format"""
        if not self.wav_info:
            self.load_info()
        return self.wav_info.format

    @property
    def sox_string(self) -> str:
        """String used for converting sound file via SoX within Kaldi"""
        if not self.wav_info:
            self.load_info()
        return self.wav_info.sox_string

    def load_wav_data(self) -> None:
        """
        Load the sound file into memory as a numpy array
        """
        self.waveform, _ = librosa.load(self.wav_path, sr=None, mono=False)

    def normalized_waveform(
        self, begin: float = 0, end: Optional[float] = None
    ) -> Tuple[np.array, np.array]:
        if self.waveform is None:
            self.load_wav_data()
        if end is None or end > self.duration:
            end = self.duration

        begin_sample = int(begin * self.sample_rate)
        end_sample = int(end * self.sample_rate)
        if len(self.waveform.shape) > 1 and self.waveform.shape[0] == 2:
            y = self.waveform[:, begin_sample:end_sample] / np.max(
                np.abs(self.waveform[:, begin_sample:end_sample]), axis=0
            )
            y[np.isnan(y)] = 0
        else:
            y = self.waveform[begin_sample:end_sample] / np.max(
                np.abs(self.waveform[begin_sample:end_sample]), axis=0
            )
        x = np.arange(start=begin_sample, stop=end_sample) / self.sample_rate
        return x, y

    def for_wav_scp(self) -> str:
        """
        Generate the string to use in feature generation

        Returns
        -------
        str
            SoX string if necessary, the sound file path otherwise
        """
        if self.sox_string:
            return self.sox_string
        return self.wav_path


class Utterance(MfaCorpusClass):
    """
    Class for information about specific utterances

    Parameters
    ----------
    speaker: :class:`~montreal_forced_aligner.corpus.classes.Speaker`
        Speaker of the utterance
    file: :class:`~montreal_forced_aligner.corpus.classes.File`
        File that the utterance belongs to
    begin: float, optional
        Start time of the utterance,
        if None, then the utterance is assumed to start at 0
    end: float, optional
        End time of the utterance,
        if None, then the utterance is assumed to end at the end of the File
    channel: int, optional
        Channel in the file, if None, then assumed to be the first/only channel
    text: str, optional
        Text transcription of the utterance

    Attributes
    ----------
    file_name: str
        Saved File.name property for reconstructing objects following serialization
    speaker_name: str
        Saved Speaker.name property for reconstructing objects following serialization
    transcription_text: str, optional
        Output of transcription is saved here
    ignored: bool
        The ignored flag is set if feature generation does not work for this utterance, or it is too short to
        be processed by Kaldi
    features: str, optional
        Feature string reference to the computed features archive
    phone_labels: list[:class:`~montreal_forced_aligner.data.CtmInterval`], optional
        Saved aligned phone labels
    word_labels: list[:class:`~montreal_forced_aligner.data.CtmInterval`], optional
        Saved aligned word labels
    oovs: list[str]
        Words not found in the dictionary for this utterance
    """

    def __init__(
        self,
        speaker: Speaker,
        file: File,
        begin: Optional[float] = None,
        end: Optional[float] = None,
        channel: Optional[int] = 0,
        text: Optional[str] = None,
    ):
        self.speaker = speaker
        self.file = file
        self.file_name = file.name
        self.speaker_name: str = speaker.name
        if begin is None:
            begin = 0
        if end is None:
            end = self.file.duration
        self.begin = begin
        self.end = end
        self.channel = channel
        self.text = text
        self.transcription_text = None
        self.ignored = False
        self.features = None
        self.phone_labels: Optional[List[CtmInterval]] = None
        self.word_labels: Optional[List[CtmInterval]] = None
        self.reference_phone_labels: Optional[List[CtmInterval]] = []
        self.oovs = set()
        self.normalized_text = []
        self.text_int = []
        self.alignment_log_likelihood = None
        self.word_error_rate = None
        self.character_error_rate = None
        self.phone_error_rate = None
        self.alignment_score = None

    def parse_transcription(self, sanitize_function=Optional[MultispeakerSanitizationFunction]):
        """
        Parse an orthographic transcription given punctuation and clitic markers

        Parameters
        ----------
        sanitize_function: :class:`~montreal_forced_aligner.dictionary.multispeaker.MultispeakerSanitizationFunction`, optional
            Function to sanitize words and strip punctuation

        """
        self.normalized_text = []
        if sanitize_function is not None:
            try:
                sanitize, split = sanitize_function.get_functions_for_speaker(self.speaker_name)
            except AttributeError:
                sanitize = sanitize_function
                split = None
            words = [
                sanitize(w)
                for w in self.text.split()
                if w and w not in sanitize.clitic_markers + sanitize.compound_markers
            ]
            self.text = " ".join(words)
            if split is not None:
                for w in words:
                    for new_w in split(w):
                        if not new_w:
                            continue
                        if split.word_set is not None and new_w not in split.word_set:
                            self.oovs.add(new_w)
                        self.normalized_text.append(new_w)

    @property
    def multiprocessing_data(self):
        return UtteranceData(
            self.speaker_name,
            self.file_name,
            self.begin,
            self.end,
            self.channel,
            self.text,
            self.normalized_text,
            self.oovs,
        )

    @classmethod
    def load_from_mp_data(cls, data: UtteranceData, file: File) -> Utterance:
        utterance = Utterance(
            speaker=Speaker(data.speaker_name),
            file=file,
            begin=data.begin,
            end=data.end,
            channel=data.channel,
            text=data.text,
        )
        if data.normalized_text:
            utterance.normalized_text = data.normalized_text
            utterance.oovs = data.oovs
        return utterance

    def __str__(self) -> str:
        """String representation"""
        return self.name

    def __repr__(self) -> str:
        """Object representation"""
        return f'<Utterance "{self.name}">'

    def __eq__(self, other: Union[Utterance, str]) -> bool:
        """Check if an Utterance is equal to another Utterance"""
        if isinstance(other, Utterance):
            return other.name == self.name
        if isinstance(other, str):
            return self.name == other
        raise TypeError("Utterances can only be compared to other utterances and strings.")

    def __lt__(self, other: Union[Utterance, str]) -> bool:
        """Check if an Utterance is less than another Utterance"""
        if isinstance(other, Utterance):
            return other.name < self.name
        if isinstance(other, str):
            return self.name < other
        raise TypeError("Utterances can only be compared to other utterances and strings.")

    def __gt__(self, other: Union[Utterance, str]) -> bool:
        """Check if an Utterance is greater than another Utterance"""
        if isinstance(other, Utterance):
            return other.name > self.name
        if isinstance(other, str):
            return self.name > other
        raise TypeError("Utterances can only be compared to other utterances and strings.")

    def __hash__(self) -> hash:
        """Compute the hash of this function"""
        return hash(self.name)

    @property
    def duration(self) -> float:
        """Duration of the utterance"""
        if self.begin is not None and self.end is not None:
            return self.end - self.begin
        return self.file.duration

    @property
    def meta(self) -> Dict[str, Any]:
        """Metadata dictionary for the utterance"""
        return {
            "speaker": self.speaker.name,
            "file": self.file.name,
            "begin": self.begin,
            "end": self.end,
            "channel": self.channel,
            "text": self.text,
            "ignored": self.ignored,
            "features": self.features,
            "normalized_text": self.normalized_text,
            "oovs": self.oovs,
            "transcription_text": self.transcription_text,
            "reference_phone_labels": self.reference_phone_labels,
            "phone_labels": self.phone_labels,
            "word_labels": self.word_labels,
            "word_error_rate": self.word_error_rate,
            "character_error_rate": self.character_error_rate,
            "phone_error_rate": self.phone_error_rate,
            "alignment_score": self.alignment_score,
            "alignment_log_likelihood": self.alignment_log_likelihood,
        }

    def set_speaker(self, speaker: Speaker) -> None:
        """
        Set the speaker of the utterance and updates other objects

        Parameters
        ----------
        speaker: :class:`~montreal_forced_aligner.corpus.classes.Speaker`
            New speaker
        """
        self.speaker = speaker
        self.speaker.add_utterance(self)
        self.file.add_utterance(self)

    @property
    def is_segment(self) -> bool:
        """Check if this utterance is a segment of a longer file"""
        return (
            self.begin is not None
            and self.end is not None
            and self.begin != 0
            and self.end != self.file.duration
        )

    def add_word_intervals(self, intervals: Union[CtmInterval, List[CtmInterval]]) -> None:
        """
        Add aligned word intervals for the utterance

        Parameters
        ----------
        intervals: Union[CtmInterval, list[CtmInterval]]
            Intervals to add
        """
        if not isinstance(intervals, list):
            intervals = [intervals]
        if self.word_labels is None:
            self.word_labels = []
        for interval in intervals:
            if self.begin is not None:
                interval.shift_times(self.begin)
        self.word_labels = intervals

    def add_phone_intervals(self, intervals: Union[CtmInterval, List[CtmInterval]]) -> None:
        """
        Add aligned phone intervals for the utterance

        Parameters
        ----------
        intervals: Union[CtmInterval, list[CtmInterval]]
            Intervals to add
        """
        if not isinstance(intervals, list):
            intervals = [intervals]
        if self.phone_labels is None:
            self.phone_labels = []
        for interval in intervals:
            if self.begin is not None:
                interval.shift_times(self.begin)
        self.phone_labels = intervals

    def text_for_scp(self) -> List[str]:
        """
        Generate the text for exporting to Kaldi's text scp

        Returns
        -------
        list[str]
            List of words
        """
        return self.text.split()

    def text_int_for_scp(self) -> Optional[List[int]]:
        """
        Generate the text for exporting to Kaldi's text int scp

        Returns
        -------
        list[int]
            List of word IDs, or None if the utterance's speaker doesn't have an associated dictionary
        """
        if self.speaker.dictionary is None:
            return
        if self.text_int:
            return self.text_int
        if self.normalized_text:
            normalized = True
            text = self.normalized_text
        else:
            normalized = False
            text = self.text_for_scp()
        self.text_int = []
        for i, t in enumerate(text):
            lookup = self.speaker.dictionary.to_int(t, normalized)
            if self.speaker.dictionary.oov_int in lookup:
                self.oovs.add(text[i])
            self.text_int.extend(lookup)
        return self.text_int

    def segment_for_scp(self) -> List[Any]:
        """
        Generate data for Kaldi's segments scp file

        Returns
        -------
        list[Any]
            Segment data
        """
        return [self.file_name.replace(" ", "_MFASPACE_"), self.begin, self.end, self.channel]

    @property
    def name(self) -> str:
        """The name of the utterance"""
        base = f"{self.file_name}"
        base = base.replace(" ", "-space-").replace(".", "-").replace("_", "-")
        if not base.startswith(f"{self.speaker_name}-"):
            base = f"{self.speaker_name}-" + base
        if self.is_segment:
            base = f"{base}-{self.begin}-{self.end}"
        return base.replace(" ", "-space-").replace(".", "-").replace("_", "-")


T = TypeVar("T", Speaker, File, Utterance)


class Collection:
    """
    Utility class for storing collections of corpus objects, allowing iteration, sorting, and
    look up via names.
    """

    CLASS_TYPE = ClassVar[MfaCorpusClass]

    def __init__(self):
        self._data: Dict[str, T] = {}

    def __getitem__(self, key: str) -> T:
        """Get an item by identifier"""
        return self._data[key]

    def __delitem__(self, key: str) -> None:
        """Delete an item by identifier"""
        del self._data[key]

    def __setitem__(self, key: str, item: T) -> None:
        """Set an item by identifier"""
        self._data[key] = item

    def __len__(self) -> int:
        """Number of items in the collection"""
        return len(self._data)

    def __bool__(self) -> bool:
        """Check for whether the collection contains any items"""
        return bool(self._data)

    def __contains__(self, item: Union[str, T]) -> bool:
        """Check for whether the collection contains a specific item"""
        if not isinstance(item, str):
            item = item.name
        return item in self._data

    def subset(self, subset_identifiers: Set[str]) -> Generator[T]:
        for item in self:
            if subset_identifiers and item.name not in subset_identifiers:
                continue
            yield item

    def __iter__(self) -> Generator[T]:
        """Iterator over the collection"""
        for v in self._data.values():
            yield v

    def update(self, other: Union[Collection, Set[T], List[T]]) -> None:
        """Update collection from another collection"""
        if isinstance(other, Collection):
            self._data.update(other._data)
        else:
            for item in other:
                self._data[item.name] = item

    def __str__(self) -> str:
        """String representation"""
        return str(self._data)

    def __repr__(self) -> str:
        """Object representation"""
        return f"<Collection of {self._data}>"


class SpeakerCollection(Collection):
    """
    Utility class for storing collections of speakers
    """

    CLASS_TYPE = Speaker

    def add_speaker(self, speaker: Speaker) -> None:
        """
        Add speaker to the collection

        Parameters
        ----------
        speaker: :class:`~montreal_forced_aligner.corpus.classes.Speaker`
            Speaker to be added
        """
        self[speaker.name] = speaker

    def __repr__(self) -> str:
        """Object representation"""
        return f"<SpeakerCollection of {self._data}>"


class FileCollection(Collection):
    """
    Utility class for storing collections of speakers
    """

    CLASS_TYPE = File

    def __init__(self):
        super(FileCollection, self).__init__()
        self.lab_count = 0
        self.textgrid_count = 0
        self.sound_file_count = 0

    def add_file(self, file: File) -> None:
        """
        Add file to the collection

        Parameters
        ----------
        speaker: :class:`~montreal_forced_aligner.corpus.classes.File`
            File to be added
        """
        self[file.name] = file
        if file.text_type == TextFileType.TEXTGRID:
            self.textgrid_count += 1
        elif file.text_type == TextFileType.LAB:
            self.lab_count += 1
        if file.sound_type != SoundFileType.NONE:
            self.sound_file_count += 1

    def __repr__(self) -> str:
        """Object representation"""
        return f"<FileCollection of {self._data}>"


class UtteranceCollection(Collection):
    """
    Utility class for storing collections of speakers
    """

    CLASS_TYPE = Utterance

    def add_utterance(self, utterance: Utterance) -> None:
        """
        Add utterance to the collection

        Parameters
        ----------
        speaker: :class:`~montreal_forced_aligner.corpus.classes.Utterance`
            Utterance to be added
        """
        self[utterance.name] = utterance

    def __iter__(self) -> Generator[Utterance]:
        """Iterator over the collection"""
        for v in self._data.values():
            if v.ignored:
                continue
            yield v

    def __repr__(self) -> str:
        """Object representation"""
        return f"<UtteranceCollection of {self._data}>"
