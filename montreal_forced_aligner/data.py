"""
Data classes
============

"""
from __future__ import annotations

import enum
import typing
from dataclasses import dataclass

from praatio.utilities.constants import Interval

from .exceptions import CtmError

__all__ = [
    "CtmInterval",
    "UtteranceData",
    "FileData",
    "TextFileType",
    "SoundFileType",
    "PhoneSetType",
]


class TextFileType(enum.Enum):
    """Enum for types of text files"""

    NONE = 0
    TEXTGRID = 1
    LAB = 2


class SoundFileType(enum.Enum):
    """Enum for types of sound files"""

    NONE = 0
    WAV = 1
    SOX = 2


class PhoneSetType(enum.Enum):
    """Enum for types of phone sets"""

    UNKNOWN = "UNKNOWN"
    AUTO = "AUTO"
    IPA = "IPA"
    ARPA = "ARPA"
    PINYIN = "PINYIN"

    def __str__(self):
        """Name of phone set"""
        return self.name


@dataclass
class Pronunciation:
    """
    Data class for information about a pronunciation string

    Parameters
    ----------
    pronunciation: tuple
        Tuple of phones
    probability: float
        Probability of pronunciation
    disambiguation: Optional[int]
        Disambiguation index within a pronunciation dictionary
    left_silence_probability: Optional[float]
        Probability of silence before the pronunciation
    right_silence_probability: Optional[float]
        Probability of silence after the pronunciation
    """

    __slots__ = [
        "pronunciation",
        "probability",
        "disambiguation",
        "left_silence_probability",
        "right_silence_probability",
    ]

    pronunciation: typing.Tuple[str, ...]
    probability: float
    disambiguation: typing.Optional[int]
    left_silence_probability: typing.Optional[float]
    right_silence_probability: typing.Optional[float]

    def __hash__(self):
        return hash(self.pronunciation)

    def __len__(self):
        return len(self.pronunciation)

    def __repr__(self):
        return f"<Pronunciation /{' '.join(self.pronunciation)}/>"

    def __str__(self):
        return f"/{' '.join(self.pronunciation)}/"

    def __eq__(self, other: Pronunciation):
        return self.pronunciation == other.pronunciation

    def __lt__(self, other: Pronunciation):
        return self.pronunciation < other.pronunciation

    def __gt__(self, other: Pronunciation):
        return self.pronunciation > other.pronunciation


@dataclass
class Word:
    """
    Data class for information about a word and its pronunciations

    Parameters
    ----------
    orthography: str
        Orthographic string for the word
    pronunciations: set[:class:`~montreal_forced_aligner.dictionary.pronunciation.Pronunciation`]
        Set of pronunciations for the word
    """

    __slots__ = ["orthography", "pronunciations"]

    orthography: str
    pronunciations: typing.Set[Pronunciation]

    def __repr__(self) -> str:
        """Word object representation"""
        pronunciation_string = ", ".join(map(str, self.pronunciations))
        return f"<Word {self.orthography} with pronunciations {pronunciation_string}>"

    def __hash__(self) -> hash:
        """Word hash"""
        return hash(self.orthography)

    def __len__(self) -> int:
        """Number of pronunciations"""
        return len(self.pronunciations)

    def __iter__(self) -> typing.Generator[Pronunciation]:
        """Iterator over pronunciations"""
        for p in self.pronunciations:
            yield p


@dataclass(order=True, frozen=True)
class UtteranceData:
    """
    Data class for utterance information

    Parameters
    ----------
    speaker_name: str
        Speaker name
    file_name: str
        File name
    begin: float, optional
        Begin timestamp
    end: float, optional
        End timestamp
    channel: int, optional
        Sound file channel
    text: str, optional
        Utterance text
    normalized_text: list[str]
        Normalized utterance text, with compounds and clitics split up
    oovs: set[str]
        Set of words not found in a look up
    """

    __slots__ = [
        "speaker_name",
        "file_name",
        "begin",
        "end",
        "channel",
        "text",
        "normalized_text",
        "oovs",
    ]
    speaker_name: str
    file_name: str
    begin: typing.Optional[float]
    end: typing.Optional[float]
    channel: typing.Optional[int]
    text: typing.Optional[str]
    normalized_text: typing.List[str]
    oovs: typing.Set[str]

    def __getstate__(self) -> typing.Dict[str, typing.Any]:
        """For pickling"""
        return dict((slot, getattr(self, slot)) for slot in self.__slots__ if hasattr(self, slot))

    def __setstate__(self, state) -> None:
        """For pickling"""
        for slot, value in state.items():
            object.__setattr__(self, slot, value)  # <- use object.__setattr__


@dataclass(order=True, frozen=True)
class FileData:
    """
    Data class for file information

    Parameters
    ----------
    name: str
        File name
    wav_path: str, optional
        Path to sound file
    text_path: str, optional
        Path to sound file
    relative_path: str
        Path relative to corpus root directory
    wav_info: dict[str, Any]
        Information dictionary about the sound file
    speaker_ordering: list[str]
        List of speakers in the file
    utterances: list[:class:`~montreal_forced_aligner.data.UtteranceData`]
        Utterance data for the file
    """

    __slots__ = [
        "name",
        "wav_path",
        "text_path",
        "relative_path",
        "wav_info",
        "speaker_ordering",
        "utterances",
    ]
    name: str
    wav_path: typing.Optional[str]
    text_path: typing.Optional[str]
    relative_path: str
    wav_info: typing.Dict[str, typing.Any]
    speaker_ordering: typing.List[str]
    utterances: typing.List[UtteranceData]

    def __getstate__(self) -> typing.Dict[str, typing.Any]:
        """For pickling"""
        return dict((slot, getattr(self, slot)) for slot in self.__slots__ if hasattr(self, slot))

    def __setstate__(self, state) -> None:
        """For pickling"""
        for slot, value in state.items():
            object.__setattr__(self, slot, value)  # <- use object.__setattr__


@dataclass
class CtmInterval:
    """
    Data class for intervals derived from CTM files

    Parameters
    ----------
    begin: float
        Start time of interval
    end: float
        End time of interval
    label: str
        Text of interval
    utterance: str
        Utterance ID that the interval belongs to
    """

    __slots__ = ["begin", "end", "label", "utterance"]

    begin: float
    end: float
    label: str
    utterance: str

    def __post_init__(self):
        """
        Check on data validity

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.CtmError`
            If begin or end are not valid
        """
        if self.end < -1 or self.begin == 1000000:
            raise CtmError(self)

    def shift_times(self, offset: float):
        """
        Shift times of the interval based on some offset (i.e., segments in Kaldi)

        Parameters
        ----------
        offset: float
            Offset to add to the interval's begin and end
        """
        self.begin += offset
        self.end += offset

    def to_tg_interval(self) -> Interval:
        """
        Converts the CTMInterval to `PraatIO's Interval class <http://timmahrt.github.io/praatIO/praatio/utilities/constants.html#Interval>`_

        Returns
        -------
        :class:`praatio.utilities.constants.Interval`
            Derived PraatIO Interval
        """
        if self.end < -1 or self.begin == 1000000:
            raise CtmError(self)
        return Interval(self.begin, self.end, self.label)
