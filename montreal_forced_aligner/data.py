"""
Data classes
============

"""
from __future__ import annotations

import dataclasses
import enum
import re
import typing

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

    @property
    def regex_detect(self) -> typing.Optional[re.Pattern]:
        if self is PhoneSetType.ARPA:
            return re.compile(r" [A-Z]{2}[012]? ")
        elif self is PhoneSetType.PINYIN:
            return re.compile(r" [a-z]{1,3}[12345]? ")
        elif self is PhoneSetType.IPA:
            return re.compile(r" [əɚʊɤʁ˥˩ɹɔɛʉɒʃɕŋʰ̚ʲɾ] ")
        return None

    @property
    def base_phone_regex(self) -> typing.Optional[re.Pattern]:
        if self is PhoneSetType.ARPA:
            return re.compile(r"([A-Z]{2})[012]")
        elif self is PhoneSetType.PINYIN:
            return re.compile(r"([a-z]{1,3})[12345]")
        elif self is PhoneSetType.IPA:
            return re.compile(r"([^̃̚ː˩˨˧˦˥̪̝̟̥̂̀̄ˑ̊ᵝ̠̹̞̩̯̬̺ˀˤ̻̙̘̰̤̜̹̑̽᷈᷄᷅̌̋̏‿̆͜͡ˌˈ̣]+)")
        return None

    @property
    def extra_short_phones(self) -> typing.Set[str]:
        if self is PhoneSetType.ARPA:
            return {"AH0", "IH0", "ER0", "UH0"}
        elif self is PhoneSetType.IPA:
            return {"ʔ", "ə", "ɚ", "ɾ", "p̚", "t̚", "k̚"}
        return set()

    @property
    def affricate_phones(self) -> typing.Set[str]:
        if self is PhoneSetType.ARPA:
            return {"CH", "JH"}
        if self is PhoneSetType.IPA:
            return {
                "ts",
                "dz",
                "tʃ",
                "dʒ",
                "tɕ",
                "dʑ",
                "tʂ",
                "ʈʂ",
                "dʐ",
                "ɖʐ",
                "cç",
                "ɟʝ",
                "kx",
                "ɡɣ",
                "tç",
                "dʝ",
            }
        return set()

    @property
    def stop_phones(self) -> typing.Set[str]:
        if self is PhoneSetType.ARPA:
            return {"B", "D", "G"}
        if self is PhoneSetType.IPA:
            return {"p", "b", "t", "d", "ʈ", "ɖ", "c", "ɟ", "k", "ɡ", "q", "ɢ"}
        return set()

    @property
    def diphthong_phones(self) -> typing.Set[str]:
        if self is PhoneSetType.ARPA:
            return {
                "AY0",
                "AY1",
                "AY2",
                "AW0",
                "AW1",
                "AW2",
                "OY0",
                "OY1",
                "OY2",
                "EY0",
                "EY1",
                "EY2",
                "OW0",
                "OW1",
                "OW2",
            }
        if self is PhoneSetType.IPA:
            return {"əw", "eɪ", "aʊ", "oʊ", "aɪ", "ɔɪ"}
        return set()

    @property
    def extra_questions(self) -> typing.Dict[str, typing.Set[str]]:
        extra_questions = {}
        if self is PhoneSetType.ARPA:
            extra_questions["bilabial_variation"] = {"P", "B"}
            extra_questions["dental_lenition"] = {"D", "DH"}
            extra_questions["flapping"] = {"T", "D"}
            extra_questions["nasal_variation"] = {"M", "N", "NG"}
            extra_questions["voiceless_sibilant_variation"] = {"CH", "SH", "S"}
            extra_questions["voiceless_sibilant_variation"] = {"JH", "ZH", "Z"}
            extra_questions["voiceless_fricative_variation"] = {"F", "TH", "HH", "K"}
            extra_questions["voiced_fricative_variation"] = {"V", "DH", "HH", "G"}
            extra_questions["dorsal_variation"] = {"HH", "K", "G"}
            extra_questions["rhotic_variation"] = {"ER0", "ER1", "ER2", "R"}

            extra_questions["low_back_variation"] = {
                "AO0",
                "AO1",
                "AO2",
                "AA0",
                "AA1",
                "AA2",
            }
            extra_questions["central_variation"] = {
                "ER0",
                "ER1",
                "ER2",
                "AH0",
                "AH1",
                "AH2",
                "UH0",
                "UH1",
                "UH2",
                "IH0",
                "IH1",
                "IH2",
            }
            extra_questions["high_back_variation"] = {
                "UW1",
                "UW2",
                "UW0",
                "UH1",
                "UH2",
                "UH0",
            }

            # extra stress questions
            vowels = [
                "AA",
                "AE",
                "AH",
                "AO",
                "AW",
                "AY",
                "EH",
                "ER",
                "EY",
                "IH",
                "IY",
                "OW",
                "OY",
                "UH",
                "UW",
            ]
            for i in range(3):
                extra_questions[f"stress_{i}"] = {f"{x}{i}" for x in vowels}
        elif self is PhoneSetType.IPA:
            extra_questions["dental_lenition"] = {"ð", "d"}
            extra_questions["flapping"] = {"d", "t", "ɾ"}
            extra_questions["glottalization"] = {"t", "ʔ", "t̚"}
            extra_questions["labial_lenition"] = {"β", "b"}
            extra_questions["velar_lenition"] = {"ɣ", "ɡ"}
            extra_questions["nasal_variation"] = {
                "m",
                "n",
                "ɲ",
                "ŋ",
                "ɴ",
                "ɳ",
                "ɱ",
                "ɴ",
                "ɾ",
                "ɰ̃",
            }
            extra_questions["trill_variation"] = {"r", "ʁ", "ɾ", "ɽ", "ɽr", "ɢ̆", "ʀ", "ɺ", "ɭ"}
            extra_questions["syllabic_rhotic_variation"] = {"ɹ", "ɝ", "ɚ", "ə", "ʁ", "ɐ"}
            extra_questions["uvular_variation"] = {"ʁ", "x", "χ", "h", "ɣ", "ɰ", "ʀ"}
            extra_questions["lateral_variation"] = {"l", "ɫ", "ʎ", "ʟ", "ɭ"}

            extra_questions["dorsal_stop_variation"] = {
                "kʰ",
                "k",
                "kʼ",
                "k͈",
                "k̚",
                "kʲ",
                "ɡ",
                "ɡʲ",
                "ɠ",
                "ɟ",
                "cʰ",
                "c",
                "cʼ",
                "q",
                "qʼ",
                "qʰ",
                "ɢ",
            }
            extra_questions["bilabial_stop_variation"] = {"pʰ", "b", "ɓ", "p", "pʼ", "p͈", "p̚"}
            extra_questions["alveolar_stop_variation"] = {
                "tʰ",
                "t",
                "tʼ",
                "d",
                "ʈʼ" "ɗ",
                "t͈",
                "t̚",
            }
            extra_questions["voiceless_fricative_variation"] = {
                "θ",
                "θʼ",
                "f",
                "fʼ",
                "ɸ",
                "ɸʼ",
                "ç",
                "çʼ",
                "x",
                "xʼ",
                "χ",
                "χʼ",
                "h",
            }
            extra_questions["voiced_fricative_variation"] = {"v", "ð", "β", "ʋ"}
            extra_questions["voiceless_affricate_variation"] = {
                "ɕ",
                "ɕʼ",
                "ʂ",
                "ʂʼ",
                "s",
                "sʼ",
                "ʃ",
                "ʃʼ",
                "tɕ",
                "tɕʼ",
                "tɕʰ",
                "tɕ͈",
                "ʈʂ",
                "ʈʂʼ",
                "ʈʂʰ",
                "ts",
                "tsʼ",
                "tsʰ",
                "tʃ",
                "tʃʼ",
                "tʃʰ",
            }
            extra_questions["voiced_affricate_variation"] = {
                "ʐ",
                "ʑ",
                "z",
                "ʒ",
                "ɖʐ",
                "dʑ",
                "dz",
                "dʒ",
            }

            extra_questions["low_vowel_variation"] = {"a", "ɐ", "ɑ", "ɔ"}
            extra_questions["mid_back_vowel_variation"] = {"oʊ", "ɤ", "o", "ɔ"}
            extra_questions["mid_front_variation"] = {"ɛ", "eɪ", "e", "œ", "ø"}
            extra_questions["high_front_variation"] = {"i", "y", "ɪ", "ʏ", "ɨ", "ʉ"}
            extra_questions["high_back_variation"] = {"ʊ", "u", "ɯ", "ɨ", "ʉ"}
            extra_questions["central_variation"] = {
                "ə",
                "ɤ",
                "ɚ",
                "ʌ",
                "ʊ",
                "ɵ",
                "ɐ",
                "ɞ",
                "ɘ",
                "ɝ",
            }
        return extra_questions


@dataclasses.dataclass
class SoundFileInformation:
    """
    Data class for sound file information with format, duration, number of channels, bit depth, and
        sox_string for use in Kaldi feature extraction if necessary

    Parameters
    ----------
    format: str
        Format of the sound file
    sample_rate: int
        Sample rate
    duration: float
        Duration
    sample_rate: int
        Sample rate
    bit_depth: int
        Bit depth
    sox_string: str
        String to use for loading with sox
    """

    __slots__ = ["format", "sample_rate", "duration", "num_channels", "bit_depth", "sox_string"]
    format: str
    sample_rate: int
    duration: float
    num_channels: int
    bit_depth: int
    sox_string: str

    @property
    def meta(self) -> typing.Dict[str, typing.Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class FileExtensions:
    """
    Data class for information about the current directory

    Parameters
    ----------
    identifiers: list[str]
        List of identifiers
    lab_files: dict[str, str]
        Mapping of identifiers to lab files
    textgrid_files: dict[str, str]
        Mapping of identifiers to TextGrid files
    wav_files: dict[str, str]
        Mapping of identifiers to wav files
    other_audio_files: dict[str, str]
        Mapping of identifiers to other audio files
    """

    __slots__ = ["identifiers", "lab_files", "textgrid_files", "wav_files", "other_audio_files"]

    identifiers: typing.List[str]
    lab_files: typing.Dict[str, str]
    textgrid_files: typing.Dict[str, str]
    wav_files: typing.Dict[str, str]
    other_audio_files: typing.Dict[str, str]


@dataclasses.dataclass
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

    def __bool__(self) -> bool:
        return bool(self.pronunciation)

    def __str__(self):
        return f"{' '.join(self.pronunciation)}"

    def __eq__(self, other: Pronunciation):
        return self.pronunciation == other.pronunciation

    def __lt__(self, other: Pronunciation):
        return self.pronunciation < other.pronunciation

    def __gt__(self, other: Pronunciation):
        return self.pronunciation > other.pronunciation


@dataclasses.dataclass
class Word:
    """
    Data class for information about a word and its pronunciations

    Parameters
    ----------
    orthography: str
        Orthographic string for the word
    pronunciations: set[:class:`~montreal_forced_aligner.data.Pronunciation`]
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


@dataclasses.dataclass
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


@dataclasses.dataclass
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
    wav_info: SoundFileInformation
    speaker_ordering: typing.List[str]
    utterances: typing.List[UtteranceData]


@dataclasses.dataclass
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

    def __add__(self, o: str):
        return self.label + o

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
        return Interval(round(self.begin, 4), round(self.end, 4), self.label)
