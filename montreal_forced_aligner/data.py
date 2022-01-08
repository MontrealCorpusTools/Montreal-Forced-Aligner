"""
Data classes
============

"""
from __future__ import annotations

import dataclasses
import enum
import itertools
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


def voiceless_variants(base_phone):
    return {base_phone + d for d in ["", "ʱ", "ʼ", "ʰ", "ʲ", "ʷ", "ˠ", "ˀ", "̚", "͈"]}


def voiced_variants(base_phone):
    return {base_phone + d for d in ["", "ʱ", "ʲ", "ʷ", "ⁿ", "ˠ", "̚"]} | {
        d + base_phone for d in ["ⁿ"]
    }


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
    def has_base_phone_regex(self):
        return self is PhoneSetType.IPA or self is PhoneSetType.ARPA or self is PhoneSetType.PINYIN

    @property
    def regex_detect(self) -> typing.Optional[re.Pattern]:
        if self is PhoneSetType.ARPA:
            return re.compile(r"[A-Z]{2}[012]")
        elif self is PhoneSetType.PINYIN:
            return re.compile(r"[a-z]{1,3}[12345]")
        elif self is PhoneSetType.IPA:
            return re.compile(
                r"[əɚʊɡɤʁɹɔɛʉɒβɲɟʝŋʃɕʰʲɾ̃̚ː˩˨˧˦˥̪̝̟̥̂̀̄ˑ̊ᵝ̠̹̞̩̯̬̺ˀˤ̻̙̘̰̤̜̹̑̽᷈᷄᷅̌̋̏‿̆͜͡ˌˈ̣]"
            )
        return None

    def get_base_phone(self, phone: str):
        if self.has_base_phone_regex:
            return self.base_phone_regex.sub("", phone)
        return phone

    @property
    def base_phone_regex(self) -> typing.Optional[re.Pattern]:
        if self is PhoneSetType.ARPA:
            return re.compile(r"[012]")
        elif self is PhoneSetType.PINYIN:
            return re.compile(r"[12345]")
        elif self is PhoneSetType.IPA:
            return re.compile(r"([ː˩˨˧˦˥̪̟̥̂̀̄ˑ̊ᵝ̠̹̞̩̯̬̺ˀˤ̻̙̘̤̜̹̑̽᷈᷄᷅̌̋̏‿̆͜͡ˌˈ̣]+)")
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
        if self is PhoneSetType.PINYIN:
            return {"z", "zh", "j", "c", "ch", "q"}
        if self is PhoneSetType.IPA:
            affricates = set()
            for p in {
                "pf",
                "ts",
                "tʃ",
                "tɕ",
                "tʂ",
                "ʈʂ",
                "cç",
                "kx",
                "tç",
            }:
                affricates |= voiceless_variants(p)
            for p in {
                "dz",
                "dʒ",
                "dʑ",
                "dʐ",
                "ɖʐ",
                "ɟʝ",
                "ɡɣ",
                "dʝ",
            }:
                affricates |= voiced_variants(p)
            return affricates
        return set()

    @property
    def stop_phones(self) -> typing.Set[str]:
        if self is PhoneSetType.ARPA:
            return {"B", "D", "G"}
        if self is PhoneSetType.PINYIN:
            return {"b", "d", "g"}
        if self is PhoneSetType.IPA:
            stops = set()
            for p in {"p", "t", "ʈ", "c", "k", "q"}:
                stops |= voiceless_variants(p)
            for p in {"b", "d", "ɖ", "ɟ", "ɡ", "ɢ"}:
                stops |= voiced_variants(p)
            return stops
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
        if self is PhoneSetType.IPA or self is PhoneSetType.PINYIN:
            return {x + y for x, y in itertools.product(self.vowels, self.vowels)}

        return set()

    @property
    def vowels(self):
        if self is PhoneSetType.PINYIN:
            return {"i", "u", "y", "e", "w", "a", "o", "e", "ü"}
        if self is PhoneSetType.IPA:
            base_vowels = {
                "i",
                "u",
                "e",
                "ə",
                "a",
                "o",
                "y",
                "ɔ",
                "j",
                "w",
                "ɪ",
                "ʊ",
                "w",
                "ʏ",
                "ɯ",
                "ɤ",
                "ɑ",
                "æ",
                "ɐ",
                "ɚ",
                "ɵ",
                "ɘ",
                "ɛ",
                "ɜ",
                "ɝ",
                "ɞ",
                "ɑ̃",
                "ɨ",
                "ɪ̈",
                "œ",
                "ɒ",
                "ɶ",
                "ø",
                "ʉ",
                "ʌ",
            }
            base_vowels |= {x + "̃" for x in base_vowels}  # Add nasals
            return {
                "i",
                "u",
                "e",
                "ə",
                "a",
                "o",
                "y",
                "ɔ",
                "j",
                "w",
                "ɪ",
                "ʊ",
                "w",
                "ʏ",
                "ɯ",
                "ɤ",
                "ɑ",
                "æ",
                "ɐ",
                "ɚ",
                "ɵ",
                "ɘ",
                "ɛ",
                "ɜ",
                "ɝ",
                "ɞ",
                "ɨ",
                "ɪ̈",
                "œ",
                "ɒ",
                "ɶ",
                "ø",
                "ʉ",
                "ʌ",
            }
        return set()

    @property
    def triphthong_phones(self) -> typing.Set[str]:
        if self is PhoneSetType.IPA or self is PhoneSetType.PINYIN:
            return {
                x + y + z for x, y, z in itertools.product(self.vowels, self.vowels, self.vowels)
            }

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
            extra_questions["high_front_variation"] = {
                "IY1",
                "IY2",
                "IY0",
                "IH0",
                "IH1",
                "IH2",
            }
            extra_questions["mid_front_variation"] = {
                "EY1",
                "EY2",
                "EY0",
                "EH0",
                "EH1",
                "EH2",
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
        elif self is PhoneSetType.PINYIN:
            for i in range(1, 6):
                extra_questions[f"tone_{i}"] = {f"{x}{i}" for x in self.vowels}
                extra_questions[f"tone_{i}"] |= {f"{x}{i}" for x in self.diphthong_phones}
                extra_questions[f"tone_{i}"] |= {f"{x}{i}" for x in self.triphthong_phones}
            extra_questions["bilabial_variation"] = {"p", "b"}
            extra_questions["nasal_variation"] = {"m", "n", "ng"}
            extra_questions["voiceless_sibilant_variation"] = {
                "z",
                "zh",
                "j",
                "c",
                "ch",
                "q",
                "s",
                "sh",
                "x",
            }
            extra_questions["dorsal_variation"] = {"h", "k", "g"}
            extra_questions["alveolar_stop_variation"] = {"t", "d"}
            extra_questions["approximant_variation"] = {"l", "r", "y", "w"}
            extra_questions["rhotic_variation"] = {"r", "sh", "e"}

        elif self is PhoneSetType.IPA:

            extra_questions["dental_lenition"] = voiced_variants("ð") | voiced_variants("d")
            extra_questions["flapping"] = {"d", "t", "ɾ"}
            extra_questions["glottalization"] = {"t", "ʔ", "t̚"}
            extra_questions["labial_lenition"] = voiced_variants("β") | voiced_variants("b")
            extra_questions["velar_lenition"] = voiced_variants("ɣ") | voiced_variants("ɡ")

            nasal_variation = (
                voiced_variants("m")
                | voiced_variants("n")
                | voiced_variants("ɲ")
                | voiced_variants("ŋ")
            )
            nasal_variation |= voiced_variants("ɴ") | voiced_variants("ɳ") | voiced_variants("ɱ")
            nasal_variation |= voiced_variants("ɰ̃") | voiced_variants("ɾ")
            extra_questions["nasal_variation"] = nasal_variation

            approximant_variation = (
                voiced_variants("w")
                | voiced_variants("ʍ")
                | voiced_variants("ɰ")
                | voiced_variants("ɥ")
            )
            approximant_variation |= (
                voiced_variants("l")
                | voiced_variants("ɭ")
                | voiced_variants("ɹ")
                | voiced_variants("ɫ")
            )
            approximant_variation |= (
                voiced_variants("ɻ") | voiced_variants("ʋ") | voiced_variants("j")
            )
            extra_questions["approximant_variation"] = approximant_variation

            front_glide_variation = {"j", "i", "ɪ", "ɥ", "ʏ", "y"}
            front_glide_variation |= {x + "̃" for x in front_glide_variation}
            extra_questions["front_glide_variation"] = front_glide_variation

            back_glide_variation = {"w", "u", "ʊ", "ɰ", "ɯ", "ʍ"}
            back_glide_variation |= {x + "̃" for x in back_glide_variation}
            extra_questions["back_glide_variation"] = back_glide_variation

            trill_variation = (
                voiced_variants("r")
                | voiced_variants("ɾ")
                | voiced_variants("ɽ")
                | voiced_variants("ɽr")
            )
            trill_variation |= voiced_variants("ʁ") | voiced_variants("ʀ") | voiced_variants("ɢ̆")
            trill_variation |= voiced_variants("ɺ") | voiced_variants("ɭ")
            extra_questions["trill_variation"] = trill_variation

            extra_questions["syllabic_rhotic_variation"] = {"ɹ", "ɝ", "ɚ", "ə", "ʁ", "ɐ"}
            uvular_variation = (
                voiced_variants("ʁ") | voiceless_variants("x") | voiceless_variants("χ")
            )
            uvular_variation |= (
                voiced_variants("ɣ")
                | voiceless_variants("h")
                | voiced_variants("ɰ")
                | voiced_variants("ʀ")
            )
            extra_questions["uvular_variation"] = uvular_variation

            lateral_variation = voiced_variants("l") | voiced_variants("ɫ") | voiced_variants("ʎ")
            lateral_variation |= (
                voiced_variants("ʟ")
                | voiced_variants("ɭ")
                | voiced_variants("ɮ")
                | voiceless_variants("ɬ")
            )
            extra_questions["lateral_variation"] = lateral_variation

            dorsal_stops = voiceless_variants("k") | voiced_variants("ɡ") | voiced_variants("ɠ")
            dorsal_stops |= voiceless_variants("c") | voiced_variants("ɟ") | voiced_variants("ʄ")
            dorsal_stops |= voiceless_variants("q") | voiced_variants("ɢ") | voiced_variants("ʛ")
            extra_questions["dorsal_stop_variation"] = dorsal_stops

            bilabial_stops = (
                voiceless_variants("p")
                | voiced_variants("b")
                | voiced_variants("ɓ")
                | voiced_variants("ʙ")
                | voiced_variants("ⱱ")
            )
            extra_questions["bilabial_stop_variation"] = bilabial_stops

            alveolar_stops = voiceless_variants("t") | voiceless_variants("ʈ")
            alveolar_stops |= (
                voiced_variants("d")
                | voiced_variants("ɗ")
                | voiced_variants("ɖ")
                | voiced_variants("ᶑ")
            )
            extra_questions["alveolar_stop_variation"] = alveolar_stops

            voiceless_fricatives = (
                voiceless_variants("θ")
                | voiceless_variants("f")
                | voiceless_variants("pf")
                | voiceless_variants("ɸ")
            )
            voiceless_fricatives |= (
                voiceless_variants("ç")
                | voiceless_variants("x")
                | voiceless_variants("kx")
                | voiceless_variants("χ")
                | voiceless_variants("h")
            )
            voiceless_fricatives |= (
                voiceless_variants("cç")
                | voiceless_variants("q")
                | voiceless_variants("qχ")
                | voiceless_variants("ɬ")
            )
            extra_questions["voiceless_fricative_variation"] = voiceless_fricatives
            voiced_fricatives = voiced_variants("v") | voiced_variants("ð")
            voiced_fricatives |= voiced_variants("β") | voiced_variants("ʋ") | voiced_variants("ɣ")
            voiced_fricatives |= (
                voiced_variants("ɟʝ") | voiced_variants("ʝ") | voiced_variants("ɮ")
            )

            extra_questions["voiced_fricative_variation"] = voiced_fricatives
            voiceless_sibilants = (
                voiceless_variants("s")
                | voiceless_variants("ʃ")
                | voiceless_variants("ɕ")
                | voiceless_variants("ʂ")
            )
            voiceless_sibilants |= (
                voiceless_variants("ts") | voiceless_variants("tʃ") | voiceless_variants("tɕ")
            )
            voiceless_sibilants |= voiceless_variants("ʈʂ") | voiceless_variants("tʂ")
            extra_questions["voiceless_sibilant_variation"] = voiceless_sibilants

            voiced_sibilants = (
                voiced_variants("z")
                | voiced_variants("ʒ")
                | voiced_variants("ʐ")
                | voiced_variants("ʑ")
            )
            voiced_sibilants |= (
                voiced_variants("dz")
                | voiced_variants("dʒ")
                | voiced_variants("dʐ")
                | voiced_variants("ɖʐ")
                | voiced_variants("dʑ")
            )
            extra_questions["voiced_sibilant_variation"] = voiced_sibilants

            extra_questions["low_vowel_variation"] = {"a", "ɐ", "ɑ", "ɔ"}
            extra_questions["mid_back_vowel_variation"] = {"oʊ", "ɤ", "o", "ɔ"}
            extra_questions["mid_front_variation"] = {"ɛ", "eɪ", "e", "œ", "ø"}
            extra_questions["high_front_variation"] = {"i", "y", "ɪ", "ʏ", "ɨ", "ʉ"}
            extra_questions["high_back_variation"] = {"ʊ", "u", "ɯ", "ɨ", "ʉ"}
            back_diphthong_variation = {"aʊ", "au", "aw", "ɐw"}
            back_diphthong_variation |= {x + "̃" for x in back_diphthong_variation}
            extra_questions["back_diphthong_variation"] = back_diphthong_variation

            front_diphthong_variation = {"aɪ", "ai", "aj", "ɐj"}
            front_diphthong_variation |= {x + "̃" for x in front_diphthong_variation}
            extra_questions["front_diphthong_variation"] = front_diphthong_variation

            mid_front_diphthong_variation = {
                "eɪ",
                "ei",
                "ej",
                "ɛɪ",
                "ɛi",
                "ɛj",
            }
            mid_front_diphthong_variation |= {x + "̃" for x in mid_front_diphthong_variation}
            extra_questions["mid_front_diphthong_variation"] = mid_front_diphthong_variation

            mid_back_diphthong_variation = {"ow", "ou", "oʊ"}
            mid_back_diphthong_variation |= {x + "̃" for x in mid_back_diphthong_variation}
            extra_questions["mid_front_diphthong_variation"] = mid_back_diphthong_variation

            mid_back_front_diphthong_variation = {"oj", "oi", "oɪ", "ɔʏ", "ɔɪ", "ɔj", "ɔi"}
            mid_back_front_diphthong_variation |= {
                x + "̃" for x in mid_back_front_diphthong_variation
            }
            extra_questions[
                "mid_back_front_diphthong_variation"
            ] = mid_back_front_diphthong_variation

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

    def __lt__(self, other: CtmInterval):
        return self.begin < other.begin

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
