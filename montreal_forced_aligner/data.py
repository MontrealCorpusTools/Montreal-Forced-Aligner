"""
Data classes
============

"""
from __future__ import annotations

import collections
import enum
import itertools
import re
import typing

import dataclassy
from praatio.utilities.constants import Interval, TextgridFormats

from .exceptions import CtmError

__all__ = [
    "MfaArguments",
    "CtmInterval",
    "TextFileType",
    "SoundFileType",
    "WordType",
    "PhoneType",
    "PhoneSetType",
    "WordData",
    "DatabaseImportData",
    "PronunciationProbabilityCounter",
]


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class DatabaseImportData:
    """
    Class for storing information on importing data into the database

    Parameters
    ----------
    speaker_objects: list[dict[str, Any]]
        List of dictionaries with :class:`~montreal_forced_aligner.db.Speaker` properties
    file_objects: list[dict[str, Any]]
        List of dictionaries with :class:`~montreal_forced_aligner.db.File` properties
    text_file_objects: list[dict[str, Any]]
        List of dictionaries with :class:`~montreal_forced_aligner.db.TextFile` properties
    sound_file_objects: list[dict[str, Any]]
        List of dictionaries with :class:`~montreal_forced_aligner.db.SoundFile` properties
    speaker_ordering_objects: list[dict[str, Any]]
        List of dictionaries with :class:`~montreal_forced_aligner.db.SpeakerOrdering` properties
    utterance_objects: list[dict[str, Any]]
        List of dictionaries with :class:`~montreal_forced_aligner.db.Utterance` properties
    """

    speaker_objects: typing.List[typing.Dict[str, typing.Any]] = dataclassy.factory(list)
    file_objects: typing.List[typing.Dict[str, typing.Any]] = dataclassy.factory(list)
    text_file_objects: typing.List[typing.Dict[str, typing.Any]] = dataclassy.factory(list)
    sound_file_objects: typing.List[typing.Dict[str, typing.Any]] = dataclassy.factory(list)
    speaker_ordering_objects: typing.List[typing.Dict[str, typing.Any]] = dataclassy.factory(list)
    utterance_objects: typing.List[typing.Dict[str, typing.Any]] = dataclassy.factory(list)

    def add_objects(self, other_import: DatabaseImportData) -> None:
        """
        Combine objects for two importers

        Parameters
        ----------
        other_import: :class:`~montreal_forced_aligner.data.DatabaseImportData`
            Other object with objects to import
        """
        self.speaker_objects.extend(other_import.speaker_objects)
        self.file_objects.extend(other_import.file_objects)
        self.text_file_objects.extend(other_import.text_file_objects)
        self.sound_file_objects.extend(other_import.sound_file_objects)
        self.speaker_ordering_objects.extend(other_import.speaker_ordering_objects)
        self.utterance_objects.extend(other_import.utterance_objects)


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class MfaArguments:
    """
    Base class for argument classes for MFA functions

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_path: str
        Path to connect to database for getting necessary information
    log_path: str
        Path to save logging information during the run
    """

    job_name: int
    db_path: str
    log_path: str


class TextFileType(enum.Enum):
    """Enum for types of text files"""

    NONE = "none"
    TEXTGRID = TextgridFormats.LONG_TEXTGRID
    SHORT_TEXTGRID = TextgridFormats.SHORT_TEXTGRID
    LAB = "lab"
    JSON = TextgridFormats.JSON

    def __str__(self):
        """Name of phone set"""
        return self.value


class SoundFileType(enum.Enum):
    """Enum for types of sound files"""

    NONE = 0
    WAV = 1
    SOX = 2


def voiceless_variants(base_phone) -> typing.Set[str]:
    """
    Generate variants of voiceless IPA phones

    Parameters
    ----------
    base_phone: str
        Voiceless IPA phone

    Returns
    -------
    set[str]
        Set of base_phone plus variants
    """
    return {base_phone + d for d in ["", "ʱ", "ʼ", "ʰ", "ʲ", "ʷ", "ˠ", "ˀ", "̚", "͈"]}


def voiced_variants(base_phone) -> typing.Set[str]:
    """
    Generate variants of voiced IPA phones

    Parameters
    ----------
    base_phone: str
        Voiced IPA phone

    Returns
    -------
    set[str]
        Set of base_phone plus variants
    """
    return {base_phone + d for d in ["", "ʱ", "ʲ", "ʷ", "ⁿ", "ˠ", "̚"]} | {
        d + base_phone for d in ["ⁿ"]
    }


class PhoneType(enum.Enum):
    """Enum for types of phones"""

    non_silence = 1
    silence = 2
    disambiguation = 3


class WordType(enum.Enum):
    """Enum for types of words"""

    speech = 1
    clitic = 2
    silence = 3
    oov = 4
    bracketed = 5
    laughter = 6
    noise = 7
    music = 8


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
    def has_base_phone_regex(self) -> bool:
        """Check for whether a base phone regex is available"""
        return self is PhoneSetType.IPA or self is PhoneSetType.ARPA or self is PhoneSetType.PINYIN

    @property
    def regex_detect(self) -> typing.Optional[re.Pattern]:
        """Pattern for detecting a phone set type"""
        if self is PhoneSetType.ARPA:
            return re.compile(r"[A-Z]{2}[012]")
        elif self is PhoneSetType.PINYIN:
            return re.compile(r"[a-z]{1,3}[12345]")
        elif self is PhoneSetType.IPA:
            return re.compile(
                r"[əɚʊɡɤʁɹɔɛʉɒβɲɟʝŋʃɕʰʲɾ̃̚ː˩˨˧˦˥̪̝̟̥̂̀̄ˑ̊ᵝ̠̹̞̩̯̬̺ˀˤ̻̙̘̰̤̜̑̽᷈᷄᷅̌̋̏‿̆͜͡ˌˈ̣]"
            )
        return None

    @property
    def suprasegmental_phone_regex(self) -> typing.Optional[re.Pattern]:
        """Regex for creating base phones"""
        if self is PhoneSetType.IPA:
            return re.compile(r"([ː̟̥̂̀̄ˑ̊ᵝ̠̹̞̩̯̬̺ˤ̻̙̘̤̜̑̽᷈᷄᷅̌̋̏‿̆͜͡ˌ̍ʱʰʲ̚ʼ͈ˈ̣]+)")
        return None

    @property
    def base_phone_regex(self) -> typing.Optional[re.Pattern]:
        """Regex for creating base phones"""
        if self is PhoneSetType.ARPA:
            return re.compile(r"[012]")
        elif self is PhoneSetType.PINYIN:
            return re.compile(r"[12345]")
        elif self is PhoneSetType.IPA:
            return re.compile(r"([ː˩˨˧˦˥̟̥̂̀̄ˑ̊ᵝ̠̹̞̩̯̬̺ˀˤ̻̙̘̤̜̑̽᷈᷄᷅̌̋̏‿̆͜͡ˌ̍ˈ]+)")
        return None

    @property
    def voiceless_obstruents(self):
        """Voiceless obstruents for the phone set"""
        if self is PhoneSetType.IPA:
            return {
                "p",
                "t",
                "ʈ",
                "k",
                "c",
                "q",
                "f",
                "s",
                "ʂ",
                "s̪",
                "ɕ",
                "x",
                "ç",
                "ɸ",
                "χ",
                "ʃ",
                "h",
                "ʜ",
                "ħ",
                "ʡ",
                "ʔ",
                "θ",
                "ɬ",
                "ɧ",
            }
        elif self is PhoneSetType.ARPA:
            return {"P", "T", "CH", "SH", "S", "F", "TH", "HH", "K"}
        return set()

    @property
    def voiced_obstruents(self):
        """Voiced obstruents for the phone set"""
        if self is PhoneSetType.IPA:
            return {
                "b",
                "d",
                "g",
                "ɖ",
                "ɡ",
                "ɟ",
                "ɢ",
                "v",
                "z̪",
                "z",
                "ʐ",
                "ʑ",
                "ɣ",
                "ʁ",
                "ʢ",
                "ʕ",
                "ʒ",
                "ʝ",
                "ɦ",
                "ð",
                "ɮ",
            }
        elif self is PhoneSetType.ARPA:
            return {"B", "D", "DH", "JH", "ZH", "Z", "V", "DH", "G"}
        return set()

    @property
    def implosive_obstruents(self):
        """Implosive obstruents for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɓ", "ɗ", "ʄ", "ɠ", "ʛ", "ᶑ", "ɗ̪"}
        return set()

    @property
    def stops(self):
        """Stops for the phone set"""
        if self is PhoneSetType.IPA:
            return {
                "p",
                "t",
                "t̪",
                "ʈ",
                "c",
                "k",
                "q",
                "kp",
                "pk",
                "b",
                "d",
                "d̪",
                "ɖ",
                "ɟ",
                "ɡ",
                "ɢ",
                "bɡ",
                "ɡb",
                "ɓ",
                "ɗ",
                "ʄ",
                "ɠ",
                "ʛ",
                "ᶑ",
                "ɗ̪",
                "ʔ",
                "ʡ",
            }
        elif self is PhoneSetType.ARPA:
            return {"B", "D", "P", "T", "G", "K"}
        return set()

    @property
    def sibilants(self):
        """Sibilants for the phone set"""
        if self is PhoneSetType.IPA:
            return {"s", "s̪", "ʃ", "ʂ", "ɕ", "z", "z̪", "ʒ", "ʑ", "ʐ", "ɧ"}
        elif self is PhoneSetType.ARPA:
            return {"SH", "S", "ZH", "Z"}
        return set()

    @property
    def affricates(self):
        """Affricates for the phone set"""
        if self is PhoneSetType.IPA:
            return {
                "pf",
                "ts",
                "t̪s̪",
                "tʃ",
                "tɕ",
                "tʂ",
                "ʈʂ",
                "cç",
                "kx",
                "tç",
                "dz",
                "d̪z̪",
                "dʒ",
                "dʑ",
                "dʐ",
                "ɖʐ",
                "ɟʝ",
                "ɡɣ",
                "dʝ",
            }
        elif self is PhoneSetType.ARPA:
            return {"JH", "CH"}
        return set()

    @property
    def fricatives(self):
        """Fricatives for the phone set"""
        if self is PhoneSetType.IPA:
            return {
                "f",
                "v",
                "ç",
                "ʝ",
                "ħ",
                "ɧ",
                "θ",
                "ð",
                "ʁ",
                "ʢ",
                "ʕ",
                "χ",
                "ʜ",
                "ʢ",
                "ɦ",
                "h",
                "ɸ",
            }
        elif self is PhoneSetType.ARPA:
            return {
                "V",
                "DH",
                "HH",
                "F",
                "TH",
            }
        return set()

    @property
    def laterals(self):
        """Laterals for the phone set"""
        if self is PhoneSetType.IPA:
            return {"l", "ɫ", "ʟ", "ʎ", "l̪"}
        elif self is PhoneSetType.ARPA:
            return {"L"}
        return set()

    @property
    def nasals(self):
        """Nasals for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɲ", "ŋ", "m", "n", "ɳ", "ɴ", "ɱ", "ŋm", "n̪"}
        elif self is PhoneSetType.ARPA:
            return {"M", "N", "NG"}
        return set()

    @property
    def trills(self):
        """Trills for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ʙ", "r", "ʀ", "r̝"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def taps(self):
        """Taps for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɾ", "ɽ", "ⱱ"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def lateral_taps(self):
        """Lateral taps for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɭ", "ɺ"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def lateral_fricatives(self):
        """Lateral fricatives for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɬ", "ɮ"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def approximants(self):
        """Approximants for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɹ", "ɻ", "ʋ", "ʍ"} | self.glides
        elif self is PhoneSetType.ARPA:
            return {"R"} | self.glides
        return set()

    @property
    def glides(self):
        """Glides for the phone set"""
        if self is PhoneSetType.IPA:
            return {"j", "w", "w̃", "j̃", "ɥ", "ɰ", "ɥ̃", "ɰ̃", "j̰"}
        elif self is PhoneSetType.ARPA:
            return {"Y", "W"}
        return set()

    @property
    def nasal_approximants(self):
        """Nasal approximants for the phone set"""
        if self is PhoneSetType.IPA:
            return {"w̃", "j̃", "ɥ̃", "ɰ̃"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def labials(self):
        """Labials for the phone set"""
        if self is PhoneSetType.IPA:
            return {"b", "p", "m", "ɸ", "β", "ɓ", "w", "ʍ"}
        elif self is PhoneSetType.ARPA:
            return {"B", "P", "M", "W"}
        return set()

    @property
    def labiodental(self):
        """Labiodentals for the phone set"""
        if self is PhoneSetType.IPA:
            return {"f", "v", "ʋ", "ⱱ", "ɱ", "pf"}
        elif self is PhoneSetType.ARPA:
            return {"F", "V"}
        return set()

    @property
    def dental(self):
        """Dentals for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ð", "θ", "t̪", "d̪", "s̪", "z̪", "t̪s̪", "d̪z̪", "n̪", "l̪", "ɗ̪"}
        elif self is PhoneSetType.ARPA:
            return {"DH", "TH"}
        return set()

    @property
    def alveolar(self):
        """Alveolars for the phone set"""
        if self is PhoneSetType.IPA:
            return {
                "t",
                "d",
                "s",
                "z",
                "n",
                "r",
                "l",
                "ɹ",
                "ɾ",
                "ɬ",
                "ɮ",
                "ɫ",
                "ts",
                "dz",
                "ɗ",
                "ɺ",
            }
        elif self is PhoneSetType.ARPA:
            return {"T", "D", "S", "Z", "N", "R", "L"}
        return set()

    @property
    def retroflex(self):
        """Retroflexes for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ʈ", "ʂ", "ʐ", "ɖ", "ɽ", "ɻ", "ɭ", "ɳ", "ʈʂ", "ɖʐ", "ᶑ"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def alveopalatal(self):
        """Alveopalatals for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ʒ", "ʃ", "dʒ", "tʃ"}
        elif self is PhoneSetType.ARPA:
            return {"ZH", "SH", "JH", "CH"}
        return set()

    @property
    def palatalized(self):
        """Palatalized phones for the phone set"""
        if self is PhoneSetType.IPA:
            palatals = set()
            palatals.update(x + "ʲ" for x in self.labials)
            palatals.update(x + "ʲ" for x in self.labiodental)
            palatals.update(x + "ʲ" for x in self.dental)
            palatals.update(x + "ʲ" for x in self.alveolar)
            palatals.update(x + "ʲ" for x in self.retroflex)
            palatals.update(x + "ʲ" for x in self.palatal)
            palatals.update(x + "ʲ" for x in self.velar)
            palatals.update(x + "ʲ" for x in self.uvular)
            palatals.update(x + "ʲ" for x in self.pharyngeal)
            palatals.update(x + "ʲ" for x in self.epiglottal)
            palatals.update(x + "ʲ" for x in self.glottal)
            return palatals
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def labialized(self):
        """Labialized phones for the phone set"""
        if self is PhoneSetType.IPA:
            palatals = set()
            palatals.update(x + "ʷ" for x in self.labials)
            palatals.update(x + "ʷ" for x in self.labiodental)
            palatals.update(x + "ʷ" for x in self.dental)
            palatals.update(x + "ʷ" for x in self.alveolar)
            palatals.update(x + "ʷ" for x in self.retroflex)
            palatals.update(x + "ʷ" for x in self.palatal)
            palatals.update(x + "ʷ" for x in self.velar)
            palatals.update(x + "ʷ" for x in self.uvular)
            palatals.update(x + "ʷ" for x in self.pharyngeal)
            palatals.update(x + "ʷ" for x in self.epiglottal)
            palatals.update(x + "ʷ" for x in self.glottal)
            return palatals
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def palatal(self):
        """Palatal phones for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ç", "c", "ɕ", "tɕ", "ɟ", "ɟʝ", "ʝ", "ɲ", "ɥ", "j", "ʎ", "ʑ", "dʑ"}
        elif self is PhoneSetType.ARPA:
            return {"Y"}
        return set()

    @property
    def velar(self):
        """Velar phones for the phone set"""
        if self is PhoneSetType.IPA:
            return {"k", "x", "ɡ", "ɠ", "ɣ", "ɰ", "ŋ"}
        elif self is PhoneSetType.ARPA:
            return {"K", "NG", "G"}
        return set()

    @property
    def uvular(self):
        """Uvular phones for the phone set"""
        if self is PhoneSetType.IPA:
            return {"q", "ɢ", "ʛ", "χ", "ʀ", "ʁ", "ʟ", "ɴ"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def pharyngeal(self):
        """Pharyngeal phones for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ʕ", "ħ"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def epiglottal(self):
        """Epiglottal phones for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ʡ", "ʢ", "ʜ"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def glottal(self):
        """Glottal phones for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ʔ", "ɦ", "h"}
        elif self is PhoneSetType.ARPA:
            return {"HH"}
        return set()

    @property
    def close_vowels(self):
        """Close vowels for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɪ", "ɨ", "ɪ̈", "ʉ", "ʊ", "i", "ĩ", "ɯ", "y", "u", "ʏ", "ũ"}
        elif self is PhoneSetType.ARPA:
            return {"IH", "UH", "IY", "UW"}
        return set()

    @property
    def close_mid_vowels(self):
        """Close-mid vowels for the phone set"""
        if self is PhoneSetType.IPA:
            return {"e", "ẽ", "ej", "eɪ", "o", "õ", "ow", "oʊ", "ɤ", "ø", "ɵ", "ɘ", "ə", "ɚ", "ʏ̈"}
        elif self is PhoneSetType.ARPA:
            return {"EY", "OW", "AH"}
        return set()

    @property
    def open_mid_vowels(self):
        """Open-mid vowels for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɛ", "ɜ", "ɞ", "œ", "ɔ", "ʌ", "ɐ", "æ", "ɛ̈", "ɔ̈", "ɝ"}
        elif self is PhoneSetType.ARPA:
            return {"EH", "AE", "ER"}
        return set()

    @property
    def open_vowels(self):
        """Open vowels for the phone set"""
        if self is PhoneSetType.IPA:
            return {"a", "ã", "ɶ", "ɒ", "ɑ"}
        elif self is PhoneSetType.ARPA:
            return {"AO", "AA"}
        return set()

    @property
    def front_vowels(self):
        """Front vowels for the phone set"""
        if self is PhoneSetType.IPA:
            return {
                "i",
                "ĩ",
                "y",
                "ɪ",
                "ʏ",
                "e",
                "ẽ",
                "ɪ",
                "ʏ",
                "ɛ̈",
                "ʏ̈",
                "ej",
                "eɪ",
                "ø",
                "ɛ",
                "œ",
                "æ",
                "ɶ",
            }
        elif self is PhoneSetType.ARPA:
            return {"IY", "EY", "EH", "AE", "IH"}
        return set()

    @property
    def central_vowels(self):
        """Central vowels for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɨ", "ʉ", "ɘ", "ɵ", "ə", "ɜ", "ɞ", "ɐ", "ɚ", "ã", "a", "ɝ"}
        elif self is PhoneSetType.ARPA:
            return {"UW", "AH", "ER"}
        return set()

    @property
    def back_vowels(self):
        """Back vowels for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɯ", "u", "ũ", "ʊ", "ɔ̈", "ɤ", "o", "õ", "ow", "oʊ", "ʌ", "ɔ", "ɑ", "ɒ"}
        elif self is PhoneSetType.ARPA:
            return {"OW", "AO", "AA", "UH"}
        return set()

    @property
    def rounded_vowels(self):
        """Rounded vowels for the phone set"""
        if self is PhoneSetType.IPA:
            return {
                "y",
                "ʏ",
                "o",
                "õ",
                "u",
                "ʊ",
                "ow",
                "oʊ",
                "ɔ",
                "ø",
                "ɵ",
                "ɞ",
                "œ",
                "ɒ",
                "ɶ",
                "ʉ",
                "ʏ̈",
                "ɔ̈",
                "ũ",
            }
        elif self is PhoneSetType.ARPA:
            return {"OW", "UW", "UH", "AO"}
        return set()

    @property
    def unrounded_vowels(self):
        """Unrounded vowels for the phone set"""
        if self is PhoneSetType.IPA:
            return {
                "i",
                "ĩ",
                "e",
                "ɛ̈",
                "ej",
                "ẽ",
                "ɤ",
                "eɪ",
                "ɨ",
                "ɯ",
                "ɘ",
                "ə",
                "ɚ",
                "ɪ",
                "ɪ̈",
                "ɛ",
                "ɜ",
                "ɝ",
                "ʌ",
                "ɐ",
                "ɑ",
                "æ",
                "ã",
                "a",
            }
        elif self is PhoneSetType.ARPA:
            return {"IY", "EY", "EH", "AH", "IH", "ER", "AE", "AA"}
        return set()

    @property
    def diphthong_phones(self) -> typing.Set[str]:
        """Diphthong phones for the phone set type (these will have 5 states in HMM topologies)"""
        if self is PhoneSetType.ARPA:
            return {
                "AY",
                "AY0",
                "AY1",
                "AY2",
                "AW",
                "AW0",
                "AW1",
                "AW2",
                "OY",
                "OY0",
                "OY1",
                "OY2",
            }
        if self is PhoneSetType.IPA or self is PhoneSetType.PINYIN:
            diphthongs = {x + y for x, y in itertools.product(self.vowels, self.vowels)}
            if self is PhoneSetType.IPA:
                diphthongs |= {x + y for x, y in itertools.product(self.glides, self.vowels)}
                diphthongs |= {x + y for x, y in itertools.product(self.vowels, self.glides)}

            return diphthongs

        return set()

    @property
    def vowels(self):
        """Vowels for the phone set type"""
        if self is PhoneSetType.PINYIN:
            return {"i", "u", "y", "e", "w", "a", "o", "e", "ü"}
        elif self is PhoneSetType.ARPA:
            return {"IH", "UH", "IY", "AE", "UW", "AH", "AO", "AA"}
        elif self is PhoneSetType.IPA:
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
        """Triphthong phones for the phone set type"""
        if self is PhoneSetType.IPA or self is PhoneSetType.PINYIN:
            triphthongs = {
                x + y + z for x, y, z in itertools.product(self.vowels, self.vowels, self.vowels)
            }
            if self is PhoneSetType.IPA:
                triphthongs |= {
                    x + y for x, y in itertools.product(self.glides, self.diphthong_phones)
                }
                triphthongs |= {
                    x + y for x, y in itertools.product(self.diphthong_phones, self.glides)
                }
            return triphthongs
        return set()

    @property
    def extra_questions(self) -> typing.Dict[str, typing.Set[str]]:
        """Extra questions for phone clustering in triphone models"""
        extra_questions = {}
        if self is PhoneSetType.ARPA:
            extra_questions["stops"] = self.stops
            extra_questions["fricatives"] = self.fricatives
            extra_questions["sibilants"] = self.sibilants | self.affricates
            extra_questions["approximants"] = self.approximants
            extra_questions["laterals"] = self.laterals
            extra_questions["nasals"] = self.nasals
            extra_questions["labials"] = self.labials | self.labiodental
            extra_questions["dental"] = self.dental | self.labiodental
            extra_questions["coronal"] = self.dental | self.alveolar | self.alveopalatal
            extra_questions["dorsal"] = self.velar | self.glottal

            extra_questions["unrounded"] = self.unrounded_vowels
            extra_questions["rounded"] = self.rounded_vowels
            extra_questions["front"] = self.front_vowels
            extra_questions["central"] = self.central_vowels
            extra_questions["back"] = self.back_vowels
            extra_questions["close"] = self.close_vowels
            extra_questions["close_mid"] = self.close_mid_vowels
            extra_questions["open_mid"] = self.open_mid_vowels
            extra_questions["open"] = self.open_vowels

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

            def add_consonant_variants(consonant_set):
                """Add consonant variants for the given set"""
                consonants = set()
                for p in consonant_set:
                    if p in self.voiceless_obstruents:
                        consonants |= voiceless_variants(p)
                    else:
                        consonants |= voiced_variants(p)
                return consonants

            extra_questions["stops"] = add_consonant_variants(self.stops)
            extra_questions["fricatives"] = add_consonant_variants(
                self.fricatives | self.lateral_fricatives
            )
            extra_questions["sibilants"] = add_consonant_variants(self.sibilants | self.affricates)
            extra_questions["approximants"] = add_consonant_variants(self.approximants)
            extra_questions["laterals"] = add_consonant_variants(self.laterals)
            extra_questions["nasals"] = add_consonant_variants(
                self.nasals | self.nasal_approximants
            )
            extra_questions["trills"] = add_consonant_variants(self.trills | self.taps)
            extra_questions["labials"] = add_consonant_variants(
                self.labials | self.labiodental | self.labialized
            )
            extra_questions["dental"] = add_consonant_variants(self.dental | self.labiodental)
            extra_questions["coronal"] = add_consonant_variants(
                self.dental | self.alveolar | self.retroflex | self.alveopalatal
            )
            extra_questions["dorsal"] = add_consonant_variants(
                self.palatal | self.velar | self.uvular
            )
            extra_questions["palatals"] = add_consonant_variants(
                self.palatal | self.alveopalatal | self.palatalized
            )
            extra_questions["pharyngeal"] = add_consonant_variants(
                self.pharyngeal | self.epiglottal | self.glottal
            )

            extra_questions["unrounded"] = add_consonant_variants(self.unrounded_vowels)
            extra_questions["rounded"] = add_consonant_variants(self.rounded_vowels)
            extra_questions["front"] = add_consonant_variants(self.front_vowels)
            extra_questions["central"] = add_consonant_variants(self.central_vowels)
            extra_questions["back"] = add_consonant_variants(self.back_vowels)
            extra_questions["close"] = add_consonant_variants(self.close_vowels)
            extra_questions["close_mid"] = add_consonant_variants(self.close_mid_vowels)
            extra_questions["open_mid"] = add_consonant_variants(self.open_mid_vowels)
            extra_questions["open"] = add_consonant_variants(self.open_vowels)

            extra_questions["front_semi_vowels"] = add_consonant_variants(
                {"j", "i", "ɪ", "ɥ", "ʏ", "y"}
            )
            extra_questions["back_semi_vowels"] = add_consonant_variants(
                {"w", "u", "ʊ", "ɰ", "ɯ", "ʍ"}
            )
            # Some language specific questions
            extra_questions["L_vocalization"] = {"ʊ", "ɫ", "u", "ʉ"}
            extra_questions["ts_z_variation"] = {"ts", "z"}
            extra_questions["rhotics"] = {"ɹ", "ɝ", "ɚ", "ə", "ʁ", "ɐ"}
            extra_questions["diphthongs"] = self.diphthong_phones
            extra_questions["triphthongs"] = self.triphthong_phones

        return extra_questions


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
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
    sox_string: str
        String to use for loading with sox
    """

    format: str
    sample_rate: int
    duration: float
    num_channels: int
    sox_string: str

    @property
    def meta(self) -> typing.Dict[str, typing.Any]:
        """Dictionary representation of sound file information"""
        return dataclassy.asdict(self)


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
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

    identifiers: typing.Set[str]
    lab_files: typing.Dict[str, str]
    textgrid_files: typing.Dict[str, str]
    wav_files: typing.Dict[str, str]
    other_audio_files: typing.Dict[str, str]


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class WordData:
    """
    Data class for information about a word and its pronunciations

    Parameters
    ----------
    orthography: str
        Orthographic string for the word
    pronunciations: set[tuple[str, ...]
        Set of tuple pronunciations for the word
    """

    orthography: str
    pronunciations: typing.Set[typing.Tuple[str, ...]]


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class PronunciationProbabilityCounter:
    """
    Data class for count information used in pronunciation probability modeling

    Parameters
    ----------
    ngram_counts: collections.defaultdict
        Counts of ngrams
    word_pronunciation_counts: collections.defaultdict
        Counts of word pronunciations
    silence_following_counts: collections.Counter
        Counts of silence following pronunciation
    non_silence_following_counts: collections.Counter
        Counts of non-silence following pronunciation
    silence_before_counts: collections.Counter
        Counts of silence before pronunciation
    non_silence_before_counts: collections.Counter
        Counts of non-silence before pronunciation

    """

    ngram_counts: collections.defaultdict = dataclassy.factory(collections.defaultdict)
    word_pronunciation_counts: collections.defaultdict = dataclassy.factory(
        collections.defaultdict
    )
    silence_following_counts: collections.Counter = dataclassy.factory(collections.Counter)
    non_silence_following_counts: collections.Counter = dataclassy.factory(collections.Counter)
    silence_before_counts: collections.Counter = dataclassy.factory(collections.Counter)
    non_silence_before_counts: collections.Counter = dataclassy.factory(collections.Counter)

    def __post_init__(self):
        """Initialize default dictionaries"""
        self.ngram_counts = collections.defaultdict(collections.Counter)
        self.word_pronunciation_counts = collections.defaultdict(collections.Counter)

    def add_counts(self, other_counter: PronunciationProbabilityCounter) -> None:
        """
        Combine counts of two :class:`~montreal_forced_aligner.data.PronunciationProbabilityCounter`

        Parameters
        ----------
        other_counter: :class:`~montreal_forced_aligner.data.PronunciationProbabilityCounter`
            Other object with pronunciation probability counts
        """
        for k, v in other_counter.ngram_counts.items():
            self.ngram_counts[k]["silence"] += v["silence"]
            self.ngram_counts[k]["non_silence"] += v["non_silence"]
        for k, v in other_counter.word_pronunciation_counts.items():
            for k2, v2 in v.items():
                self.word_pronunciation_counts[k][k2] += v2
        self.silence_following_counts.update(other_counter.silence_following_counts)
        self.non_silence_following_counts.update(other_counter.non_silence_following_counts)
        self.silence_before_counts.update(other_counter.silence_before_counts)
        self.non_silence_before_counts.update(other_counter.non_silence_before_counts)


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
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

    begin: float
    end: float
    label: str
    utterance: int

    def __lt__(self, other: CtmInterval):
        """Sorting function for CtmIntervals"""
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

    def to_tg_interval(self) -> Interval:
        """
        Converts the CTMInterval to
        `PraatIO's Interval class <http://timmahrt.github.io/praatIO/praatio/utilities/constants.html#Interval>`_

        Returns
        -------
        :class:`praatio.utilities.constants.Interval`
            Derived PraatIO Interval
        """
        if self.end < -1 or self.begin == 1000000:
            raise CtmError(self)
        return Interval(round(self.begin, 4), round(self.end, 4), self.label)
