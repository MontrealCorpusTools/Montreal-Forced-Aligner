"""
Data classes
============

"""
from __future__ import annotations

import collections
import enum
import io
import itertools
import math
import re
import typing

import dataclassy
import pynini
import pywrapfst
from praatio.utilities.constants import Interval, TextgridFormats

from montreal_forced_aligner.exceptions import CtmError

__all__ = [
    "MfaArguments",
    "CtmInterval",
    "TextFileType",
    "TextgridFormats",
    "SoundFileType",
    "WordType",
    "PhoneType",
    "PhoneSetType",
    "WordData",
    "DatabaseImportData",
    "PronunciationProbabilityCounter",
]

M_LOG_2PI = 1.8378770664093454835606594728112


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

    Attributes
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: str
        Path to save logging information during the run
    """

    job_name: int
    db_string: str
    log_path: str


class TextFileType(enum.Enum):
    """Enum for types of text files"""

    NONE = "none"  #: No text file
    TEXTGRID = TextgridFormats.LONG_TEXTGRID  #: Praat's long textgrid format
    SHORT_TEXTGRID = TextgridFormats.SHORT_TEXTGRID  #: Praat's short textgrid format
    LAB = "lab"  #: Text file
    JSON = TextgridFormats.JSON  #: JSON

    def __str__(self) -> str:
        """Name of phone set"""
        return self.value


class DatasetType(enum.Enum):
    """Enum for types of sound files"""

    NONE = 0  #: Nothing has been imported
    ACOUSTIC_CORPUS = 1  #: Imported corpus with sound files (and maybe text files)
    TEXT_CORPUS = 2  #: Imported corpus with just text files
    ACOUSTIC_CORPUS_WITH_DICTIONARY = (
        3  #: Imported corpus and pronunciation dictionary with sound files
    )
    TEXT_CORPUS_WITH_DICTIONARY = (
        4  #: Imported corpus and pronunciation dictionary with just text files
    )
    DICTIONARY = 5  #: Only imported pronunciation dictionary (for G2P)


class SoundFileType(enum.Enum):
    """Enum for types of sound files"""

    NONE = 0  #: No sound file
    WAV = 1  #: Can be read as a .wav file
    SOX = 2  #: Needs to use SoX to preprocess


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

    non_silence = 1  #: Speech sounds
    silence = 2  #: Silence phones
    oov = 3  #: Out of vocabulary/spoken noise phones
    disambiguation = 4  #: Disambiguation phones internal to Kaldi
    extra = 5  #: Phones not to be included generally, i.e., loaded from reference intervals


class WorkflowType(enum.Enum):
    """
    Enum for workflows involving corpora

    Parameters
    ----------
    reference: int
        Load alignments from reference directory
    alignment: int
        Align using corpus texts, acoustic model, and pronunciation dictionary
    transcription: int
        Transcribe using acoustic model, pronunciation dictionary, and language model
    phone_transcription: int
        Transcribe using acoustic model and phone-based language model
    per_speaker_transcription: int
        Transcribe using acoustic model, pronunciation dictionary, and per-speaker language model generated by corpus texts
    speaker_diarization: int
        Diarize speakers
    online_alignment: int
        Online alignment
    acoustic_training: int
        Acoustic model training
    acoustic_model_adaptation: int
        Acoustic model adaptation
    segmentation: int
        Segment based on speech activity
    """

    reference = 0
    alignment = 1
    transcription = 2
    phone_transcription = 3
    per_speaker_transcription = 4
    speaker_diarization = 5
    online_alignment = 6
    acoustic_training = 7
    acoustic_model_adaptation = 8
    segmentation = 9
    train_g2p = 10
    g2p = 11
    language_model_training = 12


class WordType(enum.Enum):
    """Enum for types of words"""

    speech = 1  #: General speech words
    clitic = 2  #: Clitics that must attach to words
    silence = 3  #: Words representing silence
    oov = 4  #: Words representing out of vocabulary items
    bracketed = 5  #: Words that are in brackets
    cutoff = 6  #: Words that are cutoffs of particular words or hesitations of the next word
    laughter = 7  #: Words that represent laughter
    noise = 8  #: Words that represent non-speech noise
    music = 9  #: Words that represent music
    disambiguation = 10  #: Disambiguation symbols internal to Kaldi


class DistanceMetric(enum.Enum):

    cosine = "cosine"
    plda = "plda"
    euclidean = "euclidean"


class ClusterType(enum.Enum):
    """Enum for supported clustering algorithms"""

    mfa = "mfa"
    affinity = "affinity"
    agglomerative = "agglomerative"
    spectral = "spectral"
    dbscan = "dbscan"
    hdbscan = "hdbscan"
    optics = "optics"
    kmeans = "kmeans"
    meanshift = "meanshift"


class ManifoldAlgorithm(enum.Enum):
    """Enum for supported manifold visualization algorithms"""

    tsne = "tsne"
    mds = "mds"
    spectral = "spectral"
    isomap = "isomap"


class PhoneSetType(enum.Enum):
    """Enum for types of phone sets"""

    UNKNOWN = "UNKNOWN"  #: Unknown
    AUTO = "AUTO"  #: Inspect dictionary to pick the most common phone set type
    IPA = "IPA"  #: IPA-based phoneset
    ARPA = "ARPA"  #: US English-based Arpabet
    PINYIN = "PINYIN"  #: Pinyin for Mandarin

    def __str__(self) -> str:
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
    def voiceless_obstruents(self) -> typing.Set[str]:
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
    def voiced_obstruents(self) -> typing.Set[str]:
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
    def implosive_obstruents(self) -> typing.Set[str]:
        """Implosive obstruents for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɓ", "ɗ", "ʄ", "ɠ", "ʛ", "ᶑ", "ɗ̪"}
        return set()

    @property
    def stops(self) -> typing.Set[str]:
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
    def sibilants(self) -> typing.Set[str]:
        """Sibilants for the phone set"""
        if self is PhoneSetType.IPA:
            return {"s", "s̪", "ʃ", "ʂ", "ɕ", "z", "z̪", "ʒ", "ʑ", "ʐ", "ɧ"}
        elif self is PhoneSetType.ARPA:
            return {"SH", "S", "ZH", "Z"}
        return set()

    @property
    def affricates(self) -> typing.Set[str]:
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
    def fricatives(self) -> typing.Set[str]:
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
    def laterals(self) -> typing.Set[str]:
        """Laterals for the phone set"""
        if self is PhoneSetType.IPA:
            return {"l", "ɫ", "ʟ", "ʎ", "l̪"}
        elif self is PhoneSetType.ARPA:
            return {"L"}
        return set()

    @property
    def nasals(self) -> typing.Set[str]:
        """Nasals for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɲ", "ŋ", "m", "n", "ɳ", "ɴ", "ɱ", "ŋm", "n̪"}
        elif self is PhoneSetType.ARPA:
            return {"M", "N", "NG"}
        return set()

    @property
    def trills(self) -> typing.Set[str]:
        """Trills for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ʙ", "r", "ʀ", "r̝"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def taps(self) -> typing.Set[str]:
        """Taps for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɾ", "ɽ", "ⱱ"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def lateral_taps(self) -> typing.Set[str]:
        """Lateral taps for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɭ", "ɺ"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def lateral_fricatives(self) -> typing.Set[str]:
        """Lateral fricatives for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɬ", "ɮ"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def approximants(self) -> typing.Set[str]:
        """Approximants for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɹ", "ɻ", "ʋ", "ʍ"} | self.glides
        elif self is PhoneSetType.ARPA:
            return {"R"} | self.glides
        return set()

    @property
    def glides(self) -> typing.Set[str]:
        """Glides for the phone set"""
        if self is PhoneSetType.IPA:
            return {"j", "w", "w̃", "j̃", "ɥ", "ɰ", "ɥ̃", "ɰ̃", "j̰"}
        elif self is PhoneSetType.ARPA:
            return {"Y", "W"}
        return set()

    @property
    def nasal_approximants(self) -> typing.Set[str]:
        """Nasal approximants for the phone set"""
        if self is PhoneSetType.IPA:
            return {"w̃", "j̃", "ɥ̃", "ɰ̃"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def labials(self) -> typing.Set[str]:
        """Labials for the phone set"""
        if self is PhoneSetType.IPA:
            return {"b", "p", "m", "ɸ", "β", "ɓ", "w", "ʍ"}
        elif self is PhoneSetType.ARPA:
            return {"B", "P", "M", "W"}
        return set()

    @property
    def labiodental(self) -> typing.Set[str]:
        """Labiodentals for the phone set"""
        if self is PhoneSetType.IPA:
            return {"f", "v", "ʋ", "ⱱ", "ɱ", "pf"}
        elif self is PhoneSetType.ARPA:
            return {"F", "V"}
        return set()

    @property
    def dental(self) -> typing.Set[str]:
        """Dentals for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ð", "θ", "t̪", "d̪", "s̪", "z̪", "t̪s̪", "d̪z̪", "n̪", "l̪", "ɗ̪"}
        elif self is PhoneSetType.ARPA:
            return {"DH", "TH"}
        return set()

    @property
    def alveolar(self) -> typing.Set[str]:
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
    def retroflex(self) -> typing.Set[str]:
        """Retroflexes for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ʈ", "ʂ", "ʐ", "ɖ", "ɽ", "ɻ", "ɭ", "ɳ", "ʈʂ", "ɖʐ", "ᶑ"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def alveopalatal(self) -> typing.Set[str]:
        """Alveopalatals for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ʒ", "ʃ", "dʒ", "tʃ"}
        elif self is PhoneSetType.ARPA:
            return {"ZH", "SH", "JH", "CH"}
        return set()

    @property
    def palatalized(self) -> typing.Set[str]:
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
    def labialized(self) -> typing.Set[str]:
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
    def palatal(self) -> typing.Set[str]:
        """Palatal phones for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ç", "c", "ɕ", "tɕ", "ɟ", "ɟʝ", "ʝ", "ɲ", "ɥ", "j", "ʎ", "ʑ", "dʑ"}
        elif self is PhoneSetType.ARPA:
            return {"Y"}
        return set()

    @property
    def velar(self) -> typing.Set[str]:
        """Velar phones for the phone set"""
        if self is PhoneSetType.IPA:
            return {"k", "x", "ɡ", "ɠ", "ɣ", "ɰ", "ŋ"}
        elif self is PhoneSetType.ARPA:
            return {"K", "NG", "G"}
        return set()

    @property
    def uvular(self) -> typing.Set[str]:
        """Uvular phones for the phone set"""
        if self is PhoneSetType.IPA:
            return {"q", "ɢ", "ʛ", "χ", "ʀ", "ʁ", "ʟ", "ɴ"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def pharyngeal(self) -> typing.Set[str]:
        """Pharyngeal phones for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ʕ", "ħ"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def epiglottal(self) -> typing.Set[str]:
        """Epiglottal phones for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ʡ", "ʢ", "ʜ"}
        elif self is PhoneSetType.ARPA:
            return set()
        return set()

    @property
    def glottal(self) -> typing.Set[str]:
        """Glottal phones for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ʔ", "ɦ", "h"}
        elif self is PhoneSetType.ARPA:
            return {"HH"}
        return set()

    @property
    def close_vowels(self) -> typing.Set[str]:
        """Close vowels for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɪ", "ɨ", "ɪ̈", "ʉ", "ʊ", "i", "ĩ", "ɯ", "y", "u", "ʏ", "ũ"}
        elif self is PhoneSetType.ARPA:
            return {"IH", "UH", "IY", "UW"}
        return set()

    @property
    def close_mid_vowels(self) -> typing.Set[str]:
        """Close-mid vowels for the phone set"""
        if self is PhoneSetType.IPA:
            return {"e", "ẽ", "ej", "eɪ", "o", "õ", "ow", "oʊ", "ɤ", "ø", "ɵ", "ɘ", "ə", "ɚ", "ʏ̈"}
        elif self is PhoneSetType.ARPA:
            return {"EY", "OW", "AH"}
        return set()

    @property
    def open_mid_vowels(self) -> typing.Set[str]:
        """Open-mid vowels for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɛ", "ɜ", "ɞ", "œ", "ɔ", "ʌ", "ɐ", "æ", "ɛ̈", "ɔ̈", "ɝ"}
        elif self is PhoneSetType.ARPA:
            return {"EH", "AE", "ER"}
        return set()

    @property
    def open_vowels(self) -> typing.Set[str]:
        """Open vowels for the phone set"""
        if self is PhoneSetType.IPA:
            return {"a", "ã", "ɶ", "ɒ", "ɑ"}
        elif self is PhoneSetType.ARPA:
            return {"AO", "AA"}
        return set()

    @property
    def front_vowels(self) -> typing.Set[str]:
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
    def central_vowels(self) -> typing.Set[str]:
        """Central vowels for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɨ", "ʉ", "ɘ", "ɵ", "ə", "ɜ", "ɞ", "ɐ", "ɚ", "ã", "a", "ɝ"}
        elif self is PhoneSetType.ARPA:
            return {"UW", "AH", "ER"}
        return set()

    @property
    def back_vowels(self) -> typing.Set[str]:
        """Back vowels for the phone set"""
        if self is PhoneSetType.IPA:
            return {"ɯ", "u", "ũ", "ʊ", "ɔ̈", "ɤ", "o", "õ", "ow", "oʊ", "ʌ", "ɔ", "ɑ", "ɒ"}
        elif self is PhoneSetType.ARPA:
            return {"OW", "AO", "AA", "UH"}
        return set()

    @property
    def rounded_vowels(self) -> typing.Set[str]:
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
    def unrounded_vowels(self) -> typing.Set[str]:
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
    def vowels(self) -> typing.Set[str]:
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
class NgramHistoryState:
    """
    Data class for storing ngram history
    """

    backoff_prob: float = 1.0
    word_to_prob: dict = {}


class ArpaNgramModel:
    """
    Wrapper class for ngram models, taken largely from :kaldi_utils`:`lang/internal/arpa2fst_constrained.py`
    """

    def __init__(self):
        self.orders = {0: collections.defaultdict(NgramHistoryState)}

    @classmethod
    def read(cls, input: typing.Union[io.StringIO, str]):
        """
        Read an ngram model from a stream

        Parameters
        ----------
        input: :class:`io.StringIO` or str
            Input stream or file path to read

        Returns
        -------
        :class:`~montreal_forced_aligner.data.ArpaNgramModel`
            Constructed model
        """
        cleanup = False
        if isinstance(input, str):
            cleanup = True
            input = open(input, "r", encoding="utf8")
        log10 = math.log(10.0)
        current_order = -1
        model = ArpaNgramModel()
        for line in input:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"\\(?P<order>[0-9]*)-grams:$", line)
            if m:
                current_order = int(m.group("order"))
                model.orders[current_order] = collections.defaultdict(NgramHistoryState)
                continue
            if current_order < 1:
                continue
            if line.startswith("\\"):
                continue
            col = line.split()
            prob = math.exp(float(col[0]) * log10)
            hist = tuple(col[1:current_order])
            word = col[current_order]  # a string
            backoff_prob = (
                math.exp(float(col[current_order + 1]) * log10)
                if len(col) == current_order + 2
                else None
            )

            model.orders[current_order - 1][hist].word_to_prob[word] = prob
            if backoff_prob is not None:
                model.orders[current_order][hist + (word,)].backoff_prob = backoff_prob
        if cleanup:
            input.close()
        return model

    def history_to_fst_state_mapping(
        self, min_order: int = None, max_order: int = None
    ) -> typing.Tuple[
        typing.Dict[typing.Tuple[str, ...], int], typing.List[typing.Tuple[str, ...]]
    ]:
        """

        This function, called from PrintAsFst, returns (hist_to_state,
        state_to_hist), which map from history (as a tuple of strings) to
        integer FST-state and vice versa.

        Parameters
        ----------
        min_order: int, optional
            Minimum order of ngrams to construct state mapping
        max_order: int, optional
            Maximum order of ngrams to construct state mapping

        Returns
        -------
        typing.Dict[typing.Tuple[str, ...], int]
            History to state mapping
        typing.List[typing.Tuple[str, ...]]
            State to history mapping
        """

        hist_to_state = {}
        state_to_hist = []

        # Make sure the initial bigram state comes first (and that
        # we have such a state even if it was completely pruned
        # away in the bigram LM.. which is unlikely of course)
        hist = ("<s>",)
        hist_to_state[hist] = len(state_to_hist)
        state_to_hist.append(hist)

        # create a bigram state for each of the 'real' words...  even if the LM
        # didn't naturally have such bigram states, we'll create them so that we
        # can enforce the bigram constraints supplied in 'bigrams_file' by the
        # user.
        for word in self.orders[0][()].word_to_prob:
            if word != "<s>" and word != "</s>":
                hist = (word,)
                hist_to_state[hist] = len(state_to_hist)
                state_to_hist.append(hist)

        # note: we do not allocate an FST state for the unigram state, because
        # we don't have a unigram state in the output FST, only bigram states; and
        # we don't iterate over bigram histories because we covered them all above;
        # that's why we start 'n' from 2 below instead of from 0.
        for order, history_states in self.orders.items():
            if min_order is not None and order < min_order:
                continue
            if max_order is not None and order > max_order:
                continue
            for hist in history_states.keys():
                # note: hist is a tuple of strings.
                assert hist not in hist_to_state
                hist_to_state[hist] = len(state_to_hist)
                state_to_hist.append(hist)

        return (hist_to_state, state_to_hist)

    def _get_prob(self, hist: typing.Tuple[str, ...], word: str) -> float:
        """
        Returns the probability of word 'word' in history-state 'hist'.
        Dies with error if this word is not predicted at all by the LM (not in vocab).
        history-state does not exist.

        Parameters
        ----------
        hist: tuple[str,...]
            History for ngram
        word: str
            Current word

        Returns
        -------
        float
            Probability
        """
        assert len(hist) < len(self.orders)
        if len(hist) == 0:
            word_to_prob = self.orders[0][()].word_to_prob
            return word_to_prob[word]
        else:
            if hist in self.orders[len(hist)]:
                hist_state = self.orders[len(hist)][hist]
                if word in hist_state.word_to_prob:
                    return hist_state.word_to_prob[word]
                else:
                    return hist_state.backoff_prob * self._get_prob(hist[1:], word)
            else:
                return self._get_prob(hist[1:], word)

    def _get_state_for_hist(self, hist_to_state, hist) -> int:
        """
        This gets the state corresponding to 'hist' in 'hist_to_state', but backs
        off for us if there is no such state.

        Parameters
        ----------
        hist_to_state: dict[tuple[str, ...], int]
            Mapping of history to states
        hist: tuple[str, ...]
            History to look up

        Returns
        -------
        int
            State for history
        """
        if hist in hist_to_state:
            return hist_to_state[hist]
        else:
            assert len(hist) > 1
            return self._get_state_for_hist(hist_to_state, hist[1:])

    def construct_bigram_fst(
        self,
        disambig_symbol: str,
        bigram_map: typing.Dict[str, typing.Set[str]],
        symbols: pywrapfst.SymbolTable,
    ) -> pynini.Fst:
        """

        This function prints the estimated language model as an FST.
        disambig_symbol will be something like '#0' (a symbol introduced
        to make the result determinizable).
        bigram_map represent the allowed bigrams (left-word, right-word): it's a map
        from left-word to a set of right-words (both are strings).

        Parameters
        ----------
        disambig_symbol: str
            Disambiguation symbol
        bigram_map: dict[str, set[str]]
            Mapping of left bigrams to allowed right bigrams
        symbols: :class:`pywrapfst.SymbolTable`
            Symbol table for the FST

        Returns
        -------
        :class:`pynini.Fst`
            Bigram FST
        """

        # History will map from history (as a tuple) to integer FST-state.
        (hist_to_state, state_to_hist) = self.history_to_fst_state_mapping(min_order=2)

        # The following 3 things are just for diagnostics.
        normalization_stats = [[0, 0.0] for _ in range(len(self.orders))]
        num_ngrams_allowed = 0
        num_ngrams_disallowed = 0

        fst = pynini.Fst()
        for state in range(len(state_to_hist)):
            s = fst.add_state()
            hist = state_to_hist[state]
            hist_len = len(hist)
            assert hist_len > 0
            if hist_len == 1:  # it's a bigram state...
                context_word = hist[0]
                if context_word not in bigram_map:
                    continue
                # word list is a list of words that can follow this word.  It must be nonempty.
                word_list = list(bigram_map[context_word])

                normalization_stats[hist_len][0] += 1

                for word in word_list:
                    prob = self._get_prob((context_word,), word)
                    assert prob != 0
                    normalization_stats[hist_len][1] += prob
                    cost = -math.log(prob)
                    if word == "</s>":
                        fst.set_final(s, pywrapfst.Weight(fst.weight_type(), cost))
                    else:
                        next_state = self._get_state_for_hist(hist_to_state, (context_word, word))
                        k = symbols.find(word)
                        fst.add_arc(state, pywrapfst.Arc(k, k, cost, next_state))
            else:  # it's a higher-order than bigram state.
                assert hist in self.orders[hist_len]
                hist_state = self.orders[hist_len][hist]
                most_recent_word = hist[-1]

                normalization_stats[hist_len][0] += 1
                normalization_stats[hist_len][1] += sum(
                    self._get_prob(hist, word) for word in bigram_map[most_recent_word]
                )

                for word, prob in hist_state.word_to_prob.items():
                    cost = -math.log(prob)
                    if word in bigram_map[most_recent_word]:
                        num_ngrams_allowed += 1
                    else:
                        num_ngrams_disallowed += 1
                        continue
                    if word == "</s>":
                        fst.set_final(s, pywrapfst.Weight(fst.weight_type(), cost))
                    else:
                        next_state = self._get_state_for_hist(hist_to_state, (hist) + (word,))
                        k = symbols.find(word)
                        fst.add_arc(state, pywrapfst.Arc(k, k, cost, next_state))

                assert hist in self.orders[hist_len]
                backoff_prob = self.orders[hist_len][hist].backoff_prob
                assert backoff_prob != 0.0
                cost = -math.log(backoff_prob)
                backoff_hist = hist[1:]
                backoff_state = self._get_state_for_hist(hist_to_state, backoff_hist)

                this_disambig_symbol = (
                    disambig_symbol if len(hist_state.word_to_prob) != 0 else "<eps>"
                )
                k = symbols.find(this_disambig_symbol)
                eps = symbols.find("<eps>")
                fst.add_arc(state, pywrapfst.Arc(k, eps, cost, backoff_state))
        fst.set_start(0)
        return fst

    def export_bigram_fst(
        self,
        output: typing.Union[str, io.StringIO],
        disambig_symbol: str,
        bigram_map: typing.Dict[str, typing.Set[str]],
    ) -> None:
        """

        This function prints the estimated language model as an FST.
        disambig_symbol will be something like '#0' (a symbol introduced
        to make the result determinizable).
        bigram_map represent the allowed bigrams (left-word, right-word): it's a map
        from left-word to a set of right-words (both are strings).

        Parameters
        ----------
        output: :class:`io.StringIO` or str
            Output stream or file name to export to
        disambig_symbol: str
            Disambiguation symbol to use
        bigram_map: dict[str, set[str]]
            Mapping of left bigrams to allowed right bigrams

        """

        # History will map from history (as a tuple) to integer FST-state.
        (hist_to_state, state_to_hist) = self.history_to_fst_state_mapping(min_order=2)

        # The following 3 things are just for diagnostics.
        normalization_stats = [[0, 0.0] for _ in range(len(self.orders))]
        num_ngrams_allowed = 0
        num_ngrams_disallowed = 0

        if isinstance(output, str):
            output = open(output, "w", encoding="utf8")
        for state in range(len(state_to_hist)):
            hist = state_to_hist[state]
            hist_len = len(hist)
            assert hist_len > 0
            if hist_len == 1:  # it's a bigram state...
                context_word = hist[0]
                if context_word not in bigram_map:
                    continue
                # word list is a list of words that can follow this word.  It must be nonempty.
                word_list = list(bigram_map[context_word])

                normalization_stats[hist_len][0] += 1

                for word in word_list:
                    prob = self._get_prob((context_word,), word)
                    assert prob != 0
                    normalization_stats[hist_len][1] += prob
                    cost = -math.log(prob)
                    if word == "</s>":
                        output.write(f"{state} {cost:.3f}\n")
                    else:
                        next_state = self._get_state_for_hist(hist_to_state, (context_word, word))
                        output.write(f"{state} {next_state} {word} {word} {cost:.3f}\n")
            else:  # it's a higher-order than bigram state.
                assert hist in self.orders[hist_len]
                hist_state = self.orders[hist_len][hist]
                most_recent_word = hist[-1]

                normalization_stats[hist_len][0] += 1
                normalization_stats[hist_len][1] += sum(
                    self._get_prob(hist, word) for word in bigram_map[most_recent_word]
                )

                for word, prob in hist_state.word_to_prob.items():
                    cost = -math.log(prob)
                    if word in bigram_map[most_recent_word]:
                        num_ngrams_allowed += 1
                    else:
                        num_ngrams_disallowed += 1
                        continue
                    if word == "</s>":
                        output.write(f"{state} {cost:.3f}\n")
                    else:
                        next_state = self._get_state_for_hist(hist_to_state, (hist) + (word,))
                        output.write(f"{state} {next_state} {word} {word} {cost:.3f}\n")

                assert hist in self.orders[hist_len]
                backoff_prob = self.orders[hist_len][hist].backoff_prob
                assert backoff_prob != 0.0
                cost = -math.log(backoff_prob)
                backoff_hist = hist[1:]
                backoff_state = self._get_state_for_hist(hist_to_state, backoff_hist)

                this_disambig_symbol = (
                    disambig_symbol if len(hist_state.word_to_prob) != 0 else "<eps>"
                )
                output.write(f"{state} {backoff_state} {this_disambig_symbol} <eps> {cost:.3f}")
        output.close()


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

    def __post_init__(self) -> None:
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
    confidence: float, optional
        Confidence score of the interval
    """

    begin: float
    end: float
    label: typing.Union[int, str]
    confidence: typing.Optional[float] = None

    def __lt__(self, other: CtmInterval):
        """Sorting function for CtmIntervals"""
        return self.begin < other.begin

    def __add__(self, other):
        if isinstance(other, str):
            return self.label + other
        else:
            self.begin += other
            self.end += other

    def __post_init__(self) -> None:
        """
        Check on data validity

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.CtmError`
            If begin or end are not valid
        """
        if self.end < -1 or self.begin == 1000000:
            raise CtmError(self)

    def to_tg_interval(self, file_duration=None) -> Interval:
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
        end = round(self.end, 6)
        if file_duration is not None and end > file_duration:
            end = round(file_duration, 6)
        return Interval(round(self.begin, 6), end, self.label)


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class WordCtmInterval:
    """
    Data class for word intervals derived from CTM files

    Parameters
    ----------
    begin: float
        Start time of interval
    end: float
        End time of interval
    word_id: int
        Integer id of word
    pronunciation_id: int
        Pronunciation integer id of word
    """

    begin: float
    end: float
    word_id: int
    pronunciation_id: int

    def __lt__(self, other: WordCtmInterval):
        """Sorting function for WordCtmIntervals"""
        return self.begin < other.begin
