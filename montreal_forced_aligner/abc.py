"""
Abstract Base Classes
=====================
"""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Collection, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from .config.align_config import AlignConfig
    from .config.dictionary_config import DictionaryConfig
    from .config.transcribe_config import TranscribeConfig
    from .corpus.base import Corpus
    from .dictionary.multispeaker import MultispeakerDictionary
    from .models import AcousticModel, DictionaryModel, LanguageModel


__all__ = [
    "MfaModel",
    "MfaWorker",
    "Dictionary",
    "MetaDict",
    "AcousticModelWorker",
    "IvectorExtractor",
    "Trainer",
    "Transcriber",
    "Aligner",
    "DictionaryEntryType",
    "ReversedMappingType",
    "Labels",
]

# Configuration types
MetaDict = Dict[str, Any]
Labels = List[Any]
CtmErrorDict = Dict[Tuple[str, int], str]

# Dictionary types
DictionaryEntryType = List[Dict[str, Union[Tuple[str], float, None, int]]]
ReversedMappingType = Dict[int, str]
WordsType = Dict[str, DictionaryEntryType]
MappingType = Dict[str, int]
MultiSpeakerMappingType = Dict[str, str]
IpaType = Optional[List[str]]
PunctuationType = Optional[str]

# Corpus types
SegmentsType = Dict[str, Dict[str, Union[str, float, int]]]
OneToOneMappingType = Dict[str, str]
OneToManyMappingType = Dict[str, List[str]]

CorpusMappingType = Union[OneToOneMappingType, OneToManyMappingType]
ScpType = Union[List[Tuple[str, str]], List[Tuple[str, List[Any]]]]
CorpusGroupedOneToOne = List[List[Tuple[str, str]]]
CorpusGroupedOneToMany = List[List[Tuple[str, List[Any]]]]
CorpusGroupedType = Union[CorpusGroupedOneToMany, CorpusGroupedOneToOne]


class MfaWorker(metaclass=ABCMeta):
    """Abstract class for MFA workers"""

    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    @property
    @abstractmethod
    def working_directory(self) -> str:
        """Current directory"""
        ...

    @property
    def data_directory(self) -> str:
        """Corpus data directory"""
        return self._data_directory

    @data_directory.setter
    def data_directory(self, val: str) -> None:
        self._data_directory = val

    @property
    def uses_voiced(self) -> bool:
        """Flag for using voiced features"""
        return self._uses_voiced

    @uses_voiced.setter
    def uses_voiced(self, val: bool) -> None:
        self._uses_voiced = val

    @property
    def uses_cmvn(self) -> bool:
        """Flag for using CMVN"""
        return self._uses_cmvn

    @uses_cmvn.setter
    def uses_cmvn(self, val: bool) -> None:
        self._uses_cmvn = val

    @property
    def uses_splices(self) -> bool:
        """Flag for using spliced features"""
        return self._uses_splices

    @uses_splices.setter
    def uses_splices(self, val: bool) -> None:
        self._uses_splices = val

    @property
    def speaker_independent(self) -> bool:
        """Flag for speaker independent features"""
        return self._speaker_independent

    @speaker_independent.setter
    def speaker_independent(self, val: bool) -> None:
        self._speaker_independent = val

    @property
    @abstractmethod
    def working_log_directory(self) -> str:
        """Current log directory"""
        ...

    @property
    def use_mp(self) -> bool:
        """Flag for using multiprocessing"""
        return self._use_mp

    @use_mp.setter
    def use_mp(self, val: bool) -> None:
        self._use_mp = val


class AcousticModelWorker(MfaWorker):
    """
    Abstract class for MFA classes that use acoustic models

    Parameters
    ----------
    dictionary: MultispeakerDictionary
        Dictionary for the worker docstring
    """

    def __init__(self, corpus: Corpus, dictionary: MultispeakerDictionary):
        super().__init__(corpus)
        self.dictionary: MultispeakerDictionary = dictionary


class Trainer(AcousticModelWorker):
    """
    Abstract class for MFA trainers

    Attributes
    ----------
    iteration: int
        Current iteration
    """

    def __init__(self, corpus: Corpus, dictionary: MultispeakerDictionary):
        super(Trainer, self).__init__(corpus, dictionary)
        self.iteration = 0

    @property
    @abstractmethod
    def meta(self) -> MetaDict:
        """Training configuration parameters"""
        ...

    @abstractmethod
    def train(self) -> None:
        """Perform training"""
        ...


class Aligner(AcousticModelWorker):
    """Abstract class for MFA aligners"""

    def __init__(
        self, corpus: Corpus, dictionary: MultispeakerDictionary, align_config: AlignConfig
    ):
        super().__init__(corpus, dictionary)
        self.align_config = align_config

    @abstractmethod
    def align(self, subset: Optional[int] = None) -> None:
        """Perform alignment"""
        ...

    @property
    @abstractmethod
    def model_path(self) -> str:
        """Acoustic model file path"""
        ...

    @property
    @abstractmethod
    def alignment_model_path(self) -> str:
        """Acoustic model file path for speaker-independent alignment"""
        ...


class Transcriber(AcousticModelWorker):
    """Abstract class for MFA transcribers"""

    def __init__(
        self,
        corpus: Corpus,
        dictionary: MultispeakerDictionary,
        acoustic_model: AcousticModel,
        language_model: LanguageModel,
        transcribe_config: TranscribeConfig,
    ):
        super().__init__(corpus, dictionary)
        self.acoustic_model = acoustic_model
        self.language_model = language_model
        self.transcribe_config = transcribe_config

    @abstractmethod
    def transcribe(self) -> None:
        """Perform transcription"""
        ...

    @property
    @abstractmethod
    def model_path(self) -> str:
        """Acoustic model file path"""
        ...


class IvectorExtractor(AcousticModelWorker):
    """Abstract class for MFA ivector extractors"""

    @abstractmethod
    def extract_ivectors(self) -> None:
        """Extract ivectors"""
        ...

    @property
    @abstractmethod
    def model_path(self) -> str:
        """Acoustic model file path"""
        ...

    @property
    @abstractmethod
    def ivector_options(self) -> MetaDict:
        """Ivector parameters"""
        ...

    @property
    @abstractmethod
    def dubm_path(self) -> str:
        """DUBM model file path"""
        ...

    @property
    @abstractmethod
    def ie_path(self) -> str:
        """Ivector extractor model file path"""
        ...


class Dictionary(ABC):
    """Abstract class for pronunciation dictionaries"""

    def __init__(self, dictionary_model: DictionaryModel, config: DictionaryConfig):
        self.name = dictionary_model.name
        self.dictionary_model = dictionary_model
        self.config = config


class MfaModel(ABC):
    """Abstract class for MFA models"""

    @property
    @abstractmethod
    def extensions(self) -> Collection:
        """File extensions for the model"""
        ...

    @extensions.setter
    @abstractmethod
    def extensions(self, val: Collection) -> None:
        ...

    @classmethod
    @abstractmethod
    def valid_extension(cls, filename: str) -> bool:
        """Check whether a file has a valid extensions"""
        ...

    @classmethod
    @abstractmethod
    def generate_path(cls, root: str, name: str, enforce_existence: bool = True) -> Optional[str]:
        """Generate a path from a root directory"""
        ...

    @abstractmethod
    def pretty_print(self):
        """Print the model's meta data"""
        ...

    @property
    @abstractmethod
    def meta(self) -> MetaDict:
        """Meta data for the model"""
        ...

    @abstractmethod
    def add_meta_file(self, trainer: Trainer) -> None:
        """Add meta data to the model"""
        ...
