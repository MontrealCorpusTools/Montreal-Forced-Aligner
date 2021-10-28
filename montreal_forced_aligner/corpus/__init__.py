from __future__ import annotations
from typing import TYPE_CHECKING, Union, Dict, List, Tuple, Any
from .align_corpus import AlignableCorpus
from .transcribe_corpus import TranscribeCorpus

if TYPE_CHECKING:
    from .transcribe_corpus import BaseCorpus
    CorpusType = BaseCorpus
    SegmentsType = Dict[str, Dict[str, Union[str,float, int]]]
    OneToOneMappingType = Dict[str, str]
    OneToManyMappingType = Dict[str, List[str]]

    CorpusMappingType = Union[OneToOneMappingType, OneToManyMappingType]
    ScpType = Union[List[Tuple[str, str]], List[Tuple[str, List[Any]]]]
    CorpusGroupedOneToOne = List[List[Tuple[str, str]]]
    CorpusGroupedOneToMany = List[List[Tuple[str, List[Any]]]]
    CorpusGroupedType = Union[CorpusGroupedOneToMany, CorpusGroupedOneToOne]