"""Class definitions for corpora"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

from .base import Corpus  # noqa

if TYPE_CHECKING:

    SegmentsType = Dict[str, Dict[str, Union[str, float, int]]]
    OneToOneMappingType = Dict[str, str]
    OneToManyMappingType = Dict[str, List[str]]

    CorpusMappingType = Union[OneToOneMappingType, OneToManyMappingType]
    ScpType = Union[List[Tuple[str, str]], List[Tuple[str, List[Any]]]]
    CorpusGroupedOneToOne = List[List[Tuple[str, str]]]
    CorpusGroupedOneToMany = List[List[Tuple[str, List[Any]]]]
    CorpusGroupedType = Union[CorpusGroupedOneToMany, CorpusGroupedOneToOne]
