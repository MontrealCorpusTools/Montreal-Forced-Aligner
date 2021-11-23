"""
Corpora
=======


"""
from __future__ import annotations

from .acoustic_corpus import AcousticCorpusMixin, AcousticCorpusPronunciationMixin
from .base import CorpusMixin
from .classes import File, Speaker, Utterance
from .text_corpus import DictionaryTextCorpus, DictionaryTextCorpusMixin, TextCorpusMixin

__all__ = [
    "CorpusMixin",
    "AcousticCorpusPronunciationMixin",
    "AcousticCorpusMixin",
    "TextCorpusMixin",
    "DictionaryTextCorpus",
    "DictionaryTextCorpusMixin",
    "Speaker",
    "Utterance",
    "File",
    "base",
    "helper",
    "classes",
]
