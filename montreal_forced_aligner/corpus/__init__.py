"""
Corpora
=======


"""
from __future__ import annotations

from montreal_forced_aligner.corpus.acoustic_corpus import (
    AcousticCorpus,
    AcousticCorpusMixin,
    AcousticCorpusPronunciationMixin,
)
from montreal_forced_aligner.corpus.base import CorpusMixin
from montreal_forced_aligner.corpus.classes import File, Speaker, Utterance
from montreal_forced_aligner.corpus.text_corpus import (
    DictionaryTextCorpusMixin,
    TextCorpus,
    TextCorpusMixin,
)

__all__ = [
    "base",
    "helper",
    "classes",
    "File",
    "Speaker",
    "Utterance",
    "features",
    "multiprocessing",
    "CorpusMixin",
    "ivector_corpus",
    "acoustic_corpus",
    "AcousticCorpus",
    "AcousticCorpusMixin",
    "AcousticCorpusPronunciationMixin",
    "text_corpus",
    "TextCorpus",
    "TextCorpusMixin",
    "DictionaryTextCorpusMixin",
]
