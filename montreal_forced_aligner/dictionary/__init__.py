"""
Pronunciation dictionaries
==========================

"""

from .base_dictionary import PronunciationDictionary
from .data import DictionaryData
from .multispeaker import MultispeakerDictionary

__all__ = [
    "base_dictionary",
    "multispeaker",
    "data",
    "MultispeakerDictionary",
    "PronunciationDictionary",
    "DictionaryData",
]
MultispeakerDictionary.__module__ = "montreal_forced_aligner.dictionary"
PronunciationDictionary.__module__ = "montreal_forced_aligner.dictionary"
DictionaryData.__module__ = "montreal_forced_aligner.dictionary"
