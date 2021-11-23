"""
Pronunciation dictionaries
==========================

"""

from .base_dictionary import DictionaryData, DictionaryMixin, PronunciationDictionaryMixin
from .multispeaker import MultispeakerDictionaryMixin

__all__ = [
    "base_dictionary",
    "multispeaker",
    "DictionaryData",
    "PronunciationDictionaryMixin",
    "MultispeakerDictionaryMixin",
    "DictionaryMixin",
]
