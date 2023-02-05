"""
Pronunciation dictionaries
==========================

"""

from montreal_forced_aligner.dictionary.mixins import DictionaryMixin, SanitizeFunction
from montreal_forced_aligner.dictionary.multispeaker import (
    MultispeakerDictionary,
    MultispeakerDictionaryMixin,
)

__all__ = [
    "multispeaker",
    "mixins",
    "DictionaryMixin",
    "SanitizeFunction",
    "MultispeakerDictionary",
    "MultispeakerDictionaryMixin",
]
