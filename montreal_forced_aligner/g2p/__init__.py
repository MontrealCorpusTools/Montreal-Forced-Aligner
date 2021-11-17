"""
Grapheme to phoneme (G2P)
=========================


"""

from .generator import PyniniDictionaryGenerator
from .trainer import PyniniTrainer

__all__ = ["generator", "trainer", "PyniniTrainer", "PyniniDictionaryGenerator"]

PyniniTrainer.__module__ = "montreal_forced_aligner.g2p"
PyniniDictionaryGenerator.__module__ = "montreal_forced_aligner.g2p"
