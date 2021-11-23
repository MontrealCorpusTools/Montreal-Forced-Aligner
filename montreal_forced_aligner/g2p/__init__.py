"""
Grapheme to phoneme (G2P)
=========================


"""

from .generator import (
    OrthographicCorpusGenerator,
    OrthographicWordListGenerator,
    PyniniCorpusGenerator,
    PyniniWordListGenerator,
)
from .trainer import PyniniTrainer

__all__ = [
    "generator",
    "trainer",
    "PyniniTrainer",
    "PyniniCorpusGenerator",
    "PyniniWordListGenerator",
    "OrthographicCorpusGenerator",
    "OrthographicWordListGenerator",
]
