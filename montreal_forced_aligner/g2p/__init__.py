"""
Grapheme to phoneme (G2P)
=========================


"""

from montreal_forced_aligner.g2p.generator import (
    OrthographicCorpusGenerator,
    OrthographicWordListGenerator,
    PyniniCorpusGenerator,
    PyniniWordListGenerator,
)
from montreal_forced_aligner.g2p.trainer import PyniniTrainer

__all__ = [
    "generator",
    "trainer",
    "PyniniTrainer",
    "PyniniCorpusGenerator",
    "PyniniWordListGenerator",
    "OrthographicCorpusGenerator",
    "OrthographicWordListGenerator",
]
