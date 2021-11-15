"""
Training acoustic models
========================


"""
from .base import BaseTrainer  # noqa
from .ivector_extractor import IvectorExtractorTrainer  # noqa
from .lda import LdaTrainer  # noqa
from .monophone import MonophoneTrainer  # noqa
from .sat import SatTrainer  # noqa
from .triphone import TriphoneTrainer  # noqa

__all__ = [
    "BaseTrainer",
    "IvectorExtractorTrainer",
    "LdaTrainer",
    "MonophoneTrainer",
    "SatTrainer",
    "TriphoneTrainer",
    "base",
    "ivector_extractor",
    "lda",
    "monophone",
    "sat",
    "triphone",
]

BaseTrainer.__module__ = "montreal_forced_aligner.trainers"
IvectorExtractorTrainer.__module__ = "montreal_forced_aligner.trainers"
LdaTrainer.__module__ = "montreal_forced_aligner.trainers"
MonophoneTrainer.__module__ = "montreal_forced_aligner.trainers"
SatTrainer.__module__ = "montreal_forced_aligner.trainers"
TriphoneTrainer.__module__ = "montreal_forced_aligner.trainers"
