"""
Corpora
=======


"""
from __future__ import annotations

from .base import Corpus  # noqa
from .classes import File, Speaker, Utterance

__all__ = ["Corpus", "Speaker", "Utterance", "File", "base", "helper", "classes"]

Corpus.__module__ = "montreal_forced_aligner.corpus"
Speaker.__module__ = "montreal_forced_aligner.corpus"
Utterance.__module__ = "montreal_forced_aligner.corpus"
File.__module__ = "montreal_forced_aligner.corpus"
