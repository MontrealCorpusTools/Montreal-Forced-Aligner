"""Class definitions for comparing alignments"""
from __future__ import annotations

import logging
import typing
from pathlib import Path

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import TopLevelMfaWorker
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusMixin
from montreal_forced_aligner.corpus.text_corpus import TextCorpusMixin

logger = logging.getLogger("mfa")


class AlignmentComparisonMixin(TopLevelMfaWorker):
    def __init__(self, test_directory: typing.Union[Path, str], **kwargs):
        super().__init__(**kwargs)
        self.test_directory = test_directory

    @property
    def identifier(self) -> str:
        """Identifier for the corpus"""
        return self.data_source_identifier

    @property
    def output_directory(self) -> Path:
        """Root temporary directory to store all corpus and dictionary files"""
        return config.TEMPORARY_DIRECTORY.joinpath(self.identifier)

    @property
    def working_directory(self) -> Path:
        """Working directory"""
        return self.corpus_output_directory


class AlignmentComparer(TextCorpusMixin, AlignmentComparisonMixin):
    """
    Standalone class for comparing two sets of alignments
    """

    def load_corpus(self) -> None:
        """
        Load the corpus
        """
        self.initialize_database()
        self._create_dummy_dictionary()
        self._load_corpus()
        self.initialize_jobs()
        self.create_corpus_split()
        self.load_test_alignments(self.test_directory)


class AlignmentAudioComparer(AcousticCorpusMixin, AlignmentComparisonMixin):
    """
    Standalone class for comparing two sets of alignments
    """

    def load_corpus(self) -> None:
        """
        Load the corpus
        """
        self.initialize_database()
        self._create_dummy_dictionary()
        self._load_corpus()
        self.initialize_jobs()
        self.create_corpus_split()
        self.load_test_alignments(self.test_directory)
