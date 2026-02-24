"""Class definitions for comparing alignments"""
from __future__ import annotations

import logging
import typing
from pathlib import Path

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import TopLevelMfaWorker
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusMixin
from montreal_forced_aligner.corpus.text_corpus import TextCorpusMixin
from montreal_forced_aligner.data import PhoneType
from montreal_forced_aligner.db import Phone, PhoneMapping
from montreal_forced_aligner.exceptions import MFAError
from montreal_forced_aligner.helper import load_evaluation_mapping

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

    def load_mapping(self, custom_mapping_path: typing.Union[Path, str], strict=False):
        mapping = load_evaluation_mapping(custom_mapping_path)
        with self.session() as session:
            reference_phones = {
                phone: p_id
                for phone, p_id in session.query(Phone.phone, Phone.id).filter(
                    Phone.phone_type == PhoneType.non_silence
                )
            }
            test_phones = {
                phone: p_id
                for phone, p_id in session.query(Phone.phone, Phone.id).filter(
                    Phone.phone_type == PhoneType.extra
                )
            }
            phone_mappings = []
            found_reference_phones = set()
            found_test_phones = set()
            for aligned_phones, ref_phones in mapping.items():
                if isinstance(ref_phones, str):
                    ref_phones = [ref_phones]
                for rp in ref_phones:
                    phone_mappings.append(
                        {
                            "model_phone_string": aligned_phones,
                            "reference_phone_string": rp,
                        }
                    )
                    found_reference_phones.update(rp.split())
                found_test_phones.update(aligned_phones.split())
            session.bulk_insert_mappings(PhoneMapping, phone_mappings)
            session.commit()
            unreferenced_reference_phones = sorted(
                set(reference_phones.keys()) - found_reference_phones
            )
            unreferenced_test_phones = sorted(set(test_phones.keys()) - found_test_phones)
            if unreferenced_test_phones:
                logger.warning(
                    f"Phones not referenced in mapping file: {', '.join(unreferenced_test_phones)}"
                )
            if unreferenced_reference_phones:
                logger.warning(
                    f"Reference phones not referenced in mapping file: {', '.join(unreferenced_reference_phones)}"
                )
            if strict and (unreferenced_reference_phones or unreferenced_test_phones):
                raise MFAError(
                    "The mapping file was not fully specified, see warning messages above."
                )


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

    def load_mapping(self, custom_mapping_path: typing.Union[Path, str], strict=False):
        mapping = load_evaluation_mapping(custom_mapping_path)
        with self.session() as session:
            reference_phones = {
                phone: p_id
                for phone, p_id in session.query(Phone.phone, Phone.id).filter(
                    Phone.phone_type == PhoneType.non_silence
                )
            }
            test_phones = {
                phone: p_id
                for phone, p_id in session.query(Phone.phone, Phone.id).filter(
                    Phone.phone_type == PhoneType.extra
                )
            }
            phone_mappings = []
            found_reference_phones = set()
            found_test_phones = set()
            for aligned_phones, ref_phones in mapping.items():
                if isinstance(ref_phones, str):
                    ref_phones = [ref_phones]
                for rp in ref_phones:
                    phone_mappings.append(
                        {
                            "model_phone_string": aligned_phones,
                            "reference_phone_string": rp,
                        }
                    )
                    found_reference_phones.update(rp.split())
                found_test_phones.update(aligned_phones.split())
            session.bulk_insert_mappings(PhoneMapping, phone_mappings)
            session.commit()
            unreferenced_reference_phones = sorted(
                set(reference_phones.keys()) - found_reference_phones
            )
            unreferenced_test_phones = sorted(set(test_phones.keys()) - found_test_phones)
            if unreferenced_test_phones:
                logger.warning(
                    f"Phones not referenced in mapping file: {', '.join(unreferenced_test_phones)}"
                )
            if unreferenced_reference_phones:
                logger.warning(
                    f"Reference phones not referenced in mapping file: {', '.join(unreferenced_reference_phones)}"
                )
            if strict and (unreferenced_reference_phones or unreferenced_test_phones):
                raise MFAError(
                    "The mapping file was not fully specified, see warning messages above."
                )
