"""Classes for remapping a dictionary from one phone set to another"""
from __future__ import annotations

import collections
import logging
import os
from pathlib import Path

import yaml

from montreal_forced_aligner.abc import TopLevelMfaWorker
from montreal_forced_aligner.db import Pronunciation, Word
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionaryMixin
from montreal_forced_aligner.exceptions import RemapAcousticMismatchError
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.models import AcousticModel

logger = logging.getLogger("mfa")

__all__ = ["DictionaryRemapper"]


class DictionaryRemapper(MultispeakerDictionaryMixin, TopLevelMfaWorker):
    def __init__(
        self,
        acoustic_model_path: Path,
        phone_mapping_path: Path,
        **kwargs,
    ):
        self._data_source = kwargs["dictionary_path"].stem
        super().__init__(**kwargs)
        self.acoustic_model = AcousticModel(acoustic_model_path)
        self.phone_mapping_path = phone_mapping_path
        self.phone_remapping = {}

    @property
    def data_source_identifier(self) -> str:
        """Dictionary name"""
        return self._data_source

    @property
    def data_directory(self) -> Path:
        """Data directory for trainer"""
        return self.working_directory

    def setup(self) -> None:
        """Setup for dictionary remapping"""
        super().setup()
        self.load_mapping()
        self.validate_mapping()
        if self.initialized:
            return
        self.dictionary_setup()
        os.makedirs(self.phones_dir, exist_ok=True)
        self.initialized = True

    def load_mapping(self):
        with mfa_open(self.phone_mapping_path, "r") as f:
            self.phone_remapping = yaml.load(f, Loader=yaml.Loader)

    def validate_mapping(self):
        unknown_phones = set()
        for key, value in self.phone_remapping.items():
            for p in value.split():
                if p not in self.acoustic_model.meta["phones"]:
                    unknown_phones.add(p)
        if unknown_phones:
            raise RemapAcousticMismatchError(unknown_phones, self.phone_mapping_path)

    def remap(self, output_dictionary_path: Path):
        self.setup()

        new_dictionary = collections.defaultdict(set)
        with self.session() as session:
            pronunciations = session.query(Word.word, Pronunciation.pronunciation).join(
                Pronunciation.word
            )
            skip_count = 0
            for w, pron in pronunciations:
                pron = pron.split()
                skip = False
                new_pron = []
                for p in pron:
                    if p not in self.phone_remapping:
                        if p in self.acoustic_model.meta["phones"]:
                            new_p = p
                        else:
                            skip = True
                    else:
                        new_p = self.phone_remapping[p]
                    if skip:
                        break
                    new_pron.append(new_p)
                if skip:
                    logger.debug(f"Skipping {w}: {' '.join(pron)}")
                    skip_count += 1
                    continue
                new_dictionary[w].add(" ".join(new_pron))
            logger.info(f"Skipped {skip_count} pronunciations for having unmapped phones")
            with mfa_open(output_dictionary_path, "w") as f:
                for w, prons in sorted(new_dictionary.items(), key=lambda x: x[0]):
                    for pron in sorted(prons):
                        f.write(f"{w}\t{pron}\n")
            logger.info(f"Wrote remapped dictionary to {output_dictionary_path}")
