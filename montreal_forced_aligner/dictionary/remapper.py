"""Classes for remapping a dictionary from one phone set to another"""
from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path

import yaml

from montreal_forced_aligner.abc import TopLevelMfaWorker
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionaryMixin
from montreal_forced_aligner.exceptions import RemapAcousticMismatchError
from montreal_forced_aligner.helper import format_correction, format_probability, mfa_open
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
        for key, values in self.phone_remapping.items():
            if not isinstance(values, list):
                self.phone_remapping[key] = [values]

    def validate_mapping(self):
        unknown_phones = set()
        for key, values in self.phone_remapping.items():
            for value in values:
                for p in value.split():
                    if p not in self.acoustic_model.meta["phones"]:
                        unknown_phones.add(p)
        if unknown_phones:
            raise RemapAcousticMismatchError(unknown_phones, self.phone_mapping_path)

    def remap(self, output_dictionary_path: Path):
        self.setup()

        new_dictionary = {}
        skip_count = 0
        extra_prob_keys = [
            "silence_after_probability",
            "silence_before_correction",
            "non_silence_before_correction",
        ]
        for data in self.words_for_export(probability=True):
            phones = data["pronunciation"]
            w = data["word"]
            pron = phones.split()
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
                if not isinstance(new_p, list):
                    new_p = [new_p]
                new_pron.append(new_p)
            if skip:
                logger.debug(f"Skipping {w}: {' '.join(pron)}")
                skip_count += 1
                continue
            if w not in new_dictionary:
                new_dictionary[w] = {}
            pron_combinations = list(itertools.product(*new_pron))
            for new_pron in pron_combinations:
                pron_string = " ".join(new_pron)
                if pron_string not in new_dictionary[w]:
                    new_dictionary[w][pron_string] = {
                        "count": 1,
                        "probability": data["probability"],
                        "silence_after_probability": data["silence_after_probability"],
                        "silence_before_correction": data["silence_before_correction"],
                        "non_silence_before_correction": data["non_silence_before_correction"],
                    }
                else:
                    new_dictionary[w][pron_string]["count"] += 1
                    if data["probability"] is not None:
                        if new_dictionary[w][pron_string]["probability"] is None:
                            new_dictionary[w][pron_string]["probability"] = data["probability"]
                        else:
                            new_dictionary[w][pron_string]["probability"] = max(
                                data["probability"], new_dictionary[w][pron_string]["probability"]
                            )
                    for k in extra_prob_keys:
                        if data[k] is not None:
                            if new_dictionary[w][pron_string][k] is None:
                                new_dictionary[w][pron_string][k] = data[k]
                            else:
                                new_dictionary[w][pron_string][k] += data[k]

        logger.info(f"Skipped {skip_count} pronunciations for having unmapped phones")
        with mfa_open(output_dictionary_path, "w") as f:
            for w, prons in sorted(new_dictionary.items(), key=lambda x: x[0]):
                for pron, data in sorted(prons.items(), key=lambda x: x[0]):
                    probability_string = ""
                    if data["probability"] is not None:
                        probability_string = f"\t{format_probability(data['probability'])}"

                        extra_probs = [
                            data["silence_after_probability"],
                            data["silence_before_correction"],
                            data["non_silence_before_correction"],
                        ]
                        if all(x is None for x in extra_probs):
                            continue
                        for i, x in enumerate(extra_probs):
                            if x is None:
                                continue
                            mean_value = x / data["count"]
                            if i == 0:
                                mean_value = format_correction(mean_value)
                            else:
                                mean_value = format_correction(mean_value, positive_only=False)
                            probability_string += f"\t{mean_value}"
                    f.write(f"{w}{probability_string}\t{pron}\n")
        logger.info(f"Wrote remapped dictionary to {output_dictionary_path}")
