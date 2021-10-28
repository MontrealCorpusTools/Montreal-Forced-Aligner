from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any
if TYPE_CHECKING:
    from ..config import ConfigDict
import os
import yaml
from .base_config import BaseConfig, ConfigError, DEFAULT_PUNCTUATION, DEFAULT_CLITIC_MARKERS, \
    DEFAULT_COMPOUND_MARKERS, DEFAULT_DIGRAPHS, DEFAULT_STRIP_DIACRITICS
from ..features.config import FeatureConfig


class TranscribeConfig(BaseConfig):
    def __init__(self, feature_config: FeatureConfig):
        self.transition_scale = 1.0
        self.acoustic_scale = 0.083333
        self.self_loop_scale = 0.1
        self.feature_config = feature_config
        self.silence_weight = 0.01
        self.beam = 10
        self.max_active = 7000
        self.fmllr = True
        self.fmllr_update_type = 'full'
        self.lattice_beam = 6
        self.first_beam = None
        self.first_max_active = 2000
        self.max_fmllr_jobs = 12
        self.language_model_weight = 10
        self.word_insertion_penalty = 0.5
        self.data_directory = None # Gets set later
        self.use_mp = True
        self.use_fmllr_mp = False
        self.multilingual_ipa = False
        self.no_speakers = False
        self.punctuation = DEFAULT_PUNCTUATION
        self.clitic_markers = DEFAULT_CLITIC_MARKERS
        self.compound_markers = DEFAULT_COMPOUND_MARKERS
        self.strip_diacritics = DEFAULT_STRIP_DIACRITICS
        self.digraphs = DEFAULT_DIGRAPHS
        self.overwrite = False

    def params(self) -> ConfigDict:
        return {
            'transition_scale': self.transition_scale,
            'acoustic_scale': self.acoustic_scale,
            'self_loop_scale': self.self_loop_scale,
            'silence_weight': self.silence_weight,
            'beam': self.beam,
            'max_active': self.max_active,
            'fmllr': self.fmllr,
            'fmllr_update_type': self.fmllr_update_type,
            'lattice_beam': self.lattice_beam,
            'first_beam': self.first_beam,
            'first_max_active': self.first_max_active,
            'max_fmllr_jobs': self.max_fmllr_jobs,
            'language_model_weight': self.language_model_weight,
            'word_insertion_penalty': self.word_insertion_penalty,
            'use_mp': self.use_mp,
                }

    @property
    def decode_options(self) -> ConfigDict:
        return {
            'fmllr': self.fmllr,
            'no_speakers': self.no_speakers,
            'first_beam': self.first_beam,
            'beam': self.beam,
            'first_max_active': self.first_max_active,
            'max_active': self.max_active,
            'lattice_beam': self.lattice_beam,
            'acoustic_scale': self.acoustic_scale,
        }

    @property
    def score_options(self) -> ConfigDict:
        return {
            'language_model_weight': self.language_model_weight,
            'word_insertion_penalty': self.word_insertion_penalty,
        }

    @property
    def fmllr_options(self) -> ConfigDict:
        return {
            'fmllr_update_type': self.fmllr_update_type,
            'acoustic_scale': self.acoustic_scale,
            'silence_weight': self.silence_weight,
            'lattice_beam': self.lattice_beam,
        }

    @property
    def lm_rescore_options(self) -> ConfigDict:
        return {
            'acoustic_scale': self.acoustic_scale,
        }

    @property
    def feature_file_base_name(self) -> str:
        return self.feature_config.feature_id

    def update(self, data: dict) -> None:
        for k, v in data.items():
            if k == 'use_mp':
                self.feature_config.use_mp = v
            if not hasattr(self, k):
                raise ConfigError('No field found for key {}'.format(k))
            setattr(self, k, v)


def transcribe_yaml_to_config(path: str) -> TranscribeConfig:
    with open(path, 'r', encoding='utf8') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        global_params = {}
        feature_config = FeatureConfig()
        for k, v in data.items():
            if k == 'features':
                feature_config.update(v)
            else:
                global_params[k] = v
        config = TranscribeConfig(feature_config)
        config.update(global_params)
        return config


def load_basic_transcribe() -> TranscribeConfig:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = transcribe_yaml_to_config(os.path.join(base_dir, 'basic_transcribe.yaml'))
    return config