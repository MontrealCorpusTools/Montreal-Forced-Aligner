import yaml
import os
from .base_config import BaseConfig, ConfigError


class DiarizationConfig(BaseConfig):
    def __init__(self):
        self.use_mp = True
        self.pca_dimension = -1
        self.target_energy = 0.1
        self.cluster_threshold = 0.5
        self.max_speaker_fraction = 1.0
        self.first_pass_max_utterances = 32767
        self.rttm_channel = 0
        self.read_costs = False


def diarization_yaml_to_config(path):
    with open(path, 'r', encoding='utf8') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        diarization_config = DiarizationConfig()
        if data:
            diarization_config.update(data)
        return diarization_config


def load_basic_diarization():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    diarization_config = diarization_yaml_to_config(os.path.join(base_dir, 'basic_diarization.yaml'))
    return diarization_config
