import yaml
import os
from .base_config import BaseConfig, ConfigError


class DiarizationConfig(BaseConfig):
    def __init__(self):
        pass


def diarization_yaml_to_config(path):
    with open(path, 'r', encoding='utf8') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        diarization_config = DiarizationConfig()
        diarization_config.update(data)
        return diarization_config


def load_basic_diarization():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    diarization_config = align_yaml_to_config(os.path.join(base_dir, 'basic_diarization.yaml'))
    return diarization_config
