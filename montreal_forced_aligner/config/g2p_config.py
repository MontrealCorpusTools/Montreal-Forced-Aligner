import os
import yaml
from .base_config import BaseConfig, ConfigError, DEFAULT_PUNCTUATION, DEFAULT_CLITIC_MARKERS, DEFAULT_COMPOUND_MARKERS


class G2PConfig(BaseConfig):
    def __init__(self):
        self.punctuation = DEFAULT_PUNCTUATION
        self.clitic_markers = DEFAULT_CLITIC_MARKERS
        self.compound_markers = DEFAULT_COMPOUND_MARKERS
        self.num_pronunciations = 1
        self.use_mp = True

    def update(self, data):
        for k, v in data.items():
            if k in ['punctuation', 'clitic_markers', 'compound_markers']:
                if not v:
                    continue
                if '-' in v:
                    v = '-' + v.replace('-', '')
                if ']' in v and r'\]' not in v:
                    v = v.replace(']', r'\]')
            elif not hasattr(self, k):
                raise ConfigError('No field found for key {}'.format(k))
            setattr(self, k, v)


def g2p_yaml_to_config(path):
    with open(path, 'r', encoding='utf8') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        global_params = {}
        for k, v in data.items():
            global_params[k] = v
        g2p_config = G2PConfig()
        g2p_config.update(global_params)
        return g2p_config


def load_basic_g2p_config():
    return G2PConfig()