import os
import yaml
from .base_config import BaseConfig, ConfigError
from ..features.config import FeatureConfig


class AlignConfig(BaseConfig):
    def __init__(self, feature_config):
        self.transition_scale = 1.0
        self.acoustic_scale = 0.1
        self.self_loop_scale = 0.1
        self.disable_sat = False
        self.feature_config = feature_config
        self.boost_silence = 1.0
        self.beam = 10
        self.retry_beam = 40
        self.data_directory = None # Gets set later
        self.fmllr_update_type = 'full'
        self.use_mp = True

    @property
    def feature_file_base_name(self):
        return self.feature_config.feature_id

    @property
    def align_options(self):
        return {'transition_scale': self.transition_scale,
                'acoustic_scale': self.acoustic_scale,
                'self_loop_scale': self.self_loop_scale,
                'beam': self.beam,
                'retry_beam': self.retry_beam,
                }

    def update(self, data):
        for k, v in data.items():
            if k == 'use_mp':
                self.feature_config.use_mp = v
            if not hasattr(self, k):
                raise ConfigError('No field found for key {}'.format(k))
            setattr(self, k, v)

    def update_from_args(self, args):
        super(AlignConfig, self).update_from_args(args)
        if self.retry_beam <= self.beam:
            self.retry_beam = self.beam * 4


def align_yaml_to_config(path):
    with open(path, 'r', encoding='utf8') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        global_params = {}
        feature_config = FeatureConfig()
        for k, v in data.items():
            if k == 'features':
                feature_config.update(v)
            else:
                global_params[k] = v
        align_config = AlignConfig(feature_config)
        align_config.update(global_params)
        if align_config.beam >= align_config.retry_beam:
            raise ConfigError('Retry beam must be greater than beam.')
        return align_config


def load_basic_align():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    align_config = align_yaml_to_config(os.path.join(base_dir, 'basic_align.yaml'))
    return align_config