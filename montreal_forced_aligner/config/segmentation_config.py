import yaml
import os
from .base_config import BaseConfig, ConfigError
from ..features.config import FeatureConfig


class SegmentationConfig(BaseConfig):
    def __init__(self, feature_config):
        self.use_mp = True
        self.energy_threshold = 5.5
        self.energy_mean_scale = 0.5
        self.max_segment_length = 30
        self.min_pause_duration = 0.05
        self.snap_boundary_threshold = 0.15
        self.feature_config = feature_config
        self.feature_config.use_energy = True

    def update(self, data):
        for k, v in data.items():
            if k == 'use_mp':
                self.feature_config.use_mp = v
            if not hasattr(self, k):
                raise ConfigError('No field found for key {}'.format(k))
            setattr(self, k, v)


def segmentation_yaml_to_config(path):
    with open(path, 'r', encoding='utf8') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        global_params = {}
        feature_config = FeatureConfig()
        for k, v in data.items():
            if k == 'features':
                feature_config.update(v)
            else:
                global_params[k] = v
        segmentation_config = SegmentationConfig(feature_config)
        segmentation_config.update(global_params)
        return segmentation_config


def load_basic_segmentation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    segmentation_config = segmentation_yaml_to_config(os.path.join(base_dir, 'basic_segmentation.yaml'))
    return segmentation_config