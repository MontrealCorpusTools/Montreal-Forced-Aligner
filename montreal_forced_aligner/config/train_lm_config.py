import os
import yaml
from .base_config import BaseConfig


class TrainLMConfig(BaseConfig):
    def __init__(self):
        self.order = 3
        self.method = 'kneser_ney'
        self.prune = False
        self.count_threshold = 1
        self.prune_thresh_small = 0.0000003
        self.prune_thresh_medium = 0.0000001


def train_lm_yaml_to_config(path):
    with open(path, 'r', encoding='utf8') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        config = TrainLMConfig()
        config.update(data)
    return config


def load_basic_train_lm():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_config = train_lm_yaml_to_config(os.path.join(base_dir, 'basic_train_lm.yaml'))
    return training_config
