
import os
import yaml
from .base_config import BaseConfig, ConfigError
from ..features.config import FeatureConfig
from collections import Counter

from ..trainers import MonophoneTrainer, TriphoneTrainer, LdaTrainer, SatTrainer, IvectorExtractorTrainer

from .align_config import AlignConfig


class TrainingConfig(BaseConfig):
    def __init__(self, training_configs):
        self.training_configs = training_configs
        counts = Counter([x.train_type for x in self.training_configs])
        self.training_identifiers = []
        curs = {x.train_type: 1 for x in self.training_configs}
        for t in training_configs:
            i = t.train_type
            if counts[t.train_type] != 1:
                i += str(curs[t.train_type])
                curs[t.train_type] += 1
            self.training_identifiers.append(i)

    def keys(self):
        return self.training_identifiers

    def values(self):
        return self.training_configs

    def items(self):
        return zip(self.training_identifiers, self.training_configs)

    def __getitem__(self, item):
        if item not in self.training_identifiers:
            raise KeyError('{} not a valid training identifier'.format(item))
        return self.training_configs[self.training_identifiers.index(item)]

    @property
    def uses_lda(self):
        for k in self.keys():
            if k.startswith('lda'):
                return True
        return False

    @property
    def uses_sat(self):
        for k in self.keys():
            if k.startswith('sat'):
                return True
        return False


def train_yaml_to_config(path):
    with open(path, 'r', encoding='utf8') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        global_params = {}
        training = []
        training_params = []
        global_feature_params = {}
        for k, v in data.items():
            if k == 'training':
                for t in v:
                    for k2, v2 in t.items():
                        feature_config = FeatureConfig()
                        if k2 == 'monophone':
                            training.append(MonophoneTrainer(feature_config))
                        elif k2 == 'triphone':
                            training.append(TriphoneTrainer(feature_config))
                        elif k2 == 'lda':
                            training.append(LdaTrainer(feature_config))
                        elif k2 == 'sat':
                            training.append(SatTrainer(feature_config))
                        elif k2 == 'ivector':
                            training.append(IvectorExtractorTrainer(feature_config))
                        training_params.append(v2)
            elif k == 'features':
                global_feature_params.update(v)
            else:
                global_params[k] = v
        feature_config = FeatureConfig()
        feature_config.update(global_feature_params)
        align_config = AlignConfig(feature_config)
        align_config.update(global_params)
        training_config = None
        if training:
            for i, t in enumerate(training):
                if i == 0 and t.train_type not in ['mono', 'ivector']:
                    raise ConfigError('The first round of training must be monophone.')
                t.update(global_params)
                t.update(training_params[i])
                t.feature_config.update(global_feature_params)
            training_config = TrainingConfig(training)
        align_config.feature_config.lda = training_config.uses_lda
        if training_config.uses_lda:
            align_config.feature_config.set_features_to_use_lda()
        align_config.feature_config.fmllr = training_config.uses_sat
        if align_config.beam >= align_config.retry_beam:
            raise ConfigError('Retry beam must be greater than beam.')
        return training_config, align_config


def load_basic_train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_config, align_config = train_yaml_to_config(os.path.join(base_dir, 'basic_train.yaml'))
    return training_config, align_config


def load_basic_train_ivector():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_config, align_config = train_yaml_to_config(os.path.join(base_dir, 'basic_train_ivector.yaml'))
    return training_config, align_config


def load_test_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_config, align_config = train_yaml_to_config(os.path.join(base_dir, 'test_config.yaml'))
    return training_config, align_config