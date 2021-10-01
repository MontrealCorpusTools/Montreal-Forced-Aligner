
import os
import yaml
from .base_config import BaseConfig, ConfigError, DEFAULT_PUNCTUATION, DEFAULT_CLITIC_MARKERS, DEFAULT_COMPOUND_MARKERS, PARSING_KEYS
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

        self.punctuation = DEFAULT_PUNCTUATION
        self.clitic_markers = DEFAULT_CLITIC_MARKERS
        self.compound_markers = DEFAULT_COMPOUND_MARKERS

    def update_from_align(self, align_config):
        self.training_configs[-1].overwrite = align_config.overwrite
        self.training_configs[-1].cleanup_textgrids = align_config.cleanup_textgrids

    def update(self, data):
        for k, v in data.items():
            if k in PARSING_KEYS:
                if not v:
                    continue
                if '-' in v:
                    v = '-' + v.replace('-', '')
                if ']' in v and r'\]' not in v:
                    v = v.replace(']', r'\]')
            if not hasattr(self, k):
                raise ConfigError('No field found for key {}'.format(k))
            setattr(self, k, v)
        for trainer in self.values():
            trainer.update(data)

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


def train_yaml_to_config(path, require_mono=True):
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
                if i == 0 and require_mono and t.train_type not in ['mono', 'ivector']:
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


def load_sat_adapt():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_config, align_config = train_yaml_to_config(os.path.join(base_dir, 'adapt_sat.yaml'), require_mono=False)
    training_config.training_configs[0].fmllr_iterations = range(0, training_config.training_configs[0].num_iterations)
    training_config.training_configs[0].realignment_iterations = range(0, training_config.training_configs[0].num_iterations)
    return training_config, align_config


def load_no_sat_adapt():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_config, align_config = train_yaml_to_config(os.path.join(base_dir, 'adapt_nosat.yaml'), require_mono=False)
    training_config.training_configs[0].realignment_iterations = range(0, training_config.training_configs[0].num_iterations)
    return training_config, align_config


def load_basic_train_ivector():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_config, align_config = train_yaml_to_config(os.path.join(base_dir, 'basic_train_ivector.yaml'))
    return training_config, align_config


def load_test_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_config, align_config = train_yaml_to_config(os.path.join(base_dir, 'test_config.yaml'))
    return training_config, align_config