import os
from collections import Counter
import yaml

from ..exceptions import ConfigError

TEMP_DIR = os.path.expanduser('~/Documents/MFA')

from ..trainers import MonophoneTrainer, TriphoneTrainer, LdaTrainer, SatTrainer


def make_safe(value):
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


class BaseConfig(object):
    def update(self, data):
        for k, v in data.items():
            if not hasattr(self, k):
                raise ConfigError('No field found for key {}'.format(k))
            setattr(self, k, v)


class TrainingConfig(BaseConfig):
    def __init__(self, training_configs, feature_config):
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
        self.feature_config = feature_config

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


class AlignConfig(BaseConfig):
    def __init__(self, feature_config):
        self.scale_opts = ['--transition-scale=1.0',
                           '--acoustic-scale=0.1',
                           '--self-loop-scale=0.1']
        self.feature_config = feature_config
        self.boost_silence = 1.0
        self.beam = 10
        self.retry_beam = 40


class FeatureConfig(BaseConfig):
    '''
    Class to store configuration information about MFCC generation

    The ``config_dict`` currently stores one key ``'use-energy'`` which
    defaults to False

    Parameters
    ----------
    output_directory : str
        Path to directory to save configuration files for Kaldi
    kwargs : dict, optional
        If specified, updates ``config_dict`` with this dictionary

    Attributes
    ----------
    config_dict : dict
        Dictionary of configuration parameters
    '''

    def __init__(self):
        self.type = 'mfcc'
        self.use_energy = False
        self.frame_shift = 10
        self.pitch = False

    def write(self, output_directory, job, extra_params=None):
        '''
        Write configuration dictionary to a file for use in Kaldi binaries
        '''
        f = '{}.{}.conf'.format(self.type, job)
        path = os.path.join(output_directory, 'config')
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f)
        with open(path, 'w', encoding='utf8') as f:
            f.write('--{}={}\n'.format('use-energy', make_safe(self.use_energy)))
            f.write('--{}={}\n'.format('frame-shift', make_safe(self.frame_shift)))
            if extra_params is not None:
                for k, v in extra_params.items():
                    f.write('--{}={}\n'.format(k, make_safe(v)))
        return path

    def params(self):
        return {'type': self.type,
                'use_energy': self.use_energy,
                'frame_shift': self.frame_shift,
                'pitch': self.pitch}


def train_yaml_to_config(path):
    with open(path, 'r', encoding='utf8') as f:
        data = yaml.load(f)
        global_params = {}
        training = []
        training_params = []
        feature_config = FeatureConfig()
        for k, v in data.items():
            if k == 'training':
                for t in v:
                    for k2, v2 in t.items():
                        if k2 == 'monophone':
                            training.append(MonophoneTrainer())
                        elif k2 == 'triphone':
                            training.append(TriphoneTrainer())
                        elif k2 == 'lda':
                            training.append(LdaTrainer())
                        elif k2 == 'sat':
                            training.append(SatTrainer())
                        training_params.append(v2)
            elif k == 'features':
                feature_config.update(v)
            else:
                global_params[k] = v
        align_config = AlignConfig(feature_config)
        align_config.update(global_params)
        training_config = None
        if training:
            for i, t in enumerate(training):
                if i == 0 and t.train_type != 'mono':
                    raise ConfigError('The first round of training must be monophone.')
                t.update(global_params)
                t.update(training_params[i])
            training_config = TrainingConfig(training, feature_config)
        return training_config, align_config


def align_yaml_to_config(path):
    with open(path, 'r', encoding='utf8') as f:
        data = yaml.load(f)
        global_params = {}
        feature_config = FeatureConfig()
        for k, v in data.items():
            if k == 'features':
                feature_config.update(v)
            else:
                global_params[k] = v
        align_config = AlignConfig(feature_config)
        align_config.update(global_params)
        return align_config


def load_basic_align():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    align_config = align_yaml_to_config(os.path.join(base_dir, 'basic_align.yaml'))
    return align_config


def load_basic_train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_config, align_config = train_yaml_to_config(os.path.join(base_dir, 'basic_train.yaml'))
    return training_config, align_config
