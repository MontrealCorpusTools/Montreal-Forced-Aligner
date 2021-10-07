import os
import yaml
from .base_config import BaseConfig, save_config, ConfigError
from .align_config import AlignConfig, load_basic_align, align_yaml_to_config, FeatureConfig
from .speaker_classification_config import SpeakerClassificationConfig, classification_yaml_to_config, load_basic_classification
from .train_config import TrainingConfig, load_basic_train, load_basic_train_ivector, load_test_config, \
    train_yaml_to_config
from .train_lm_config import TrainLMConfig, load_basic_train_lm, train_lm_yaml_to_config
from .transcribe_config import TranscribeConfig, load_basic_transcribe, transcribe_yaml_to_config
from .segmentation_config import SegmentationConfig, segmentation_yaml_to_config, load_basic_segmentation
from .command_config import load_command_configuration

TEMP_DIR = os.path.expanduser('~/Documents/MFA')

def generate_config_path():
    return os.path.join(TEMP_DIR, 'global_config.yaml')

def generate_command_history_path():
    return os.path.join(TEMP_DIR, 'command_history.yaml')

def load_command_history():
    path = generate_command_history_path()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf8') as f:
            history = yaml.safe_load(f)
    else:
        history = []
    if not history:
        history = []
    return history


def update_command_history(command_data):
    try:
        if command_data['command'].split(' ')[1] == 'history':
            return
    except Exception:
        return
    history = load_command_history()
    path = generate_command_history_path()
    history.append(command_data)
    history = history[-50:]
    with open(path, 'w', encoding='utf8') as f:
        yaml.safe_dump(history, f)

def update_global_config(args):
    global_configuration_file = generate_config_path()
    default_config = {
        'clean': False,
        'verbose': False,
        'debug': False,
        'overwrite': False,
        'cleanup_textgrids': True,
        'num_jobs': 3,
        'use_mp': True,
        'temp_directory': TEMP_DIR
    }
    if os.path.exists(global_configuration_file):
        with open(global_configuration_file, 'r', encoding='utf8') as f:
            data = yaml.safe_load(f)
            default_config.update(data)
    if args.always_clean:
        default_config['clean'] = True
    if args.never_clean:
        default_config['clean'] = False
    if args.always_verbose:
        default_config['verbose'] = True
    if args.never_verbose:
        default_config['verbose'] = False
    if args.always_debug:
        default_config['debug'] = True
    if args.never_debug:
        default_config['debug'] = False
    if args.always_overwrite:
        default_config['overwrite'] = True
    if args.never_overwrite:
        default_config['overwrite'] = False
    if args.disable_mp:
        default_config['use_mp'] = False
    if args.enable_mp:
        default_config['use_mp'] = True
    if args.disable_textgrid_cleanup:
        default_config['cleanup_textgrids'] = False
    if args.enable_textgrid_cleanup:
        default_config['cleanup_textgrids'] = True
    if args.num_jobs and args.num_jobs > 0:
        default_config['num_jobs'] = args.num_jobs
    if args.temp_directory:
        default_config['temp_directory'] = args.temp_directory
    with open(global_configuration_file, 'w', encoding='utf8') as f:
        yaml.dump(default_config, f)


def load_global_config():
    global_configuration_file = generate_config_path()
    default_config = {
        'clean': False,
        'verbose': False,
        'debug': False,
        'overwrite': False,
        'cleanup_textgrids': True,
        'num_jobs': 3,
        'use_mp': True,
        'temp_directory': TEMP_DIR
    }
    if os.path.exists(global_configuration_file):
        with open(global_configuration_file, 'r', encoding='utf8') as f:
            data = yaml.safe_load(f)
            default_config.update(data)
    return default_config

