import os
import yaml
from .config import TEMP_DIR
from .models import AcousticModel, G2PModel, IvectorExtractor, LanguageModel
from .exceptions import ArgumentError


def get_available_acoustic_languages():
    pretrained_dir = os.path.join(TEMP_DIR, 'pretrained_models', 'acoustic')
    os.makedirs(pretrained_dir, exist_ok=True)
    languages = []
    for f in os.listdir(pretrained_dir):
        if f.endswith(AcousticModel.extension):
            languages.append(os.path.splitext(f)[0])
    return languages


def get_available_g2p_languages():
    pretrained_dir = os.path.join(TEMP_DIR, 'pretrained_models', 'g2p')
    os.makedirs(pretrained_dir, exist_ok=True)
    languages = []
    for f in os.listdir(pretrained_dir):
        if f.endswith(G2PModel.extension):
            languages.append(os.path.splitext(f)[0])
    return languages


def get_available_ivector_languages():
    pretrained_dir = os.path.join(TEMP_DIR, 'pretrained_models', 'ivector')
    os.makedirs(pretrained_dir, exist_ok=True)
    languages = []
    for f in os.listdir(pretrained_dir):
        if f.endswith(IvectorExtractor.extension):
            languages.append(os.path.splitext(f)[0])
    return languages


def get_available_lm_languages():
    pretrained_dir = os.path.join(TEMP_DIR, 'pretrained_models', 'language_model')
    os.makedirs(pretrained_dir, exist_ok=True)
    languages = []
    for f in os.listdir(pretrained_dir):
        if f.endswith(LanguageModel.extension):
            languages.append(os.path.splitext(f)[0])
    return languages


def get_available_dict_languages():
    extension = '.dict'
    pretrained_dir = os.path.join(TEMP_DIR, 'pretrained_models', 'dictionary')
    os.makedirs(pretrained_dir, exist_ok=True)
    languages = []
    for f in os.listdir(pretrained_dir):
        if f.endswith(extension):
            languages.append(os.path.splitext(f)[0])
    return languages


def get_pretrained_acoustic_path(language):
    pretrained_dir = os.path.join(TEMP_DIR, 'pretrained_models', 'acoustic')
    os.makedirs(pretrained_dir, exist_ok=True)
    return os.path.join(pretrained_dir, language + AcousticModel.extension)


def get_pretrained_ivector_path(language):
    pretrained_dir = os.path.join(TEMP_DIR, 'pretrained_models', 'ivector')
    os.makedirs(pretrained_dir, exist_ok=True)
    return os.path.join(pretrained_dir, language + IvectorExtractor.extension)


def get_pretrained_language_model_path(language):
    pretrained_dir = os.path.join(TEMP_DIR, 'pretrained_models', 'language_model')
    os.makedirs(pretrained_dir, exist_ok=True)
    return os.path.join(pretrained_dir, language + LanguageModel.extension)


def get_pretrained_g2p_path(language):
    pretrained_dir = os.path.join(TEMP_DIR, 'pretrained_models', 'g2p')
    os.makedirs(pretrained_dir, exist_ok=True)
    return os.path.join(pretrained_dir, language + G2PModel.extension)


def get_dictionary_path(language):
    pretrained_dir = os.path.join(TEMP_DIR, 'pretrained_models', 'dictionary')
    os.makedirs(pretrained_dir, exist_ok=True)
    return os.path.join(pretrained_dir, language + '.dict')

def validate_dictionary_arg(dictionary_path, download_dictionaries):
    if dictionary_path.lower().endswith('.yaml'):
        with open(dictionary_path, 'r', encoding='utf8') as f:
            data = yaml.safe_load(f)
            found_default = False
            for speaker, path in data.items():
                if speaker == 'default':
                    found_default = True
                if path.lower() in download_dictionaries:
                    path = get_dictionary_path(path.lower())
                if not os.path.exists(path):
                    raise ArgumentError('Could not find the dictionary file {} for speaker {}'.format(path, speaker))
                if not os.path.isfile(path):
                    raise ArgumentError('The specified dictionary path ({} for speaker {}) is not a text file.'.format(path, speaker))
            if not found_default:
                raise ArgumentError('No "default" dictionary was found.')
    else:
        if dictionary_path.lower() in download_dictionaries:
            dictionary_path = get_dictionary_path(dictionary_path.lower())
        if not os.path.exists(dictionary_path):
            raise ArgumentError('Could not find the dictionary file {}'.format(dictionary_path))
        if not os.path.isfile(dictionary_path):
            raise ArgumentError('The specified dictionary path ({}) is not a text file.'.format(dictionary_path))
    return dictionary_path
