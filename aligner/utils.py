import os
from .config import TEMP_DIR
from .models import AcousticModel, G2PModel


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


def get_pretrained_g2p_path(language):
    pretrained_dir = os.path.join(TEMP_DIR, 'pretrained_models', 'g2p')
    os.makedirs(pretrained_dir, exist_ok=True)
    return os.path.join(pretrained_dir, language + G2PModel.extension)


def get_dictionary_path(language):
    pretrained_dir = os.path.join(TEMP_DIR, 'pretrained_models', 'dictionary')
    os.makedirs(pretrained_dir, exist_ok=True)
    return os.path.join(pretrained_dir, language + '.dict')