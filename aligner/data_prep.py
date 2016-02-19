import os
import shutil
import re

from .prep import prepare_dict, prepare_train_data, prepare_mfccs, prepare_config

from .validation import validate_training_directory

def data_prep(data_directory):
    """
    Prepares data for alignment from a directory of sound files with
    TextGrids (or label files)

    Parameters
    ----------
    source_dir : str
        Path to directory of sound files to align
    temp_dir : str
        Path to directory to temporary store files used in alignment
    dict_path : str
        Path to a pronunciation dictionary
    """
    config_dir = os.path.join(data_directory, 'conf')
    if not os.path.exists(config_dir):
        print('Creating a config directory...')
        prepare_config(config_dir)
        print('Done!')
    else:
        print('Using existing config directory.')

    lang_dir = os.path.join(data_directory, 'lang')
    if not os.path.exists(lang_dir):
        print('Preparing dictionary...')
        prepare_dict(data_directory)
        print('Done!')
    else:
        print('Using existing dictionary.')

    train_dir = os.path.join(data_directory, 'train')
    if not os.path.exists(train_dir):
        print('Preparing training data...')
        files_dir = os.path.join(data_directory, 'files')
        prepare_train_data(files_dir, train_dir)
        print('Done!')
    else:
        print('Using existing training set up.')

    validate_training_directory(data_directory, fail_ok = True)

    mfcc_dir = os.path.join(data_directory, 'mfcc')
    if not os.path.exists(mfcc_dir):
        print('Creating mfccs...')
        mfcc_config = os.path.join(config_dir, 'mfcc.conf')
        prepare_mfccs(train_dir, mfcc_dir, mfcc_config, num_jobs = 6)
        print('Done!')
    else:
        print('Using existing mfccs.')
