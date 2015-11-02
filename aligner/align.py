import shutil
import os

from .config import *

from .prep.helper import (load_scp, load_utt2spk, find_best_groupings,
                        utt2spk_to_spk2utt, save_scp, load_oov_int,
                        load_word_to_int, load_text)

from .data_split import setup_splits

from .multiprocessing import compile_train_graphs, align, calc_fmllr

def align_si(data_directory, model_directory, output_directory, num_jobs = 4):
    lang_directory = os.path.join(data_directory, 'lang')
    optional_silence = load_text(os.path.join(lang_directory, 'phones', 'optional_silence.csl'))
    train_directory = os.path.join(data_directory, 'train')

    oov = load_oov_int(lang_directory)

    log_dir = os.path.join(output_directory, 'log')
    os.makedirs(log_dir, exist_ok = True)
    split_directory = os.path.join(train_directory, 'split{}'.format(num_jobs))
    if not os.path.exists(split_directory):
        setup_splits(train_directory, split_directory, lang_directory, num_jobs)

    shutil.copy(os.path.join(model_directory, 'tree'), output_directory)
    shutil.copy(os.path.join(model_directory, 'final.mdl'),
                                os.path.join(output_directory, '0.mdl'))
    shutil.copy(os.path.join(model_directory, 'final.occs'),
                        os.path.join(output_directory, '0.occs'))

    feat_type = 'delta'

    compile_train_graphs(output_directory, lang_directory, split_directory, num_jobs)
    align(0, output_directory, split_directory,
                optional_silence, num_jobs)
    os.rename(os.path.join(output_directory, '0.mdl'), os.path.join(output_directory, 'final.mdl'))
    os.rename(os.path.join(output_directory, '0.occs'), os.path.join(output_directory, 'final.occs'))

def align_sgmm():
    pass

def align_fmllr(data_directory, model_directory, output_directory, num_jobs = 4):
    lang_directory = os.path.join(data_directory, 'lang')
    train_directory = os.path.join(data_directory, 'train')
    align_si(data_directory, model_directory, output_directory, num_jobs)
    split_directory = os.path.join(train_directory, 'split{}'.format(num_jobs))
    sil_phones = load_text(os.path.join(lang_directory, 'phones', 'silence.csl'))

    calc_fmllr(output_directory, split_directory, sil_phones, num_jobs)
    optional_silence = load_text(os.path.join(lang_directory, 'phones', 'optional_silence.csl'))
    align(-1, output_directory, split_directory,
                optional_silence, num_jobs, fmllr = True)
