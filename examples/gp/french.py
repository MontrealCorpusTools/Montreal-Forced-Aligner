
import os
import shutil
import re
import sys
base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0,base)

from aligner.data_prep import data_prep

from aligner.train import train_mono

from gp_utils import lang_encodings, globalphone_prep, globalphone_dict_prep


source_dir = os.path.normpath(r'/home/michael/dev/kaldi-gp-alignment/alignment/GlobalPhone/French')

lang_code = 'FR'

data_dir = os.path.join(source_dir, 'kaldi_align_data')

dict_path = os.path.join(source_dir, 'French_Dict', 'French-GPDict.txt')

lm_path = os.path.join(source_dir, 'French_languageModel', 'FR.3gram.lm')

if __name__ == '__main__':
    globalphone_prep(source_dir, data_dir, lang_code)

    globalphone_dict_prep(dict_path, data_dir, lang_code)

    data_prep(data_dir, lm_path)

    train_mono(data_dir)
