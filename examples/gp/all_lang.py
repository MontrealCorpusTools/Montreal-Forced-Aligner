
import os
import shutil
import re
import sys
base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0,base)

from aligner.corpus import Corpus
from aligner.config import MfccConfig
from aligner.dictionary import Dictionary
from aligner.aligner import BaseAligner

from gp_utils import lang_encodings, globalphone_prep, globalphone_dict_prep

full_names = {
                'AR': 'Arabic',
                'BG': 'Bulgarian',
                'CH': 'Mandarin',
                'WU': 'Cantonese',
                'CR': 'Croatian',
                'CZ': 'Czech',
                'FR': 'French',
                'GE': 'German',
                'HA': 'Hausa',
                'JA': 'Japanese',
                'KO': 'Korean',
                'RU': 'Russian',
                'PO': 'Portuguese',
                'PL': 'Polish',
                'SP': 'Spanish',
                'SA': 'Swahili',
                'SW': 'Swedish',
                'TA': 'Tamil',
                'TH': 'Thai',
                'TU': 'Turkish',
                'VN': 'Vietnamese',
                'UA': 'Ukrainian'
                }

globalphone_dir = r'D:\Data\GlobalPhone'

base_dirs = {k: os.path.join(globalphone_dir, v) for k,v in full_names.items()}

source_dirs = {k: os.path.join(base_dirs[k], v) for k,v in full_names.items()}

data_directory = r'D:\Data\kaldi-gp-data'

data_dirs = {k: os.path.join(data_directory, k) for k,v in source_dirs.items()}

dict_paths = {k: os.path.join(base_dirs[k],
                            '{}_Dict'.format(v),
                            '{}-GPDict.txt'.format(v))
                    for k,v in full_names.items()}

if __name__ == '__main__':
    for k in sorted(full_names.keys()):
        if not os.path.exists(source_dirs[k]):
            print(source_dirs[k],"not found")
            continue
        print(k)

        globalphone_dict_prep(dict_paths[k], data_dirs[k], k)

        globalphone_prep(source_dirs[k], data_dirs[k], k)

        output_dir = os.path.join(data_dirs[k], 'textgrid_output')

        corpus_dir = os.path.join(data_dirs[k], 'files')
        dict_path = os.path.join(data_dirs[k], 'dict', 'lexicon.txt')

        temp_dir = os.path.join(data_dirs[k], 'new_temp')

        dictionary = Dictionary(dict_path, temp_dir)
        dictionary.write()



        c = MfccConfig(temp_dir)
        corpus = Corpus(corpus_dir, temp_dir, c, num_jobs = 4)
        corpus.write()
        corpus.create_mfccs()
        corpus.setup_splits(dictionary)

        a = BaseAligner(corpus, dictionary, output_dir,
                            temp_directory = temp_dir, num_jobs = 4)
        a.train_mono()
        a.train_tri()
        a.train_tri_fmllr()
        a.export_textgrids()
