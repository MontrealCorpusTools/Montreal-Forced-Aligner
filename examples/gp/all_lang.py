
import os
import shutil
import re
import sys
base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0,base)

from aligner.data_prep import data_prep

from aligner.train import train_mono, train_tri, train_tri_fmllr

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
                'SW': 'Swedish',
                'TA': 'Tamil',
                'TH': 'Thai',
                'TU': 'Turkish',
                'VN': 'Vietnamese'
                }

globalphone_dir = r'D:\Data\GlobalPhone'

source_dirs = {k: os.path.join(globalphone_dir, v) for k,v in full_names.items()}

data_dirs = {k: os.path.join(v, 'kaldi_align_data') for k,v in source_dirs.items()}

dict_paths = {k: os.path.join(source_dirs[k],
                            '{}_Dict'.format(v),
                            '{}-GPDict.txt'.format(v))
                    for k,v in full_names.items()}

lm_paths = {k: os.path.join(source_dirs[k],
                            '{}_languageModel'.format(v),
                            '{}.3gram.lm'.format(k))
                    for k,v in full_names.items()}

if __name__ == '__main__':
    for k in sorted(full_names.keys()):
        if not os.path.exists(source_dirs[k]):
            print(source_dirs[k],"not found")
            continue
        print(k)

        globalphone_prep(source_dirs[k], data_dirs[k], k)

        globalphone_dict_prep(dict_paths[k], data_dirs[k], k)

        data_prep(data_dirs[k], lm_paths[k])

        train_mono(data_dirs[k])
        train_tri(data_dirs[k], num_jobs = 6)
        train_tri_fmllr(data_dirs[k], num_jobs = 6)
