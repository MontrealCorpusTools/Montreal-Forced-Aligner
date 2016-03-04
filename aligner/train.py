import os
import subprocess
import re
import shutil

from .prep.helper import (load_scp, load_utt2spk, find_best_groupings,
                        utt2spk_to_spk2utt, save_scp, load_oov_int,
                        load_word_to_int, load_text)

from .multiprocessing import (align, mono_align_equal, compile_train_graphs,
                            acc_stats, tree_stats, convert_alignments,
                             convert_ali_to_textgrids, calc_fmllr)

from .align import align_si, align_fmllr

from .data_split import setup_splits, get_feat_dim

from .config import *

def get_num_gauss(mono_directory):
    with open(os.devnull, 'w') as devnull:
        proc = subprocess.Popen(['gmm-info','--print-args=false',
                    os.path.join(mono_directory, '0.mdl')],
                    stderr = devnull,
                    stdout = subprocess.PIPE)
        stdout, stderr = proc.communicate()
        num = stdout.decode('utf8')
        matches = re.search(r'gaussians (\d+)', num)
        num = int(matches.groups()[0])
    return num



def train_sgmm_sat():
    pass
