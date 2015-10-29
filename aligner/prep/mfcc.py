import os
from aligner.multiprocessing import mfcc

import subprocess
import shutil

from .helper import split_scp

def combine_feats(train_directory, mfcc_directory, num_jobs):
    feat_path = os.path.join(train_directory, 'feats.scp')
    with open(feat_path, 'w') as outf:
        for i in range(num_jobs):
            path = os.path.join(mfcc_directory, 'raw_mfcc.{}.scp'.format(i+1))
            with open(path,'r') as inf:
                for line in inf:
                    outf.write(line)
            #os.remove(path)

def calc_cmvn(train_directory, mfcc_directory):
    spk2utt = os.path.join(train_directory, 'spk2utt')
    feats = os.path.join(train_directory, 'feats.scp')
    cmvn_directory = os.path.join(mfcc_directory, 'cmvn')
    os.makedirs(cmvn_directory, exist_ok = True)
    cmvn_ark = os.path.join(cmvn_directory, 'cmvn.ark')
    cmvn_scp = os.path.join(cmvn_directory, 'cmvn.scp')
    subprocess.call(['compute-cmvn-stats','--spk2utt=ark:'+spk2utt,
                    'scp:'+feats, 'ark,scp:{},{}'.format(cmvn_ark, cmvn_scp)])
    shutil.copy(cmvn_scp, os.path.join(train_directory, 'cmvn.scp'))

def prepare_mfccs(train_directory, mfcc_directory, mfcc_config_path, num_jobs = None):
    if num_jobs is None:
        num_jobs = 6
    log_directory = os.path.join(mfcc_directory, 'log')
    os.makedirs(log_directory, exist_ok = True)
    split_scp(os.path.join(train_directory, 'wav.scp'), log_directory, num_jobs)
    mfcc(mfcc_directory, log_directory, num_jobs, mfcc_config_path)
    combine_feats(train_directory, mfcc_directory, num_jobs)
    calc_cmvn(train_directory, mfcc_directory)
