import os
from aligner.multiprocessing import mfcc

def load_utt2spk(train_directory):
    utt2spk = []
    with open(os.path.join(train_directory, 'utt2spk'), 'r', encoding = 'utf8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            utt2spk.append(line.split())
    return utt2spk

def load_wavscp(train_directory):
    wavscp = {}
    with open(os.path.join(train_directory, 'wav.scp'), 'r', encoding = 'utf8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            utt, wav = line.split()
            wavscp[utt] = wav
    return wavscp

def find_best_groupings(utt2spk, num_jobs):
    num_utt = len(utt2spk)

    interval = int(num_utt / num_jobs)
    groups = []
    current_ind = 0
    for i in range(num_jobs):
        if i == num_jobs - 1:
            end_ind = -1
        else:
            end_ind = current_ind + interval
            spk = utt2spk[end_ind][1]
            for j in range(end_ind, num_utt):
                if utt2spk[j][1] != spk:
                    j -= 1
                    break
            else:
                j = num_utt - 1
            for k in range(end_ind, 0, -1):
                if utt2spk[k][1] != spk:
                    k += 1
                    break

            if j - end_ind < i - end_ind:
                end_ind = j
            else:
                end_ind = k
        groups.append(utt2spk[current_ind:end_ind])
        current_ind = end_ind
    return groups

def save_groups(groups, seg_dir, wavscp):
    for i, g in enumerate(groups):
        with open(os.path.join(seg_dir, 'wav.{}.scp'.format(i+1)), 'w', encoding = 'utf8') as f:
            for utt in g:
                wav = wavscp[utt[0]]
                f.write('{} {}\n'.format(utt[0], wav))

def split_scp(train_directory, seg_dir, num_jobs):
    utt2spk = load_utt2spk(train_directory)
    groups = find_best_groupings(utt2spk, num_jobs)
    wavscp = load_wavscp(train_directory)
    save_groups(groups, seg_dir, wavscp)

def combine_feats(train_directory, mfcc_directory, num_jobs):
    feat_path = os.path.join(train_directory, 'feats.scp')
    with open(feat_path, 'w') as outf:
        for i in range(num_jobs):
            path = os.path.join(mfcc_directory, 'raw_mfcc.{}.scp'.format(i+1))
            with open(path,'r') as inf:
                for line in inf:
                    outf.write(line)
            os.remove(path)

def calc_cmvn(train_directory, mfcc_directory):
    spk2utt = os.path.join(train_directory, 'spk2utt')
    feats = os.path.join(train_directory, 'feats.scp')
    cmvn_directory = os.path.join(mfcc_directory, 'cmvn')
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
    split_scp(train_directory, log_directory, num_jobs)
    mfcc(mfcc_directory, log_directory, num_jobs, mfcc_config_path)
    combine_feats(train_directory, mfcc_directory, num_jobs)
    calc_cmvn(train_directory, mfcc_directory)
