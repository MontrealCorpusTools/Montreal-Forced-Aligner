import os
import subprocess
import re
import shutil


from .prep.helper import (load_scp, load_utt2spk, find_best_groupings,
                        utt2spk_to_spk2utt, save_scp, load_oov_int,
                        load_word_to_int)

def filter_scp(scp, group_mapping):
    groups = [[] for x in range(len(group_mapping))]
    for t in scp:
        groups[group_mapping[t[0]]].append(t)
    return groups

def make_mapping_for_groups(groups):
    mapping = {}
    for i, g in enumerate(groups):
        for line in g:
            mapping[line[0]] = i
    return mapping

def text_to_int(text, words, oov):
    newtext = []
    for i, t in enumerate(text):
        line = t[1:]
        new_line = []
        for w in line:
            if w in words:
                new_line.append(str(words[w]))
            else:
                new_line.append(str(oov))
        newtext.append([text[i][0]] + new_line)
    return newtext

def feats_gen(directory):
    kwargs = {'cmvn_opts': '',
                'utt2spk_path': os.path.join(directory, 'utt2spk'),
                'cmvn_path': os.path.join(directory, 'cmvn.scp'),
                'feats_path': os.path.join(directory, 'feats.scp')}
    return feat_template.format(**kwargs)

def make_feats_file(directory):
    path = os.path.join(directory, 'cmvndeltafeats')
    if not os.path.exists(path):
        with open(path, 'wb') as outf, open(os.devnull, 'w') as devnull:
            cmvn_proc = subprocess.Popen(['apply-cmvn',
                        '--utt2spk=ark:'+os.path.join(directory, 'utt2spk'),
                        'scp:'+os.path.join(directory, 'cmvn.scp'),
                        'scp:'+os.path.join(directory, 'feats.scp'),
                        'ark:-'], stdout = subprocess.PIPE,
                        stderr = devnull)
            deltas_proc = subprocess.Popen(['add-deltas', 'ark:-', 'ark:-'],
                                    stdin = cmvn_proc.stdout,
                                    stdout = outf,
                                    stderr = devnull)
            deltas_proc.communicate()
        with open(path, 'rb') as inf, open(path+'_sub','wb') as outf:
            subprocess.call(["subset-feats", "--n=10", "ark:-", "ark:-"],
                        stdin = inf, stdout = outf)

def get_feat_dim(split_directory):
    cmvn_opts = ''
    directory = os.path.join(split_directory, '1')

    make_feats_file(directory)
    path = os.path.join(directory, 'cmvndeltafeats')
    with open(path, 'rb') as f:
        dim_proc = subprocess.Popen(['feat-to-dim', 'ark,s,cs:-', '-'],
                                    stdin = f,
                                    stdout = subprocess.PIPE)
        stdout, stderr = dim_proc.communicate()
        feats = stdout.decode('utf8').strip()

    return feats

def setup_splits(train_directory, split_directory, lang_directory, num_jobs):
    utt2spk = load_utt2spk(train_directory)
    utt2spks = find_best_groupings(utt2spk, num_jobs)
    utt_mapping = make_mapping_for_groups(utt2spks)
    spk2utts = [utt2spk_to_spk2utt(x) for x in utt2spks]
    spk_mapping = make_mapping_for_groups(spk2utts)

    words = load_word_to_int(lang_directory)
    oov = load_oov_int(lang_directory)

    wavs = load_scp(os.path.join(train_directory, 'wav.scp'))
    feats = load_scp(os.path.join(train_directory, 'feats.scp'))
    text = load_scp(os.path.join(train_directory, 'text'))

    wav_groups = filter_scp(wavs, utt_mapping)
    feat_groups = filter_scp(feats, utt_mapping)
    text_groups = filter_scp(text, utt_mapping)

    cmvns = load_scp(os.path.join(train_directory, 'cmvn.scp'))
    cmvn_groups = filter_scp(cmvns, spk_mapping)
    for i in range(num_jobs):
        job_dir = os.path.join(split_directory, str(i+1))
        os.makedirs(job_dir, exist_ok = True)
        wav_path = os.path.join(job_dir, 'wav.scp')
        utt2spk_path = os.path.join(job_dir, 'utt2spk')
        spk2utt_path = os.path.join(job_dir, 'spk2utt')
        feats_path = os.path.join(job_dir, 'feats.scp')
        cmvn_path = os.path.join(job_dir, 'cmvn.scp')
        text_path = os.path.join(job_dir, 'text')
        text_int_path = os.path.join(job_dir, 'text.int')
        save_scp(wav_groups[i], wav_path)
        save_scp(utt2spks[i], utt2spk_path)
        save_scp(spk2utts[i], spk2utt_path)
        save_scp(feat_groups[i], feats_path)
        save_scp(cmvn_groups[i], cmvn_path)
        save_scp(text_groups[i], text_path)
        save_scp(text_to_int(text_groups[i], words, oov), text_int_path)
        save_scp(text_groups[i], text_path)
        make_feats_file(job_dir)
