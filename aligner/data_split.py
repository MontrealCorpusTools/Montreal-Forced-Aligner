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
                        #stderr = devnull
                        )
            deltas_proc = subprocess.Popen(['add-deltas', 'ark:-', 'ark:-'],
                                    stdin = cmvn_proc.stdout,
                                    stdout = outf,
                                    #stderr = devnull
                                    )
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
