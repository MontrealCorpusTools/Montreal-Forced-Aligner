import os

from collections import defaultdict

def load_text(path):
    with open(path, 'r', encoding = 'utf8') as f:
        text = f.read().strip()
    return text

def load_phone_to_int(lang_directory):
    path = os.path.join(lang_directory, 'phones.txt')
    mapping = {}
    with open(path, 'r', encoding = 'utf8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            symbol, i = line.split(' ')
            mapping[symbol] = i
    return mapping

def load_word_to_int(lang_directory):
    path = os.path.join(lang_directory, 'words.txt')
    mapping = {}
    with open(path, 'r', encoding = 'utf8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            symbol, i = line.split(' ')
            mapping[symbol] = i
    return mapping

def save_groups(groups, seg_dir, pattern):
    for i, g in enumerate(groups):
        path = os.path.join(seg_dir, pattern.format(i+1))
        save_scp(g, path)

def save_scp(scp, path):
    with open(path, 'w', encoding = 'utf8') as f:
        for line in scp:
            f.write('{}\n'.format(' '.join(line)))

def load_scp(path):
    scp = []
    with open(path, 'r', encoding = 'utf8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            scp.append(line.split())
    return scp

def utt2spk_to_spk2utt(utt2spk):
    mapping = defaultdict(list)
    for line in utt2spk:
        mapping[line[1]].append(line[0])
    spk2utt = []
    for k in sorted(mapping.keys()):
        for v in sorted(mapping[k]):
            spk2utt.append((k, v))
    return spk2utt


def find_best_groupings(scp, num_jobs):
    num_utt = len(scp)

    interval = int(num_utt / num_jobs)
    groups = []
    current_ind = 0
    for i in range(num_jobs):
        if i == num_jobs - 1:
            end_ind = num_utt
        else:
            end_ind = current_ind + interval
            utt = scp[end_ind][0]
            if '_' not in utt:
                spk = utt
            else:
                spk = utt.split('_')[0]
            print(spk)
            for j in range(end_ind, num_utt):
                if not scp[j][0].startswith(spk):
                    j -= 1
                    break
            else:
                j = num_utt - 1
            for k in range(end_ind, 0, -1):
                if not scp[k][0].startswith(spk):
                    k += 1
                    break
            if j - end_ind < i - end_ind:
                end_ind = j
            else:
                end_ind = k
        groups.append(scp[current_ind:end_ind])
        current_ind = end_ind
    return groups

def load_utt2spk(train_directory):
    utt2spk = load_scp(os.path.join(train_directory, 'utt2spk'))
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

def split_scp(scp_path, seg_dir, num_jobs):
    base = os.path.basename(scp_path)
    name = os.path.splitext(base)[0]
    pattern = name + '.{}.scp'
    scp = load_scp(scp_path)
    groups = find_best_groupings(scp, num_jobs)
    save_groups(groups, seg_dir, pattern)

def load_oov_int(lang_directory):
    return load_text(os.path.join(lang_directory,'oov.int'))
