import os
import subprocess
import re

from .prep.helper import (load_scp, load_utt2spk, find_best_groupings,
                        utt2spk_to_spk2utt, save_scp, load_oov_int,
                        load_word_to_int)

from .multiprocessing import mono_realign, mono_align_equal

num_iters = 40

scale_opts = ['--transition-scale=1.0',
                '--acoustic-scale=0.1',
                '--self-loop-scale=0.1']

max_iter_inc = 30
totgauss = 1000
boost_silence = 1.0
realign_iters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 38]
stage = -4
power = 0.25

feat_template="ark,s,cs:apply-cmvn {cmvn_opts} --utt2spk=ark:{utt2spk_path} scp:{cmvn_path} scp:{feats_path} ark:- | add-deltas ark:- ark:- |"

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
    #Not working
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

def init_mono(split_directory, lang_directory, mono_directory, num_jobs):
    tree_path = os.path.join(mono_directory,'tree')
    mdl_path = os.path.join(mono_directory,'0.mdl')

    feat_dim = str(get_feat_dim(split_directory))
    directory = os.path.join(split_directory, '1')
    path = os.path.join(directory, 'cmvndeltafeats_sub')
    feat_path = os.path.join(directory, 'cmvndeltafeats')
    shared_phones_opt = "--shared-phones=" + os.path.join(lang_directory,'phones', 'sets.int')
    log_path = os.path.join(directory, 'log')
    with open(path, 'rb') as f, open(log_path, 'w') as logf:
        subprocess.call(['gmm-init-mono',shared_phones_opt,
                        "--train-feats=ark:-",
                        os.path.join(lang_directory,'topo'), feat_dim,
                        mdl_path,
                        tree_path],
                        stdin = f,
                        stderr = logf)
    num_gauss = get_num_gauss(mono_directory)
    mono_align_equal(mono_directory, lang_directory, split_directory, num_jobs)
    log_path = os.path.join(mono_directory, 'log', 'update.0.log')
    with open(log_path, 'w') as logf:
        pattern = r'^0.*.acc$'
        pattern = re.compile(pattern)
        acc_files = [os.path.join(mono_directory, x) for x in os.listdir(mono_directory) if pattern.match(x) is not None]
        est_proc = subprocess.Popen(['gmm-est', '--min-gaussian-occupancy=3',
                '--mix-up={}'.format(num_gauss), '--power={}'.format(power),
                mdl_path, "gmm-sum-accs - {}|".format(' '.join(acc_files)),
                os.path.join(mono_directory,'1.mdl')],
                stderr = logf)
        est_proc.communicate()

def do_training(mono_directory, split_directory, lang_directory, num_jobs):
    num_gauss = get_num_gauss(mono_directory)
    inc_gauss = int((totgauss - num_gauss) / max_iter_inc)
    for i in range(1, num_iters):
        if i in realign_iters:
            mono_realign(i, mono_directory, split_directory,
                        lang_directory, scale_opts, num_jobs, boost_silence, num_gauss)

        log_path = os.path.join(mono_directory, 'log', 'update.{}.log'.format(i))
        occs_path = os.path.join(mono_directory, '{}.occs'.format(i+1))
        model_path = os.path.join(mono_directory,'{}.mdl'.format(i))
        next_model_path = os.path.join(mono_directory,'{}.mdl'.format(i+1))
        with open(log_path, 'w') as logf:
            pattern = r'^{}.*.acc$'.format(i)
            pattern = re.compile(pattern)
            acc_files = [os.path.join(mono_directory, x) for x in os.listdir(mono_directory) if pattern.match(x) is not None]
            est_proc = subprocess.Popen(['gmm-est', '--write-occs='+occs_path,
                    '--mix-up='+str(num_gauss), '--power='+str(power), model_path,
                    "gmm-sum-accs - {}|".format(' '.join(acc_files)), next_model_path],
                    stderr = logf)
        if i < max_iter_inc:
            num_gauss += inc_gauss


def train_mono(data_directory, num_jobs = 6):
    lang_directory = os.path.join(data_directory, 'lang')
    train_directory = os.path.join(data_directory, 'train')
    mono_directory = os.path.join(data_directory, 'mono')
    os.makedirs(os.path.join(mono_directory, 'log'), exist_ok = True)
    split_directory = os.path.join(train_directory, 'split{}'.format(num_jobs))
    if not os.path.exists(split_directory):
        setup_splits(train_directory, split_directory, lang_directory, num_jobs)

    init_mono(split_directory, lang_directory, mono_directory, num_jobs)
    do_training(mono_directory, split_directory, lang_directory, num_jobs)

def train_tri():
    pass

def train_tri_fmllr():
    pass

def train_sgmm_sat():
    pass
