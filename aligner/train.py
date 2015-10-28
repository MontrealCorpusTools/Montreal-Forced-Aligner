import os

num_iters = 40

scale_opts = ['--transition-scale=1.0',
                '--acoustic-scale=0.1',
                '--self-loop-scale=0.1']

max_iter_inc = 30
totgauss = 1000
boost_silence = 1.0
realign_iters = "1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 23 26 29 32 35 38"
stage = -4
power = 0.25

feat_template="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"

def setup_splits(train_directory, num_jobs):
    split_dir = os.path.join(train_directory, 'split{}'.format(num_jobs))
    if os.path.exists(split_dir):
        return
    for i in range(num_jobs):
        os.makedirs(os.path.join(split_dir,str(i+1)))

def train_mono(train_directory, lang_directory, mono_directory):
    os.makedirs(mono_directory, exist_ok = True)

    feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"


def train_tri():
    pass

def train_tri_fmllr():
    pass

def train_sgmm_sat():
    pass
