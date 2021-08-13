import subprocess
import os
import shutil

from ..helper import thirdparty_binary, make_safe, save_groups, load_scp

from ..multiprocessing import run_mp, run_non_mp


def mfcc_func(directory, job_name, mfcc_options):
    log_directory = os.path.join(directory, 'log')
    raw_mfcc_path = os.path.join(directory, 'raw_mfcc.{}.ark'.format(job_name))
    raw_scp_path = os.path.join(directory, 'feats.{}.scp'.format(job_name))
    lengths_path = os.path.join(directory, 'utterance_lengths.{}.scp'.format(job_name))
    log_path = os.path.join(log_directory, 'make_mfcc.{}.log'.format(job_name))
    segment_path = os.path.join(directory, 'segments.{}'.format(job_name))
    scp_path = os.path.join(directory, 'wav.{}.scp'.format(job_name))
    utt2num_frames_path = os.path.join(directory, 'utt2num_frames.{}'.format(job_name))
    mfcc_base_command = [thirdparty_binary('compute-mfcc-feats'), '--verbose=2']
    for k, v in mfcc_options.items():
        mfcc_base_command.append('--{}={}'.format(k.replace('_', '-'), make_safe(v)))
    with open(log_path, 'w') as log_file:
        if os.path.exists(segment_path):
            mfcc_base_command += ['ark:-', 'ark:-']
            seg_proc = subprocess.Popen([thirdparty_binary('extract-segments'),
                                         'scp,p:' + scp_path, segment_path, 'ark:-'],
                                        stdout=subprocess.PIPE, stderr=log_file)
            comp_proc = subprocess.Popen(mfcc_base_command,
                                         stdout=subprocess.PIPE, stderr=log_file, stdin=seg_proc.stdout)
        else:
            mfcc_base_command += ['scp,p:' + scp_path, 'ark:-']
            comp_proc = subprocess.Popen(mfcc_base_command,
                                         stdout=subprocess.PIPE, stderr=log_file)
        copy_proc = subprocess.Popen([thirdparty_binary('copy-feats'),
                                      '--compress=true', '--write-num-frames=ark,t:' + utt2num_frames_path,
                                      'ark:-',
                                      'ark,scp:{},{}'.format(raw_mfcc_path, raw_scp_path)],
                                     stdin=comp_proc.stdout, stderr=log_file)
        copy_proc.communicate()

        utt_lengths_proc = subprocess.Popen([thirdparty_binary('feat-to-len'),
                                             'scp:' + raw_scp_path, 'ark,t:' + lengths_path],
                                            stderr=log_file)
        utt_lengths_proc.communicate()


def mfcc(mfcc_directory, num_jobs, feature_config):
    """
    Multiprocessing function that converts wav files into MFCCs

    See http://kaldi-asr.org/doc/feat.html and
    http://kaldi-asr.org/doc/compute-mfcc-feats_8cc.html for more details on how
    MFCCs are computed.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/make_mfcc.sh
    for the bash script this function was based on.

    Parameters
    ----------
    mfcc_directory : str
        Directory to save MFCC feature matrices
    num_jobs : int
        The number of processes to use in calculation
    feature_config : :class:`~montreal_forced_aligner.features_config.FeatureConfig`
        Configuration object for generating MFCCs

    Raises
    ------
    CorpusError
        If the files per speaker exceeds the number of files that are
        allowed to be open on the computer (for Unix-based systems)
    """
    log_directory = os.path.join(mfcc_directory, 'log')
    os.makedirs(log_directory, exist_ok=True)

    jobs = [(mfcc_directory, x, feature_config.mfcc_options())
            for x in range(num_jobs)]
    if feature_config.use_mp:
        run_mp(mfcc_func, jobs, log_directory)
    else:
        run_non_mp(mfcc_func, jobs, log_directory)


def compute_vad_func(directory, vad_config, job_name):
    feats_path = os.path.join(directory, 'feats.{}.scp'.format(job_name))
    vad_scp_path = os.path.join(directory, 'vad.{}.scp'.format(job_name))
    with open(os.path.join(directory, 'log', 'vad.{}.log'.format(job_name)), 'w') as log_file:
        vad_proc = subprocess.Popen([thirdparty_binary('compute-vad'),
                                     '--vad-energy-mean-scale={}'.format(vad_config['energy_mean_scale']),
                                     '--vad-energy-threshold={}'.format(vad_config['energy_threshold']),
                                     'scp:' + feats_path,
                                     'ark,t:{}'.format(vad_scp_path)],
                                    stderr=log_file
                                    )
        vad_proc.communicate()


def calc_cmvn(corpus):
    split_dir = corpus.split_directory()
    spk2utt = os.path.join(corpus.output_directory, 'spk2utt')
    feats = os.path.join(corpus.output_directory, 'feats.scp')
    cmvn_directory = os.path.join(corpus.features_directory, 'cmvn')
    os.makedirs(cmvn_directory, exist_ok=True)
    cmvn_ark = os.path.join(cmvn_directory, 'cmvn.ark')
    cmvn_scp = os.path.join(cmvn_directory, 'cmvn.scp')
    log_path = os.path.join(cmvn_directory, 'cmvn.log')
    with open(log_path, 'w') as logf:
        subprocess.call([thirdparty_binary('compute-cmvn-stats'),
                         '--spk2utt=ark:' + spk2utt,
                         'scp:' + feats, 'ark,scp:{},{}'.format(cmvn_ark, cmvn_scp)],
                        stderr=logf)
    shutil.copy(cmvn_scp, os.path.join(corpus.output_directory, 'cmvn.scp'))
    corpus.cmvn_mapping = load_scp(cmvn_scp)
    pattern = 'cmvn.{}.scp'
    save_groups(corpus.grouped_cmvn, split_dir, pattern)


def compute_vad(directory, num_jobs, use_mp, vad_config=None):
    log_directory = os.path.join(directory, 'log')
    os.makedirs(log_directory, exist_ok=True)
    if vad_config is None:
        vad_config = {'energy_threshold': 5.5,
                      'energy_mean_scale': 0.5}
    jobs = [(directory, vad_config, x)
            for x in range(num_jobs)]
    if use_mp:
        run_mp(compute_vad_func, jobs, log_directory)
    else:
        run_non_mp(compute_vad_func, jobs, log_directory)
