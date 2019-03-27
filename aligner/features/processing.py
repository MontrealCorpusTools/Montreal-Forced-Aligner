import multiprocessing as mp
import subprocess
import shutil
import os

from ..helper import make_path_safe, thirdparty_binary, filter_scp
from ..exceptions import CorpusError


def mfcc_func(directory, job_name, mfcc_config_path):  # pragma: no cover
    log_directory = os.path.join(directory, 'log')
    raw_mfcc_path = os.path.join(directory, 'raw_mfcc.{}.ark'.format(job_name))
    raw_scp_path = os.path.join(directory, 'feats.{}.scp'.format(job_name))
    log_path = os.path.join(log_directory, 'make_mfcc.{}.log'.format(job_name))
    segment_path = os.path.join(directory, 'segments.{}'.format(job_name))
    scp_path = os.path.join(directory, 'wav.{}.scp'.format(job_name))

    with open(log_path, 'w') as f:
        if os.path.exists(segment_path):
            seg_proc = subprocess.Popen([thirdparty_binary('extract-segments'),
                                         'scp,p:' + scp_path, segment_path, 'ark:-'],
                                        stdout=subprocess.PIPE, stderr=f)
            comp_proc = subprocess.Popen([thirdparty_binary('compute-mfcc-feats'), '--verbose=2',
                                          '--config=' + mfcc_config_path,
                                          'ark:-', 'ark:-'],
                                         stdout=subprocess.PIPE, stderr=f, stdin=seg_proc.stdout)
        else:

            comp_proc = subprocess.Popen([thirdparty_binary('compute-mfcc-feats'), '--verbose=2',
                                          '--config=' + mfcc_config_path,
                                          'scp,p:' + scp_path, 'ark:-'],
                                         stdout=subprocess.PIPE, stderr=f)
        copy_proc = subprocess.Popen([thirdparty_binary('copy-feats'),
                                      '--compress=true', 'ark:-',
                                      'ark,scp:{},{}'.format(raw_mfcc_path, raw_scp_path)],
                                     stdin=comp_proc.stdout, stderr=f)
        copy_proc.wait()


def init(env):
    os.environ = env

def mfcc(mfcc_directory, num_jobs, feature_config, frequency_configs):
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
    log_directory : str
        Directory to store log files
    num_jobs : int
        The number of processes to use in calculation
    mfcc_configs : list of :class:`~aligner.config.MfccConfig`
        Configuration object for generating MFCCs

    Raises
    ------
    CorpusError
        If the files per speaker exceeds the number of files that are
        allowed to be open on the computer (for Unix-based systems)
    """
    child_env = os.environ.copy()

    os.makedirs(os.path.join(mfcc_directory, 'log'), exist_ok=True)
    paths = []
    for j, p in frequency_configs:
        paths.append(feature_config.write(mfcc_directory, j, p))
    jobs = [(mfcc_directory, x, paths[x])
            for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(child_env,)) as pool:
        r = False
        try:
            results = [pool.apply_async(mfcc_func, args=i) for i in jobs]
            output = [p.get() for p in results]
        except OSError as e:
            print(dir(e))
            if e.errno == 24:
                r = True
            else:
                raise
    if r:
        raise (CorpusError(
            'There were too many files per speaker to process based on your OS settings.  Please try to split your data into more speakers.'))


def apply_cmvn_func(directory, job_name, config):
    normed_scp_path = os.path.join(directory, config.raw_feature_id + '.{}.scp'.format(job_name))
    normed_ark_path = os.path.join(directory, config.raw_feature_id + '.{}.ark'.format(job_name))
    with open(os.path.join(directory, 'log', 'norm.{}.log'.format(job_name)), 'w') as logf:
        utt2spkpath = os.path.join(directory, 'utt2spk.{}'.format(job_name))
        cmvnpath = os.path.join(directory, 'cmvn.{}.scp'.format(job_name))
        featspath = os.path.join(directory, 'feats.{}.scp'.format(job_name))
        if not os.path.exists(normed_scp_path):
            cmvn_proc = subprocess.Popen([thirdparty_binary('apply-cmvn'),
                                          '--utt2spk=ark:' + utt2spkpath,
                                          'scp:' + cmvnpath,
                                          'scp:' + featspath,
                                          'ark,scp:{},{}'.format(normed_ark_path, normed_scp_path)],
                                         stderr=logf
                                         )
            cmvn_proc.communicate()


def apply_cmvn(directory, num_jobs, config):
    child_env = os.environ.copy()
    jobs = [(directory, x, config)
            for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(child_env,)) as pool:
        results = [pool.apply_async(apply_cmvn_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def add_deltas_func(directory, job_name, config):
    normed_scp_path = os.path.join(directory, config.raw_feature_id + '.{}.scp'.format(job_name))
    ark_path = os.path.join(directory, config.feature_id + '.{}.ark'.format(job_name))
    scp_path = os.path.join(directory, config.feature_id + '.{}.scp'.format(job_name))
    with open(os.path.join(directory, 'log', 'add_deltas.{}.log'.format(job_name)), 'w') as logf:
        if config.fmllr_path is not None and os.path.exists(config.fmllr_path):
            deltas_proc = subprocess.Popen([thirdparty_binary('add-deltas'),
                                            'scp:' + normed_scp_path, 'ark:-'],
                                           stderr=logf,
                                           stdout=subprocess.PIPE)
            trans_proc = subprocess.Popen([thirdparty_binary('transform-feats'),
                                           'ark:' + config.fmllr_path, 'ark:-',
                                           'ark,scp:{},{}'.format(ark_path, scp_path)],
                                          stdin=deltas_proc.stdout,
                                          stderr=logf)
            trans_proc.communicate()
        else:
            deltas_proc = subprocess.Popen([thirdparty_binary('add-deltas'),
                                            'scp:' + normed_scp_path, 'ark,scp:{},{}'.format(ark_path, scp_path)],
                                           stderr=logf)
            deltas_proc.communicate()


def add_deltas(directory, num_jobs, config):
    child_env = os.environ.copy()
    jobs = [(directory, x, config)
            for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(child_env,)) as pool:
        results = [pool.apply_async(add_deltas_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def apply_lda_func(directory, job_name, config):
    normed_scp_path = os.path.join(directory, config.raw_feature_id + '.{}.scp'.format(job_name))
    ark_path = os.path.join(directory, config.feature_id + '.{}.ark'.format(job_name))
    scp_path = os.path.join(directory, config.feature_id + '.{}.scp'.format(job_name))
    ivector_scp_path = os.path.join(directory, 'ivector.{}.scp'.format(job_name))
    with open(os.path.join(directory, 'log', 'lda.{}.log'.format(job_name)), 'a') as logf:
        if os.path.exists(config.lda_path):
            splice_feats_proc = subprocess.Popen([thirdparty_binary('splice-feats'),
                                                  '--left-context={}'.format(config.splice_left_context),
                                                  '--right-context={}'.format(config.splice_right_context),
                                                  'scp:' + normed_scp_path,
                                                  'ark:-'],
                                                 stdout=subprocess.PIPE,
                                                 stderr=logf)
            if config.ivectors and os.path.exists(ivector_scp_path):
                transform_feats_proc = subprocess.Popen([thirdparty_binary("transform-feats"),
                                                         config.lda_path,
                                                         'ark:-',
                                                         'ark:-'],
                                                        stdin=splice_feats_proc.stdout,
                                                        stdout=subprocess.PIPE,
                                                        stderr=logf)
                paste_proc = subprocess.Popen([thirdparty_binary('paste-feats'),
                                               'ark:-',
                                               'scp:' + ivector_scp_path,
                                               'ark,scp:{},{}'.format(ark_path, scp_path)],
                                              stdin=transform_feats_proc.stdout,
                                              stderr=logf)
                paste_proc.communicate()
            else:
                transform_feats_proc = subprocess.Popen([thirdparty_binary("transform-feats"),
                                                         config.lda_path,
                                                         'ark:-',
                                                         'ark,scp:{},{}'.format(ark_path, scp_path)],
                                                        stdin=splice_feats_proc.stdout,
                                                        stderr=logf)
                transform_feats_proc.communicate()
        else:
            logf.write('could not find "{}"\n'.format(config.lda_path))
            splice_feats_proc = subprocess.Popen([thirdparty_binary('splice-feats'),
                                                  '--left-context={}'.format(config.splice_left_context),
                                                  '--right-context={}'.format(config.splice_right_context),
                                                  'scp:' + normed_scp_path,
                                                  'ark,scp:{},{}'.format(ark_path, scp_path)],
                                                 stderr=logf)
            splice_feats_proc.communicate()


def apply_lda(directory, num_jobs, config):
    jobs = [(directory, x, config)
            for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(apply_lda_func, args=i) for i in jobs]
        output = [p.get() for p in results]
