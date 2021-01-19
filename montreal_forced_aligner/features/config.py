import os
import shutil
import subprocess
from ..exceptions import ConfigError
from .processing import mfcc, add_deltas, apply_cmvn, apply_lda, compute_vad, select_voiced, compute_ivector_features

from ..helper import thirdparty_binary, load_scp, save_groups


def make_safe(value):
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


class FeatureConfig(object):
    """
    Class to store configuration information about MFCC generation

    Parameters
    ----------
    directory : str
        Path to directory to save configuration files for Kaldi

    Attributes
    ----------
    directory : str
        Path of the directory to store outputs
    type : str
        Feature type, defaults to "mfcc"
    deltas : bool
        Flag for whether deltas from previous frames are included in the features, defaults to True
    lda : bool
        Flag for whether LDA is run on the features, requires an lda.mat to generate, defaults to False
    fmllr : bool
        Flag for whether speaker adaptation should be run, defaults to False
    use_energy : bool
        Flag for whether first coefficient should be used, defaults to False
    frame_shift : int
        number of milliseconds between frames, defaults to 10
    pitch : bool
        Flag for including pitch in features, currently nonfunctional, defaults to False
    splice_left_context : int or None
        Number of frames to splice on the left for calculating LDA
    splice_right_context : int or None
        Number of frames to splice on the right for calculating LDA
    use_mp : bool
        Flag for using multiprocessing, defaults to True
    """

    def __init__(self, directory=None):
        self.directory = directory
        self.type = 'mfcc'
        self.deltas = True
        self.lda = False
        self.fmllr = False
        self.use_energy = False
        self.frame_shift = 10
        self.pitch = False
        self.splice_left_context = None
        self.splice_right_context = None
        self.use_mp = True
        self.job_specific_configuration = {}

    def params(self):
        return {'type': self.type,
                'use_energy': self.use_energy,
                'frame_shift': self.frame_shift,
                'pitch': self.pitch,
                'deltas': self.deltas,
                'lda': self.lda,
                'fmllr': self.fmllr,
                'splice_left_context': self.splice_left_context,
                'splice_right_context': self.splice_right_context,
                }

    def set_features_to_use_lda(self):
        self.lda = True
        self.deltas = False

    @property
    def lda_options(self):
        return {'splice_left_context': self.splice_left_context, 'splice_right_context': self.splice_right_context}

    def add_job_specific_config(self, job_name, config):
        self.job_specific_configuration[job_name] = config

    def mfcc_options(self, job_name):
        options = {'use_energy': self.use_energy}
        options.update(self.job_specific_configuration[job_name])
        return options

    def update(self, data):
        for k, v in data.items():
            if not hasattr(self, k):
                raise ConfigError('No field found for key {}'.format(k))
            setattr(self, k, v)
        if self.lda:
            self.deltas = False

    def write(self, output_directory, job, extra_params=None):
        """
        Write configuration dictionary to a file for use in Kaldi binaries
        """
        f = '{}.{}.conf'.format(self.type, job)
        path = os.path.join(output_directory, 'config')
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f)
        with open(path, 'w', encoding='utf8') as f:
            f.write('--{}={}\n'.format('use-energy', make_safe(self.use_energy)))
            f.write('--{}={}\n'.format('frame-shift', make_safe(self.frame_shift)))
            if extra_params is not None:
                for k, v in extra_params.items():
                    f.write('--{}={}\n'.format(k, make_safe(v)))
        return path

    def calc_cmvn(self, corpus):
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
        apply_cmvn(split_dir, corpus.num_jobs, self)

    def compute_vad(self, corpus, logger=None):
        if logger is None:
            log_func = print
        else:
            log_func = logger.info
        split_directory = corpus.split_directory()
        if os.path.exists(os.path.join(split_directory, 'vad.0.scp')):
            log_func('VAD already computed, skipping!')
            return
        log_func('Computing VAD...')
        compute_vad(split_directory, corpus.num_jobs, self.use_mp)

    @property
    def raw_feature_id(self):
        name = 'features_{}'.format(self.type)
        if self.type == 'mfcc':
            name += '_cmvn'
        return name

    @property
    def feature_id(self):
        name = 'features_{}'.format(self.type)
        if self.type == 'mfcc':
            name += '_cmvn'
        if self.deltas:
            name += '_deltas'
        elif self.lda:
            name += '_lda'
        if self.fmllr:
            name += '_fmllr'
        return name

    @property
    def voiced_feature_id(self):
        name = 'feats_voiced'
        return name

    @property
    def pre_ivector_feature_id(self):
        name = 'feats_for_ivector'
        return name

    @property
    def fmllr_path(self):
        return os.path.join(self.directory, 'trans.{}')

    @property
    def lda_path(self):
        return os.path.join(self.directory, 'lda.mat')

    def generate_base_features(self, corpus, logger=None, compute_cmvn=True):
        if logger is None:
            log_func = print
        else:
            log_func = logger.info
        split_directory = corpus.split_directory()
        for job_name, config in enumerate(corpus.frequency_configs):
            self.add_job_specific_config(job_name, config[1])
        if compute_cmvn:
            feat_id = self.raw_feature_id
        else:
            feat_id = 'feats'
        if not os.path.exists(os.path.join(split_directory, feat_id + '.0.scp')):
            log_func('Generating base features ({})...'.format(self.type))
            if self.type == 'mfcc':
                mfcc(split_directory, corpus.num_jobs, self)
            corpus.combine_feats()
            if compute_cmvn:
                log_func('Calculating CMVN...')
                self.calc_cmvn(corpus)
        #corpus.parse_features_logs()

    def generate_features(self, corpus, data_directory=None, overwrite=False, logger=None):
        if data_directory is None:
            data_directory = corpus.split_directory()
        if self.directory is None:
            self.directory = data_directory
        if not overwrite and os.path.exists(os.path.join(data_directory, self.feature_id + '.0.scp')):
            return
        self.generate_base_features(corpus, logger=logger)
        if self.deltas:
            add_deltas(data_directory, corpus.num_jobs, self)
        elif self.lda:
            apply_lda(data_directory, corpus.num_jobs, self)

    def generate_ivector_extract_features(self, corpus, data_directory=None, overwrite=False, apply_cmn=False, logger=None):
        if logger is None:
            log_func = print
        else:
            log_func = logger.info
        if data_directory is None:
            data_directory = corpus.split_directory()
        data_directory = os.path.join(data_directory, 'subsegments')
        if self.directory is None:
            self.directory = data_directory
        if not overwrite and os.path.exists(os.path.join(data_directory, self.pre_ivector_feature_id + '.0.scp')):
            log_func('Features for ivector already exist, skipping!')
            return
        compute_ivector_features(data_directory, corpus.num_jobs, self, apply_cmn=apply_cmn)
        log_func('Finished generating features for ivector extraction!')

    def generate_voiced_features(self, corpus, data_directory=None, overwrite=False, apply_cmn=False, logger=None):
        if logger is None:
            log_func = print
        else:
            log_func = logger.info
        if data_directory is None:
            data_directory = corpus.split_directory()
        if self.directory is None:
            self.directory = data_directory
        if not overwrite and os.path.exists(os.path.join(data_directory, self.voiced_feature_id + '.0.scp')):
            log_func('Voiced features already selected, skipping!')
            return
        select_voiced(data_directory, corpus.num_jobs, self, apply_cmn=apply_cmn)
        log_func('Finished selecting voiced features!')
