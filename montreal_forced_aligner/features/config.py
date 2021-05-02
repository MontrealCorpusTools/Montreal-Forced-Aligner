import os
from ..exceptions import ConfigError
from .processing import mfcc, compute_vad, calc_cmvn
from ..config import BaseConfig


def make_safe(value):
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


class FeatureConfig(BaseConfig):
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
        self.snip_edges = True
        self.pitch = False
        self.low_frequency = 20
        self.high_frequency = 7800
        self.sample_frequency = 16000
        self.allow_downsample = True
        self.allow_upsample = True
        self.splice_left_context = 3
        self.splice_right_context = 3
        self.use_mp = True

    def params(self):
        return {'type': self.type,
                'use_energy': self.use_energy,
                'frame_shift': self.frame_shift,
                'snip_edges': self.snip_edges,
                'low_frequency': self.low_frequency,
                'high_frequency': self.high_frequency,
                'sample_frequency': self.sample_frequency,
                'allow_downsample': self.allow_downsample,
                'allow_upsample': self.allow_upsample,
                'pitch': self.pitch,
                'deltas': self.deltas,
                'lda': self.lda,
                'fmllr': self.fmllr,
                'splice_left_context': self.splice_left_context,
                'splice_right_context': self.splice_right_context,
                }

    def mfcc_options(self):
        """Return dictionary of parameters to use in computing MFCC features."""
        return {'use-energy': self.use_energy, 'frame-shift': self.frame_shift, 'low-freq': self.low_frequency,
                'high-freq': self.high_frequency, 'sample-frequency': self.sample_frequency,
                'allow-downsample': self.allow_downsample, 'allow-upsample': self.allow_upsample,
                'snip-edges': self.snip_edges}

    def set_features_to_use_lda(self):
        self.lda = True
        self.deltas = False

    @property
    def splice_options(self):
        return {'splice_left_context': self.splice_left_context, 'splice_right_context': self.splice_right_context}

    def update(self, data):
        for k, v in data.items():
            if not hasattr(self, k):
                raise ConfigError('No field found for key {}'.format(k))
            setattr(self, k, v)
        if self.lda:
            self.deltas = False

    def compute_vad(self, corpus, logger=None, vad_config=None):
        if logger is None:
            log_func = print
        else:
            log_func = logger.info
        split_directory = corpus.split_directory()
        if os.path.exists(os.path.join(split_directory, 'vad.0.scp')):
            log_func('VAD already computed, skipping!')
            return
        log_func('Computing VAD...')
        compute_vad(split_directory, corpus.num_jobs, self.use_mp, vad_config=vad_config)

    @property
    def feature_id(self):
        return 'feats'

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
        feat_id = 'feats'
        if not os.path.exists(os.path.join(split_directory, feat_id + '.0.scp')):
            log_func('Generating base features ({})...'.format(self.type))
            if self.type == 'mfcc':
                mfcc(split_directory, corpus.num_jobs, self)
            corpus.combine_feats()
            if compute_cmvn:
                log_func('Calculating CMVN...')
                calc_cmvn(corpus)

    def construct_feature_proc_string(self, data_directory, model_directory, job_name, splice=False, voiced=False, cmvn=True):
        if self.directory is None:
            self.directory = data_directory
        lda_mat_path = None
        fmllr_trans_path = None
        if model_directory is not None:
            lda_mat_path = os.path.join(model_directory, 'lda.mat')
            if not os.path.exists(lda_mat_path):
                lda_mat_path = None
            fmllr_trans_path = os.path.join(model_directory, 'trans.{}'.format(job_name))
            if not os.path.exists(fmllr_trans_path):
                fmllr_trans_path = None
        if job_name is not None:
            utt2spk_path = os.path.join(data_directory, 'utt2spk.{}'.format(job_name))
            cmvn_path = os.path.join(data_directory, 'cmvn.{}.scp'.format(job_name))
            feat_path = os.path.join(data_directory, 'feats.{}.scp'.format(job_name))
            vad_path = os.path.join(data_directory, 'vad.{}.scp'.format(job_name))
        else:
            utt2spk_path = os.path.join(data_directory, 'utt2spk')
            cmvn_path = os.path.join(data_directory, 'cmvn.scp')
            feat_path = os.path.join(data_directory, 'feats.scp')
            vad_path = os.path.join(data_directory, 'vad.scp')
        if voiced:
            feats = 'ark,s,cs:add-deltas scp:{} ark:- |'.format(feat_path)
            if cmvn:
                feats += ' apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |'
            feats += ' select-voiced-frames ark:- scp,s,cs:{} ark:- |'.format(vad_path)
        elif not os.path.exists(cmvn_path) and cmvn:
            feats = 'ark,s,cs:add-deltas scp:{} ark:- |'.format(feat_path)
            if cmvn:
                feats += ' apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |'
        else:
            feats = "ark,s,cs:apply-cmvn --utt2spk=ark:{} scp:{} scp:{} ark:- |".format(utt2spk_path, cmvn_path, feat_path)
            if lda_mat_path is not None:
                if not os.path.exists(lda_mat_path):
                    raise Exception('Could not find {}'.format(lda_mat_path))
                feats += ' splice-feats --left-context={} --right-context={} ark:- ark:- |'.format(self.splice_left_context,
                                                                                                   self.splice_right_context)
                feats += " transform-feats {} ark:- ark:- |".format(lda_mat_path)
            elif splice:
                feats += ' splice-feats --left-context={} --right-context={} ark:- ark:- |'.format(self.splice_left_context,
                                                                                                   self.splice_right_context)
            elif self.deltas:
                feats += " add-deltas ark:- ark:- |"

            if fmllr_trans_path is not None:
                if not os.path.exists(fmllr_trans_path):
                    raise Exception('Could not find {}'.format(fmllr_trans_path))
                feats += " transform-feats --utt2spk=ark:{} ark,s,cs:{} ark:- ark:- |".format(utt2spk_path, fmllr_trans_path)
        return feats

    def generate_features(self, corpus, data_directory=None, overwrite=False, logger=None, cmvn=True):
        if data_directory is None:
            data_directory = corpus.split_directory()
        if self.directory is None:
            self.directory = data_directory
        if not overwrite and os.path.exists(os.path.join(data_directory, 'feats.0.scp')):
            return
        self.generate_base_features(corpus, logger=logger, compute_cmvn=cmvn)


