import os
import yaml
from .base_config import BaseConfig, ConfigError, DEFAULT_PUNCTUATION, DEFAULT_CLITIC_MARKERS, \
    DEFAULT_COMPOUND_MARKERS, DEFAULT_DIGRAPHS, DEFAULT_STRIP_DIACRITICS
from ..features.config import FeatureConfig


class AlignConfig(BaseConfig):
    """
    Configuration object for alignment

    Attributes
    ----------
    transition_scale : float
        Transition scale, defaults to 1.0
    acoustic_scale : float
        Acoustic scale, defaults to 0.1
    self_loop_scale : float
        Self-loop scale, defaults to 0.1
    disable_sat : bool
        Flag for disabling speaker adaptation, defaults to False
    feature_config : :class:`~montreal_forced_aligner.features.config.FeatureConfig`
        Configuration object for feature generation
    boost_silence : float
        Factor to boost silence probabilities, 1.0 is no boost or reduction
    beam : int
        Size of the beam to use in decoding, defaults to 10
    retry_beam : int
        Size of the beam to use in decoding if it fails with the initial beam width, defaults to 40
    data_directory : str
        Path to save feature files
    fmllr_update_type : str
        Type of update for fMLLR, defaults to full
    use_mp : bool
        Flag for whether to use multiprocessing in feature generation
    """

    def __init__(self, feature_config):
        self.transition_scale = 1.0
        self.acoustic_scale = 0.1
        self.self_loop_scale = 0.1
        self.disable_sat = False
        self.feature_config = feature_config
        self.boost_silence = 1.0
        self.beam = 10
        self.retry_beam = 40
        self.data_directory = None # Gets set later
        self.fmllr_update_type = 'full'
        self.punctuation = DEFAULT_PUNCTUATION
        self.clitic_markers = DEFAULT_CLITIC_MARKERS
        self.compound_markers = DEFAULT_COMPOUND_MARKERS
        self.strip_diacritics = DEFAULT_STRIP_DIACRITICS
        self.digraphs = DEFAULT_DIGRAPHS
        self.use_mp = True
        self.use_fmllr_mp = False
        self.multilingual_ipa = False

    @property
    def feature_file_base_name(self):
        return self.feature_config.feature_id

    @property
    def align_options(self):
        return {'transition_scale': self.transition_scale,
                'acoustic_scale': self.acoustic_scale,
                'self_loop_scale': self.self_loop_scale,
                'beam': self.beam,
                'retry_beam': self.retry_beam,
                }

    def update(self, data):
        for k, v in data.items():
            if k == 'use_mp':
                self.feature_config.use_mp = v
            elif k in ['punctuation', 'clitic_markers', 'compound_markers']:
                if not v:
                    continue
                if '-' in v:
                    v = '-' + v.replace('-', '')
                if ']' in v and r'\]' not in v:
                    v = v.replace(']', r'\]')
            elif not hasattr(self, k):
                raise ConfigError('No field found for key {}'.format(k))
            setattr(self, k, v)

    def update_from_args(self, args):
        super(AlignConfig, self).update_from_args(args)
        self.feature_config.update_from_args(args)
        if self.retry_beam <= self.beam:
            self.retry_beam = self.beam * 4


def align_yaml_to_config(path):
    with open(path, 'r', encoding='utf8') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        global_params = {}
        feature_config = FeatureConfig()
        for k, v in data.items():
            if k == 'features':
                feature_config.update(v)
            else:
                global_params[k] = v
        align_config = AlignConfig(feature_config)
        align_config.update(global_params)
        if align_config.beam >= align_config.retry_beam:
            raise ConfigError('Retry beam must be greater than beam.')
        return align_config


def load_basic_align():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    align_config = align_yaml_to_config(os.path.join(base_dir, 'basic_align.yaml'))
    return align_config