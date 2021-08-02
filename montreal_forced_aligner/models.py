import os
import yaml

from shutil import copy, copyfile, rmtree, make_archive, unpack_archive

from . import __version__
from .exceptions import PronunciationAcousticMismatchError

# default format for output
FORMAT = "zip"


class Archive(object):
    """
    Class representing data in a directory or archive file (zip, tar,
    tar.gz/tgz)

    Based on the prosodylab-aligner
    (https://github.com/prosodylab/Prosodylab-Aligner) archive class.
    """

    extension = '.zip'

    def __init__(self, source, root_directory=None):
        from .config import TEMP_DIR
        if root_directory is None:
            root_directory = TEMP_DIR
        self.root_directory = root_directory
        self._meta = {}
        self.name, _ = os.path.splitext(os.path.basename(source))
        if os.path.isdir(source):
            self.dirname = os.path.abspath(source)
        else:
            base = root_directory
            self.dirname = os.path.join(root_directory, self.name)
            if not os.path.exists(self.dirname):
                os.makedirs(root_directory, exist_ok=True)
                unpack_archive(source, base)

    @property
    def meta(self):
        if not self._meta:
            meta_path = os.path.join(self.dirname, 'meta.yaml')
            with open(meta_path, 'r', encoding='utf8') as f:
                self._meta = yaml.load(f, Loader=yaml.SafeLoader)
        return self._meta

    def add_meta_file(self, trainer):
        with open(os.path.join(self.dirname, 'meta.yaml'), 'w', encoding='utf8') as f:
            yaml.dump(trainer.meta, f)

    @classmethod
    def empty(cls, head, root_directory=None):
        """
        Initialize an archive using an empty directory
        """
        from .config import TEMP_DIR
        if root_directory is None:
            root_directory = TEMP_DIR

        os.makedirs(root_directory, exist_ok=True)
        source = os.path.join(root_directory, head)
        os.makedirs(source, exist_ok=True)
        return cls(source, True)

    def add(self, source):
        """
        Add file into archive
        """
        copy(source, self.dirname)

    def __repr__(self):
        return "{}(dirname={!r})".format(self.__class__.__name__,
                                         self.dirname)

    def clean_up(self):
        rmtree(self.dirname)

    def dump(self, sink, archive_fmt=FORMAT):
        """
        Write archive to disk, and return the name of final archive
        """
        return make_archive(sink, archive_fmt,
                            *os.path.split(self.dirname))


class AcousticModel(Archive):
    files = ['final.mdl', 'final.occs', 'lda.mat', 'tree']
    def add_meta_file(self, aligner):
        with open(os.path.join(self.dirname, 'meta.yaml'), 'w', encoding='utf8') as f:
            yaml.dump(aligner.meta, f)

    @property
    def feature_config(self):
        from .features.config import FeatureConfig
        fc = FeatureConfig(self.dirname)
        fc.update(self.meta['features'])
        return fc

    def adaptation_config(self):
        from .config.train_config import load_sat_adapt, load_no_sat_adapt
        if self.meta['features']['fmllr']:
            train, align =load_sat_adapt()
        else:
            train, align = load_no_sat_adapt()
        return train

    @property
    def meta(self):
        default_features = {'type': 'mfcc',
                            'use_energy': False,
                            'frame_shift': 10,
                            'pitch': False,
                            'fmllr': True}
        if not self._meta:
            meta_path = os.path.join(self.dirname, 'meta.yaml')
            if not os.path.exists(meta_path):
                self._meta = {'version': '0.9.0',
                              'architecture': 'gmm-hmm',
                              'multilingual_ipa': False,
                              'features': default_features
                              }
            else:
                with open(meta_path, 'r', encoding='utf8') as f:
                    self._meta = yaml.load(f, Loader=yaml.SafeLoader)
                if self._meta['features'] == 'mfcc+deltas':
                    self._meta['features'] = default_features
            if 'uses_lda' not in self._meta:  # Backwards compatibility
                self._meta['uses_lda'] = False
            if 'multilingual_ipa' not in self._meta:
                self._meta['multilingual_ipa'] = False
            if 'uses_sat' not in self._meta:
                self._meta['uses_sat'] = False
            if 'phone_type' not in self._meta:
                self._meta['phone_type'] = 'triphone'
            self._meta['phones'] = set(self._meta.get('phones', []))
        return self._meta

    def add_model(self, source):
        """
        Add file into archive
        """
        for f in self.files:
            if os.path.exists(os.path.join(source, f)):
                copyfile(os.path.join(source, f), os.path.join(self.dirname, f))

    def export_model(self, destination):
        """
        """
        os.makedirs(destination, exist_ok=True)
        for f in self.files:
            if os.path.exists(os.path.join(self.dirname, f)):
                copyfile(os.path.join(self.dirname, f), os.path.join(destination, f))

    def log_details(self, logger):
        logger.debug('')
        logger.debug('====ACOUSTIC MODEL INFO====')
        logger.debug('Acoustic model root directory: ' + self.root_directory)
        logger.debug('Acoustic model dirname: ' + self.dirname)
        meta_path = os.path.join(self.dirname, 'meta.yaml')
        logger.debug('Acoustic model meta path: ' + meta_path)
        if not os.path.exists(meta_path):
            logger.debug('META.YAML DOES NOT EXIST, this may cause issues in validating the model')
        logger.debug('Acoustic model meta information:')
        stream = yaml.dump(self.meta)
        logger.debug(stream)
        logger.debug('')

    def validate(self, dictionary):
        if isinstance(dictionary, G2PModel):
            missing_phones = dictionary.meta['phones'] - set(self.meta['phones'])
        else:
            missing_phones = dictionary.nonsil_phones - set(self.meta['phones'])
        if missing_phones:
            print('dictionary phones: {}'.format(dictionary.nonsil_phones))
            print('model phones: {}'.format(self.meta['phones']))
            raise (PronunciationAcousticMismatchError(missing_phones))


class IvectorExtractor(Archive):
    """
    Archive for i-vector extractors
    """
    model_files = ['final.ie', 'final.ubm', 'final.dubm', 'plda', 'mean.vec', 'trans.mat',
                   'speaker_classifier.mdl', 'speaker_labels.txt']

    def add_model(self, source):
        """
        Add file into archive
        """
        for filename in self.model_files:
            if os.path.exists(os.path.join(source, filename)):
                copyfile(os.path.join(source, filename), os.path.join(self.dirname, filename))

    def export_model(self, destination):
        """
        """
        os.makedirs(destination, exist_ok=True)
        for filename in self.model_files:
            if os.path.exists(os.path.join(self.dirname, filename)):
                copyfile(os.path.join(self.dirname, filename), os.path.join(destination, filename))

    @property
    def feature_config(self):
        from .features.config import FeatureConfig
        fc = FeatureConfig(self.dirname)
        fc.update(self.meta['features'])
        return fc


class G2PModel(Archive):
    def add_meta_file(self, dictionary, architecture):
        with open(os.path.join(self.dirname, 'meta.yaml'), 'w', encoding='utf8') as f:
            meta = {'phones': sorted(dictionary.nonsil_phones),
                    'graphemes': sorted(dictionary.graphemes),
                    'architecture': architecture,
                    'version': __version__}
            yaml.dump(meta, f)

    @property
    def meta(self):
        if not self._meta:
            meta_path = os.path.join(self.dirname, 'meta.yaml')
            if not os.path.exists(meta_path):
                self._meta = {'version': '0.9.0',
                              'architecture': 'phonetisaurus'}
            else:
                with open(meta_path, 'r', encoding='utf8') as f:
                    self._meta = yaml.load(f, Loader=yaml.SafeLoader)
            self._meta['phones'] = set(self._meta.get('phones', []))
            self._meta['graphemes'] = set(self._meta.get('graphemes', []))
        return self._meta

    @property
    def fst_path(self):
        return os.path.join(self.dirname, 'model.fst')

    @property
    def sym_path(self):
        return os.path.join(self.dirname, 'phones.sym')

    def add_sym_path(self, source):
        """
        Add file into archive
        """
        if not os.path.exists(self.sym_path):
            copyfile(os.path.join(source, 'phones.sym'), self.sym_path)

    def add_fst_model(self, source):
        """
        Add file into archive
        """
        if not os.path.exists(self.fst_path):
            copyfile(os.path.join(source, 'model.fst'), self.fst_path)

    def export_fst_model(self, destination):
        os.makedirs(destination, exist_ok=True)
        copy(self.fst_path, destination)

    def validate(self, word_list):
        graphemes = set()
        for w in word_list:
            graphemes.update(w)
        missing_graphemes = graphemes - self.meta['graphemes']
        if missing_graphemes:
            print('WARNING! The following graphemes were not found in the specified G2P model: '
                  '{}'.format(' '.join(sorted(missing_graphemes))))
            return False
        else:
            return True


class LanguageModel(Archive):
    extension = '.arpa'

    def __init__(self, source, root_directory=None):
        from .config import TEMP_DIR
        if root_directory is None:
            root_directory = TEMP_DIR
        self.root_directory = root_directory
        self._meta = {}
        self.name, _ = os.path.splitext(os.path.basename(source))
        if os.path.isdir(source):
            self.dirname = os.path.abspath(source)
        elif source.endswith(FORMAT):
            base = root_directory
            self.dirname = os.path.join(root_directory, self.name)
            if not os.path.exists(self.dirname):
                os.makedirs(root_directory, exist_ok=True)
                unpack_archive(source, base)
        else:
            self.dirname = os.path.join(root_directory, self.name)
            if not os.path.exists(self.dirname):
                os.makedirs(self.dirname, exist_ok=True)
            copy(source, self.large_arpa_path)

    @property
    def decode_arpa_path(self):
        """
        Use the smallest language model for decoding
        """
        for path in [self.small_arpa_path, self.medium_arpa_path, self.large_arpa_path]:
            if os.path.exists(path):
                return path
        raise Exception('Could not find a suitable language model')

    @property
    def carpa_path(self):
        """
        Use the largest language model for rescoring
        """
        for path in [self.large_arpa_path, self.medium_arpa_path, self.small_arpa_path]:
            if os.path.exists(path):
                return path
        raise Exception('Could not find a suitable language model')

    @property
    def small_arpa_path(self):
        return os.path.join(self.dirname, self.name + '_small' + self.extension)

    @property
    def medium_arpa_path(self):
        return os.path.join(self.dirname, self.name + '_med' + self.extension)

    @property
    def large_arpa_path(self):
        return os.path.join(self.dirname, self.name + self.extension)

    def add_arpa_file(self, arpa_path):
        name = os.path.basename(arpa_path)
        copyfile(arpa_path, os.path.join(self.dirname, name))
