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
    def add_meta_file(self, aligner):
        with open(os.path.join(self.dirname, 'meta.yaml'), 'w', encoding='utf8') as f:
            yaml.dump(aligner.meta, f)

    @property
    def feature_config(self):
        from .features.config import FeatureConfig
        fc = FeatureConfig(self.dirname)
        fc.update(self.meta['features'])
        return fc

    @property
    def meta(self):
        default_features = {'type': 'mfcc',
                            'use_energy': False,
                            'frame_shift': 10,
                            'pitch': False}
        if not self._meta:
            meta_path = os.path.join(self.dirname, 'meta.yaml')
            if not os.path.exists(meta_path):
                self._meta = {'version': '0.9.0',
                              'architecture': 'gmm-hmm',
                              'features': default_features
                              }
            else:
                with open(meta_path, 'r', encoding='utf8') as f:
                    self._meta = yaml.load(f, Loader=yaml.SafeLoader)
                if self._meta['features'] == 'mfcc+deltas':
                    self._meta['features'] = default_features
            if 'uses_lda' not in self._meta:  # Backwards compatibility
                self._meta['uses_lda'] = False
            if 'uses_sat' not in self._meta:
                self._meta['uses_sat'] = False
            if 'phone_type' not in self._meta:
                self._meta['phone_type'] = 'triphone'
            self._meta['phones'] = set(self._meta.get('phones', []))
        return self._meta

    def add_lda_matrix(self, source):
        copyfile(os.path.join(source, 'lda.mat'), os.path.join(self.dirname, 'lda.mat'))

    def add_model(self, source):
        """
        Add file into archive
        """
        copyfile(os.path.join(source, 'final.mdl'), os.path.join(self.dirname, 'final.mdl'))
        if os.path.exists(os.path.join(source, 'final.occs')):
            copyfile(os.path.join(source, 'final.occs'), os.path.join(self.dirname, 'final.occs'))
        copyfile(os.path.join(source, 'tree'), os.path.join(self.dirname, 'tree'))

    def export_model(self, destination):
        """
        """
        os.makedirs(destination, exist_ok=True)
        copyfile(os.path.join(self.dirname, 'final.mdl'), os.path.join(destination, 'final.mdl'))
        if os.path.exists(os.path.join(self.dirname, 'final.occs')):
            copyfile(os.path.join(self.dirname, 'final.occs'), os.path.join(destination, 'final.occs'))
        copyfile(os.path.join(self.dirname, 'tree'), os.path.join(destination, 'tree'))

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
    model_files = ['final.ie', 'final.ubm', 'plda', 'mean.vec', 'trans.mat',
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
            print(meta_path)
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
            copy(source, self.arpa_path)

    @property
    def arpa_path(self):
        print(os.listdir(self.dirname))
        return os.path.join(self.dirname, self.name + self.extension)

    def add_arpa_file(self, arpa_path):
        name = os.path.basename(arpa_path)
        copyfile(arpa_path, os.path.join(self.dirname, name))
