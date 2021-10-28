from __future__ import annotations
import os
import yaml
from typing import TYPE_CHECKING, Union, Collection, Dict, Any
if TYPE_CHECKING:
    from .trainers import BaseTrainer
    from .aligner.adapting import AdaptingAligner
    from .features.config import FeatureConfig
    from .config.train_config import TrainingConfig
    from logging import Logger
    from .dictionary import Dictionary
    from .lm.trainer import LmTrainer

    TrainerType = Union[BaseTrainer, LmTrainer, AdaptingAligner]
    MetaDict = Dict[str, Any]
from typing import Optional
import json

from shutil import copy, copyfile, rmtree, make_archive, unpack_archive, move

from . import __version__
from .exceptions import PronunciationAcousticMismatchError, ModelLoadError, LanguageModelNotFoundError

from .helper import TerminalPrinter

# default format for output
FORMAT = "zip"


class Archive(object):
    """
    Class representing data in a directory or archive file (zip, tar,
    tar.gz/tgz)

    Based on the prosodylab-aligner
    (https://github.com/prosodylab/Prosodylab-Aligner) archive class.
    """

    extensions = ['.zip']

    def __init__(self, source: str, root_directory: Optional[str]=None):
        from .config import TEMP_DIR
        if root_directory is None:
            root_directory = TEMP_DIR
        self.root_directory = root_directory
        self.source = source
        self._meta = {}
        self.name, _ = os.path.splitext(os.path.basename(source))
        if os.path.isdir(source):
            self.dirname = os.path.abspath(source)
        else:
            self.dirname = os.path.join(root_directory, self.name)
            if not os.path.exists(self.dirname):
                os.makedirs(root_directory, exist_ok=True)
                unpack_archive(source, self.dirname)
                files = os.listdir(self.dirname)
                old_dir_path = os.path.join(self.dirname, files[0])
                if len(files) == 1 and os.path.isdir(old_dir_path):  # Backwards compatibility
                    for f in os.listdir(old_dir_path):
                        move(os.path.join(old_dir_path, f), os.path.join(self.dirname, f))
                    os.rmdir(old_dir_path)

    def get_subclass_object(self) -> Union[AcousticModel, G2PModel, LanguageModel, IvectorExtractor]:
        for f in os.listdir(self.dirname):
            if f == 'tree':
                return AcousticModel(self.dirname, self.root_directory)
            elif f == 'phones.sym':
                return G2PModel(self.dirname, self.root_directory)
            elif f.endswith('.arpa'):
                return LanguageModel(self.dirname, self.root_directory)
            elif f == 'final.ie':
                return IvectorExtractor(self.dirname, self.root_directory)
        raise ModelLoadError(self.source)

    @classmethod
    def valid_extension(cls, filename: str) -> bool:
        if os.path.splitext(filename)[1] in cls.extensions:
            return True
        return False

    @classmethod
    def generate_path(cls, root: str, name: str) -> Optional[str]:
        for ext in cls.extensions:
            path = os.path.join(root, name + ext)
            if os.path.exists(path):
                return path
        return None

    def pretty_print(self):
        print(json.dumps(self.meta, indent=4))

    @property
    def meta(self) -> dict:
        if not self._meta:
            meta_path = os.path.join(self.dirname, 'meta.yaml')
            with open(meta_path, 'r', encoding='utf8') as f:
                self._meta = yaml.safe_load(f)
        return self._meta

    def add_meta_file(self, trainer: TrainerType) -> None:
        with open(os.path.join(self.dirname, 'meta.yaml'), 'w', encoding='utf8') as f:
            yaml.dump(trainer.meta, f)

    @classmethod
    def empty(cls, head: str, root_directory: Optional[str]=None) -> Archive:
        """
        Initialize an archive using an empty directory
        """
        from .config import TEMP_DIR
        if root_directory is None:
            root_directory = TEMP_DIR

        os.makedirs(root_directory, exist_ok=True)
        source = os.path.join(root_directory, head)
        os.makedirs(source, exist_ok=True)
        return cls(source)

    def add(self, source: str):
        """
        Add file into archive
        """
        copy(source, self.dirname)

    def __repr__(self) -> str:
        return "{}(dirname={!r})".format(self.__class__.__name__,
                                         self.dirname)

    def clean_up(self) -> None:
        rmtree(self.dirname)

    def dump(self, path: str, archive_fmt: str=FORMAT) -> str:
        """
        Write archive to disk, and return the name of final archive
        """
        return make_archive(os.path.splitext(path)[0], archive_fmt,
                            *os.path.split(self.dirname))


class AcousticModel(Archive):
    files = ['final.mdl', 'final.alimdl', 'final.occs', 'lda.mat', 'tree']
    extensions = ['.zip', '.am']

    def add_meta_file(self, aligner: TrainerType) -> None:
        with open(os.path.join(self.dirname, 'meta.yaml'), 'w', encoding='utf8') as f:
            yaml.dump(aligner.meta, f)

    @property
    def feature_config(self) -> FeatureConfig:
        from .features.config import FeatureConfig
        fc = FeatureConfig()
        fc.update(self.meta['features'])
        return fc

    def adaptation_config(self) -> TrainingConfig:
        from .config.train_config import load_sat_adapt, load_no_sat_adapt
        if self.meta['features']['fmllr']:
            train, align = load_sat_adapt()
        else:
            train, align = load_no_sat_adapt()
        return train

    @property
    def meta(self) -> MetaDict:
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
                    self._meta = yaml.safe_load(f)
                if self._meta['features'] == 'mfcc+deltas':
                    self._meta['features'] = default_features
            if 'uses_lda' not in self._meta:  # Backwards compatibility
                self._meta['uses_lda'] = os.path.exists(os.path.join(self.dirname, 'lda.mat'))
            if 'multilingual_ipa' not in self._meta:
                self._meta['multilingual_ipa'] = False
            if 'uses_sat' not in self._meta:
                self._meta['uses_sat'] = False
            if 'phone_type' not in self._meta:
                self._meta['phone_type'] = 'triphone'
            self._meta['phones'] = set(self._meta.get('phones', []))
            self._meta['has_speaker_independent_model'] = os.path.exists(os.path.join(self.dirname, 'final.alimdl'))
        return self._meta

    def pretty_print(self) -> None:
        printer = TerminalPrinter()
        configuration_data = {
            'Acoustic model': {
                'name': (self.name, 'green'),
                'data': {}
            }
        }
        version_color = 'green'
        if self.meta['version'] != __version__:
            version_color = 'red'
        configuration_data['Acoustic model']['data']['Version'] = (self.meta['version'], version_color)

        if 'citation' in self.meta:
            configuration_data['Acoustic model']['data']['Citation'] = self.meta['citation']
        if 'train_date' in self.meta:
            configuration_data['Acoustic model']['data']['Train date'] = self.meta['train_date']
        configuration_data['Acoustic model']['data']['Architecture'] = self.meta['architecture']
        configuration_data['Acoustic model']['data']['Phone type'] = self.meta['phone_type']
        configuration_data['Acoustic model']['data']['Features'] = {
            'Type': self.meta['features']['type'],
            'Frame shift': self.meta['features']['frame_shift'],
        }
        if self.meta['phones']:
            configuration_data['Acoustic model']['data']['Phones'] = self.meta['phones']
        else:
            configuration_data['Acoustic model']['data']['Phones'] = ('None found!', 'red')


        configuration_data['Acoustic model']['data']['Configuration options'] = {
            'Multilingual IPA': self.meta['multilingual_ipa'],
            'Performs speaker adaptation': self.meta['uses_sat'],
            'Has speaker-independent model': self.meta['has_speaker_independent_model'],
            'Performs LDA on features': self.meta['uses_lda'],
        }
        printer.print_config(configuration_data)


    def add_model(self, source: str) -> None:
        """
        Add file into archive
        """
        for f in self.files:
            if os.path.exists(os.path.join(source, f)):
                copyfile(os.path.join(source, f), os.path.join(self.dirname, f))

    def export_model(self, destination: str) -> None:
        """
        """
        os.makedirs(destination, exist_ok=True)
        for f in self.files:
            if os.path.exists(os.path.join(self.dirname, f)):
                copyfile(os.path.join(self.dirname, f), os.path.join(destination, f))

    def log_details(self, logger:Logger) -> None:
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

    def validate(self, dictionary: Union[Dictionary, G2PModel]) -> None:
        if isinstance(dictionary, G2PModel):
            missing_phones = dictionary.meta['phones'] - set(self.meta['phones'])
        else:
            missing_phones = dictionary.nonsil_phones - set(self.meta['phones'])
        if missing_phones:
            raise (PronunciationAcousticMismatchError(missing_phones))


class IvectorExtractor(Archive):
    """
    Archive for job_name-vector extractors
    """
    model_files = ['final.ie', 'final.ubm', 'final.dubm', 'plda', 'mean.vec', 'trans.mat',
                   'speaker_classifier.mdl', 'speaker_labels.txt']
    extensions = ['.zip', '.ivector']

    def add_model(self, source: str) -> None:
        """
        Add file into archive
        """
        for filename in self.model_files:
            if os.path.exists(os.path.join(source, filename)):
                copyfile(os.path.join(source, filename), os.path.join(self.dirname, filename))

    def export_model(self, destination: str) -> None:
        """
        """
        os.makedirs(destination, exist_ok=True)
        for filename in self.model_files:
            if os.path.exists(os.path.join(self.dirname, filename)):
                copyfile(os.path.join(self.dirname, filename), os.path.join(destination, filename))

    @property
    def feature_config(self) -> FeatureConfig:
        from .features.config import FeatureConfig
        fc = FeatureConfig()
        fc.update(self.meta['features'])
        return fc


class G2PModel(Archive):
    extensions = ['.zip', '.g2p']

    def add_meta_file(self, dictionary: Dictionary, architecture: str) -> None:
        with open(os.path.join(self.dirname, 'meta.yaml'), 'w', encoding='utf8') as f:
            meta = {'phones': sorted(dictionary.nonsil_phones),
                    'graphemes': sorted(dictionary.graphemes),
                    'architecture': architecture,
                    'version': __version__}
            yaml.dump(meta, f)

    @property
    def meta(self) -> dict:
        if not self._meta:
            meta_path = os.path.join(self.dirname, 'meta.yaml')
            if not os.path.exists(meta_path):
                self._meta = {'version': '0.9.0',
                              'architecture': 'phonetisaurus'}
            else:
                with open(meta_path, 'r', encoding='utf8') as f:
                    self._meta = yaml.safe_load(f)
            self._meta['phones'] = set(self._meta.get('phones', []))
            self._meta['graphemes'] = set(self._meta.get('graphemes', []))
        return self._meta

    @property
    def fst_path(self) -> str:
        return os.path.join(self.dirname, 'model.fst')

    @property
    def sym_path(self) -> str:
        return os.path.join(self.dirname, 'phones.sym')

    def add_sym_path(self, source: str) -> None:
        """
        Add file into archive
        """
        if not os.path.exists(self.sym_path):
            copyfile(os.path.join(source, 'phones.sym'), self.sym_path)

    def add_fst_model(self, source: str) -> None:
        """
        Add file into archive
        """
        if not os.path.exists(self.fst_path):
            copyfile(os.path.join(source, 'model.fst'), self.fst_path)

    def export_fst_model(self, destination: str) -> None:
        os.makedirs(destination, exist_ok=True)
        copy(self.fst_path, destination)

    def validate(self, word_list: Collection) -> bool:
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

    arpa_extension = '.arpa'
    extensions = [f".{FORMAT}", arpa_extension, '.lm']

    def __init__(self, source: str, root_directory: Optional[str]=None):
        from .config import TEMP_DIR
        if root_directory is None:
            root_directory = TEMP_DIR
        self.root_directory = root_directory
        self._meta = {}
        self.name, _ = os.path.splitext(os.path.basename(source))
        if os.path.isdir(source):
            self.dirname = os.path.abspath(source)
        elif source.endswith(self.arpa_extension):
            self.dirname = os.path.join(root_directory, self.name)
            if not os.path.exists(self.dirname):
                os.makedirs(self.dirname, exist_ok=True)
            copy(source, self.large_arpa_path)
        elif any(source.endswith(x) for x in self.extensions):
            base = root_directory
            self.dirname = os.path.join(root_directory, self.name)
            if not os.path.exists(self.dirname):
                os.makedirs(root_directory, exist_ok=True)
                unpack_archive(source, base)

    @property
    def decode_arpa_path(self) -> str:
        """
        Use the smallest language model for decoding
        """
        for path in [self.small_arpa_path, self.medium_arpa_path, self.large_arpa_path]:
            if os.path.exists(path):
                return path
        raise LanguageModelNotFoundError()

    @property
    def carpa_path(self) -> str:
        """
        Use the largest language model for rescoring
        """
        for path in [self.large_arpa_path, self.medium_arpa_path, self.small_arpa_path]:
            if os.path.exists(path):
                return path
        raise LanguageModelNotFoundError()

    @property
    def small_arpa_path(self) -> str:
        return os.path.join(self.dirname, self.name + '_small' + self.arpa_extension)

    @property
    def medium_arpa_path(self) -> str:
        return os.path.join(self.dirname, self.name + '_med' + self.arpa_extension)

    @property
    def large_arpa_path(self) -> str:
        return os.path.join(self.dirname, self.name + self.arpa_extension)

    def add_arpa_file(self, arpa_path: str) -> None:
        name = os.path.basename(arpa_path)
        copyfile(arpa_path, os.path.join(self.dirname, name))


class DictionaryModel(Archive):
    extensions = ['.dict', f".{FORMAT}", '.txt', '.yaml']


MODEL_TYPES = {'acoustic': AcousticModel,
               'g2p': G2PModel,
               'dictionary': DictionaryModel,
               'language_model': LanguageModel,
               'ivector': IvectorExtractor}
