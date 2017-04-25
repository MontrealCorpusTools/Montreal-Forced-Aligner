import os
import pickle
import yaml

from tempfile import mkdtemp
from shutil import copy, copyfile, rmtree, make_archive, unpack_archive

# default format for output
FORMAT = "zip"

from . import __version__
from .exceptions import PronunciationAcousticMismatchError, PronunciationOrthographyMismatchError


class Archive(object):
    """
    Class representing data in a directory or archive file (zip, tar,
    tar.gz/tgz)

    Largely duplicated from the prosodylab-aligner
    (https://github.com/prosodylab/Prosodylab-Aligner) archive class.
    """

    def __init__(self, source, is_tmpdir=False):
        self._meta = {}
        self.name, _ = os.path.splitext(os.path.basename(source))
        if os.path.isdir(source):
            self.dirname = os.path.abspath(source)
            self.is_tmpdir = is_tmpdir  # trust caller
        else:
            base = mkdtemp(dir=os.environ.get("TMPDIR", None))
            unpack_archive(source, base)
            (head, tail, _) = next(os.walk(base))
            if not tail:
                raise ValueError("'{}' is empty.".format(source))
            if len(tail) > 1:
                raise ValueError("'{}' is a bomb.".format(source))
            self.dirname = os.path.join(head, tail[0])
            self.is_tmpdir = True  # ignore caller

    @classmethod
    def empty(cls, head):
        """
        Initialize an archive using an empty directory
        """
        base = mkdtemp(dir=os.environ.get("TMPDIR", None))
        source = os.path.join(base, head)
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

    def dump(self, sink, archive_fmt=FORMAT):
        """
        Write archive to disk, and return the name of final archive
        """
        return make_archive(sink, archive_fmt,
                            *os.path.split(self.dirname))

    def __del__(self):
        if self.is_tmpdir:
            rmtree(self.dirname)


class AcousticModel(Archive):
    def add_meta_file(self, aligner):
        with open(os.path.join(self.dirname, 'meta.yaml'), 'w') as f:
            yaml.dump(aligner.meta, f)

    @property
    def meta(self):
        if not self._meta:
            meta_path = os.path.join(self.dirname, 'meta.yaml')
            if not os.path.exists(meta_path):
                self._meta = {'version': '0.9.0',
                              'architecture': 'gmm-hmm'}
            else:
                with open(meta_path, 'r') as f:
                    self._meta = yaml.load(f)
            self._meta['phones'] = set(self._meta.get('phones', []))
        return self._meta

    def add_triphone_model(self, source):
        """
        Add file into archive
        """
        copyfile(os.path.join(source, 'final.mdl'), os.path.join(self.dirname, 'ali-final.mdl'))
        copyfile(os.path.join(source, 'final.occs'), os.path.join(self.dirname, 'ali-final.occs'))
        copyfile(os.path.join(source, 'tree'), os.path.join(self.dirname, 'ali-tree'))

    def add_triphone_fmllr_model(self, source):
        """
        Add file into archive
        """
        copy(os.path.join(source, 'final.mdl'), self.dirname)
        copy(os.path.join(source, 'final.occs'), self.dirname)
        copy(os.path.join(source, 'tree'), self.dirname)

    def export_triphone_model(self, destination):
        """
        """
        os.makedirs(destination, exist_ok=True)
        ali_model_path = os.path.join(self.dirname, 'ali-final.mdl')
        if False and os.path.exists(ali_model_path):
            copyfile(ali_model_path, os.path.join(destination, 'final.mdl'))
            copyfile(os.path.join(self.dirname, 'ali-final.occs'), os.path.join(destination, 'final.occs'))
            copyfile(os.path.join(self.dirname, 'ali-tree'), os.path.join(destination, 'tree'))
        else:
            copyfile(os.path.join(self.dirname, 'final.mdl'), os.path.join(destination, 'final.mdl'))
            copyfile(os.path.join(self.dirname, 'final.occs'), os.path.join(destination, 'final.occs'))
            copyfile(os.path.join(self.dirname, 'tree'), os.path.join(destination, 'tree'))

    def export_triphone_fmllr_model(self, destination):
        """
        """
        os.makedirs(destination, exist_ok=True)
        copy(os.path.join(self.dirname, 'final.mdl'), destination)
        copy(os.path.join(self.dirname, 'final.occs'), destination)
        copy(os.path.join(self.dirname, 'tree'), destination)

    def validate(self, dictionary):
        if isinstance(dictionary, G2PModel):
            if self.meta['phones'] < dictionary.meta['phones']:
                missing_phones = dictionary.meta['phones'] - set(self.meta['phones'])
                raise (PronunciationAcousticMismatchError(missing_phones))
        else:
            if self.meta['phones'] < dictionary.nonsil_phones:
                missing_phones = dictionary.nonsil_phones - set(self.meta['phones'])
                raise (PronunciationAcousticMismatchError(missing_phones))


class G2PModel(Archive):
    def add_meta_file(self, dictionary):
        with open(os.path.join(self.dirname, 'meta.yaml'), 'w') as f:
            meta = {'phones': sorted(dictionary.nonsil_phones),
                    'graphemes': sorted(dictionary.graphemes),
                    'architecture': 'phonetisaurus',
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
                with open(meta_path, 'r') as f:
                    self._meta = yaml.load(f)
            self._meta['phones'] = set(self._meta.get('phones', []))
            self._meta['graphemes'] = set(self._meta.get('graphemes', []))
        return self._meta

    @property
    def fst_path(self):
        return os.path.join(self.dirname, 'model.fst')

    def add_fst_model(self, source):
        """
        Add file into archive
        """
        copyfile(os.path.join(source, 'model.fst'), self.fst_path)

    def export_fst_model(self, destination):
        os.makedirs(destination, exist_ok=True)
        copy(self.fst_path, destination)

    def validate(self, corpus):
        return True  # FIXME add actual validation
