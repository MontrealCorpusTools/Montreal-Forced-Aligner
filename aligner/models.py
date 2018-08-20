import os
import pickle
import yaml
import glob

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

    extension = '.zip'

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
            name = tail[0]
            if len(tail) > 1:
                if tail[0] != '__MACOSX':   # Zipping from Mac adds a directory
                    raise ValueError("'{}' is a bomb.".format(source))
                else:
                    name = tail[1]
            self.dirname = os.path.join(head, name)
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
        with open(os.path.join(self.dirname, 'meta.yaml'), 'w', encoding='utf8') as f:
            yaml.dump(aligner.meta, f)

    @property
    def feature_config(self):
        from .features.config import FeatureConfig
        fc = FeatureConfig(self.dirname)
        fc.update(self._meta['features'])
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
                    self._meta = yaml.load(f)
                if self._meta['features'] == 'mfcc+deltas':
                    self._meta['features'] = default_features
            if 'uses_lda' not in self._meta: # Backwards compat
                self._meta['uses_lda'] = False
            if 'uses_sat' not in self._meta:
                self._meta['uses_sat'] = False
            if 'phone_type' not in self._meta:
                self._meta['phone_type'] = 'triphone'
            self._meta['phones'] = set(self._meta.get('phones', []))
        return self._meta

    def add_lda_matrix(self, source):
        copyfile(os.path.join(source, 'lda.mat'), os.path.join(self.dirname, 'lda.mat'))

    def add_ivector_model(self, source):
        copyfile(os.path.join(source, 'final.ie'), os.path.join(self.dirname, 'final.ie'))
        copyfile(os.path.join(source, 'final.dubm'), os.path.join(self.dirname, 'final.dubm'))

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
            #print('dictionary phones: {}'.format(dictionary.meta['phones']))
            print('dictionary phones: {}'.format(dictionary.nonsil_phones))
            print('model phones: {}'.format(self.meta['phones']))
            raise (PronunciationAcousticMismatchError(missing_phones))


class G2PModel(Archive):
    def add_meta_file(self, dictionary):
        with open(os.path.join(self.dirname, 'meta.yaml'), 'w', encoding='utf8') as f:
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
                with open(meta_path, 'r', encoding='utf8') as f:
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

class IvectorExtractor(Archive):
    '''
    Archive for i-vector extractors (used with DNNs)
    '''
    def add_meta_file(self, trainer):
        with open(os.path.join(self.dirname, 'meta.yaml'), 'w', encoding='utf8') as f:
            yaml.dump(trainer.meta, f)

    @property
    def meta(self):
        if not self._meta:
            meta_path = os.path.join(self.dirname, 'meta.yaml')
            with open(meta_path, 'r', encoding='utf8') as f:
                self._meta = yaml.load(f)
        return self._meta

    def add_model(self, source):
        """
        Add file into archive
        """
        copyfile(os.path.join(source, 'final.ie'), os.path.join(self.dirname, 'final.ie'))
        copyfile(os.path.join(source, 'final.dubm'), os.path.join(self.dirname, 'final.dubm'))
        copyfile(os.path.join(source, 'lda.mat'), os.path.join(self.dirname, 'lda.mat'))

    def export_model(self, destination):
        os.makedirs(destination, exist_ok=True)
        copy(os.path.join(self.dirname, 'final.ie'), destination)           # i-vector extractor itself
        copy(os.path.join(self.dirname, 'final.dubm'), destination)         # Diag UBM itself
        copy(os.path.join(self.dirname, 'lda.mat'), destination)          # LDA matrix


        # Write a "cmvn config" file (this is blank in the actual kaldi code, but it needs the argument passed)
        cmvn_config = os.path.join(destination, 'online_cmvn.conf')
        with open(cmvn_config, 'w', newline='') as cconf:
            cconf.write("")

        # Write a "splice config" file
        splice_config = os.path.join(destination, 'splice.conf')
        with open(splice_config, 'w', newline='') as sconf:
            sconf.write('--left-context={}'.format(self.meta['splice_left_context']))
            sconf.write('\n')
            sconf.write('--right-context={}'.format(self.meta['splice_right_context']))

        # Write a "config" file to input to the extraction binary
        ext_config = os.path.join(destination, 'ivector_extractor.conf')
        with open(ext_config, 'w', newline='') as ieconf:
            ieconf.write('--cmvn-config={}\n'.format(cmvn_config))
            ieconf.write('--ivector-period={}\n'.format(self.meta['ivector_period']))
            ieconf.write('--splice-config={}\n'.format(splice_config))
            ieconf.write('--lda-matrix={}\n'.format(os.path.join(destination, 'lda.mat')))
            ieconf.write('--global-cmvn-stats={}\n'.format(os.path.join(destination, 'global_cmvn.stats')))
            ieconf.write('--diag-ubm={}\n'.format(os.path.join(destination, 'final.dubm')))
            ieconf.write('--ivector-extractor={}\n'.format(os.path.join(destination, 'final.ie')))
            ieconf.write('--num-gselect={}\n'.format(self.meta['num_gselect']))
            ieconf.write('--min-post={}\n'.format(self.meta['min_post']))
            ieconf.write('--posterior-scale={}\n'.format(self.meta['posterior_scale']))
            ieconf.write('--max-remembered-frames=1000\n')
            ieconf.write('--max-count={}\n'.format(0))
        return ext_config
