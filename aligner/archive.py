

import os
import pickle

from tempfile import mkdtemp
from shutil import copy, rmtree, make_archive, unpack_archive

# default format for output
FORMAT = "zip"


class Archive(object):

    """
    Class representing data in a directory or archive file (zip, tar,
    tar.gz/tgz)

    Largely duplicated from the prosodylab-aligner
    (https://github.com/prosodylab/Prosodylab-Aligner) archive class.
    """

    def __init__(self, source, is_tmpdir=False):
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
        os.makedirs(source, exist_ok = True)
        return cls(source, True)

    def add(self, source):
        """
        Add file into archive
        """
        copy(source, self.dirname)

    def add_triphone_model(self, source):
        """
        Add file into archive
        """
        copy(os.path.join(source, 'final.mdl'), self.dirname)
        copy(os.path.join(source, 'final.occs'), self.dirname)
        copy(os.path.join(source, 'tree'), self.dirname)

    def export_triphone_model(self, destination):
        """
        Add file into archive
        """
        os.makedirs(destination, exist_ok = True)
        copy(os.path.join(self.dirname, 'final.mdl'), destination)
        copy(os.path.join(self.dirname, 'final.occs'), destination)
        copy(os.path.join(self.dirname, 'tree'), destination)

    def add_dictionary(self, source):
        with open(os.path.join(self.dirname, 'dictionary'), 'wb') as f:
            pickle.dump(source, f)

    def load_dictionary(self):
        dict_path = os.path.join(self.dirname, 'dictionary')
        if not os.path.exists(dict_path):
            return None
        with open(dict_path, 'rb') as f:
            return pickle.load(f)

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
