import os
import logging
import shutil

from .. import __version__
from ..multiprocessing import compile_information
from ..config import TEMP_DIR

from ..multiprocessing import convert_ali_to_textgrids
from ..dictionary import MultispeakerDictionary


class BaseAligner(object):
    """
    Base aligner class for common aligner functions

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.AlignableCorpus`
        Corpus object for the dataset
    dictionary : :class:`~montreal_forced_aligner.dictionary.Dictionary`
        Dictionary object for the pronunciation dictionary
    align_config : :class:`~montreal_forced_aligner.config.AlignConfig`
        Configuration for alignment
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    call_back : callable, optional
        Specifies a call back function for alignment
    debug : bool
        Flag for running in debug mode, defaults to false
    verbose : bool
        Flag for running in verbose mode, defaults to false
    """

    def __init__(self, corpus, dictionary, align_config, temp_directory=None,
                 call_back=None, debug=False, verbose=False, logger=None):
        self.align_config = align_config
        self.corpus = corpus
        self.dictionary = dictionary
        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = temp_directory
        os.makedirs(self.temp_directory, exist_ok=True)
        self.log_file = os.path.join(self.temp_directory, 'aligner.log')
        if logger is None:
            self.logger = logging.getLogger('corpus_setup')
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_file, 'w', 'utf-8')
            handler.setFormatter = logging.Formatter('%(name)s %(message)s')
            self.logger.addHandler(handler)
        else:
            self.logger = logger
        self.call_back = call_back
        if self.call_back is None:
            self.call_back = print
        self.verbose = verbose
        self.debug = debug
        self.setup()

    def setup(self):
        self.dictionary.write()
        self.corpus.initialize_corpus(self.dictionary, self.align_config.feature_config)

    @property
    def use_mp(self):
        return self.align_config.use_mp

    @property
    def meta(self):
        data = {'phones': sorted(self.dictionary.nonsil_phones),
                'version': __version__,
                'architecture': 'gmm-hmm',
                'features': 'mfcc+deltas',
                }
        return data

    def dictionaries_for_job(self, job_name):
        if isinstance(self.dictionary, MultispeakerDictionary):
            dictionary_names = []
            for name in self.dictionary.dictionary_mapping.keys():
                if os.path.exists(os.path.join(self.corpus.split_directory(), 'utt2spk.{}.{}'.format(job_name, name))):
                    dictionary_names.append(name)
            return dictionary_names
        return None

    @property
    def align_directory(self):
        return os.path.join(self.temp_directory, 'align')

    @property
    def backup_output_directory(self):
        return os.path.join(self.align_directory, 'textgrids')

    def compile_information(self, output_directory):
        model_directory = self.align_directory
        issues, average_log_like = compile_information(model_directory, self.corpus, self.corpus.num_jobs, self)
        errors_path = os.path.join(output_directory, 'output_errors.txt')
        if os.path.exists(errors_path):
            self.logger.warning('There were errors when generating the textgrids. See the output_errors.txt in the '
                                'output directory for more details.')
        if issues:
            issue_path = os.path.join(output_directory, 'unaligned.txt')
            with open(issue_path, 'w', encoding='utf8') as f:
                for u, r in sorted(issues.items()):
                    f.write('{}\t{}\n'.format(u, r))
            self.logger.warning('There were {} segments/files not aligned.  Please see {} for more details on why '
                                'alignment failed for these files.'.format(len(issues), issue_path))
        if os.path.exists(self.backup_output_directory) and os.listdir(self.backup_output_directory):
            self.logger.info(f'Some TextGrids were not output in the output directory to avoid overwriting existing files. '
                             f'You can find them in {self.backup_output_directory}, and if you would like to disable this '
                             f'behavior, you can rerun with the --overwrite flag or run `mfa configure --always_overwrite`.')

    def export_textgrids(self, output_directory):
        """
        Export a TextGrid file for every sound file in the dataset
        """
        if os.path.exists(self.backup_output_directory):
            shutil.rmtree(self.backup_output_directory, ignore_errors=True)
        convert_ali_to_textgrids(self.align_config, output_directory, self.align_directory, self.dictionary,
                                 self.corpus, self.corpus.num_jobs)
        self.compile_information(output_directory)
