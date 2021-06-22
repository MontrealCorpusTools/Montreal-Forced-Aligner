import os
import logging

from .. import __version__
from ..multiprocessing import compile_information
from ..config import TEMP_DIR

from ..helper import log_kaldi_errors, load_scp
from ..exceptions import KaldiProcessingError


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
        self.corpus.initialize_corpus(self.dictionary)
        try:
            self.align_config.feature_config.generate_features(self.corpus, logger=self.logger)
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise

    @property
    def meta(self):
        data = {'phones': sorted(self.dictionary.nonsil_phones),
                'version': __version__,
                'architecture': 'gmm-hmm',
                'features': 'mfcc+deltas',
                }
        return data

    def compile_information(self, model_directory, output_directory):
        issues = compile_information(model_directory, self.corpus, self.corpus.num_jobs, self)
        if issues:
            issue_path = os.path.join(output_directory, 'unaligned.txt')
            with open(issue_path, 'w', encoding='utf8') as f:
                for u, r in sorted(issues.items()):
                    f.write('{}\t{}\n'.format(u, r))
            self.logger.warning('There were {} segments/files not aligned.  Please see {} for more details on why '
                                'alignment failed for these files.'.format(len(issues), issue_path))

        log_like = 0
        tot_frames = 0
        for j in range(self.corpus.num_jobs):
            score_path = os.path.join(model_directory, 'ali.{}.scores'.format(j))
            scores = load_scp(score_path, data_type=float)
            for k, v in scores.items():
                log_like += v
                tot_frames += self.corpus.utterance_lengths[k]
        if tot_frames:
            self.logger.debug('Average per frame likelihood (this might not actually mean anything): {}'.format(log_like/tot_frames))
        else:
            self.logger.debug('No files were aligned, this likely indicates serious problems with the aligner.')

    def export_textgrids(self, output_directory):
        """
        Export a TextGrid file for every sound file in the dataset
        """
        raise NotImplementedError
