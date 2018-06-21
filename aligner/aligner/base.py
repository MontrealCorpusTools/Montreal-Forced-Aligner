import os
import shutil
import glob
import subprocess
import re
import io
import math
import numpy as np
from tqdm import tqdm
from shutil import copy, copyfile, rmtree, make_archive, unpack_archive
from contextlib import redirect_stdout
from aligner.models import IvectorExtractor
from random import shuffle

from ..helper import thirdparty_binary, make_path_safe, awk_like, filter_scp

from ..multiprocessing import (align, mono_align_equal, compile_train_graphs,
                               acc_stats, tree_stats, convert_alignments,
                               convert_ali_to_textgrids, calc_fmllr,
                               lda_acc_stats,
                               calc_lda_mllt, gmm_gselect, acc_global_stats,
                               gauss_to_post, acc_ivector_stats, get_egs,
                               get_lda_nnet, nnet_train_trans, nnet_train,
                               nnet_align, nnet_get_align_feats, extract_ivectors,
                               compute_prob, get_average_posteriors, relabel_egs)
#from ..accuracy_graph import get_accuracy_graph
                               convert_ali_to_textgrids, calc_fmllr,
                               compile_information)


from ..exceptions import NoSuccessfulAlignments

from .. import __version__

from ..config import TEMP_DIR


class BaseAligner(object):
    '''
    Base aligner class for common aligner functions

    Parameters
    ----------
    corpus : :class:`~aligner.corpus.Corpus`
        Corpus object for the dataset
    dictionary : :class:`~aligner.dictionary.Dictionary`
        Dictionary object for the pronunciation dictionary
    align_config : :class:`~aligner.config.AlignConfig`
        Configuration for alignment
    output_directory : str
        Path to export aligned TextGrids
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    num_jobs : int, optional
        Number of processes to use, defaults to 3
    call_back : callable, optional
        Specifies a call back function for alignment
    '''

    def __init__(self, corpus, dictionary, align_config, output_directory, temp_directory=None, num_jobs=3,
                 call_back=None, debug=False, skip_input=False, verbose=False):
        self.align_config = align_config
        self.corpus = corpus
        self.dictionary = dictionary
        self.output_directory = output_directory
        self.num_jobs = num_jobs
        if self.corpus.num_jobs != num_jobs:
            self.num_jobs = self.corpus.num_jobs
        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = temp_directory
        self.call_back = call_back
        if self.call_back is None:
            self.call_back = print
        self.verbose = verbose
        self.debug = debug
        self.skip_input = skip_input
        self.setup()

    def setup(self):
        self.dictionary.write()
        self.corpus.initialize_corpus(self.dictionary, skip_input=self.skip_input, feature_config=self.align_config.feature_config)
        print(self.corpus.speaker_utterance_info())

    @property
    def meta(self):
        data = {'phones': sorted(self.dictionary.nonsil_phones),
                'version': __version__,
                'architecture': 'gmm-hmm',
                'features': 'mfcc+deltas',
                }
        return data


    def compile_information(self, model_directory):
        issues = compile_information(model_directory, self.corpus, self.num_jobs)
        if issues:
            issue_path = os.path.join(self.output_directory, 'unaligned.txt')
            with open(issue_path, 'w', encoding='utf8') as f:
                for u, r in sorted(issues.items()):
                    f.write('{}\t{}\n'.format(u, r))
            print('There were {} segments/files not aligned. '
                  'Please see {} for more details on why alignment failed for these files.'.format(len(issues),
                                                                                                   issue_path))

    def export_textgrids(self):
        '''
        Export a TextGrid file for every sound file in the dataset
        '''
        if os.path.exists(self.nnet_basic_final_model_path):
            model_directory = self.nnet_basic_directory
        elif os.path.exists(self.tri_fmllr_final_model_path):
            model_directory = self.tri_fmllr_directory
        elif os.path.exists(self.tri_final_model_path):
            model_directory = self.tri_directory
        elif os.path.exists(self.mono_final_model_path):
            model_directory = self.mono_directory

        convert_ali_to_textgrids(self.output_directory, model_directory, self.dictionary,
                                 self.corpus, self.num_jobs)
        self.compile_information(model_directory)

