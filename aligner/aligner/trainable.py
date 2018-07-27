import os
import shutil
import subprocess
import re
import math
from tqdm import tqdm

from ..helper import thirdparty_binary, make_path_safe

from ..multiprocessing import (align, mono_align_equal, compile_train_graphs,
                               acc_stats, tree_stats, convert_alignments,
                               convert_ali_to_textgrids)

from ..exceptions import NoSuccessfulAlignments

from .base import BaseAligner

from ..models import AcousticModel


class TrainableAligner(BaseAligner):
    '''
    Aligner that aligns and trains acoustics models on a large dataset

    Parameters
    ----------
    corpus : :class:`~aligner.corpus.Corpus`
        Corpus object for the dataset
    dictionary : :class:`~aligner.dictionary.Dictionary`
        Dictionary object for the pronunciation dictionary
    training_config : :class:`~aligner.config.TrainingConfig`
        Configuration to train a model
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

    def __init__(self, corpus, dictionary, training_config, align_config, output_directory, temp_directory=None,
                 call_back=None, debug=False, verbose=False):
        super(TrainableAligner, self).__init__(corpus, dictionary, align_config, output_directory, temp_directory,
                                               call_back, debug, verbose)
        self.training_config = training_config

    def save(self, path):
        '''
        Output an acoustic model and dictionary to the specified path

        Parameters
        ----------
        path : str
            Path to save acoustic model and dictionary
        '''
        self.training_config.values()[-1].save(path)
        print('Saved model to {}'.format(path))

    @property
    def meta(self):
        from .. import __version__
        data = {'phones': sorted(self.dictionary.nonsil_phones),
                'version': __version__,
                'architecture': self.training_config.values()[-1].architecture,
                'phone_type': self.training_config.values()[-1].phone_type,
                'features': self.training_config.feature_config.params(),
                }
        return data

    def train(self):
        previous = None
        for identifier, trainer in self.training_config.items():
            if previous is not None:
                previous.align(trainer.subset)
            trainer.init_training(identifier, self.temp_directory, self.corpus, self.dictionary, previous)
            trainer.train(call_back=print)
            previous = trainer
        previous.align(None)

    def export_textgrids(self):
        '''
        Export a TextGrid file for every sound file in the dataset
        '''
        ali_directory = self.training_config.values()[-1].align_directory
        convert_ali_to_textgrids(self.align_config, self.output_directory, ali_directory, self.dictionary,
                                 self.corpus, self.corpus.num_jobs)
        self.compile_information(ali_directory)
