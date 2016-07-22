import os

from .base import BaseAligner, TEMP_DIR, TriphoneFmllrConfig, TriphoneConfig

from ..dictionary import Dictionary

class PretrainedAligner(BaseAligner):
    '''
    Class for aligning a dataset using a pretrained acoustic model

    Parameters
    ----------
    archive : :class:`~aligner.archive.Archive`
        Archive containing the acoustic model and pronunciation dictionary
    corpus : :class:`~aligner.corpus.Corpus`
        Corpus object for the dataset
    output_directory : str
        Path to directory to save TextGrids
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    num_jobs : int, optional
        Number of processes to use, defaults to 3
    call_back : callable, optional
        Specifies a call back function for alignment
    '''
    def __init__(self, archive, corpus, output_directory,
                    temp_directory = None, num_jobs = 3, call_back = None):

        if temp_directory is None:
            temp_directory = TEMP_DIR
        self.temp_directory = temp_directory
        self.output_directory = output_directory
        self.corpus = corpus

        self.dictionary = Dictionary(archive.dictionary_path, os.path.join(temp_directory, 'dictionary'))

        self.dictionary.write()
        archive.export_triphone_model(self.tri_directory)

        if self.corpus.num_jobs != num_jobs:
            num_jobs = self.corpus.num_jobs
        self.num_jobs = num_jobs
        self.call_back = call_back
        if self.call_back is None:
            self.call_back = print
        self.verbose = False
        self.tri_fmllr_config = TriphoneFmllrConfig(**{'realign_iters': [1, 2],
                                                        'fmllr_iters': [1],
                                                        'num_iters': 3})
        self.tri_config = TriphoneConfig()

    def do_align(self):
        '''
        Perform alignment while calculating speaker transforms (fMLLR estimation)
        '''
        self.train_tri_fmllr()
