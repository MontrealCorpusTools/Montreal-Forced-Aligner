from ..multiprocessing import convert_ali_to_textgrids
from .base import BaseAligner


class TrainableAligner(BaseAligner):
    """
    Aligner that aligns and trains acoustics models on a large dataset

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.AlignableCorpus`
        Corpus object for the dataset
    dictionary : :class:`~montreal_forced_aligner.dictionary.Dictionary`
        Dictionary object for the pronunciation dictionary
    training_config : :class:`~montreal_forced_aligner.config.TrainingConfig`
        Configuration to train a model
    align_config : :class:`~montreal_forced_aligner.config.AlignConfig`
        Configuration for alignment
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    call_back : callable, optional
        Specifies a call back function for alignment
    """

    def __init__(self, corpus, dictionary, training_config, align_config, temp_directory=None,
                 call_back=None, debug=False, verbose=False, logger=None, pretrained_aligner=None):
        self.training_config = training_config
        self.pretrained_aligner = pretrained_aligner
        super(TrainableAligner, self).__init__(corpus, dictionary, align_config, temp_directory,
                                               call_back, debug, verbose, logger)

    def setup(self):
        if self.dictionary is not None:
            self.dictionary.set_word_set(self.corpus.word_set)
            self.dictionary.write()
        first_trainer = list(self.training_config.items())[0][1]
        self.corpus.initialize_corpus(self.dictionary, first_trainer.feature_config)

    def save(self, path, root_directory=None):
        """
        Output an acoustic model and dictionary to the specified path

        Parameters
        ----------
        path : str
            Path to save acoustic model and dictionary
        root_directory : str or None
            Path for root directory of temporary files
        """
        self.training_config.values()[-1].save(path, root_directory)
        self.logger.info('Saved model to {}'.format(path))

    @property
    def meta(self):
        from .. import __version__
        data = {'phones': sorted(self.dictionary.nonsil_phones),
                'version': __version__,
                'architecture': self.training_config.values()[-1].architecture,
                'phone_type': self.training_config.values()[-1].phone_type,
                'features': self.align_config.feature_config.params(),
                }
        return data

    def train(self):
        previous = self.pretrained_aligner
        for identifier, trainer in self.training_config.items():
            trainer.debug = self.debug
            trainer.logger = self.logger
            if previous is not None:
                previous.align(trainer.subset)
            trainer.init_training(identifier, self.temp_directory, self.corpus, self.dictionary, previous)
            trainer.train(call_back=print)
            previous = trainer
        previous.align(None)

    def export_textgrids(self, output_directory):
        """
        Export a TextGrid file for every sound file in the dataset
        """
        ali_directory = self.training_config.values()[-1].align_directory
        convert_ali_to_textgrids(self.align_config, output_directory, ali_directory, self.dictionary,
                                 self.corpus, self.corpus.num_jobs, self)
        self.compile_information(ali_directory, output_directory)
