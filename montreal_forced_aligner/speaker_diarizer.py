import os
from .config import TEMP_DIR


class SpeakerDiarizer(object):
    """
    Class for performing speaker diarization

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.TranscribeCorpus`
        Corpus object for the dataset
    ivector_extractor : :class:`~montreal_forced_aligner.models.IvectorExtractor`
        Configuration for alignment
    diarization_config : :class:`~montreal_forced_aligner.config.DiarizationConfig`
        Configuration for alignment
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    call_back : callable, optional
        Specifies a call back function for diarization
    debug : bool
        Flag for running in debug mode, defaults to false
    verbose : bool
        Flag for running in verbose mode, defaults to false
    """
    def __init__(self, corpus, ivector_extractor, diarization_config, compute_segments=False,
                 temp_directory=None, call_back=None, debug=False, verbose=False, logger=None):
        self.corpus = corpus
        self.ivector_extractor = ivector_extractor

        self.ivector_extractor.export_model(self.diarize_directory)
        self.diarization_config = diarization_config

        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = temp_directory
        self.call_back = call_back
        if self.call_back is None:
            self.call_back = print
        self.debug = debug
        self.compute_segments = compute_segments
        self.verbose = verbose
        self.logger = logger

        self.setup()

    @property
    def diarize_directory(self):
        return os.path.join(self.temp_directory, 'speaker_diarization')

    def setup(self):
        self.corpus.initialize_corpus()
        # Compute VAD if compute_segments
        config = self.ivector_extractor.meta
        fc = self.ivector_extractor.feature_config
        fc.compute_vad(self.corpus, logger=self.logger)
        # Subsegment existing segments from TextGrids or VAD
        if self.corpus.segments:
            self.corpus.create_subsegments()
        fc.generate_ivector_extract_features(self.corpus, apply_cmn=config['apply_cmn'], logger=self.logger)
        # Sliding window CMVN over segments
        # Extract ivectors
        pass

    def diarize(self):

        pass
