import os
import shutil
import subprocess
from .config import TEMP_DIR
from .helper import thirdparty_binary, make_path_safe, log_kaldi_errors, parse_logs
from .exceptions import KaldiProcessingError

from .multiprocessing import extract_ivectors, transform_ivectors, score_plda, cluster


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

    @property
    def ivector_options(self):
        return self.ivector_extractor.meta

    @property
    def plda_options(self):
        return {'target_energy': self.diarization_config.target_energy}

    @property
    def cluster_options(self):
        return {
            'cluster_threshold': self.diarization_config.cluster_threshold,
            'max_speaker_fraction': self.diarization_config.max_speaker_fraction,
            'first_pass_max_utterances': self.diarization_config.first_pass_max_utterances,
            'rttm_channel': self.diarization_config.rttm_channel,
            'read_costs': self.diarization_config.read_costs,
        }

    @property
    def use_mp(self):
        return self.diarization_config.use_mp

    def setup(self):
        done_path = os.path.join(self.diarize_directory, 'done')
        if os.path.exists(done_path):
            self.logger.info('Diarization already done, skipping initialization.')
            return
        dirty_path = os.path.join(self.diarize_directory, 'dirty')
        if os.path.exists(dirty_path):
            shutil.rmtree(self.diarize_directory)
        log_dir = os.path.join(self.diarize_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        self.ivector_extractor.export_model(self.diarize_directory)
        try:
            self.corpus.initialize_corpus()
            # Compute VAD if compute_segments
            config = self.ivector_extractor.meta
            fc = self.ivector_extractor.feature_config
            fc.generate_base_features(self.corpus, logger=self.logger, compute_cmvn=False)
            fc.compute_vad(self.corpus, logger=self.logger)
            # create VAD segments
            self.corpus.create_vad_segments(fc)
            # Subsegment existing segments from TextGrids or VAD
            self.corpus.create_subsegments(fc)
            # Sliding window CMVN over segments
            fc.generate_ivector_extract_features(self.corpus, apply_cmn=config['apply_cmn'], logger=self.logger)
            # Extract ivectors
            extract_ivectors(self.diarize_directory, self.corpus.split_directory(), self, self.corpus.num_jobs)
            if False:
                ivector_path = os.path.join(self.diarize_directory, 'ivectors.scp')
                mean_path = os.path.join(self.diarize_directory, 'mean.vec')
                transform_path = os.path.join(self.diarize_directory, 'trans.mat')
                with open(ivector_path, 'w', encoding='utf8') as out_f:
                    for i in range(self.corpus.num_jobs):
                        with open(os.path.join(self.diarize_directory, 'ivectors.{}.scp'.format(i)), 'r', encoding='utf8') as in_f:
                            for line in in_f:
                                out_f.write(line)
                with open(os.path.join(log_dir, 'mean.log'), 'w', encoding='utf8') as log_file:
                    mean_proc = subprocess.Popen([thirdparty_binary('ivector-mean'), 'scp:'+ivector_path, mean_path], stderr=log_file)
                    mean_proc.communicate()
                    est_proc = subprocess.Popen([thirdparty_binary('est-pca'),
                                                 '--read-vectors=true', '--normalize-mean=false',
                                                 '--normalize-variance=true',
                                                 '--dim={}'.format(self.diarization_config.pca_dimension),
                                                 'scp:'+ivector_path, transform_path], stderr=log_file)
                    est_proc.communicate()
                parse_logs(log_dir)
            transform_ivectors(self.diarize_directory, self, self.corpus.num_jobs)
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
            raise

    def diarize(self):
        # Train PLDA models
        log_directory = os.path.join(self.diarize_directory, 'log')
        dirty_path = os.path.join(self.diarize_directory, 'dirty')
        done_path = os.path.join(self.diarize_directory, 'done')
        if os.path.exists(done_path):
            self.logger.info('Diarization already done, skipping.')
            return

        try:
            score_plda(self.diarize_directory, self.corpus.split_directory(), self, self.corpus.num_jobs)
            cluster(self.diarize_directory, self.corpus.split_directory(), self, self.corpus.num_jobs)
            parse_logs(log_directory)
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
            raise
