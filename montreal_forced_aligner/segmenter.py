import os
import shutil
from decimal import Decimal
from praatio import textgrid
from .config import TEMP_DIR
from .helper import log_kaldi_errors, parse_logs
from .exceptions import KaldiProcessingError


class Segmenter(object):
    """
    Class for performing speaker classification

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.TranscribeCorpus`
        Corpus object for the dataset
    segmentation_config : :class:`~montreal_forced_aligner.config.SegmentationConfig`
        Configuration for alignment
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    call_back : callable, optional
        Specifies a call back function for segmentation
    debug : bool
        Flag for running in debug mode, defaults to false
    verbose : bool
        Flag for running in verbose mode, defaults to false
    """

    def __init__(self, corpus, segmentation_config,
                 temp_directory=None, call_back=None, debug=False, verbose=False, logger=None):
        self.corpus = corpus
        self.segmentation_config = segmentation_config

        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = temp_directory
        self.call_back = call_back
        if self.call_back is None:
            self.call_back = print
        self.debug = debug
        self.verbose = verbose
        self.logger = logger
        self.setup()

    @property
    def segmenter_directory(self):
        return os.path.join(self.temp_directory, 'segmentation')

    @property
    def vad_options(self):
        return {'energy_threshold': self.segmentation_config.energy_threshold,
                'energy_mean_scale': self.segmentation_config.energy_mean_scale}

    @property
    def segmentation_options(self):
        return {'max_segment_length': self.segmentation_config.max_segment_length,
                'min_pause_duration': self.segmentation_config.min_pause_duration,
                'snap_boundary_threshold': self.segmentation_config.snap_boundary_threshold,
                'frame_shift': round(self.segmentation_config.feature_config.frame_shift / 1000, 2)}

    @property
    def use_mp(self):
        return self.segmentation_config.use_mp

    def setup(self):
        done_path = os.path.join(self.segmenter_directory, 'done')
        if os.path.exists(done_path):
            self.logger.info('Classification already done, skipping initialization.')
            return
        dirty_path = os.path.join(self.segmenter_directory, 'dirty')
        if os.path.exists(dirty_path):
            shutil.rmtree(self.segmenter_directory)
        log_dir = os.path.join(self.segmenter_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        try:
            self.corpus.initialize_corpus(None, None)
            fc = self.segmentation_config.feature_config
            fc.generate_features(self.corpus, logger=self.logger, cmvn=False)
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise

    def segment(self):
        log_directory = os.path.join(self.segmenter_directory, 'log')
        dirty_path = os.path.join(self.segmenter_directory, 'dirty')
        done_path = os.path.join(self.segmenter_directory, 'done')
        if os.path.exists(done_path):
            self.logger.info('Classification already done, skipping.')
            return
        try:
            fc = self.segmentation_config.feature_config
            fc.compute_vad(self.corpus, logger=self.logger, vad_config=self.vad_options)
            self.corpus.create_vad_segments(self)
            parse_logs(log_directory)
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
        with open(done_path, 'w'):
            pass

    def export_segments(self, output_directory):
        file_dict = {}
        for utt, segment in self.corpus.vad_segments.items():
            filename, utt_begin, utt_end = segment
            utt_begin = Decimal(utt_begin)
            utt_end = Decimal(utt_end)
            if filename not in file_dict:
                file_dict[filename] = {}
            speaker = 'segments'
            text = 'speech'
            if speaker not in file_dict[filename]:
                file_dict[filename][speaker] = []
            file_dict[filename][speaker].append([utt_begin, utt_end, text])
        for filename, speaker_dict in file_dict.items():
            try:
                speaker_directory = os.path.join(output_directory, self.corpus.file_directory_mapping[filename])
            except KeyError:
                speaker_directory = output_directory
            os.makedirs(speaker_directory, exist_ok=True)
            max_time = self.corpus.get_wav_duration(filename)
            tg = textgrid.Textgrid()
            tg.minTimestamp = 0
            tg.maxTimestamp = max_time
            for speaker in sorted(speaker_dict.keys()):
                words = speaker_dict[speaker]
                entry_list = []
                for w in words:
                    if w[1] > max_time:
                        w[1] = max_time
                    entry_list.append(w)
                tier = textgrid.IntervalTier(speaker, entry_list, minT=0, maxT=max_time)
                tg.addTier(tier)
            tg.save(os.path.join(speaker_directory, filename + '.TextGrid'), includeBlankSpaces=True, format='long_textgrid')
