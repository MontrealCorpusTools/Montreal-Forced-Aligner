from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Optional
if TYPE_CHECKING:
    from .corpus import TranscribeCorpus
    from .config import SpeakerClassificationConfig, ConfigDict
    from .models import IvectorExtractor, MetaDict

import os
import logging
import shutil
from joblib import load
import numpy as np
import time
from collections import Counter
from .corpus.classes import Speaker, Utterance
from decimal import Decimal
from praatio import textgrid
from .config import TEMP_DIR
from .utils import log_kaldi_errors, parse_logs
from .exceptions import KaldiProcessingError

from .multiprocessing import extract_ivectors, classify_speakers

from .helper import load_scp, save_scp


class SpeakerClassifier(object):
    """
    Class for performing speaker classification

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.TranscribeCorpus`
        Corpus object for the dataset
    ivector_extractor : :class:`~montreal_forced_aligner.models.IvectorExtractor`
        Configuration for alignment
    classification_config : :class:`~montreal_forced_aligner.config.SpeakerClassificationConfig`
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
    def __init__(self, corpus: TranscribeCorpus, ivector_extractor: IvectorExtractor, classification_config:SpeakerClassificationConfig,
                 compute_segments: Optional[bool]=False,
                 num_speakers: Optional[int]= None, cluster: Optional[bool]=False,
                 temp_directory: Optional[str]=None, debug: Optional[bool]=False, verbose: Optional[bool]=False,
                 logger: Optional[logging.Logger]=None):
        self.corpus = corpus
        self.ivector_extractor = ivector_extractor
        self.feature_config = self.ivector_extractor.feature_config
        self.classification_config = classification_config

        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = temp_directory
        os.makedirs(self.temp_directory, exist_ok=True)
        self.debug = debug
        self.compute_segments = compute_segments
        self.verbose = verbose
        if logger is None:
            self.log_file = os.path.join(self.temp_directory, 'speaker_classifier.log')
            self.logger = logging.getLogger('speaker_classifier')
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_file, 'w', 'utf-8')
            handler.setFormatter = logging.Formatter('%(name)s %(message)s')
            self.logger.addHandler(handler)
        else:
            self.logger = logger
        self.classifier = None
        self.speaker_labels = {}
        self.ivectors = {}
        self.num_speakers = num_speakers
        self.cluster = cluster
        self.uses_voiced = False
        self.uses_cmvn = True
        self.uses_splices = False
        self.setup()

    @property
    def classify_directory(self) -> str:
        return os.path.join(self.temp_directory, 'speaker_classification')

    @property
    def data_directory(self) -> str:
        return self.corpus.split_directory

    @property
    def working_directory(self) -> str:
        return self.classify_directory

    @property
    def ie_path(self) -> str:
        return os.path.join(self.working_directory, 'final.ie')

    @property
    def speaker_classification_model_path(self) -> str:
        return os.path.join(self.working_directory, 'speaker_classifier.mdl')

    @property
    def speaker_labels_path(self) -> str:
        return os.path.join(self.working_directory, 'speaker_labels.txt')

    @property
    def model_path(self) -> str:
        return os.path.join(self.working_directory, 'final.mdl')

    @property
    def dubm_path(self) -> str:
        return os.path.join(self.working_directory, 'final.dubm')

    @property
    def working_log_directory(self) -> str:
        return self.log_directory

    @property
    def ivector_options(self) -> MetaDict:
        data = self.ivector_extractor.meta
        data['silence_weight'] = 0.0
        data['posterior_scale'] = 0.1
        data['max_count'] = 100
        data['sil_phones'] = None
        return data

    @property
    def log_directory(self) -> str:
        return os.path.join(self.classify_directory, 'log')

    @property
    def use_mp(self) -> bool:
        return self.classification_config.use_mp

    def setup(self) -> None:
        done_path = os.path.join(self.classify_directory, 'done')
        if os.path.exists(done_path):
            self.logger.info('Classification already done, skipping initialization.')
            return
        dirty_path = os.path.join(self.classify_directory, 'dirty')
        if os.path.exists(dirty_path):
            shutil.rmtree(self.classify_directory)
        log_dir = os.path.join(self.classify_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        self.ivector_extractor.export_model(self.classify_directory)
        try:
            self.corpus.initialize_corpus(None, self.feature_config)
            extract_ivectors(self)
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise

    def classify(self) -> None:
        log_directory = os.path.join(self.classify_directory, 'log')
        dirty_path = os.path.join(self.classify_directory, 'dirty')
        done_path = os.path.join(self.classify_directory, 'done')
        if os.path.exists(done_path):
            self.logger.info('Classification already done, skipping.')
            return
        try:
            if not self.cluster:
                classify_speakers(self)
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

    def load_ivectors(self) -> None:
        self.ivectors = {}
        for j in self.corpus.jobs:
            ivectors_args = j.extract_ivector_arguments(self)
            for ivectors_path in ivectors_args.ivector_paths.values():
                ivec = load_scp(ivectors_path)
                for utt, ivector in ivec.items():
                    ivector = [float(x) for x in ivector]
                    self.ivectors[utt] = ivector

    def load_classifier(self) -> None:
        import warnings
        mdl_path = os.path.join(self.classify_directory, 'speaker_classifier.mdl')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.classifier = load(mdl_path)

        labels_path = os.path.join(self.classify_directory, 'speaker_labels.txt')
        with open(labels_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip().split()
                speaker, speak_ind = line
                self.speaker_labels[int(speak_ind)] = speaker
        speakers = {}
        with open(labels_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip().split()
                speaker, speak_ind = line
                speakers[int(speak_ind)] = speaker

    def cluster_utterances(self) -> None:
        from sklearn.cluster import KMeans
        if not self.ivectors:
            self.load_ivectors()
        x = []
        for k, v in self.ivectors.items():
            x.append(v)
        x = np.array(x)
        clust = KMeans(self.num_speakers).fit(x)
        y = clust.labels_
        for i, u in enumerate(self.ivectors.keys()):
            speaker_name = y[i]
            utterance = self.corpus.utterances[u]
            if speaker_name not in self.corpus.speakers:
                self.corpus.speakers[speaker_name] = Speaker(speaker_name)
            utterance.set_speaker(self.corpus.speakers[speaker_name])

    def classify_utterances(self, utterances: List[Utterance], valid_speakers: Optional[List[Speaker]]=None):
        if not self.classifier:
            self.load_classifier()
        if not self.ivectors:
            self.load_ivectors()
        x = []
        for u in utterances:
            x.append(self.ivectors[u.name])
        x = np.array(x)
        y = self.classifier.predict_proba(x)
        if valid_speakers:
            for i in range(y.shape[1]):
                if self.speaker_labels[i] not in valid_speakers:
                    y[:,i] = 0
        inds = y.argmax(axis=1)
        for i, u in enumerate(utterances):
            speaker_name = self.speaker_labels[inds[i]]
            if speaker_name not in self.corpus.speakers:
                self.corpus.speakers[speaker_name] = Speaker(speaker_name)
            u.set_speaker(self.corpus.speakers[speaker_name])

    def get_classification_stats(self) -> None:
        begin = time.time()
        counts = Counter()
        for speaker in self.corpus.speakers.values():
            counts[speaker] = len(speaker.utterances)

        if self.num_speakers:
            valid_speakers = sorted(counts.keys(), key=lambda x: counts[x])[:self.num_speakers]
        else:
            valid_speakers = [x for x in counts.keys() if counts[x] > 1]
        if not valid_speakers:  # Only single utterance count speakers
            valid_speakers = counts.keys()
        reanalyze_utts = []
        for speaker, c in counts.items():
            if c == 1 or speaker not in valid_speakers:
                utts = speaker.utterances
                for u in utts:
                    reanalyze_utts.append(u)
        if reanalyze_utts:
            self.classify_utterances(reanalyze_utts, valid_speakers)
        self.logger.debug(f'Analyzing stats and reclassification took {time.time() - begin} seconds')

    def export_classification(self, output_directory: str) -> None:
        backup_output_directory = None
        if not self.classification_config.overwrite:
            backup_output_directory = os.path.join(self.classify_directory, 'output')
            os.makedirs(backup_output_directory, exist_ok=True)
        if self.cluster:
            self.cluster_utterances()
        else:
            self.get_classification_stats()

        for file in self.corpus.files.values():
            file.save(output_directory, backup_output_directory)