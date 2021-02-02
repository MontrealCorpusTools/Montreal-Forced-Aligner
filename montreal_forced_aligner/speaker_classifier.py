import os
import shutil
import subprocess
from joblib import load
import numpy as np
import time
from .config import TEMP_DIR
from .helper import thirdparty_binary, make_path_safe, log_kaldi_errors, parse_logs
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
    def __init__(self, corpus, ivector_extractor, classification_config, compute_segments=False,
                 num_speakers = None, cluster=False,
                 temp_directory=None, call_back=None, debug=False, verbose=False, logger=None):
        self.corpus = corpus
        self.ivector_extractor = ivector_extractor
        self.feature_config = self.ivector_extractor.feature_config
        self.classification_config = classification_config

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
        self.classifier = None
        self.speaker_labels = {}
        self.ivectors = {}
        self.num_speakers = num_speakers
        self.cluster = cluster
        self.setup()

    @property
    def classify_directory(self):
        return os.path.join(self.temp_directory, 'speaker_classification')

    @property
    def ivector_options(self):
        return self.ivector_extractor.meta

    @property
    def use_mp(self):
        return self.classification_config.use_mp

    def setup(self):
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
            self.corpus.initialize_corpus()
            self.feature_config.generate_features(self.corpus, logger=self.logger, cmvn=False)
            extract_ivectors(self.classify_directory, self.corpus.split_directory(), self, self.corpus.num_jobs)
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
            raise

    def classify(self):
        log_directory = os.path.join(self.classify_directory, 'log')
        dirty_path = os.path.join(self.classify_directory, 'dirty')
        done_path = os.path.join(self.classify_directory, 'done')
        if os.path.exists(done_path):
            self.logger.info('Classification already done, skipping.')
            return
        try:
            if not self.cluster:
                classify_speakers(self.classify_directory, self, self.corpus.num_jobs)
            parse_logs(log_directory)
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
            raise
        with open(done_path, 'w'):
            pass

    def load_ivectors(self):
        self.ivectors = {}
        for j in range(self.corpus.num_jobs):
            ivectors_path = os.path.join(self.classify_directory, 'ivectors.{}'.format(j))
            ivec = load_scp(ivectors_path)
            for utt, ivector in ivec.items():
                ivector = [float(x) for x in ivector]
                self.ivectors[utt] = ivector

    def load_classifier(self):
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

    def cluster_utterances(self):
        from sklearn.cluster import KMeans
        if not self.ivectors:
            self.load_ivectors()
        x = []
        for k, v in self.ivectors.items():
            x.append(v)
        x = np.array(x)
        clust = KMeans(self.num_speakers).fit(x)
        y = clust.labels_
        spk2utt_path = os.path.join(self.classify_directory, 'spk2utt')
        utt2spk_path = os.path.join(self.classify_directory, 'utt2spk')
        utt2spk = {}
        spk2utt = {}
        for i, u in enumerate(self.ivectors.keys()):
            speaker = y[i]
            utt2spk[u] = speaker
            if speaker not in spk2utt:
                spk2utt[speaker] = []
            spk2utt[speaker].append(speaker)
        save_scp(([k, v] for k,v in spk2utt.items()), spk2utt_path)
        save_scp(([k, v] for k,v in utt2spk.items()), utt2spk_path)

    def classify_utterances(self, utterances, valid_speakers=None):
        if not self.classifier:
            self.load_classifier()
        if not self.ivectors:
            self.load_ivectors()
        x = []
        for u in utterances:
            x.append(self.ivectors[u])
        x = np.array(x)
        y = self.classifier.predict_proba(x)
        if valid_speakers:
            for i in range(y.shape[1]):
                if self.speaker_labels[i] not in valid_speakers:
                    y[:,i] = 0
        output = {}
        inds = y.argmax(axis=1)
        for i, u in enumerate(utterances):
            output[u] = self.speaker_labels[inds[i]]
        return output

    def get_classification_stats(self):
        begin = time.time()
        from collections import Counter
        counts = Counter()
        utt2spk = {}
        spk2utt = {}
        for j in range(self.corpus.num_jobs):
            utt2spk_path = os.path.join(self.classify_directory, 'utt2spk.{}'.format(j))
            utt2spk.update(load_scp(utt2spk_path))
        for j in range(self.corpus.num_jobs):
            spk2utt_path = os.path.join(self.classify_directory, 'spk2utt.{}'.format(j))
            spk2utt.update(load_scp(spk2utt_path))
        spk2utt_path = os.path.join(self.classify_directory, 'spk2utt')
        utt2spk_path = os.path.join(self.classify_directory, 'utt2spk')
        for speak, utts in spk2utt.items():
            if not isinstance(utts, list):
                spk2utt[speak] = [utts]
            counts[speak] = len(spk2utt[speak])

        if self.num_speakers:
            valid_speakers = sorted(counts.keys(), key=lambda x: counts[x])[:self.num_speakers]
        else:
            valid_speakers = [x for x in counts.keys() if counts[x] > 1]
        if not valid_speakers:  # Only single utterance count speakers
            valid_speakers = [x for x in counts.keys()]
        reanalyze_utts = []
        for speak, c in counts.items():
            if c == 1 or speak not in valid_speakers:
                utts = spk2utt[speak]
                for u in utts:
                    reanalyze_utts.append(u)

        spk2utt = {k: v for k, v in spk2utt.items() if k in valid_speakers}
        new_utt2spk = self.classify_utterances(reanalyze_utts, valid_speakers)
        for u, spk in new_utt2spk.items():
            utt2spk[u] = spk
            spk2utt[spk].append(u)
        save_scp(([k, v] for k,v in spk2utt.items()), spk2utt_path)
        save_scp(([k, v] for k,v in utt2spk.items()), utt2spk_path)
        self.logger.debug('Analyzing stats and reclassification took {} seconds'.format(time.time() - begin))

    def export_classification(self, output_directory):
        if self.cluster:
            self.cluster_utterances()
        else:
            self.get_classification_stats()
        from decimal import Decimal
        from textgrid import TextGrid, IntervalTier
        spk2utt_path = os.path.join(self.classify_directory, 'spk2utt')
        utt2spk_path = os.path.join(self.classify_directory, 'utt2spk')
        if self.corpus.segments:
            utt2spk = load_scp(utt2spk_path)
            file_dict = {}
            for utt, segment in self.corpus.segments.items():

                filename, utt_begin, utt_end = segment.split(' ')
                utt_begin = Decimal(utt_begin)
                utt_end = Decimal(utt_end)
                if filename not in file_dict:
                    file_dict[filename] = {}
                speaker = utt2spk[utt]
                text = self.corpus.text_mapping[utt]
                if speaker not in file_dict[filename]:
                    file_dict[filename][speaker] = []
                file_dict[filename][speaker].append([utt_begin, utt_end, text])
            for filename, speaker_dict in file_dict.items():
                try:
                    speaker_directory = os.path.join(output_directory, self.corpus.file_directory_mapping[filename])
                except KeyError:
                    speaker_directory = output_directory
                max_time = self.corpus.get_wav_duration(filename)
                tg = TextGrid(maxTime=max_time)
                for speaker in sorted(speaker_dict.keys()):
                    words = speaker_dict[speaker]
                    tier = IntervalTier(name=speaker, maxTime=max_time)
                    for w in words:
                        if w[1] > max_time:
                            w[1] = max_time
                        tier.add(*w)
                    tg.append(tier)
                tg.write(os.path.join(speaker_directory, filename + '.TextGrid'))

        else:
            spk2utt = load_scp(spk2utt_path)
            for speaker, utts in spk2utt.items():
                speaker_dir = os.path.join(output_directory, speaker)
                os.makedirs(speaker_dir, exist_ok=True)
                with open(os.path.join(speaker_dir, 'utterances.txt'), 'w', encoding='utf8') as f:
                    for u in utts:
                        f.write('{}\n'.format(u))