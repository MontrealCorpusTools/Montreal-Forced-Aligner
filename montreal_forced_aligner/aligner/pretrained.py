import os
import re
from collections import Counter

from .base import BaseAligner
from ..multiprocessing import (align, convert_ali_to_textgrids, compile_train_graphs,
                               calc_fmllr, generate_pronunciations)
from ..exceptions import KaldiProcessingError
from ..helper import log_kaldi_errors, load_scp


def parse_transitions(path, phones_path):
    state_extract_pattern = re.compile(r'Transition-state (\d+): phone = (\w+)')
    id_extract_pattern = re.compile(r'Transition-id = (\d+)')
    cur_phone = None
    current = 0
    with open(path, encoding='utf8') as f, open(phones_path, 'w', encoding='utf8') as outf:
        outf.write('{} {}\n'.format('<eps>', 0))
        for line in f:
            line = line.strip()
            if line.startswith('Transition-state'):
                m = state_extract_pattern.match(line)
                _, phone = m.groups()
                if phone != cur_phone:
                    current = 0
                    cur_phone = phone
            else:
                m = id_extract_pattern.match(line)
                transition_id = m.groups()[0]
                outf.write('{}_{} {}\n'.format(phone, current, transition_id))
                current += 1


class PretrainedAligner(BaseAligner):
    """
    Class for aligning a dataset using a pretrained acoustic model

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.AlignableCorpus`
        Corpus object for the dataset
    dictionary : :class:`~montreal_forced_aligner.dictionary.Dictionary`
        Dictionary object for the pronunciation dictionary
    acoustic_model : :class:`~montreal_forced_aligner.models.AcousticModel`
        Archive containing the acoustic model and pronunciation dictionary
    align_config : :class:`~montreal_forced_aligner.config.AlignConfig`
        Configuration for alignment
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    call_back : callable, optional
        Specifies a call back function for alignment
    """

    def __init__(self, corpus, dictionary, acoustic_model, align_config,
                 temp_directory=None,
                 call_back=None, debug=False, verbose=False, logger=None):
        self.acoustic_model = acoustic_model
        super(PretrainedAligner, self).__init__(corpus, dictionary, align_config, temp_directory,
                                                call_back, debug, verbose, logger)
        self.align_config.data_directory = corpus.split_directory()
        self.acoustic_model.export_model(self.align_directory)
        log_dir = os.path.join(self.align_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        self.logger.info('Done with setup!')

    @property
    def model_directory(self):
        return os.path.join(self.temp_directory, 'model')

    @property
    def align_directory(self):
        return os.path.join(self.temp_directory, 'align')

    def setup(self):
        self.dictionary.nonsil_phones = self.acoustic_model.meta['phones']
        super(PretrainedAligner, self).setup()

    def align(self, subset=None):
        done_path = os.path.join(self.align_directory, 'done')
        dirty_path = os.path.join(self.align_directory, 'dirty')
        if os.path.exists(done_path):
            self.logger.info('Alignment already done, skipping.')
            return
        try:
            compile_train_graphs(self.align_directory, self.dictionary.output_directory,
                                 self.align_config.data_directory, self.corpus.num_jobs, self)
            self.acoustic_model.feature_config.generate_features(self.corpus)
            log_dir = os.path.join(self.align_directory, 'log')
            os.makedirs(log_dir, exist_ok=True)
            self.logger.info('Performing first-pass alignment...')
            align('final', self.align_directory, self.align_config.data_directory,
                  self.dictionary.optional_silence_csl,
                  self.corpus.num_jobs, self.align_config)

            log_like = 0
            tot_frames = 0
            for j in range(self.corpus.num_jobs):
                score_path = os.path.join(self.align_directory, 'ali.{}.scores'.format(j))
                scores = load_scp(score_path, data_type=float)
                for k, v in scores.items():
                    log_like += v
                    tot_frames += self.corpus.utterance_lengths[k]
            if tot_frames:
                self.logger.debug('Prior to SAT, average per frame likelihood (this might not actually mean anything): {}'.format(log_like/tot_frames))
            else:
                self.logger.debug('No files were aligned, this likely indicates serious problems with the aligner.')
            if not self.align_config.disable_sat and self.acoustic_model.feature_config.fmllr \
                    and not os.path.exists(os.path.join(self.align_directory, 'trans.0')):
                self.logger.info('Calculating fMLLR for speaker adaptation...')
                calc_fmllr(self.align_directory, self.align_config.data_directory,
                      self.dictionary.optional_silence_csl, self.corpus.num_jobs, self.align_config, initial=True, iteration='final')
                self.logger.info('Performing second-pass alignment...')
                align('final', self.align_directory, self.align_config.data_directory,
                      self.dictionary.optional_silence_csl,
                      self.corpus.num_jobs, self.align_config)

                log_like = 0
                tot_frames = 0
                for j in range(self.corpus.num_jobs):
                    score_path = os.path.join(self.align_directory, 'ali.{}.scores'.format(j))
                    scores = load_scp(score_path, data_type=float)
                    for k, v in scores.items():
                        log_like += v
                        tot_frames += self.corpus.utterance_lengths[k]
                if tot_frames:
                    self.logger.debug('Following SAT, average per frame likelihood (this might not actually mean anything): {}'.format(log_like/tot_frames))
                else:
                    self.logger.debug('No files were aligned, this likely indicates serious problems with the aligner.')
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
        with open(done_path, 'w'):
            pass

    def export_textgrids(self, output_directory):
        """
        Export a TextGrid file for every sound file in the dataset
        """
        ali_directory = self.align_directory
        convert_ali_to_textgrids(self.align_config, output_directory, ali_directory, self.dictionary,
                                 self.corpus, self.corpus.num_jobs, self)
        self.compile_information(ali_directory, output_directory)

    def generate_pronunciations(self, output_path, calculate_silence_probs=False, min_count=1):
        pron_counts, utt_mapping = generate_pronunciations(self.align_config, self.align_directory, self.dictionary, self.corpus, self.corpus.num_jobs)
        if calculate_silence_probs:
            sil_before_counts = Counter()
            nonsil_before_counts = Counter()
            sil_after_counts = Counter()
            nonsil_after_counts = Counter()
            sils = ['<s>', '</s>', '<eps>']
            for u, v in utt_mapping.items():
                for i, w in enumerate(v):
                    if w in sils:
                        continue
                    prev_w = v[i - 1]
                    next_w = v[i + 1]
                    if prev_w in sils:
                        sil_before_counts[w] += 1
                    else:
                        nonsil_before_counts[w] += 1
                    if next_w in sils:
                        sil_after_counts[w] += 1
                    else:
                        nonsil_after_counts[w] += 1

        self.dictionary.pronunciation_probabilities = True
        for word, prons in self.dictionary.words.items():
            if word not in pron_counts:
                for p in prons:
                    p['probability'] = 1
            else:
                print(word)
                print(pron_counts[word])
                total = 0
                best_pron = 0
                best_count = 0
                for p in prons:
                    p['probability'] = min_count
                    if p['pronunciation'] in pron_counts[word]:
                        p['probability'] += pron_counts[word][p['pronunciation']]
                    total += p['probability']
                    if p['probability'] > best_count:
                        best_pron = p['pronunciation']
                        best_count = p['probability']
                print(total)
                print(prons)
                for p in prons:
                    if p['pronunciation'] == best_pron:
                        p['probability'] = 1
                    else:
                        p['probability'] /= total
                self.dictionary.words[word] = prons
                print(self.dictionary.words[word])
        self.dictionary.export_lexicon(output_path, probability=True)