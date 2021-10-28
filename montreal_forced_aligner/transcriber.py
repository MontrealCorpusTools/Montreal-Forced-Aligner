from __future__ import annotations
import subprocess
import os
import shutil
import re
from typing import TYPE_CHECKING, Callable, Optional, List, TextIO, Tuple
if TYPE_CHECKING:
    from .config.transcribe_config import TranscribeConfig
    from .dictionary import DictionaryType
    from .corpus import TranscribeCorpus
    from .models import AcousticModel, LanguageModel
    from logging import Logger
import multiprocessing as mp
from praatio import textgrid
from praatio.utilities.constants import Interval
from .config import TEMP_DIR
from .utils import thirdparty_binary, log_kaldi_errors, parse_logs
from .multiprocessing.transcription import transcribe, transcribe_fmllr, create_hclgs, score_transcriptions
from .helper import score
from .exceptions import KaldiProcessingError



class Transcriber(object):
    """
    Class for performing transcription.

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.TranscribeCorpus` or :class:`~montreal_forced_aligner.corpus.AlignableCorpus`
        Corpus to transcribe
    dictionary: :class:`~montreal_forced_aligner.dictionary.Dictionary`
        Pronunciation dictionary to use as a lexicon
    acoustic_model : :class:`~montreal_forced_aligner.models.AcousticModel`
        Acoustic model to use
    language_model : :class:`~montreal_forced_aligner.models.LanguageModel`
        Language model to use
    transcribe_config : :class:`~montreal_forced_aligner.config.TranscribeConfig`
        Language model to use
    """
    min_language_model_weight = 7
    max_language_model_weight = 17
    word_insertion_penalties = [0, 0.5, 1.0]

    def __init__(self, corpus: TranscribeCorpus, dictionary: DictionaryType,
                 acoustic_model: AcousticModel,
                 language_model: LanguageModel,
                 transcribe_config: TranscribeConfig, temp_directory: Optional[str]=None,
                 call_back:Optional[Callable]=None, debug: bool=False, verbose: bool=False, evaluation_mode: bool=False,
                 logger: Optional[Logger]=None):
        self.logger = logger
        self.corpus = corpus
        self.dictionary = dictionary
        self.acoustic_model = acoustic_model
        self.language_model = language_model
        self.transcribe_config = transcribe_config

        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = temp_directory
        self.call_back = call_back
        if self.call_back is None:
            self.call_back = print
        self.verbose = verbose
        self.debug = debug
        self.evaluation_mode = evaluation_mode
        self.acoustic_model.export_model(self.model_directory)
        self.acoustic_model.export_model(self.working_directory)
        self.log_dir = os.path.join(self.transcribe_directory, 'log')
        self.uses_voiced = False
        self.uses_splices = False
        self.uses_cmvn = True
        self.speaker_independent = True
        os.makedirs(self.log_dir, exist_ok=True)
        self.setup()

    @property
    def transcribe_directory(self) -> str:
        return os.path.join(self.temp_directory, 'transcribe')

    @property
    def evaluation_directory(self):
        eval_string = f'eval_{self.transcribe_config.language_model_weight}_{self.transcribe_config.word_insertion_penalty}'
        path = os.path.join(self.working_directory, eval_string)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def working_directory(self) -> str:
        return self.transcribe_directory

    @property
    def evaluation_log_directory(self) -> str:
        return os.path.join(self.evaluation_directory, 'log')

    @property
    def working_log_directory(self) -> str:
        return os.path.join(self.working_directory, 'log')

    @property
    def data_directory(self) -> str:
        return self.corpus.split_directory

    @property
    def model_directory(self) -> str:
        return os.path.join(self.temp_directory, 'models')

    @property
    def model_path(self) -> str:
        return os.path.join(self.working_directory, 'final.mdl')

    @property
    def alignment_model_path(self) -> str:
        path = os.path.join(self.working_directory, 'final.alimdl')
        if os.path.exists(path):
            return path
        return self.model_path

    @property
    def fmllr_options(self):
        data = self.transcribe_config.fmllr_options
        data['sil_phones'] = self.dictionary.silence_csl
        return data

    @property
    def hclg_options(self):
        context_width, central_pos = self.get_tree_info()
        return {'context_width': context_width,
                'central_pos': central_pos,
                'self_loop_scale': self.transcribe_config.self_loop_scale,
                'transition_scale': self.transcribe_config.transition_scale,
                }

    def get_tree_info(self) -> Tuple[int, int]:
        tree_proc = subprocess.Popen([thirdparty_binary('tree-info'),
                                      os.path.join(self.model_directory, 'tree')], text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = tree_proc.communicate()
        context_width = 1
        central_pos = 0
        for line in stdout.split('\n'):
            text = line.strip().split(' ')
            if text[0] == 'context-width':
                context_width = int(text[1])
            elif text[0] == 'central-position':
                central_pos = int(text[1])
        return context_width, central_pos

    def setup(self) -> None:
        dirty_path = os.path.join(self.model_directory, 'dirty')

        if os.path.exists(dirty_path):  # if there was an error, let's redo from scratch
            shutil.rmtree(self.model_directory)
        log_dir = os.path.join(self.model_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        self.dictionary.write(disambig=True)
        for dict_name, output_directory in self.dictionary.output_paths.items():
            words_path = os.path.join(self.model_directory, f'words.{dict_name}.txt')
            shutil.copyfile(os.path.join(output_directory, 'words.txt'), words_path)
        self.corpus.initialize_corpus(self.dictionary, self.acoustic_model.feature_config)

        big_arpa_path = self.language_model.carpa_path
        small_arpa_path = self.language_model.small_arpa_path
        medium_arpa_path = self.language_model.medium_arpa_path
        if not os.path.exists(small_arpa_path) or not os.path.exists(medium_arpa_path):
            self.logger.info('Parsing large ngram model...')
            mod_path = os.path.join(self.model_directory, 'base_lm.mod')
            new_carpa_path = os.path.join(self.model_directory, 'base_lm.arpa')
            with open(big_arpa_path, 'r', encoding='utf8') as inf, open(new_carpa_path, 'w', encoding='utf8') as outf:
                for line in inf:
                    outf.write(line.lower())
            big_arpa_path = new_carpa_path
            subprocess.call(['ngramread', '--ARPA', big_arpa_path, mod_path])

            if not os.path.exists(small_arpa_path):
                self.logger.info("Generating small model from the large ARPA with a pruning threshold of 3e-7")
                prune_thresh_small = 0.0000003
                small_mod_path = mod_path.replace('.mod', '_small.mod')
                subprocess.call(['ngramshrink', '--method=relative_entropy',
                                 f'--theta={prune_thresh_small}',
                                 mod_path, small_mod_path])
                subprocess.call(['ngramprint', '--ARPA', small_mod_path, small_arpa_path])

            if not os.path.exists(medium_arpa_path):
                self.logger.info("Generating medium model from the large ARPA with a pruning threshold of 1e-7")
                prune_thresh_medium = 0.0000001
                med_mod_path = mod_path.replace('.mod', '_med.mod')
                subprocess.call(['ngramshrink', '--method=relative_entropy',
                                 f'--theta={prune_thresh_medium}',
                                 mod_path, med_mod_path])
                subprocess.call(['ngramprint', '--ARPA', med_mod_path, medium_arpa_path])
        try:
            create_hclgs(self)
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise

    def transcribe(self) -> None:
        self.logger.info('Beginning transcription...')
        dirty_path = os.path.join(self.transcribe_directory, 'dirty')
        if os.path.exists(dirty_path):
            shutil.rmtree(self.transcribe_directory, ignore_errors=True)
        os.makedirs(self.log_dir,exist_ok=True)
        try:
            transcribe(self)
            if self.acoustic_model.feature_config.fmllr and not self.transcribe_config.no_speakers and self.transcribe_config.fmllr:
                self.logger.info('Performing speaker adjusted transcription...')
                transcribe_fmllr(self)
            score_transcriptions(self)
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise

    def evaluate(self):
        self.logger.info('Evaluating transcripts...')
        self._load_transcripts()
        # Sentence-level measures

        correct = 0
        incorrect = 0
        # Word-level measures
        total_edits = 0
        total_length = 0
        issues = []
        with mp.Pool(self.corpus.num_jobs) as pool:
            to_comp = []
            for utt_name, utterance in self.corpus.utterances.items():
                g = utterance.text.split()
                if not utterance.transcription_text:
                    incorrect += 1
                    gold = gold.split()
                    total_edits += len(gold)
                    total_length += len(gold)

                h = utterance.transcription_text.split()
                if g != h:
                    issues.append((utt_name, g, h))
                to_comp.append((g, h))
            gen = pool.starmap(score, to_comp)
            for (edits, length) in gen:
                if edits == 0:
                    correct += 1
                else:
                    incorrect += 1
                total_edits += edits
                total_length += length
        ser = 100 * incorrect / (correct + incorrect)
        wer = 100 * total_edits / total_length
        output_path = os.path.join(self.evaluation_directory, 'transcription_issues.csv')
        with open(output_path, 'w', encoding='utf8') as f:
            for utt, g, h in issues:
                g = ' '.join(g)
                h = ' '.join(h)
                f.write(f'{utt},{g},{h}\n')
        self.logger.info(f'SER: {ser:.2f}%, WER: {wer:.2f}%')
        return ser, wer

    def _load_transcripts(self):
        for j in self.corpus.jobs:
            score_arguments = j.score_arguments(self)
            for dict_name, tra_path in score_arguments.tra_paths.items():

                with open(tra_path, 'r', encoding='utf8') as f:
                    for line in f:
                        t = line.strip().split(' ')
                        utt = t[0]
                        utterance = self.corpus.utterances[utt]
                        speaker = utterance.speaker
                        lookup = speaker.dictionary.reversed_word_mapping
                        ints = t[1:]
                        if not ints:
                            continue
                        transcription = []
                        for i in ints:
                            transcription.append(lookup[int(i)])
                        utterance.transcription_text = ' '.join(transcription)


    def export_transcriptions(self, output_directory):
        backup_output_directory = None
        if not self.transcribe_config.overwrite:
            backup_output_directory = os.path.join(self.transcribe_directory, 'transcriptions')
            os.makedirs(backup_output_directory, exist_ok=True)
        self._load_transcripts()
        for file_name, file in self.corpus.files.items():
            file.save(output_directory, backup_output_directory)
