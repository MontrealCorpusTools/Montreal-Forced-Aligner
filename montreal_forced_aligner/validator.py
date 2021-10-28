from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from .corpus import AlignableCorpus
    from .dictionary import DictionaryType
import os
from decimal import Decimal
import subprocess

from .utils import thirdparty_binary, log_kaldi_errors
from .helper import load_scp, edit_distance
from .exceptions import KaldiProcessingError, CorpusError
from .multiprocessing import run_mp, run_non_mp

from .trainers import MonophoneTrainer
from .config import FeatureConfig
from .aligner.pretrained import PretrainedAligner


def test_utterances_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    words_paths: Dict[str, str],
    graphs_paths: Dict[str, str],
    text_int_paths: Dict[str, str],
    edits_paths: Dict[str, str],
    out_int_paths: Dict[str, str],
    model_path: str):
    """

    Parameters
    ----------
    validator : :class:`~montreal_forced_aligner.validator.CorpusValidator`
    job_name : int

    Returns
    -------

    """
    acoustic_scale = 0.1
    beam = 15.0
    lattice_beam = 8.0
    max_active = 750
    with open(log_path, 'w') as logf:
        for dict_name in dictionaries:
            words_path = words_paths[dict_name]
            graphs_path = graphs_paths[dict_name]
            feature_string = feature_strings[dict_name]
            edits_path = edits_paths[dict_name]
            text_int_path = text_int_paths[dict_name]
            out_int_path = out_int_paths[dict_name]
            latgen_proc = subprocess.Popen([thirdparty_binary('gmm-latgen-faster'),
                                            f'--acoustic-scale={acoustic_scale}',
                                            f'--beam={beam}',
                                            f'--max-active={max_active}', f'--lattice-beam={lattice_beam}',
                                            f'--word-symbol-table={words_path}',
                                            model_path, 'ark:' + graphs_path, feature_string, 'ark:-'],
                                           stderr=logf, stdout=subprocess.PIPE)

            oracle_proc = subprocess.Popen([thirdparty_binary('lattice-oracle'),
                                            'ark:-', f'ark,t:{text_int_path}',
                                            f'ark,t:{out_int_path}', f'ark,t:{edits_path}'],
                                           stderr=logf, stdin=latgen_proc.stdout)
            oracle_proc.communicate()


def compile_utterance_train_graphs_func(
    log_path: str,
    dictionaries: List[str],
    disambig_int_paths: Dict[str, str],
    disambig_L_fst_paths: Dict[str, str],
    fst_paths: Dict[str, str],
    graphs_paths: Dict[str, str],
    model_path: str,
    tree_path: str):  # pragma: no cover
    """

    Parameters
    ----------
    validator : :class:`~aligner.aligner.validator.CorpusValidator`
    job_name : int

    Returns
    -------

    """

    with open(log_path, 'w') as logf:
        for dict_name in dictionaries:
            disambig_int_path = disambig_int_paths[dict_name]
            disambig_L_fst_path = disambig_L_fst_paths[dict_name]
            fst_path = fst_paths[dict_name]
            graphs_path = graphs_paths[dict_name]
            proc = subprocess.Popen([thirdparty_binary('compile-train-graphs-fsts'),
                                     '--transition-scale=1.0', '--self-loop-scale=0.1',
                                     f'--read-disambig-syms={disambig_int_path}',
                                     tree_path, model_path,
                                     disambig_L_fst_path,
                                     f"ark:{fst_path}", f"ark:{graphs_path}"],
                                    stderr=logf)

            proc.communicate()


class CorpusValidator(object):
    """
    Aligner that aligns and trains acoustics models on a large dataset

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.Corpus`
        Corpus object for the dataset
    dictionary : :class:`~montreal_forced_aligner.dictionary.Dictionary`
        Dictionary object for the pronunciation dictionary
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``

    Attributes
    ----------
    trainer : :class:`~montreal_forced_aligner.trainers.monophone.MonophoneTrainer`
    """

    corpus_analysis_template = '''
    =========================================Corpus=========================================
    {} sound files
    {} sound files with .lab transcription files
    {} sound files with TextGrids transcription files
    {} additional sound files ignored (see below)
    {} speakers
    {} utterances
    {} seconds total duration
    
    DICTIONARY
    ----------
    {}
    
    SOUND FILE READ ERRORS
    ----------------------
    {}
    
    FEATURE CALCULATION
    -------------------
    {}
    
    FILES WITHOUT TRANSCRIPTIONS
    ----------------------------
    {}
    
    TRANSCRIPTIONS WITHOUT FILES
    --------------------
    {}
      
    TEXTGRID READ ERRORS
    --------------------
    {}
    
    UNREADABLE TEXT FILES
    --------------------
    {}
    '''

    alignment_analysis_template = '''
    =======================================Alignment========================================
    {}
    '''

    transcription_analysis_template = '''
    ====================================Transcriptions======================================
    {}
    '''

    def __init__(self, corpus: AlignableCorpus, dictionary: DictionaryType, temp_directory=None, ignore_acoustics=False, test_transcriptions=False,
                 use_mp=True, logger=None):
        self.dictionary = dictionary
        self.corpus = corpus
        self.temp_directory = temp_directory
        self.test_transcriptions = test_transcriptions
        self.ignore_acoustics = ignore_acoustics
        self.trainer: MonophoneTrainer = MonophoneTrainer(FeatureConfig())
        self.logger = logger
        self.trainer.logger = logger
        self.trainer.update({"use_mp": use_mp})
        self.setup()

    def setup(self):
        self.dictionary.set_word_set(self.corpus.word_set)
        self.dictionary.write()
        if self.test_transcriptions:
            self.dictionary.write(disambig=True)
        if self.ignore_acoustics:
            fc = None
            if self.logger is not None:
                self.logger.info('Skipping acoustic feature generation')
        else:
            fc = self.trainer.feature_config
        try:
            self.corpus.initialize_corpus(self.dictionary, fc)
            if self.test_transcriptions:
                self.corpus.initialize_utt_fsts()
        except CorpusError:
            if self.logger is not None:
                self.logger.warning('There was an error when initializing the corpus, likely due to missing sound files. Ignoring acoustic generation...')
            self.ignore_acoustics = True

    def analyze_setup(self):
        total_duration = sum(x.duration for x in self.corpus.files.values())
        total_duration = Decimal(str(total_duration)).quantize(Decimal('0.001'))

        ignored_count = len(self.corpus.no_transcription_files)
        ignored_count += len(self.corpus.textgrid_read_errors)
        ignored_count += len(self.corpus.decode_error_files)
        self.logger.info(self.corpus_analysis_template.format(sum(1 for x in self.corpus.files.values() if x.wav_path is not None),
                                                              sum(1 for x in self.corpus.files.values() if x.text_type == 'lab'),
                                                              sum(1 for x in self.corpus.files.values() if x.text_type == 'textgrid'),
                                                              ignored_count,
                                                              len(self.corpus.speakers),
                                                              self.corpus.num_utterances,
                                                              total_duration,
                                                              self.analyze_oovs(),
                                                              self.analyze_wav_errors(),
                                                              self.analyze_missing_features(),
                                                              self.analyze_files_with_no_transcription(),
                                                              self.analyze_transcriptions_with_no_wavs(),
                                                              self.analyze_textgrid_read_errors(),
                                                              self.analyze_unreadable_text_files()
                                                              ))

    def analyze_oovs(self):
        output_dir = self.corpus.output_directory
        oov_types = self.dictionary.oovs_found
        oov_path = os.path.join(output_dir, 'oovs_found.txt')
        utterance_oov_path = os.path.join(output_dir, 'utterance_oovs.txt')
        if oov_types:
            total_instances = 0
            with open(utterance_oov_path, 'w', encoding='utf8') as f:
                for utt, utterance in sorted(self.corpus.utterances.items()):
                    if not utterance.oovs:
                        continue
                    total_instances += len(utterance.oovs)
                    f.write(f"{utt} {', '.join(utterance.oovs)}\n")
            self.dictionary.save_oovs_found(output_dir)
            message = f'There were {len(oov_types)} word types not found in the dictionary with a total of {total_instances} instances.\n\n' \
                      f'    Please see \n\n        {oov_path}\n\n    for a full list of the word types and \n\n        {utterance_oov_path}\n\n    for a by-utterance breakdown of ' \
                      f'missing words.'
        else:
            message = 'There were no missing words from the dictionary. If you plan on using the a model trained ' \
                      'on this dataset to align other datasets in the future, it is recommended that there be at ' \
                      'least some missing words.'
        return message

    def analyze_wav_errors(self):
        output_dir = self.corpus.output_directory
        wav_read_errors = self.corpus.sound_file_errors
        if wav_read_errors:
            path = os.path.join(output_dir, 'sound_file_errors.csv')
            with open(path, 'w') as f:
                for p in wav_read_errors:
                    f.write(f'{p}\n')

            message = f'There were {len(wav_read_errors)} sound files that could not be read. ' \
                      f'Please see {path} for a list.'
        else:
            message = 'There were no sound files that could not be read.'

        return message

    def analyze_missing_features(self):
        if self.ignore_acoustics:
            return 'Acoustic feature generation was skipped.'
        output_dir = self.corpus.output_directory
        missing_features = [x for x in self.corpus.utterances.values() if x.ignored]
        if missing_features:
            path = os.path.join(output_dir, 'missing_features.csv')
            with open(path, 'w') as f:
                for utt in missing_features:
                    if utt.begin is not None:

                        f.write(f'{utt.file.wav_path},{utt.begin},{utt.end}\n')
                    else:
                        f.write(f'{utt.file.wav_path}\n')

            message = f'There were {len(missing_features)} utterances missing features. ' \
                      f'Please see {path} for a list.'
        else:
            message = 'There were no utterances missing features.'
        return message

    def analyze_files_with_no_transcription(self):
        output_dir = self.corpus.output_directory
        if self.corpus.no_transcription_files:
            path = os.path.join(output_dir, 'missing_transcriptions.csv')
            with open(path, 'w') as f:
                for file_path in self.corpus.no_transcription_files:
                    f.write(f'{file_path}\n')
            message = f'There were {len(self.corpus.no_transcription_files)} sound files missing transcriptions. ' \
                      f'Please see {path} for a list.'
        else:
            message = 'There were no sound files missing transcriptions.'
        return message

    def analyze_transcriptions_with_no_wavs(self):
        output_dir = self.corpus.output_directory
        if self.corpus.transcriptions_without_wavs:
            path = os.path.join(output_dir, 'transcriptions_missing_sound_files.csv')
            with open(path, 'w') as f:
                for file_path in self.corpus.transcriptions_without_wavs:
                    f.write(f'{file_path}\n')
            message = f'There were {len(self.corpus.transcriptions_without_wavs)} transcription files missing sound files. ' \
                      f'Please see {path} for a list.'
        else:
            message = 'There were no transcription files missing sound files.'
        return message

    def analyze_textgrid_read_errors(self):
        output_dir = self.corpus.output_directory
        if self.corpus.textgrid_read_errors:
            path = os.path.join(output_dir, 'textgrid_read_errors.txt')
            with open(path, 'w') as f:
                for k, v in self.corpus.textgrid_read_errors.items():
                    f.write(f'The TextGrid file {k} gave the following error on load:\n\n{v}\n\n\n')
            message = f'There were {len(self.corpus.textgrid_read_errors)} TextGrid files that could not be parsed. ' \
                      f'Please see {path} for a list.'
        else:
            message = 'There were no issues reading TextGrids.'
        return message

    def analyze_unreadable_text_files(self):
        output_dir = self.corpus.output_directory
        if self.corpus.decode_error_files:
            path = os.path.join(output_dir, 'utf8_read_errors.csv')
            with open(path, 'w') as f:
                for file_path in self.corpus.decode_error_files:
                    f.write(f'{file_path}\n')
            message = f'There were {len(self.corpus.decode_error_files)} text files that could not be parsed. ' \
                      f'Please see {path} for a list.'
        else:
            message = 'There were no issues reading text files.'
        return message

    def analyze_unaligned_utterances(self):
        unaligned_utts = self.trainer.get_unaligned_utterances()
        num_utterances = self.corpus.num_utterances
        if unaligned_utts:
            path = os.path.join(self.corpus.output_directory, 'unalignable_files.csv')
            with open(path, 'w') as f:
                f.write('File path,begin,end,duration,text length\n')
                for utt in unaligned_utts:
                    utterance = self.corpus.utterances[utt]
                    utt_duration = utterance.duration
                    utt_length_words = utterance.text.count(' ') + 1
                    if utterance.begin is not None:
                        f.write(f'{utterance.file.wav_path},{utterance.begin},{utterance.end},{utt_duration},{utt_length_words}\n')
                    else:
                        f.write(f'{utterance.file.wav_path},,,{utt_duration},{utt_length_words}\n')
            message = f'There were {len(unaligned_utts)} unalignable utterances out of {num_utterances} after the initial training. ' \
                      f'Please see {path} for a list.'
        else:
            message = f'All {num_utterances} utterances were successfully aligned!'
        print(self.alignment_analysis_template.format(message))

    def validate(self):
        self.analyze_setup()
        if self.ignore_acoustics:
            print('Skipping test alignments.')
            return
        if not isinstance(self.trainer, PretrainedAligner):
            self.trainer.init_training(self.trainer.train_type, self.temp_directory, self.corpus, self.dictionary, None)
            self.trainer.train()
        self.trainer.align(None)
        self.analyze_unaligned_utterances()
        if self.test_transcriptions:
            self.test_utterance_transcriptions()

    def test_utterance_transcriptions(self):
        self.logger.info('Checking utterance transcriptions...')

        split_directory = self.corpus.split_directory
        model_directory = self.trainer.align_directory
        log_directory = os.path.join(model_directory, 'log')

        try:

            jobs = [x.compile_utterance_train_graphs_arguments(self)
                    for x in self.corpus.jobs]
            if self.trainer.feature_config.use_mp:
                run_mp(compile_utterance_train_graphs_func, jobs, log_directory)
            else:
                run_non_mp(compile_utterance_train_graphs_func, jobs, log_directory)
            self.logger.info('Utterance FSTs compiled!')
            self.logger.info('Decoding utterances (this will take some time)...')
            jobs = [x.test_utterances_arguments(self)
                    for x in self.corpus.jobs]
            if self.trainer.feature_config.use_mp:
                run_mp(test_utterances_func, jobs, log_directory)
            else:
                run_non_mp(test_utterances_func, jobs, log_directory)
            self.logger.info('Finished decoding utterances!')

            word_mapping = self.dictionary.reversed_word_mapping
            errors = {}

            for job in jobs:
                for dict_name in job.dictionaries:
                    aligned_int = load_scp(job.out_int_paths[dict_name])
                    for utt, line in sorted(aligned_int.items()):
                        text = []
                        for t in line:
                            text.append(word_mapping[int(t)])
                        ref_text = self.corpus.utterances[utt].text.split()
                        edits = edit_distance(text, ref_text)

                        if edits:
                            errors[utt] = (edits, ref_text, text)
            if not errors:
                message = 'There were no utterances with transcription issues.'
            else:
                out_path = os.path.join(self.corpus.output_directory, 'transcription_problems.csv')
                with open(out_path, 'w') as problemf:
                    problemf.write('Utterance,Edits,Reference,Decoded\n')
                    for utt, (edits, ref_text, text) in sorted(errors.items(),
                                                               key=lambda x: -1 * (
                                                                       len(x[1][1]) + len(x[1][2]))):
                        problemf.write(f"{utt},{edits},{' '.join(ref_text)},{' '.join(text)}\n")
                message = f'There were {len(errors)} of {self.corpus.num_utterances} utterances with at least one transcription issue. ' \
                          f'Please see the outputted csv file {out_path}.'

            self.logger.info(self.transcription_analysis_template.format(message))

        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
