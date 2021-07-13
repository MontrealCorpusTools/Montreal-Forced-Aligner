import os
from decimal import Decimal
import subprocess

from .helper import thirdparty_binary, load_scp, edit_distance, log_kaldi_errors
from .exceptions import KaldiProcessingError, CorpusError
from .multiprocessing import run_mp, run_non_mp

from .trainers import MonophoneTrainer
from .config import FeatureConfig
from .aligner.pretrained import PretrainedAligner


def test_utterances_func(validator, job_name):
    """

    Parameters
    ----------
    validator : :class:`~montreal_forced_aligner.validator.CorpusValidator`
    job_name : int

    Returns
    -------

    """
    aligner = validator.trainer
    log_path = os.path.join(aligner.align_directory, 'log', 'decode.0.{}.log'.format(job_name))
    words_path = os.path.join(validator.dictionary.output_directory, 'words.txt')
    mdl_path = os.path.join(aligner.align_directory, 'final.mdl')

    split_directory = validator.corpus.split_directory()
    feat_string = aligner.feature_config.construct_feature_proc_string(split_directory, aligner.align_directory, job_name)
    feat_path = os.path.join(validator.corpus.split_directory(),
                             aligner.feature_file_base_name + '.{}.scp'.format(job_name))
    graphs_path = os.path.join(aligner.align_directory, 'utterance_graphs.{}.fst'.format(job_name))

    text_int_path = os.path.join(validator.corpus.split_directory(), 'text.{}.int'.format(job_name))
    edits_path = os.path.join(aligner.align_directory, 'edits.{}.txt'.format(job_name))
    out_int_path = os.path.join(aligner.align_directory, 'aligned.{}.int'.format(job_name))
    acoustic_scale = 0.1
    beam = 15.0
    lattice_beam = 8.0
    max_active = 750
    lat_path = os.path.join(aligner.align_directory, 'lat.{}'.format(job_name))
    with open(log_path, 'w') as logf:
        latgen_proc = subprocess.Popen([thirdparty_binary('gmm-latgen-faster'),
                                        '--acoustic-scale={}'.format(acoustic_scale),
                                        '--beam={}'.format(beam),
                                        '--max-active={}'.format(max_active), '--lattice-beam={}'.format(lattice_beam),
                                        '--word-symbol-table=' + words_path,
                                        mdl_path, 'ark:' + graphs_path, feat_string, 'ark:' + lat_path],
                                       stderr=logf)
        latgen_proc.communicate()

        oracle_proc = subprocess.Popen([thirdparty_binary('lattice-oracle'),
                                        'ark:' + lat_path, 'ark,t:' + text_int_path,
                                        'ark,t:' + out_int_path, 'ark,t:' + edits_path],
                                       stderr=logf)
        oracle_proc.communicate()


def compile_utterance_train_graphs_func(validator, job_name):  # pragma: no cover
    """

    Parameters
    ----------
    validator : :class:`~aligner.aligner.validator.CorpusValidator`
    job_name : int

    Returns
    -------

    """
    aligner = validator.trainer
    disambig_int_path = os.path.join(validator.dictionary.output_directory, 'phones', 'disambig.int')
    tree_path = os.path.join(aligner.align_directory, 'tree')
    mdl_path = os.path.join(aligner.align_directory, 'final.mdl')
    lexicon_fst_path = os.path.join(validator.dictionary.output_directory, 'L_disambig.fst')
    fsts_path = os.path.join(validator.corpus.split_directory(), 'utt2fst.{}'.format(job_name))
    graphs_path = os.path.join(aligner.align_directory, 'utterance_graphs.{}.fst'.format(job_name))

    log_path = os.path.join(aligner.align_directory, 'log', 'compile-graphs-fst.0.{}.log'.format(job_name))

    with open(log_path, 'w') as logf:
        proc = subprocess.Popen([thirdparty_binary('compile-train-graphs-fsts'),
                                 '--transition-scale=1.0', '--self-loop-scale=0.1',
                                 '--read-disambig-syms={}'.format(disambig_int_path),
                                 tree_path, mdl_path,
                                 lexicon_fst_path,
                                 "ark:" + fsts_path, "ark:" + graphs_path],
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

    def __init__(self, corpus, dictionary, temp_directory=None, ignore_acoustics=False, test_transcriptions=False,
                 use_mp=True, logger=None):
        self.dictionary = dictionary
        self.corpus = corpus
        self.temp_directory = temp_directory
        self.test_transcriptions = test_transcriptions
        self.ignore_acoustics = ignore_acoustics
        self.trainer = MonophoneTrainer(FeatureConfig())
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
            self.logger.info('Skipping acoustic feature generation')
        else:
            fc = self.trainer.feature_config
        try:
            self.corpus.initialize_corpus(self.dictionary, fc)
        except CorpusError:
            self.logger.warning('There was an error when initializing the corpus, likely due to missing sound files. Ignoring acoustic generation...')
            self.ignore_acoustics = True

    def analyze_setup(self):
        total_duration = sum(x[2] for x in self.corpus.wav_info.values())
        total_duration = Decimal(str(total_duration)).quantize(Decimal('0.001'))

        ignored_count = len(self.corpus.no_transcription_files)
        ignored_count += len(self.corpus.textgrid_read_errors)
        ignored_count += len(self.corpus.decode_error_files)
        self.logger.info(self.corpus_analysis_template.format(len(self.corpus.wav_files),
                                                              self.corpus.lab_count,
                                                              self.corpus.tg_count,
                                                              ignored_count,
                                                              len(self.corpus.speak_utt_mapping),
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
        utterance_oovs = self.corpus.utterance_oovs
        oov_path = os.path.join(output_dir, 'oovs_found.txt')
        utterance_oov_path = os.path.join(output_dir, 'utterance_oovs.txt')
        if utterance_oovs:
            with open(utterance_oov_path, 'w', encoding='utf8') as f:
                for k in sorted(utterance_oovs.keys()):
                    oovs = utterance_oovs[k]
                    f.write('{} {}\n'.format(k, ', '.join(oovs)))
            self.dictionary.save_oovs_found(output_dir)
            total_instances = sum(len(x) for x in utterance_oovs.values())
            message = 'There were {} word types not found in the dictionary with a total of {} instances.\n\n' \
                      '    Please see \n\n        {}\n\n    for a full list of the word types and \n\n        {}\n\n    for a by-utterance breakdown of ' \
                      'missing words.'.format(len(oov_types), total_instances, oov_path, utterance_oov_path)
        else:
            message = 'There were no missing words from the dictionary. If you plan on using the a model trained ' \
                      'on this dataset to align other datasets in the future, it is recommended that there be at ' \
                      'least some missing words.'
        return message

    def analyze_wav_errors(self):
        output_dir = self.corpus.output_directory
        wav_read_errors = self.corpus.wav_read_errors
        unsupported_bit_depths = self.corpus.unsupported_bit_depths
        if wav_read_errors:
            path = os.path.join(output_dir, 'wav_read_errors.csv')
            with open(path, 'w') as f:
                for p in wav_read_errors:
                    f.write('{}\n'.format(p))

            message = 'There were {} sound files that could not be read. ' \
                      'Please see {} for a list.'.format(len(wav_read_errors), path)
        else:
            message = 'There were no sound files that could not be read.'
        if unsupported_bit_depths:
            path = os.path.join(output_dir, 'bit_depth_errors.csv')
            with open(path, 'w') as f:
                for p in unsupported_bit_depths:
                    f.write('{}\n'.format(p))

            message += ' There were {} sound files that had unsupported (!=16) bit depths. ' \
                       'Kaldi only supports 16-bit wav files, ' \
                       'please use sox, praat, or ffmpeg to convert your files. ' \
                       'Please see {} for a full list.'.format(len(unsupported_bit_depths), path)
        else:
            message += ' There were no sound files that had unsupported bit depths.'

        return message

    def analyze_missing_features(self):
        if self.ignore_acoustics:
            return 'Acoustic feature generation was skipped.'
        output_dir = self.corpus.output_directory
        missing_features = self.corpus.ignored_utterances
        if missing_features:
            path = os.path.join(output_dir, 'missing_features.csv')
            with open(path, 'w') as f:
                for utt in missing_features:
                    if utt in self.corpus.segments:
                        seg = self.corpus.segments[utt]
                        file_path = self.corpus.utt_wav_mapping[seg['file_name']]
                        f.write('{},{},{}\n'.format(file_path, seg['begin'], seg['end']))
                    else:
                        file_path = self.corpus.utt_wav_mapping[utt]
                        f.write('{}\n'.format(file_path))

            message = 'There were {} utterances missing features. ' \
                      'Please see {} for a list.'.format(len(missing_features), path)
        else:
            message = 'There were no utterances missing features.'
        return message

    def analyze_files_with_no_transcription(self):
        output_dir = self.corpus.output_directory
        if self.corpus.no_transcription_files:
            path = os.path.join(output_dir, 'missing_transcriptions.csv')
            with open(path, 'w') as f:
                for file_path in self.corpus.no_transcription_files:
                    f.write('{}\n'.format(file_path))
            message = 'There were {} sound files missing transcriptions. ' \
                      'Please see {} for a list.'.format(len(self.corpus.no_transcription_files), path)
        else:
            message = 'There were no sound files missing transcriptions.'
        return message

    def analyze_transcriptions_with_no_wavs(self):
        output_dir = self.corpus.output_directory
        if self.corpus.transcriptions_without_wavs:
            path = os.path.join(output_dir, 'transcriptions_missing_sound_files.csv')
            with open(path, 'w') as f:
                for file_path in self.corpus.transcriptions_without_wavs:
                    f.write('{}\n'.format(file_path))
            message = 'There were {} transcription files missing sound files. ' \
                      'Please see {} for a list.'.format(len(self.corpus.transcriptions_without_wavs), path)
        else:
            message = 'There were no transcription files missing sound files.'
        return message

    def analyze_textgrid_read_errors(self):
        output_dir = self.corpus.output_directory
        if self.corpus.textgrid_read_errors:
            path = os.path.join(output_dir, 'textgrid_read_errors.txt')
            with open(path, 'w') as f:
                for k, v in self.corpus.textgrid_read_errors.items():
                    f.write('The TextGrid file {} gave the following error on load:\n\n{}\n\n\n'.format(k, v))
            message = 'There were {} TextGrid files that could not be parsed. ' \
                      'Please see {} for a list.'.format(len(self.corpus.textgrid_read_errors), path)
        else:
            message = 'There were no issues reading TextGrids.'
        return message

    def analyze_unreadable_text_files(self):
        output_dir = self.corpus.output_directory
        if self.corpus.decode_error_files:
            path = os.path.join(output_dir, 'utf8_read_errors.csv')
            with open(path, 'w') as f:
                for file_path in self.corpus.decode_error_files:
                    f.write('{}\n'.format(file_path))
            message = 'There were {} text files that could not be parsed. ' \
                      'Please see {} for a list.'.format(len(self.corpus.decode_error_files), path)
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
                    utt_duration = self.corpus.utterance_lengths[utt] * self.trainer.feature_config.frame_shift
                    utt_duration /= 1000
                    utt_duration = Decimal(str(utt_duration)).quantize(Decimal('0.001'))
                    utt_length_words = self.corpus.text_mapping[utt].count(' ') + 1
                    if utt in self.corpus.segments:
                        seg = self.corpus.segments[utt]
                        file_path = self.corpus.utt_wav_mapping[seg['file_name']]
                        f.write('{},{},{},{},{}\n'.format(file_path, seg['begin'], seg['end'], utt_duration, utt_length_words))
                    else:
                        file_path = self.corpus.utt_wav_mapping[utt]
                        f.write('{},,,{},{}\n'.format(file_path, utt_duration, utt_length_words))
            message = 'There were {} unalignable utterances out of {} after the initial training. ' \
                      'Please see {} for a list.'.format(len(unaligned_utts), num_utterances, path)
        else:
            message = 'All {} were successfully aligned!'.format(num_utterances)
        print(self.alignment_analysis_template.format(message))

    def validate(self):
        self.analyze_setup()
        if self.ignore_acoustics:
            print('Skipping test alignments.')
            return
        if not isinstance(self.trainer, PretrainedAligner):
            self.trainer.init_training(self.trainer.train_type, self.temp_directory, self.corpus, self.dictionary, None)
            self.trainer.train(call_back=print)
        self.trainer.align(None)
        self.analyze_unaligned_utterances()
        if self.test_transcriptions:
            self.test_utterance_transcriptions()

    def test_utterance_transcriptions(self):
        self.logger.info('Checking utterance transcriptions...')

        split_directory = self.corpus.split_directory()
        model_directory = self.trainer.align_directory
        log_directory = os.path.join(model_directory, 'log')

        try:

            jobs = [(self, x)
                    for x in range(self.corpus.num_jobs)]
            if self.trainer.feature_config.use_mp:
                run_mp(compile_utterance_train_graphs_func, jobs, log_directory)
            else:
                run_non_mp(compile_utterance_train_graphs_func, jobs, log_directory)
            self.logger.info('Utterance FSTs compiled!')
            self.logger.info('Decoding utterances (this will take some time)...')
            if self.trainer.feature_config.use_mp:
                run_mp(test_utterances_func, jobs, log_directory)
            else:
                run_non_mp(test_utterances_func, jobs, log_directory)
            self.logger.info('Finished decoding utterances!')

            word_mapping = self.dictionary.reversed_word_mapping
            errors = {}

            for job in range(self.corpus.num_jobs):
                text_path = os.path.join(split_directory, 'text.{}'.format(job))
                texts = load_scp(text_path)
                aligned_int = load_scp(os.path.join(model_directory, 'aligned.{}.int'.format(job)))
                with open(os.path.join(model_directory, 'aligned.{}'.format(job)), 'w') as outf:
                    for utt, line in sorted(aligned_int.items()):
                        text = []
                        for t in line:
                            text.append(word_mapping[int(t)])
                        outf.write('{} {}\n'.format(utt, ' '.join(text)))
                        ref_text = texts[utt]
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
                        problemf.write('{},{},{},{}\n'.format(utt, edits,
                                                              ' '.join(ref_text), ' '.join(text)))
                message = 'There were {} of {} utterances with at least one transcription issue. ' \
                          'Please see the outputted csv file {}.'.format(len(errors), self.corpus.num_utterances, out_path)

            self.logger.info(self.transcription_analysis_template.format(message))

        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
