import subprocess
import os
import shutil
import re
import multiprocessing as mp
from praatio import tgio
from .config import TEMP_DIR
from .helper import thirdparty_binary
from .multiprocessing import transcribe, transcribe_fmllr
from .corpus import AlignableCorpus
from .dictionary import MultispeakerDictionary
from .helper import score, log_kaldi_errors, parse_logs
from .exceptions import KaldiProcessingError


def compose_lg(model_directory, dictionary_path, small_g_path, lg_path, log_file):
    if os.path.exists(lg_path):
        return
    temp_compose_path = os.path.join(model_directory, 'LG.temp')
    compose_proc = subprocess.Popen([thirdparty_binary('fsttablecompose'),
                                     dictionary_path, small_g_path, temp_compose_path],
                                    stderr=log_file)
    compose_proc.communicate()

    temp2_compose_path = os.path.join(model_directory, 'LG.temp2')
    determinize_proc = subprocess.Popen([thirdparty_binary('fstdeterminizestar'),
                                         '--use-log=true', temp_compose_path, temp2_compose_path],
                                        stderr=log_file)
    determinize_proc.communicate()
    os.remove(temp_compose_path)

    minimize_proc = subprocess.Popen([thirdparty_binary('fstminimizeencoded'),
                                      temp2_compose_path, temp_compose_path],
                                     stdout=subprocess.PIPE, stderr=log_file)
    minimize_proc.communicate()
    os.remove(temp2_compose_path)
    push_proc = subprocess.Popen([thirdparty_binary('fstpushspecial'), temp_compose_path, lg_path],
                                 stderr=log_file)
    push_proc.communicate()
    os.remove(temp_compose_path)

def compose_clg(in_disambig, out_disambig, context_width, central_pos, ilabels_temp, lg_path, clg_path, log_file):
    compose_proc = subprocess.Popen([thirdparty_binary('fstcomposecontext'),
                                     '--context-size={}'.format(context_width),
                                     '--central-position={}'.format(central_pos),
                                     '--read-disambig-syms={}'.format(in_disambig),
                                     '--write-disambig-syms={}'.format(out_disambig),
                                     ilabels_temp, lg_path], stdout=subprocess.PIPE, stderr=log_file)
    sort_proc = subprocess.Popen([thirdparty_binary('fstarcsort'), '--sort_type=ilabel', '-', clg_path],
                                 stdin=compose_proc.stdout, stderr=log_file)
    sort_proc.communicate()

def compose_hclg(model_directory, ilabels_temp, transition_scale, clg_path, hclga_path, log_file):
    model_path = os.path.join(model_directory, 'final.mdl')
    tree_path = os.path.join(model_directory, 'tree')
    ha_path = os.path.join(model_directory, 'Ha.fst')
    ha_out_disambig = os.path.join(model_directory, 'disambig_tid.int')
    make_h_proc = subprocess.Popen([thirdparty_binary('make-h-transducer'),
                                    '--disambig-syms-out={}'.format(ha_out_disambig),
                                    '--transition-scale={}'.format(transition_scale),
                                    ilabels_temp, tree_path, model_path, ha_path],
                                   stderr=log_file, stdout=log_file)
    make_h_proc.communicate()

    temp_compose_path = os.path.join(model_directory, 'HCLGa.temp')
    compose_proc = subprocess.Popen([thirdparty_binary('fsttablecompose'), ha_path,
                                     clg_path, temp_compose_path], stderr=log_file)
    compose_proc.communicate()

    determinize_proc = subprocess.Popen([thirdparty_binary('fstdeterminizestar'),
                                         '--use-log=true', temp_compose_path],
                                        stdout=subprocess.PIPE, stderr=log_file)
    rmsymbols_proc = subprocess.Popen([thirdparty_binary('fstrmsymbols'), ha_out_disambig],
                                      stdin=determinize_proc.stdout, stdout=subprocess.PIPE,
                                      stderr=log_file)
    rmeps_proc = subprocess.Popen([thirdparty_binary('fstrmepslocal')],
                                  stdin=rmsymbols_proc.stdout, stdout=subprocess.PIPE, stderr=log_file)
    minimize_proc = subprocess.Popen([thirdparty_binary('fstminimizeencoded'), '-', hclga_path],
                                     stdin=rmeps_proc.stdout, stderr=log_file)
    minimize_proc.communicate()
    os.remove(temp_compose_path)

def compose_g(arpa_path, words_path, g_path, log_file):
    arpafst_proc = subprocess.Popen([thirdparty_binary('arpa2fst'), '--disambig-symbol=#0',
                                     '--read-symbol-table=' + words_path,
                                     arpa_path, g_path], stderr=log_file,
                                    stdout=log_file)
    arpafst_proc.communicate()

def compose_g_carpa(in_carpa_path, temp_carpa_path, dictionary, carpa_path, log_file):
    bos_symbol = dictionary.words_mapping['<s>']
    eos_symbol = dictionary.words_mapping['</s>']
    unk_symbol = dictionary.words_mapping['<unk>']
    with open(in_carpa_path, 'r', encoding='utf8') as f, \
            open(temp_carpa_path, 'w', encoding='utf8') as outf:
        current_order = -1
        num_oov_lines = 0
        for line in f:
            line = line.strip()
            col = line.split()
            if current_order == -1 and not re.match(r'^\\data\\$', line):
                continue
            if re.match(r'^\\data\\$', line):
                log_file.write(r'Processing data...\n')
                current_order = 0
                outf.write(line + '\n')
            elif re.match(r'^\\[0-9]*-grams:$', line):
                current_order = int(re.sub(r'\\([0-9]*)-grams:$', r'\1', line))
                log_file.write('Processing {} grams...\n'.format(current_order))
                outf.write(line + '\n')
            elif re.match(r'^\\end\\$', line):
                outf.write(line + '\n')
            elif not line:
                if current_order >= 1:
                    outf.write('\n')
            else:
                if current_order == 0:
                    outf.write(line + '\n')
                else:
                    if len(col) > 2 + current_order or len(col) < 1 + current_order:
                        raise Exception('Bad line in arpa lm "{}"'.format(line))
                    prob = col.pop(0)
                    is_oov = False
                    for i in range(current_order):
                        try:
                            col[i] = str(dictionary.words_mapping[col[i]])
                        except KeyError:
                            is_oov = True
                            num_oov_lines += 1
                            break
                    if not is_oov:
                        rest_of_line = ' '.join(col)
                        outf.write('{}\t{}\n'.format(prob, rest_of_line))
    carpa_proc = subprocess.Popen([thirdparty_binary('arpa-to-const-arpa'),
                                   '--bos-symbol={}'.format(bos_symbol), '--eos-symbol={}'.format(eos_symbol),
                                   '--unk-symbol={}'.format(unk_symbol),
                                   temp_carpa_path, carpa_path], stdin=subprocess.PIPE,
                                  stderr=log_file,
                                  stdout=log_file)
    carpa_proc.communicate()
    os.remove(temp_carpa_path)

class Transcriber(object):
    min_language_model_weight = 7
    max_language_model_weight = 17
    word_insertion_penalties = [0, 0.5, 1.0]

    def __init__(self, corpus, dictionary, acoustic_model, language_model, transcribe_config, temp_directory=None,
                 call_back=None, debug=False, verbose=False, evaluation_mode=False, logger=None):
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
        self.log_dir = os.path.join(self.transcribe_directory, 'log')
        os.makedirs(self.log_dir, exist_ok=True)
        self.setup()

    @property
    def transcribe_directory(self):
        return os.path.join(self.temp_directory, 'transcribe')

    @property
    def model_directory(self):
        return os.path.join(self.temp_directory, 'models')

    def get_tree_info(self):
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

    def dictionaries_for_job(self, job_name):
        if isinstance(self.dictionary, MultispeakerDictionary):
            dictionary_names = []
            for name in self.dictionary.dictionary_mapping.keys():
                if os.path.exists(os.path.join(self.corpus.split_directory(), 'utt2spk.{}.{}'.format(job_name, name))):
                    dictionary_names.append(name)
            return dictionary_names
        return None

    def setup(self):
        dirty_path = os.path.join(self.model_directory, 'dirty')
        if os.path.exists(dirty_path):  # if there was an error, let's redo from scratch
            shutil.rmtree(self.model_directory)
        self.dictionary.write(disambig=True)
        self.corpus.initialize_corpus(self.dictionary, self.acoustic_model.feature_config)
        log_dir = os.path.join(self.model_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        context_width, central_pos = self.get_tree_info()
        ldet_temp_path = os.path.join(self.model_directory, 'Ldet.fst_temp')
        ldet_path = os.path.join(self.model_directory, 'Ldet.fst')
        log_path = os.path.join(log_dir, 'hclg.log')
        ilabels_temp = os.path.join(self.model_directory, 'ilabels_{}_{}'.format(context_width, central_pos))
        model_path = os.path.join(self.model_directory, 'final.mdl')
        lg_path = os.path.join(self.model_directory, 'LG.fst')
        clg_path = os.path.join(self.model_directory, 'CLG_{}_{}.fst'.format(context_width, central_pos))
        hclga_path = os.path.join(self.model_directory, 'HCLGa.fst')
        hclg_path = os.path.join(self.model_directory, 'HCLG.fst')
        dirty_path = os.path.join(self.model_directory, 'dirty')
        out_disambig = os.path.join(self.model_directory,
                                    'disambig_ilabels_{}_{}.int'.format(context_width, central_pos))

        try:

            with open(log_path, 'w') as log_file:
                if not self.dictionary.has_multiple:
                    small_g_path = os.path.join(self.model_directory, 'small_G.fst')
                    med_g_path = os.path.join(self.model_directory, 'med_G.fst')
                    carpa_path = os.path.join(self.model_directory, 'G.carpa')
                    temp_carpa_path = os.path.join(self.model_directory, 'G.carpa_temp')
                    words_path = os.path.join(self.model_directory, 'words.txt')
                    shutil.copyfile(self.dictionary.words_symbol_path, words_path)
                    if os.path.exists(hclg_path):
                        return
                    self.logger.info('Generating decoding graph...')
                        #if not os.path.exists(ldet_path):
                        #    self.logger.info('Generating Ldet.fst...')
                        #    with open(ldet_temp_path, 'w', encoding='utf8') as f:
                        #        print_proc = subprocess.Popen([thirdparty_binary('fstprint'), self.dictionary.disambig_path],
                        #                                      stdout=subprocess.PIPE, stderr=log_file, text=True)
                        #        for line in iter(print_proc.stdout.readline,''):
                        #
                        #            print(line)
                        #        error
                    if not os.path.exists(small_g_path):
                        self.logger.info('Generating small_G.fst...')
                        compose_g(self.language_model.small_arpa_path, words_path, small_g_path, log_file)
                        self.logger.info('Done!')
                    if not os.path.exists(med_g_path):
                        self.logger.info('Generating med_G.fst...')
                        compose_g(self.language_model.medium_arpa_path, words_path, med_g_path, log_file)
                        self.logger.info('Done!')
                    if not os.path.exists(carpa_path):
                        self.logger.info('Generating G.carpa...')
                        compose_g_carpa(self.language_model.carpa_path, temp_carpa_path, self.dictionary, carpa_path, log_file)
                        self.logger.info('Done!')
                    if not os.path.exists(lg_path):
                        self.logger.info('Generating LG.fst...')
                        compose_lg(self.model_directory, self.dictionary.disambig_path, small_g_path, lg_path, log_file)
                        self.logger.info('Done!')
                    if not os.path.exists(clg_path):
                        in_disambig = os.path.join(self.dictionary.phones_dir, 'disambig.int')
                        self.logger.info('Generating CLG.fst...')
                        compose_clg(in_disambig, out_disambig, context_width, central_pos, ilabels_temp, lg_path, clg_path,
                                    log_file)
                        self.logger.info('Done!')
                    if not os.path.exists(hclga_path):
                        self.logger.info('Generating HCLGa.fst...')
                        compose_hclg(self.model_directory, ilabels_temp, self.transcribe_config.transition_scale,
                                     clg_path, hclga_path, log_file)
                        self.logger.info('Done!')
                    self.logger.info('Generating HCLG.fst...')
                    self_loop_proc = subprocess.Popen([thirdparty_binary('add-self-loops'),
                                                       '--self-loop-scale={}'.format(
                                                           self.transcribe_config.self_loop_scale),
                                                       '--reorder=true', model_path, hclga_path],
                                                      stdout=subprocess.PIPE, stderr=log_file)
                    convert_proc = subprocess.Popen([thirdparty_binary('fstconvert'), '--fst_type=const', '-', hclg_path],
                                                    stdin=self_loop_proc.stdout, stderr=log_file)
                    convert_proc.communicate()
                else:
                    found_all = True
                    for name in self.dictionary.dictionary_mapping.keys():
                        hclg_path = os.path.join(self.model_directory, name + '_HCLG.fst')
                        if not os.path.exists(hclg_path):
                            found_all = False
                    if found_all:
                        return
                    self.logger.info('Generating decoding graph...')

                    for name, d in self.dictionary.dictionary_mapping.items():
                        words_path = os.path.join(self.model_directory, name + '_words.txt')
                        shutil.copyfile(d.words_symbol_path, words_path)
                        small_g_path = os.path.join(self.model_directory, name + '_small_G.fst')
                        med_g_path = os.path.join(self.model_directory, name + '_med_G.fst')
                        carpa_path = os.path.join(self.model_directory, name + '_G.carpa')
                        temp_carpa_path = os.path.join(self.model_directory, name + '_G.carpa_temp')
                        if not os.path.exists(small_g_path):
                            self.logger.info(f'Generating small_G.fst for {name}...')
                            compose_g(self.language_model.small_arpa_path, words_path, small_g_path, log_file)
                            self.logger.info('Done!')
                        if not os.path.exists(med_g_path):
                            self.logger.info(f'Generating med_G.fst for {name}...')
                            compose_g(self.language_model.medium_arpa_path, words_path, med_g_path, log_file)
                            self.logger.info('Done!')
                        if not os.path.exists(carpa_path):
                            self.logger.info(f'Generating G.carpa for {name}...')
                            compose_g_carpa(self.language_model.carpa_path, temp_carpa_path, d, carpa_path, log_file)
                            self.logger.info('Done!')
                        lg_path = os.path.join(self.model_directory, name + '_LG.fst')
                        if not os.path.exists(lg_path):
                            self.logger.info(f'Generating {name}_LG.fst...')
                            compose_lg(self.model_directory, d.disambig_path, small_g_path, lg_path, log_file)
                            self.logger.info('Done!')
                        clg_path = os.path.join(self.model_directory, name + '_CLG_{}_{}.fst'.format(context_width, central_pos))
                        if not os.path.exists(clg_path):
                            in_disambig = os.path.join(d.phones_dir, 'disambig.int')
                            self.logger.info(f'Generating {name}_CLG.fst...')
                            compose_clg(in_disambig, out_disambig, context_width, central_pos, ilabels_temp, lg_path, clg_path, log_file)
                            self.logger.info('Done!')
                        hclga_path = os.path.join(self.model_directory, name + '_HCLGa.fst')
                        if not os.path.exists(hclga_path):
                            self.logger.info(f'Generating {name}_HCLGa.fst...')
                            compose_hclg(self.model_directory, ilabels_temp, self.transcribe_config.transition_scale,
                                         clg_path, hclga_path, log_file)
                            self.logger.info('Done!')
                        hclg_path = os.path.join(self.model_directory, name + '_HCLG.fst')
                        if not os.path.exists(hclg_path):
                            self.logger.info(f'Generating {name}_HCLG.fst...')
                            self_loop_proc = subprocess.Popen([thirdparty_binary('add-self-loops'),
                                                               '--self-loop-scale={}'.format(self.transcribe_config.self_loop_scale),
                                                               '--reorder=true', model_path, hclga_path],
                                                              stdout=subprocess.PIPE, stderr=log_file)
                            convert_proc = subprocess.Popen([thirdparty_binary('fstconvert'), '--fst_type=const', '-', hclg_path],
                                                            stdin=self_loop_proc.stdout, stderr=log_file)
                            convert_proc.communicate()
                            self.logger.info('Done!')

                        parse_logs(log_dir)
                        self.logger.info(f'Finished graph construction for {name}!')
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise

    def transcribe(self):
        self.logger.info('Beginning transcription...')
        dirty_path = os.path.join(self.transcribe_directory, 'dirty')
        if os.path.exists(dirty_path):
            shutil.rmtree(self.transcribe_directory, ignore_errors=True)
        os.makedirs(self.log_dir,exist_ok=True)
        try:
            transcribe(self)
            if self.transcribe_config.fmllr and not self.transcribe_config.no_speakers:
                self.logger.info('Performing speaker adjusted transcription...')
                transcribe_fmllr(self)
        except Exception as e:
            with open(dirty_path, 'w'):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise

    def evaluate(self, output_directory, input_directory=None):
        self.logger.info('Evaluating transcripts...')
        transcripts = self._load_transcripts(input_directory)
        # Sentence-level measures

        correct = 0
        incorrect = 0
        # Word-level measures
        total_edits = 0
        total_length = 0
        issues = []
        with mp.Pool(self.corpus.num_jobs) as pool:
            to_comp = []
            for utt, pred in transcripts.items():
                g = self.corpus.text_mapping[utt].split()
                h = pred.split()
                if g != h:
                    issues.append((utt, g, h))
                to_comp.append((g, h))
            gen = pool.map(score, to_comp)
            for (edits, length) in gen:
                if edits == 0:
                    correct += 1
                else:
                    incorrect += 1
                total_edits += edits
                total_length += length
            for utt, gold in self.corpus.text_mapping.items():
                if utt not in transcripts:
                    incorrect += 1
                    gold = gold.split()
                    total_edits += len(gold)
                    total_length += len(gold)
        ser = 100 * incorrect / (correct + incorrect)
        wer = 100 * total_edits / total_length
        output_path = os.path.join(output_directory, 'transcription_issues.csv')
        with open(output_path, 'w', encoding='utf8') as f:
            for utt, g, h in issues:
                g = ' '.join(g)
                h = ' '.join(h)
                f.write('{},{},{}\n'.format(utt, g, h))
        self.logger.info('SER: {:.2f}%, WER: {:.2f}%'.format(ser, wer))
        return ser, wer

    def _load_transcripts(self, input_directory=None):
        transcripts = {}
        lookup = self.dictionary.reversed_word_mapping
        if input_directory is None:
            input_directory = self.transcribe_directory
            if self.transcribe_config.fmllr and not self.transcribe_config.no_speakers:
                input_directory = os.path.join(input_directory, 'fmllr')
        for j in range(self.corpus.num_jobs):
            tra_path = os.path.join(input_directory, 'tra.{}'.format(j))
            if os.path.exists(tra_path):

                with open(tra_path, 'r', encoding='utf8') as f:
                    for line in f:
                        t = line.strip().split(' ')
                        utt = t[0]
                        ints = t[1:]
                        if not ints:
                            continue
                        transcription = []
                        for i in ints:
                            transcription.append(lookup[int(i)])
                        transcripts[utt] = ' '.join(transcription)
            else:
                for name in self.dictionary.dictionary_mapping.keys():
                    tra_path = os.path.join(input_directory, 'tra.{}.{}'.format(j, name))
                    if not os.path.exists(tra_path):
                        continue
                    with open(tra_path, 'r', encoding='utf8') as f:
                        for line in f:
                            t = line.strip().split(' ')
                            utt = t[0]
                            ints = t[1:]
                            if not ints:
                                continue
                            transcription = []
                            for i in ints:
                                transcription.append(lookup[int(i)])
                            transcripts[utt] = ' '.join(transcription)

        return transcripts

    def export_transcriptions(self, output_directory, source=None):
        transcripts = self._load_transcripts(source)
        if not self.corpus.segments:
            for utt, t in transcripts.items():
                relative = self.corpus.file_directory_mapping[utt]
                if relative:
                    speaker_directory = os.path.join(output_directory, relative)
                else:
                    speaker_directory = output_directory
                os.makedirs(speaker_directory, exist_ok=True)
                outpath = os.path.join(speaker_directory, utt + '.lab')
                with open(outpath, 'w', encoding='utf8') as f:
                    f.write(t)

        else:

            for filename in self.corpus.file_directory_mapping.keys():
                maxtime = self.corpus.get_wav_duration(filename)
                speaker_directory = output_directory
                try:
                    if self.corpus.file_directory_mapping[filename]:
                        speaker_directory = os.path.join(output_directory, self.corpus.file_directory_mapping[filename])
                except KeyError:
                    pass
                tiers = {}
                if self.transcribe_config.no_speakers:
                    speaker = 'speech'
                    tiers[speaker] = tgio.IntervalTier(speaker, [], minT=0, maxT=maxtime)
                else:
                    for speaker in self.corpus.speaker_ordering[filename]:
                        tiers[speaker] = tgio.IntervalTier(speaker, [], minT=0, maxT=maxtime)

                tg = tgio.Textgrid()
                tg.maxTimestamp = maxtime
                for utt_name, text in transcripts.items():
                    seg = self.corpus.segments[utt_name]
                    utt_filename, begin, end = seg['file_name'], seg['begin'], seg['end']
                    if utt_filename != filename:
                        continue
                    if self.transcribe_config.no_speakers:
                        speaker = 'speech'
                    else:
                        speaker = self.corpus.utt_speak_mapping[utt_name]
                    begin = float(begin)
                    end = float(end)
                    tiers[speaker].entryList.append((begin, end, text))
                for t in tiers.values():
                    tg.addTier(t)
                tg.save(os.path.join(speaker_directory, filename + '.TextGrid'), useShortForm=False)
