import subprocess
import os
import shutil
import multiprocessing as mp
from textgrid import TextGrid, IntervalTier
from .config import TEMP_DIR
from .helper import thirdparty_binary
from .multiprocessing import transcribe, transcribe_fmllr
from .corpus import AlignableCorpus
from .helper import score


class Transcriber(object):
    min_language_model_weight = 7
    max_language_model_weight = 17
    word_insertion_penalties = [0, 0.5, 1.0]

    def __init__(self, corpus, dictionary, acoustic_model, language_model, transcribe_config, temp_directory=None,
                 call_back=None, debug=False, verbose=False, evaluation_mode=False):
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
        self.acoustic_model.export_model(self.transcribe_directory)
        self.log_dir = os.path.join(self.transcribe_directory, 'log')
        os.makedirs(self.log_dir, exist_ok=True)
        self.setup()

    @property
    def transcribe_directory(self):
        return os.path.join(self.temp_directory, 'transcribe')

    def get_tree_info(self):
        tree_proc = subprocess.Popen([thirdparty_binary('tree-info'),
                            os.path.join(self.transcribe_directory, 'tree')], text=True,
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

    def setup(self):
        self.dictionary.write(disambig=True)
        if isinstance(self.corpus, AlignableCorpus):
            self.corpus.initialize_corpus(self.dictionary)
        else:
            self.corpus.initialize_corpus()
        self.acoustic_model.feature_config.generate_features(self.corpus)

        context_width, central_pos = self.get_tree_info()
        g_path = os.path.join(self.transcribe_directory, 'G.fst')
        lg_path = os.path.join(self.transcribe_directory, 'LG.fst')
        clg_path = os.path.join(self.transcribe_directory, 'CLG_{}_{}.fst'.format(context_width, central_pos))
        log_path = os.path.join(self.log_dir, 'hclg.log')
        in_disambig = os.path.join(self.dictionary.phones_dir, 'disambig.int')
        out_disambig = os.path.join(self.transcribe_directory,
                                    'disambig_ilabels_{}_{}.int'.format(context_width, central_pos))
        ha_out_disambig = os.path.join(self.transcribe_directory, 'disambig_tid.int')
        ilabels_temp = os.path.join(self.transcribe_directory, 'ilabels_{}_{}'.format(context_width, central_pos))
        tree_path = os.path.join(self.transcribe_directory, 'tree')
        model_path = os.path.join(self.transcribe_directory, 'final.mdl')
        ha_path = os.path.join(self.transcribe_directory, 'Ha.fst')
        hclga_path = os.path.join(self.transcribe_directory, 'HCLGa.fst')
        hclg_path = os.path.join(self.transcribe_directory, 'HCLG.fst')
        shutil.copyfile(self.dictionary.words_symbol_path, os.path.join(self.transcribe_directory, 'words.txt'))
        if os.path.exists(hclg_path):
            return
        print('Generating decoding graph...')
        with open(log_path, 'w') as log_file:
            if not os.path.exists(g_path):
                print('Generating G.fst...')
                arpafst_proc = subprocess.Popen([thirdparty_binary('arpa2fst'), '--disambig-symbol=#0',
                                 '--read-symbol-table=' + self.dictionary.words_symbol_path,
                                 self.language_model.arpa_path, g_path], stderr=log_file, stdout=log_file)
                arpafst_proc.communicate()
                print('Done!')
            if not os.path.exists(lg_path):
                print('Generating LG.fst...')
                temp_compose_path = os.path.join(self.transcribe_directory, 'LG.temp')
                compose_proc = subprocess.Popen([thirdparty_binary('fsttablecompose'),
                                                 self.dictionary.disambig_path, g_path, temp_compose_path],
                                                stderr=log_file)
                compose_proc.communicate()

                temp2_compose_path = os.path.join(self.transcribe_directory, 'LG.temp2')
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
                print('Done!')
            if not os.path.exists(clg_path):
                print('Generating CLG.fst...')
                compose_proc = subprocess.Popen([thirdparty_binary('fstcomposecontext'),
                                                 '--context-size={}'.format(context_width),
                                                 '--central-position={}'.format(central_pos),
                                                 '--read-disambig-syms={}'.format(in_disambig),
                                                 '--write-disambig-syms={}'.format(out_disambig),
                                                 ilabels_temp, lg_path], stdout=subprocess.PIPE, stderr=log_file)
                sort_proc = subprocess.Popen([thirdparty_binary('fstarcsort'), '--sort_type=ilabel', '-', clg_path],
                                             stdin=compose_proc.stdout, stderr=log_file)
                sort_proc.communicate()
                print('Done!')
            if not os.path.exists(hclga_path):
                print('Generating HCLGa.fst...')
                make_h_proc = subprocess.Popen([thirdparty_binary('make-h-transducer'),
                                                '--disambig-syms-out={}'.format(ha_out_disambig),
                                                '--transition-scale={}'.format(self.transcribe_config.transition_scale),
                                                ilabels_temp, tree_path, model_path, ha_path],
                                               stderr=log_file, stdout=log_file)
                make_h_proc.communicate()

                temp_compose_path = os.path.join(self.transcribe_directory, 'HCLGa.temp')
                compose_proc = subprocess.Popen([thirdparty_binary('fsttablecompose'), ha_path,
                                                 clg_path, temp_compose_path], stderr=log_file)
                compose_proc.communicate()

                determinize_proc = subprocess.Popen([thirdparty_binary('fstdeterminizestar'),
                                                     '--use-log=true', temp_compose_path],
                                                    stdout=subprocess.PIPE, stderr=log_file)
                rmsymbols_proc = subprocess.Popen([thirdparty_binary('fstrmsymbols'), ha_out_disambig],
                                                  stdin=determinize_proc.stdout, stdout=subprocess.PIPE, stderr=log_file)
                rmeps_proc = subprocess.Popen([thirdparty_binary('fstrmepslocal')],
                                              stdin=rmsymbols_proc.stdout, stdout=subprocess.PIPE, stderr=log_file)
                minimize_proc = subprocess.Popen([thirdparty_binary('fstminimizeencoded'), '-', hclga_path],
                                                 stdin=rmeps_proc.stdout, stderr=log_file)
                minimize_proc.communicate()
                os.remove(temp_compose_path)
                print('Done!')
            print('Finishing up...')
            self_loop_proc = subprocess.Popen([thirdparty_binary('add-self-loops'),
                                               '--self-loop-scale={}'.format(self.transcribe_config.self_loop_scale),
                                               '--reorder=true', model_path, hclga_path],
                                              stdout=subprocess.PIPE, stderr=log_file)
            convert_proc = subprocess.Popen([thirdparty_binary('fstconvert'), '--fst_type=const', '-', hclg_path],
                                            stdin=self_loop_proc.stdout, stderr=log_file)
            convert_proc.communicate()
            print('Finished graph construction!')

    def transcribe(self):
        print('Beginning transcription...')
        transcribe(self)
        if self.transcribe_config.fmllr:
            print('Performing speaker adjusted transcription...')
            transcribe_fmllr(self)

    def evaluate(self, output_directory, input_directory=None):
        print('Evaluating transcripts...')
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
        print('SER: {:.2f}%, WER: {:.2f}%'.format(ser, wer))
        return ser, wer

    def _load_transcripts(self, input_directory=None):
        transcripts = {}
        lookup = self.dictionary.reversed_word_mapping
        if input_directory is None:
            input_directory = self.transcribe_directory
            if self.transcribe_config.fmllr:
                input_directory = os.path.join(input_directory, 'fmllr')
        for i in range(self.corpus.num_jobs):
            with open(os.path.join(input_directory, 'tra.{}'.format(i)), 'r', encoding='utf8') as f:
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
        print(self.corpus.file_directory_mapping)
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
            for filename in self.corpus.speaker_ordering.keys():
                maxtime = self.corpus.get_wav_duration(filename)
                try:
                    speaker_directory = os.path.join(output_directory, self.corpus.file_directory_mapping[filename])
                except KeyError:
                    speaker_directory = output_directory
                tiers = {}
                for speaker in self.corpus.speaker_ordering[filename]:
                    tiers[speaker] = IntervalTier(name=speaker, maxTime=maxtime)

                tg = TextGrid(maxTime=maxtime)
                for utt_name, text in transcripts.items():
                    utt_filename, begin, end = self.corpus.segments[utt_name].split(' ')
                    if utt_filename != filename:
                        continue
                    speaker = self.corpus.utt_speak_mapping[utt_name]
                    begin = float(begin)
                    end = float(end)
                    tiers[speaker].add(begin, end, text)
                for t in tiers.values():
                    tg.append(t)
                tg.write(os.path.join(speaker_directory, filename + '.TextGrid'))