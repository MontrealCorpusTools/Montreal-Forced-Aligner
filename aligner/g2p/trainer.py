import subprocess
import os
import random
import re
import tempfile
from ..dictionary import Dictionary

from ..helper import thirdparty_binary

from ..config import TEMP_DIR

from ..models import G2PModel

from ..exceptions import G2PError


class PhonetisaurusTrainer(object):
    """Train a g2p model from a pronunciation dictionary

    Parameters
    ----------
    language: str
        the path and language code
    input_dict : str
        path to the pronunciation dictionary

    """

    def __init__(self, dictionary, model_path, temp_directory=None, window_size=2, evaluate=False):
        super(PhonetisaurusTrainer, self).__init__()
        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = os.path.join(temp_directory, 'G2P')

        self.name, _ = os.path.splitext(os.path.basename(model_path))
        self.temp_directory = os.path.join(temp_directory, self.name)
        os.makedirs(self.temp_directory, exist_ok=True)
        self.model_path = model_path
        self.grapheme_window_size = 2
        self.phoneme_window_size = window_size
        self.evaluate = evaluate
        self.dictionary = dictionary

    def train(self, word_dict=None):
        input_path = os.path.join(self.temp_directory, 'input.txt')
        if word_dict is None:
            word_dict = self.dictionary.words
        with open(input_path, "w", encoding='utf8') as f2:
            for word, v in word_dict.items():
                if re.match(r'\W', word) is not None:
                    continue
                for v2 in v:
                    f2.write(word + "\t" + " ".join(v2[0]) + "\n")

        corpus_path = os.path.join(self.temp_directory, 'full.corpus')
        sym_path = os.path.join(self.temp_directory, 'full.syms')
        far_path = os.path.join(self.temp_directory, 'full.far')
        cnts_path = os.path.join(self.temp_directory, 'full.cnts')
        mod_path = os.path.join(self.temp_directory, 'full.mod')
        arpa_path = os.path.join(self.temp_directory, 'full.arpa')
        fst_path = os.path.join(self.temp_directory, 'model.fst')

        align_proc = subprocess.Popen([thirdparty_binary('phonetisaurus-align'),
                                       '--seq1_max={}'.format(self.grapheme_window_size),
                                       '--seq2_max={}'.format(self.phoneme_window_size),
                                       '--input=' + input_path, '--ofile=' + corpus_path],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
        stdout, stderr = align_proc.communicate()
        #if stderr:
        #    raise G2PError('There was an error in {}: {}'.format('phonetisaurus-align', stderr.decode('utf8')))

        ngramsymbols_proc = subprocess.Popen([thirdparty_binary('ngramsymbols'),
                                              corpus_path, sym_path],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE)
        stdout, stderr = ngramsymbols_proc.communicate()
        if stderr:
            raise G2PError('There was an error in {}: {}'.format('ngramsymbols', stderr.decode('utf8')))

        farcompile_proc = subprocess.Popen([thirdparty_binary('farcompilestrings'),
                                            '--symbols=' + sym_path, '--keep_symbols=1',
                                            corpus_path, far_path],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
        stdout, stderr = farcompile_proc.communicate()
        if stderr:
            raise G2PError('There was an error in {}: {}'.format('farcompilestrings', stderr.decode('utf8')))

        ngramcount_proc = subprocess.Popen([thirdparty_binary('ngramcount'),
                                            '--order=7', far_path, cnts_path],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
        stdout, stderr = ngramcount_proc.communicate()
        if stderr:
            raise G2PError('There was an error in {}: {}'.format('ngramcount', stderr.decode('utf8')))

        ngrammake_proc = subprocess.Popen([thirdparty_binary('ngrammake'),
                                           '--method=kneser_ney', cnts_path, mod_path],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
        stdout, stderr = ngrammake_proc.communicate()
        if stderr:
            raise G2PError('There was an error in {}: {}'.format('ngrammake', stderr.decode('utf8')))

        ngramprint_proc = subprocess.Popen([thirdparty_binary('ngramprint'),
                                            '--ARPA', mod_path, arpa_path],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
        stdout, stderr = ngramprint_proc.communicate()
        if stderr:
            raise G2PError('There was an error in {}: {}'.format('ngramprint', stderr.decode('utf8')))

        arpa2wfst_proc = subprocess.Popen([thirdparty_binary('phonetisaurus-arpa2wfst'),
                                           '--lm=' + arpa_path, '--ofile=' + fst_path],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
        stdout, stderr = arpa2wfst_proc.communicate()

        #if stderr:
        #    raise G2PError('There was an error in {}: {}'.format('phonetisaurus-arpa2wfst', stderr.decode('utf8')))

        directory, filename = os.path.split(self.model_path)
        basename, _ = os.path.splitext(filename)
        model = G2PModel.empty(basename)
        model.add_meta_file(self.dictionary)
        model.add_fst_model(self.temp_directory)
        os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(self.model_path)
        model.dump(basename)
        print('Saved model to {}'.format(self.model_path))

    def validate(self):
        from .generator import PhonetisaurusDictionaryGenerator
        from ..models import G2PModel
        print('Performing validation...')
        word_dict = self.dictionary.words
        validation = 0.1
        words = word_dict.keys()
        total_items = len(words)
        validation_items = int(total_items * validation)
        validation_words = random.sample(words, validation_items)
        training_dictionary = {k: v for k, v in word_dict.items() if k not in validation_words}
        validation_dictionary = {k: v for k, v in word_dict.items() if k in validation_words}
        self.train(training_dictionary)

        model = G2PModel(self.model_path)
        output_path = os.path.join(self.temp_directory, 'validation.txt')
        validation_errors_path = os.path.join(self.temp_directory, 'validation_errors.txt')
        gen = PhonetisaurusDictionaryGenerator(model, validation_dictionary.keys(),
                                               output_path,
                                               temp_directory=os.path.join(self.temp_directory, 'validation'))
        gen.generate()
        count_right = 0

        with open(output_path, 'r', encoding='utf8') as f, \
                open(validation_errors_path, 'w', encoding='utf8') as outf:
            for line in f:
                line = line.strip().split()
                word = line[0]
                pron = ' '.join(line[1:])
                actual_prons = set(' '.join(x[0]) for x in validation_dictionary[word])
                if pron not in actual_prons:
                    outf.write('{}\t{}\t{}\n'.format(word, pron, ', '.join(actual_prons)))
                else:
                    count_right += 1
        accuracy = count_right / validation_items
        print('Accuracy was: {}'.format(accuracy))
        os.remove(self.model_path)

        return accuracy
