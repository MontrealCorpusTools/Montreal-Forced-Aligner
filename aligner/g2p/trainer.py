import subprocess
import os
import random
import tempfile
from ..dictionary import Dictionary

from ..helper import thirdparty_binary

from ..config import TEMP_DIR

from ..models import G2PModel


class PhonetisaurusTrainer(object):
    """Train a g2p model from a pronunciation dictionary

    Parameters
    ----------
    language: str
        the path and language code
    input_dict : str
        path to the pronunciation dictionary

    """

    def __init__(self, dictionary, model_path, temp_directory=None, korean=False, evaluate=False):
        super(PhonetisaurusTrainer, self).__init__()
        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = os.path.join(temp_directory, 'G2P')

        self.name, _ = os.path.splitext(os.path.basename(model_path))
        self.temp_directory = os.path.join(temp_directory, self.name)
        os.makedirs(self.temp_directory, exist_ok=True)
        self.model_path = model_path
        self.korean = korean
        self.evaluate = evaluate
        self.dictionary = dictionary

    def train(self, word_dict = None):
        if self.korean:
            try:
                from jamo import h2j, j2hcj
            except ImportError:
                raise (Exception('Cannot parse hangul into jamo for increased accuracy, please run `pip install jamo`.'))
        input_path = os.path.join(self.temp_directory, 'input.txt')
        if word_dict is None:
            word_dict = self.dictionary.words
        with open(input_path, "w") as f2:
            for word, v in word_dict.items():
                for v2 in v:
                    if self.korean:
                        word = j2hcj(h2j(word))
                    f2.write(word + "\t" + " ".join(v2[0]) + "\n")

        corpus_path = os.path.join(self.temp_directory, 'full.corpus')
        sym_path = os.path.join(self.temp_directory, 'full.syms')
        far_path = os.path.join(self.temp_directory, 'full.far')
        cnts_path = os.path.join(self.temp_directory, 'full.cnts')
        mod_path = os.path.join(self.temp_directory, 'full.mod')
        arpa_path = os.path.join(self.temp_directory, 'full.arpa')
        fst_path = os.path.join(self.temp_directory, 'model.fst')

        subprocess.call([thirdparty_binary('phonetisaurus-align'),
                         '--input=' + input_path, '--ofile=' + corpus_path])

        subprocess.call([thirdparty_binary('ngramsymbols'), corpus_path, sym_path])

        subprocess.call(
                [thirdparty_binary('farcompilestrings'), '--symbols=' + sym_path, '--keep_symbols=1',
                 corpus_path, far_path])

        subprocess.call([thirdparty_binary('ngramcount'), '--order=7', far_path, cnts_path])

        subprocess.call([thirdparty_binary('ngrammake'), '--method=kneser_ney', cnts_path, mod_path])

        subprocess.call([thirdparty_binary('ngramprint'), '--ARPA', mod_path, arpa_path])

        subprocess.call(
                [thirdparty_binary('phonetisaurus-arpa2wfst'), '--lm=' + arpa_path, '--ofile=' + fst_path])

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
        word_dict = self.dictionary.words
        validation = 0.1
        words = word_dict.keys()
        total_items = len(words)
        validation_items = int(total_items * validation)
        validation_words = random.sample(words, validation_items)
        training_dictionary = {k:v for k,v in word_dict.items() if k not in validation_words}
        validation_dictionary = {k:v for k,v in word_dict.items() if k in validation_words}
        self.train(training_dictionary)
        count_right = 0
        incorrect = []
        for k,v in validation_dictionary.items():
            actual = list(self.g2p([k])[0])
            if actual == v:
                count_right += 1
            else:
                incorrect.append((k, v, actual))
        print('Accuracy was: {}'.format(count_right/validation_items))
        print(incorrect[:100])
