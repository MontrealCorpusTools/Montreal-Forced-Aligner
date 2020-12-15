import os
import random
from ..models import LanguageModel
from ..helper import thirdparty_binary
from ..exceptions import LMError
from ..config import TEMP_DIR


class LmTrainer(object):
    """
    Train a language model from a corpus with text
    """

    def __init__(self, corpus, config, output_model_path, dictionary=None, temp_directory=None, num_jobs=3):
        if not temp_directory:
            temp_directory = TEMP_DIR
        temp_directory = os.path.join(temp_directory, 'LM')

        self.name, _ = os.path.splitext(os.path.basename(output_model_path))
        self.temp_directory = os.path.join(temp_directory, self.name)
        os.makedirs(self.temp_directory, exist_ok=True)
        self.corpus = corpus
        self.dictionary = dictionary
        self.output_model_path = output_model_path
        self.config = config
        self.num_jobs = num_jobs

    def init_training(self):
        random.seed(self.config['seed'])

        print(self.config)
        # Set up utterances
        utterances = [x for x in self.corpus.text_mapping.values()]
        random.shuffle(utterances)
        train_directory = os.path.join(self.temp_directory, 'train')
        os.makedirs(train_directory, exist_ok=True)
        real_dev_path = os.path.join(self.temp_directory, 'dev.txt')
        dev_path = os.path.join(train_directory, 'dev.txt')
        train_path = os.path.join(train_directory, 'train.txt')
        if self.dictionary is not None:
            word_list = self.dictionary.words.keys()
            compute_dict = False
        else:
            word_list = set()
            compute_dict = True
        with open(dev_path, 'w', encoding='utf8') as devf, \
            open(real_dev_path, 'w', encoding='utf8') as realdevf, \
            open(train_path, 'w', encoding='utf8') as trainf:
            for i, u in enumerate(utterances):
                if i < self.config['num_dev_utterances']:
                    devf.write(u)
                    devf.write('\n')
                elif self.config['num_dev_utterances'] <= i < self.config['num_dev_utterances'] * 2:
                    realdevf.write(u)
                    realdevf.write('\n')
                else:
                    trainf.write(u)
                    trainf.write('\n')
                if compute_dict:
                    word_list.update(u.split(' '))
        with open(os.path.join(self.temp_directory, 'wordlist'), 'w', encoding='utf8') as f:
            for w in sorted(word_list):
                f.write(w)
                f.write('\n')

    def train(self):
        self.init_training()



