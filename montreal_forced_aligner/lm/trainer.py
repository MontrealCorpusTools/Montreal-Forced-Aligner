import os
import subprocess
import random
from ..models import LanguageModel
from ..helper import thirdparty_binary
from ..exceptions import LMError
from ..config import TEMP_DIR


class LmTrainer(object):
    """
    Train a language model from a corpus with text
    """

    def __init__(self, corpus, config, output_model_path, dictionary=None, temp_directory=None, num_jobs=3,
                 supplemental_model_path=None, supplemental_model_weight=1):
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
        self.supplemental_model_path = supplemental_model_path
        self.source_model_weight = 1
        self.supplemental_model_weight = 1

    def train(self):
        sym_path = os.path.join(self.temp_directory, self.name + '.sym')
        far_path = os.path.join(self.temp_directory, self.name + '.far')
        cnts_path = os.path.join(self.temp_directory, self.name + '.cnts')
        mod_path = os.path.join(self.temp_directory, self.name + '.mod')
        training_path = os.path.join(self.temp_directory, 'training.txt')
        with open(training_path, 'w', encoding='utf8') as f:
            for utt, text in self.corpus.text_mapping.items():
                f.write(text + "\n")

        subprocess.call(['ngramsymbols', training_path, sym_path])
        subprocess.call(['farcompilestrings', '--fst_type=compact',
                         '--symbols=' + sym_path, '--keep_symbols', training_path, far_path])
        subprocess.call(['ngramcount', '--order={}'.format(self.config.order), far_path,  cnts_path])
        subprocess.call(['ngrammake', '--method={}'.format(self.config.method), cnts_path, mod_path])
        if self.supplemental_model_path:
            supplemental_path = os.path.join(self.temp_directory, 'extra.mod')
            merged_path = os.path.join(self.temp_directory, 'merged.mod')
            subprocess.call(['ngramread', '--ARPA', self.supplemental_model_path, supplemental_path])
            subprocess.call(['ngrammerge', '--normalize',
                             '--alpha={}'.format(self.source_model_weight),
                             '--beta={}'.format(self.supplemental_model_weight),
                             mod_path, supplemental_path, merged_path])
            mod_path = merged_path

        subprocess.call(['ngramprint', '--ARPA', mod_path, self.output_model_path])

        if self.config.prune:
            small_mod_path = mod_path.replace('.mod', '_small.mod')
            med_mod_path = mod_path.replace('.mod', '_med.mod')
            subprocess.call(['ngramshrink', '--method=relative_entropy',
                             '--theta={}'.format(self.config.prune_thresh_small),
                             mod_path, small_mod_path])
            subprocess.call(['ngramshrink', '--method=relative_entropy',
                             '--theta={}'.format(self.config.prune_thresh_medium),
                             mod_path, med_mod_path])
            small_output_path = self.output_model_path.replace('.arpa', '_small.arpa')
            med_output_path = self.output_model_path.replace('.arpa', '_med.arpa')
            subprocess.call(['ngramprint', '--ARPA', small_mod_path, small_output_path])
            subprocess.call(['ngramprint', '--ARPA', med_mod_path, med_output_path])





