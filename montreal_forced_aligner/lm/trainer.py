import os
import subprocess
import random
from ..models import LanguageModel
from ..corpus import AlignableCorpus
from ..helper import thirdparty_binary
from ..exceptions import LMError
from ..config import TEMP_DIR


class LmTrainer(object):
    """
    Train a language model from a corpus with text, or convert an existing ARPA-format language model to MFA format

    Parameters
    ----------
    source: class:`~montreal_forced_aligner.corpus.AlignableCorpus` or str
        Either a alignable corpus or a path to an ARPA format language model
    config : class:`~montreal_forced_aligner.config.TrainLMConfig`
        Config class for training language model
    output_model_path : str
        Path to output trained model
    dictionary : class:`~montreal_forced_aligner.dictionary.Dictionary`, optional
        Optional dictionary to calculate unknown words
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    supplemental_model_path: str, optional
        Path to second language model to merge with the trained model
    supplemental_model_weight : float, optional
        Weight of supplemental model when merging, defaults to 1
    """

    def __init__(self, source, config, output_model_path, dictionary=None, temp_directory=None,
                 supplemental_model_path=None, supplemental_model_weight=1):
        if not temp_directory:
            temp_directory = TEMP_DIR
        temp_directory = os.path.join(temp_directory, 'LM')

        self.name, _ = os.path.splitext(os.path.basename(output_model_path))
        self.temp_directory = os.path.join(temp_directory, self.name)
        os.makedirs(self.temp_directory, exist_ok=True)
        self.source = source
        self.dictionary = dictionary
        self.output_model_path = output_model_path
        self.config = config
        self.supplemental_model_path = supplemental_model_path
        self.source_model_weight = 1
        self.supplemental_model_weight = supplemental_model_weight

    def train(self):
        mod_path = os.path.join(self.temp_directory, self.name + '.mod')
        if isinstance(self.source, AlignableCorpus):
            sym_path = os.path.join(self.temp_directory, self.name + '.sym')
            far_path = os.path.join(self.temp_directory, self.name + '.far')
            cnts_path = os.path.join(self.temp_directory, self.name + '.cnts')
            training_path = os.path.join(self.temp_directory, 'training.txt')

            with open(training_path, 'w', encoding='utf8') as f:
                for text in self.source.normalized_text_iter(self.dictionary, self.config.count_threshold):
                    f.write(text + "\n")

            subprocess.call(['ngramsymbols', training_path, sym_path])
            subprocess.call(['farcompilestrings', '--fst_type=compact',
                             '--symbols=' + sym_path, '--keep_symbols', training_path, far_path])
            subprocess.call(['ngramcount', '--order={}'.format(self.config.order), far_path,  cnts_path])
            subprocess.call(['ngrammake', '--method={}'.format(self.config.method), cnts_path, mod_path])
        else:
            temp_text_path = os.path.join(self.temp_directory, 'input.arpa')
            with open(self.source, 'r', encoding='utf8') as inf, open(temp_text_path, 'w', encoding='utf8') as outf:
                for line in inf:
                    outf.write(line.lower())
            subprocess.call(['ngramread', '--ARPA', temp_text_path, mod_path])
            os.remove(temp_text_path)
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





