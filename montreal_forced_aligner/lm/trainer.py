import os
import re
import subprocess
from ..models import LanguageModel
from ..corpus import AlignableCorpus
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
                 supplemental_model_path=None, supplemental_model_weight=1, debug=False, logger=None):
        if not temp_directory:
            temp_directory = TEMP_DIR
        self.logger = logger
        temp_directory = os.path.join(temp_directory, 'LM')
        self.debug = debug
        self.name, _ = os.path.splitext(os.path.basename(output_model_path))
        self.temp_directory = os.path.join(temp_directory, self.name)
        self.models_temp_dir = os.path.join(self.temp_directory, 'models')
        self.log_directory = os.path.join(self.temp_directory, 'logs')
        os.makedirs(self.log_directory, exist_ok=True)
        self.source = source
        self.dictionary = dictionary
        self.output_model_path = output_model_path
        self.config = config
        self.supplemental_model_path = supplemental_model_path
        self.source_model_weight = 1
        self.supplemental_model_weight = supplemental_model_weight

    @property
    def meta(self):
        from .. import __version__
        return {'type': 'ngram',
                    'order': self.config.order,
                    'method': self.config.method,
                    'prune': self.config.prune,
                    'version': __version__}

    def evaluate(self):
        log_path = os.path.join(self.log_directory, 'evaluate.log')
        mod_path = os.path.join(self.temp_directory, self.name + '.mod')
        far_path = os.path.join(self.temp_directory, self.name + '.far')
        small_mod_path = mod_path.replace('.mod', '_small.mod')
        med_mod_path = mod_path.replace('.mod', '_med.mod')
        with open(log_path, 'w', encoding='utf8') as log_file:
            perplexity_proc = subprocess.Popen(['ngramperplexity',
                                                '--OOV_symbol="<unk>"',
                                                mod_path, far_path], stdout=subprocess.PIPE, stderr=log_file, text=True)
            stdout, stderr = perplexity_proc.communicate()
            num_sentences = None
            num_words = None
            num_oovs = None
            perplexity = None
            for line in stdout.splitlines():
                m = re.search(r'(\d+) sentences', line)
                if m:
                    num_sentences = m.group(0)
                m = re.search(r'(\d+) words', line)
                if m:
                    num_words = m.group(0)
                m = re.search(r'(\d+) OOVs', line)
                if m:
                    num_oovs = m.group(0)
                m = re.search(r'perplexity = ([\d.]+)', line)
                if m:
                    perplexity = m.group(0)

            self.logger.info('{} sentences, {} words, {} oovs'.format(num_sentences, num_words, num_oovs))
            self.logger.info('Perplexity of large model: {}'.format(perplexity))

            perplexity_proc = subprocess.Popen(['ngramperplexity',
                                                '--OOV_symbol="<unk>"',
                                                med_mod_path, far_path], stdout=subprocess.PIPE, stderr=log_file, text=True)
            stdout, stderr = perplexity_proc.communicate()

            perplexity = None
            for line in stdout.splitlines():
                m = re.search(r'perplexity = ([\d.]+)', line)
                if m:
                    perplexity = m.group(0)
            self.logger.info('Perplexity of medium model: {}'.format(perplexity))
            perplexity_proc = subprocess.Popen(['ngramperplexity',
                                                '--OOV_symbol="<unk>"',
                                                small_mod_path, far_path], stdout=subprocess.PIPE, stderr=log_file, text=True)
            stdout, stderr = perplexity_proc.communicate()

            perplexity = None
            for line in stdout.splitlines():
                m = re.search(r'perplexity = ([\d.]+)', line)
                if m:
                    perplexity = m.group(0)
            self.logger.info('Perplexity of small model: {}'.format(perplexity))



    def train(self):
        mod_path = os.path.join(self.temp_directory, self.name + '.mod')
        large_model_path = os.path.join(self.temp_directory, self.name + '.arpa')
        small_output_path = large_model_path.replace('.arpa', '_small.arpa')
        med_output_path = large_model_path.replace('.arpa', '_med.arpa')
        if isinstance(self.source, AlignableCorpus):
            self.logger.info('Beginning training large ngram model...')
            sym_path = os.path.join(self.temp_directory, self.name + '.sym')
            far_path = os.path.join(self.temp_directory, self.name + '.far')
            cnts_path = os.path.join(self.temp_directory, self.name + '.cnts')
            training_path = os.path.join(self.temp_directory, 'training.txt')

            with open(training_path, 'w', encoding='utf8') as f:
                for text in self.source.normalized_text_iter(self.dictionary, self.config.count_threshold):
                    f.write(text + "\n")

            if self.dictionary is not None:
                self.dictionary.save_oovs_found(self.temp_directory)

            subprocess.call(['ngramsymbols',
                             '--OOV_symbol="<unk>"', training_path, sym_path])
            subprocess.call(['farcompilestrings', '--fst_type=compact',
                             '--unknown_symbol="<unk>"',
                             '--symbols=' + sym_path, '--keep_symbols', training_path, far_path])
            subprocess.call(['ngramcount', '--order={}'.format(self.config.order), far_path,  cnts_path])
            subprocess.call(['ngrammake', '--method={}'.format(self.config.method), cnts_path, mod_path])
            self.logger.info('Done!')
        else:
            self.logger.info('Parsing large ngram model...')
            temp_text_path = os.path.join(self.temp_directory, 'input.arpa')
            with open(self.source, 'r', encoding='utf8') as inf, open(temp_text_path, 'w', encoding='utf8') as outf:
                for line in inf:
                    outf.write(line.lower())
            subprocess.call(['ngramread', '--ARPA', temp_text_path, mod_path])
            os.remove(temp_text_path)
        if self.supplemental_model_path:
            self.logger.info('Parsing supplemental ngram model...')
            supplemental_path = os.path.join(self.temp_directory, 'extra.mod')
            merged_path = os.path.join(self.temp_directory, 'merged.mod')
            subprocess.call(['ngramread', '--ARPA', self.supplemental_model_path, supplemental_path])
            self.logger.info('Merging both ngram models to create final large model...')
            subprocess.call(['ngrammerge', '--normalize',
                             '--alpha={}'.format(self.source_model_weight),
                             '--beta={}'.format(self.supplemental_model_weight),
                             mod_path, supplemental_path, merged_path])
            mod_path = merged_path

        subprocess.call(['ngramprint', '--ARPA', mod_path, large_model_path])

        self.logger.info('Large ngam model created!')
        directory, filename = os.path.split(self.output_model_path)
        basename, _ = os.path.splitext(filename)

        if self.config.prune:
            self.logger.info('Pruning large ngram model to medium and small versions...')
            small_mod_path = mod_path.replace('.mod', '_small.mod')
            med_mod_path = mod_path.replace('.mod', '_med.mod')
            subprocess.call(['ngramshrink', '--method=relative_entropy',
                             '--theta={}'.format(self.config.prune_thresh_small),
                             mod_path, small_mod_path])
            subprocess.call(['ngramshrink', '--method=relative_entropy',
                             '--theta={}'.format(self.config.prune_thresh_medium),
                             mod_path, med_mod_path])
            subprocess.call(['ngramprint', '--ARPA', small_mod_path, small_output_path])
            subprocess.call(['ngramprint', '--ARPA', med_mod_path, med_output_path])
            self.logger.info('Done!')
        self.evaluate()
        model = LanguageModel.empty(basename, root_directory=self.models_temp_dir)
        model.add_meta_file(self)
        model.add_arpa_file(large_model_path)
        if self.config.prune:
            model.add_arpa_file(med_output_path)
            model.add_arpa_file(small_output_path)
        basename, _ = os.path.splitext(self.output_model_path)
        model.dump(basename)
        print(model.dirname)
        #model.clean_up()





