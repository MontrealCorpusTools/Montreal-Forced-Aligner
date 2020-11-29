import os
import re
import subprocess
import logging
from typing import Iterator, Set, Tuple, Union
import multiprocessing as mp
import pynini
import pywrapfst
import tqdm
import queue
from ..helper import thirdparty_binary
import traceback
import sys
import time

from ..config import TEMP_DIR
from ..exceptions import G2PError
from ..multiprocessing import Stopped, Counter


TokenType = Union[str, pynini.SymbolTable]


class Rewriter:
    """Helper object for rewriting."""

    def __init__(
        self,
        fst: pynini.Fst,
        input_token_type: TokenType,
        output_token_type: TokenType,
    ):
        self.fst = fst
        self.input_token_type = input_token_type
        self.output_token_type = output_token_type
        self.logger = logging.getLogger('g2p')

    def rewrite(self, i: str) -> Tuple[str, str]:
        lattice = (
            pynini.acceptor(i, token_type=self.input_token_type) @ self.fst
        )
        if lattice.start() == pynini.NO_STATE_ID:
            logging.error("Composition failure: %s", i)
            return "<composition failure>"
        lattice.project(True).rmepsilon()
        return pynini.shortestpath(lattice).string(self.output_token_type)


def parse_errors(error_output):
    missing_symbols = []
    line_regex = re.compile("Symbol: '(.+?)' not found in input symbols table")
    for line in error_output.splitlines():
        m = line_regex.match(line)
        if m is not None:
            missing_symbols.append(m.groups()[0])
    return missing_symbols


def parse_output(output):
    for line in output.splitlines():
        line = line.strip().split("\t")
        if len(line) == 2:
            line += [None]
        yield line[0], line[2]


class RewriterWorker(mp.Process):
    def __init__(self, job_q, return_dict, rewriter, counter, stopped):
        mp.Process.__init__(self)
        self.job_q = job_q
        self.return_dict = return_dict
        self.rewriter = rewriter
        self.counter = counter
        self.stopped = stopped

    def run(self):
        while True:
            try:
                word = self.job_q.get(timeout=1)
            except queue.Empty:
                break
            self.job_q.task_done()
            if self.stopped.stop_check():
                continue
            try:
                rep = self.rewriter.rewrite(word)
                self.return_dict[word] = rep
            except Exception as e:
                self.stopped.stop()
                self.return_dict['error'] = word, Exception(traceback.format_exception(*sys.exc_info()))
            self.counter.increment()
        return


def clean_up_word(word, graphemes):
    new_word = []
    missing_graphemes = []
    for c in word:
        if c not in graphemes:
            missing_graphemes.append(c)
        else:
            new_word.append(c)
    return ''.join(new_word), missing_graphemes


class PyniniDictionaryGenerator(object):
    def __init__(self, g2p_model, word_set, temp_directory=None, num_jobs=3):
        super(PyniniDictionaryGenerator, self).__init__()
        if not temp_directory:
            temp_directory = TEMP_DIR
        temp_directory = os.path.join(temp_directory, 'G2P')
        self.model = g2p_model

        self.temp_directory = os.path.join(temp_directory, self.model.name)
        log_dir = os.path.join(self.temp_directory, 'logging')
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'g2p.log')
        self.logger = logging.getLogger('g2p')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_file, 'w', 'utf-8')
        handler.setFormatter = logging.Formatter('%(name)s %(message)s')
        self.logger.addHandler(handler)
        self.words = word_set
        self.num_jobs = num_jobs

    def generate(self):
        if self.model.meta['architecture'] == 'phonetisaurus':
            raise G2PError('Previously trained Phonetisaurus models from 1.1 and earlier are not currently supported. '
                           'Please retrain your model using 2.0+')

        input_token_type = 'utf8'
        fst = pynini.Fst.read(self.model.fst_path)

        output_token_type = 'utf8'
        if self.model.sym_path is not None and os.path.exists(self.model.sym_path):
            output_token_type = pynini.SymbolTable.read_text(self.model.sym_path)
        print(output_token_type)
        print(self.model.meta['architecture'])
        print(self.model.fst_path)
        rewriter = Rewriter(fst, input_token_type, output_token_type)

        stopped = Stopped()
        job_queue = mp.JoinableQueue(100)
        ind = 0
        num_words = len(self.words)
        words = sorted(self.words)
        begin = time.time()
        last_value = 0
        missing_graphemes = set()
        print('Generating pronunciations...')
        with tqdm.tqdm(total=num_words) as pbar:
            while True:
                if ind == num_words:
                    break
                try:
                    w, m = clean_up_word(words[ind], self.model.meta['graphemes'])
                    missing_graphemes.update(m)
                    if not w:
                        ind += 1
                        continue
                    job_queue.put(w, False)
                except queue.Full:
                    break
                ind += 1
            manager = mp.Manager()
            return_dict = manager.dict()
            procs = []
            counter = Counter()
            for i in range(self.num_jobs):
                p = RewriterWorker(job_queue, return_dict, rewriter, counter, stopped)
                procs.append(p)
                p.start()
            while True:
                if ind == num_words:
                    break
                w, m = clean_up_word(words[ind], self.model.meta['graphemes'])
                missing_graphemes.update(m)
                if not w:
                    ind += 1
                    continue
                job_queue.put(w)
                value = counter.value()
                pbar.update(value-last_value)
                last_value = value
                ind += 1
            job_queue.join()
        for p in procs:
            p.join()
        if 'error' in return_dict:
            element, exc = return_dict['error']
            print(element)
            raise exc
        to_return = {}
        to_return.update(return_dict)
        print(to_return)
        print('Processed {} in {} seconds'.format(len(self.words), time.time()-begin))
        self.logger.debug('Processed {} in {} seconds'.format(len(self.words), time.time()-begin))
        return to_return

    def output(self, outfile):
        results = self.generate()
        with open(outfile, "w", encoding='utf8') as f:
            for (word, pronunciation) in results.items():
                f.write('{}\t{}\n'.format(word, pronunciation))


class PhonetisaurusDictionaryGenerator(object):
    """creates a Dictionary from a g2pfst model

    Parameters
    ----------
        g2p_model: :class:`~aligner.models.G2PModel`
            path to the models
        corpus : :class:`~aligner.corpus.Corpus`
            Corpus object to get word list from
        outfile: str
            destination for the dictionary
    """

    def __init__(self, g2p_model, word_set, use_unk = False, temp_directory=None):
        super(PhonetisaurusDictionaryGenerator, self).__init__()
        if not temp_directory:
            temp_directory = TEMP_DIR
        temp_directory = os.path.join(temp_directory, 'G2P')

        self.model = g2p_model

        self.temp_directory = os.path.join(temp_directory, self.model.name)
        log_dir = os.path.join(self.temp_directory, 'logging')
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'g2p.log')

        self.logger = logging.getLogger('g2p')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_file, 'w', 'utf-8')
        handler.setFormatter = logging.Formatter('%(name)s %(message)s')
        self.logger.addHandler(handler)
        with open(self.word_list_path, 'w', encoding='utf8') as f:
            for word in sorted(word_set):
                f.write(word.strip() + '\n')


    @property
    def word_list_path(self):
        return os.path.join(self.temp_directory, 'words.txt')

    def output(self, outfile):
        results = self.generate()
        with open(outfile, "w", encoding='utf8') as f:
            for (word, pronunciation) in results:
                f.write('{}\t{}\n'.format(word, pronunciation))

    def generate(self):
        """
        runs the phonetisaurus-g2pfst binary with the language and all the words in the corpus
        """

        proc = subprocess.Popen([thirdparty_binary('phonetisaurus-g2pfst'),
                                 '--model=' + self.model.fst_path, '--wordlist=' + self.word_list_path],
                                stderr=subprocess.PIPE,
                                stdout=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        results = stdout.decode('utf8')
        errors = stderr.decode('utf8')
        missing_symbols = parse_errors(errors)
        if missing_symbols:
            print("There were {} unmatched symbols in your transcriptions, "
                  "please see the log file ({}) for a list.".format(len(missing_symbols), self.log_file))
            self.logger.warning(
                'The following symbols were not found in the G2P model: {}.'.format(', '.join(missing_symbols)))
        output = []
        for word, pronunciation in parse_output(results):
            if pronunciation is None:
                continue
            output.append((word, pronunciation))
        return output
