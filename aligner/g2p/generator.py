import os
import re
import subprocess
import sys
import tempfile
import logging
from textgrid import TextGrid, IntervalTier
import traceback

from ..corpus import parse_transcription, load_text

from ..helper import thirdparty_binary

from ..config import TEMP_DIR


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

    def __init__(self, g2p_model, word_set, outfile, use_unk = False, temp_directory=None):
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

        self.outfile = outfile

    @property
    def word_list_path(self):
        return os.path.join(self.temp_directory, 'words.txt')

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

        with open(self.outfile, "w", encoding='utf8') as f:
            for word, pronunciation in parse_output(results):
                if pronunciation is None:
                    continue
                f.write('{}\t{}\n'.format(word, pronunciation))
