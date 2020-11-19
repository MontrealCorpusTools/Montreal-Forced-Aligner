import subprocess
import os
import random
import re
import logging
import multiprocessing
import operator
import functools
import time
import shutil
import sys
from typing import Iterator, Set, Tuple, Union

import pynini
import pywrapfst

from ..helper import thirdparty_binary

from ..config import TEMP_DIR

from ..models import G2PModel

from ..exceptions import G2PError

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

TokenType = Union[str, pynini.SymbolTable]

TOKEN_TYPES = ["byte", "utf8"]
DEV_NULL = open(os.devnull, "w")
INF = float("inf")
RAND_MAX = 32767


def _reader(path: str) -> Iterator[str]:
    """Reads strings from a single-column filepath."""
    with open(path, "r") as source:
        for line in source:
            yield line.rstrip()


class PairNGramAligner:
    """Produces FSA alignments for pair n-gram model training."""

    _compactor = functools.partial(
        pywrapfst.convert, fst_type="compact_string"
    )

    def __init__(self, temp_directory):
        self.tempdir = temp_directory
        self.g_path = os.path.join(self.tempdir, "g.far")
        self.p_path = os.path.join(self.tempdir, "p.far")
        self.c_path = os.path.join(self.tempdir, "c.fst")
        self.align_path = os.path.join(self.tempdir, "align.fst")
        self.afst_path = os.path.join(self.tempdir, "afst.far")

    def __del__(self):
        # self.tempdir.cleanup()
        pass

    def align(
            self,
            # Input TSV path.
            tsv_path: str,
            # Output FAR path.
            far_path: str,
            encoder_path: str,
            # Arguments for constructing the lexicon and covering grammar.
            input_token_type: TokenType,
            input_epsilon: bool,
            output_token_type: TokenType,
            output_epsilon: bool,
            # Arguments used during the alignment phase.
            cores: int,
            random_starts: int,
            seed: int,
            delta: str = "",
            fst_default_cache_gc: str = "",
            fst_default_cache_gc_limit: str = "",
            max_iters: str = "",
    ):
        """Runs the entire alignment regimen."""
        self._lexicon_covering(
            tsv_path,
            input_token_type,
            input_epsilon,
            output_token_type,
            output_epsilon,
        )
        self._alignments(
            cores,
            random_starts,
            seed,
            delta,
            fst_default_cache_gc,
            fst_default_cache_gc_limit,
            max_iters,
        )
        self._encode(far_path, encoder_path)
        logging.info(
            "Success! FAR path: %s; encoder path: %s", far_path, encoder_path
        )

    @staticmethod
    def _label_union(labels: Set[int], epsilon: bool) -> pynini.Fst:
        """Creates FSA over a union of the labels."""
        side = pynini.Fst()
        src = side.add_state()
        side.set_start(src)
        dst = side.add_state()
        if epsilon:
            labels.add(0)
        one = pynini.Weight.one(side.weight_type())
        for label in labels:
            side.add_arc(src, pynini.Arc(label, label, one, dst))
        side.set_final(dst)
        assert side.verify(), "FST is ill-formed"
        return side

    @staticmethod
    def _narcs(f: pynini.Fst) -> int:
        """Computes the number of arcs in an FST."""
        return sum(f.num_arcs(state) for state in f.states())

    NON_SYMBOL = ("byte", "utf8")

    def _lexicon_covering(
            self,
            tsv_path: str,
            input_token_type: TokenType,
            input_epsilon: bool,
            output_token_type: TokenType,
            output_epsilon: bool,
    ) -> None:
        """Builds covering grammar and lexicon FARs."""
        # Sets of labels for the covering grammar.
        g_labels: Set[int] = set()
        p_labels: Set[int] = set()
        # Curries compiler functions for the FARs.
        icompiler = functools.partial(
            pynini.acceptor, token_type=input_token_type
        )
        ocompiler = functools.partial(
            pynini.acceptor, token_type=output_token_type
        )
        logging.info("Constructing grapheme and phoneme FARs")
        g_writer = pywrapfst.FarWriter.create(self.g_path)
        p_writer = pywrapfst.FarWriter.create(self.p_path)
        with open(tsv_path, "r") as source:
            for (linenum, line) in enumerate(source, 1):
                key = f"{linenum:08x}"
                (g, p) = line.rstrip().split("\t", 1)
                # For both G and P, we compile a FSA, store the labels, and
                # then write the compact version to the FAR.
                g_fst = icompiler(g)
                g_labels.update(g_fst.paths().ilabels())
                g_writer[key] = self._compactor(g_fst)
                p_fst = ocompiler(p)
                p_labels.update(p_fst.paths().ilabels())
                p_writer[key] = self._compactor(p_fst)
        logging.info("Processed %s examples", f"{linenum:,d}")
        logging.info("Constructing covering grammar")
        logging.info("%d unique graphemes", len(g_labels))
        g_side = self._label_union(g_labels, input_epsilon)
        logging.info("%d unique phones", len(p_labels))
        p_side = self._label_union(p_labels, output_epsilon)
        # The covering grammar is given by (G x P)^*.
        print(dir(pynini))
        covering = pynini.cross(g_side, p_side).closure().optimize()
        assert covering.num_states() == 1, "Covering grammar FST is ill-formed"
        logging.info(
            "Covering grammar has %s arcs",
            f"{PairNGramAligner._narcs(covering):,d}",
        )
        covering.write(self.c_path)

    @staticmethod
    def _random_start(*args: str) -> Tuple[str, float]:
        """Performs a single random start."""
        (*cmd, idx) = args
        start = time.time()
        likelihood = INF
        logging.debug("Subprocess call: %s", cmd)
        with subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True) as proc:
            # Parses STDERR to capture the likelihood.
            for line in proc.stderr:
                match = re.match(
                    r"INFO: Best likelihood:\s(-?\d*(\.\d*))", line
                )
                if match:
                    likelihood = float(match.group(1))
                    logging.info(
                        "Random start %d; likelihood: %f; time elapsed: %ds",
                        idx,
                        likelihood,
                        time.time() - start,
                    )
        return (cmd[-1], likelihood)

    def _alignments(
            self,
            cores: int,
            random_starts: int,
            seed: int,
            delta: str = "",
            fst_default_cache_gc: str = "",
            fst_default_cache_gc_limit: str = "",
            max_iters: str = "",
    ) -> None:
        """Trains the aligner and constructs the alignments FAR."""
        logging.info("Training aligner")
        cmd_fixed = ["baumwelchtrain", "--normalize_ilabel"]
        if delta:
            cmd_fixed.append(f"--delta={delta}")
        if fst_default_cache_gc:
            cmd_fixed.append(f"--fst_default_cache_gc={fst_default_cache_gc}")
        if fst_default_cache_gc_limit:
            cmd_fixed.append(
                f"--fst_default_cache_gc_limit={fst_default_cache_gc_limit}"
            )
        if max_iters:
            cmd_fixed.append(f"--max_iters={max_iters}")
        # Adds more arguments shared across all commands.
        if max_iters:
            cmd_fixed.append(f"--max_iters={max_iters}")
        # cmd_fixed.append("--remove_zero_arcs=false")
        # cmd_fixed.append("--flat_start=false")
        # cmd_fixed.append("--random_starts=1")
        # Constructs the actual command vectors (plus an index for logging
        # purposes).
        random.seed(seed)
        idx = 0
        command = (
            *cmd_fixed,
            # f"--seed={seed}",
            self.g_path,
            self.p_path,
            self.c_path,
            self.align_path
        )
        start = time.time()
        with subprocess.Popen(command, stderr=subprocess.PIPE, text=True) as proc:
            # Parses STDERR to capture the likelihood.
            for line in proc.stderr:
                print(line)
                match = re.match(
                    r"INFO: Best likelihood:\s(-?\d*(\.\d*))", line
                )
                if match:
                    likelihood = float(match.group(1))
                    logging.info(
                        "Random start %d; likelihood: %f; time elapsed: %ds",
                        idx,
                        likelihood,
                        time.time() - start,
                    )

                    logging.info("Likelihood: %f", likelihood)
        # Moves best likelihood solution to the requested location.
        logging.info("Computing alignments")
        cmd = ["baumwelchdecode"]
        if fst_default_cache_gc:
            cmd.append(f"--fst_default_cache_gc={fst_default_cache_gc}")
        if fst_default_cache_gc_limit:
            cmd.append(
                f"--fst_default_cache_gc_limit={fst_default_cache_gc_limit}"
            )
        cmd.append(self.g_path)
        cmd.append(self.p_path)
        cmd.append(self.align_path)
        cmd.append(self.afst_path)
        logging.debug("Subprocess call: %s", cmd)
        subprocess.check_call(cmd)

    def _encode(self, far_path: str, encoder_path: str) -> None:
        """Encodes the alignments."""
        logging.info("Encoding the alignments as FSAs")
        encoder = pywrapfst.EncodeMapper(encode_labels=True)
        a_reader = pywrapfst.FarReader.open(self.afst_path)
        a_writer = pywrapfst.FarWriter.create(far_path)
        # Curries converter function for the FAR.
        converter = functools.partial(pywrapfst.convert, fst_type="vector")
        while not a_reader.done():
            key = a_reader.get_key()
            fst = converter(a_reader.get_fst())
            fst.encode(encoder)
            a_writer[key] = self._compactor(fst)
            a_reader.next()
        encoder.write(encoder_path)


class PyniniTrainer(object):
    def __init__(self, dictionary, model_path, temp_directory=None, order=7, window_size=2, evaluate=False,
                 input_epsilon=True, output_epsilon=True, num_jobs=3, random_starts=10, seed=1234,
                 max_iters=None, smoothing_method='kneser_ney', pruning_method='relative_entropy',
                 model_size=1000000):
        super(PyniniTrainer, self).__init__()
        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = os.path.join(temp_directory, 'G2P')

        self.name, _ = os.path.splitext(os.path.basename(model_path))
        self.temp_directory = os.path.join(self.temp_directory, self.name)
        os.makedirs(self.temp_directory, exist_ok=True)
        self.model_path = model_path
        self.fst_path = os.path.join(self.temp_directory, 'model.fst')
        self.far_path = os.path.join(self.temp_directory, self.name + '.far')
        self.encoder_path = os.path.join(self.temp_directory, self.name + '.enc')
        self.grapheme_window_size = 2
        self.phoneme_window_size = window_size
        self.evaluate = evaluate
        self.dictionary = dictionary
        self.input_epsilon = input_epsilon
        self.output_epsilon = output_epsilon
        self.num_jobs = num_jobs
        self.random_starts = random_starts
        self.seed = seed
        self.max_iters = max_iters
        if self.max_iters is None:
            self.max_iters = ''
        else:
            self.max_iters = str(self.max_iters)
        self.delta = ''
        self.fst_default_cache_gc = ''
        self.fst_default_cache_gc_limit = ''
        self.order = order
        self.model_log_path = os.path.join(self.temp_directory, 'model.log')
        self.smoothing_method = smoothing_method
        self.pruning_method = pruning_method
        self.model_size = model_size
        self.sym_path = os.path.join(self.temp_directory, 'phones.sym')
        self.output_token_type = None

    def generate_model(self):
        assert os.path.exists(self.far_path)
        with open(self.model_log_path, 'w', encoding='utf8') as logf:
            ngram_count_path = os.path.join(self.temp_directory, 'ngram.count')
            ngram_make_path = os.path.join(self.temp_directory, 'ngram.make')
            ngram_shrink_path = os.path.join(self.temp_directory, 'ngram.shrink')
            ngramcount_proc = subprocess.Popen(['ngramcount', "--require_symbols=false",
                                           '--order={}'.format(self.order),
                                           self.far_path, ngram_count_path],
                                           stderr=logf)
            ngramcount_proc.communicate()

            ngrammake_proc = subprocess.Popen(['ngrammake',
                                           '--method='+self.smoothing_method, ngram_count_path, ngram_make_path],
                                          stderr=logf)
            ngrammake_proc.communicate()

            ngramshrink_proc = subprocess.Popen(['ngramshrink',
                                           '--method='+self.pruning_method,
                                           '--target_number_of_ngrams={}'.format(self.model_size),
                                                 ngram_make_path, ngram_shrink_path
                                                 ],
                                          stderr=logf)
            ngramshrink_proc.communicate()

            fstencode_proc = subprocess.Popen(['fstencode',
                                           '--decode', ngram_shrink_path,
                                               self.encoder_path,
                                               self.fst_path],
                                          stderr=logf)
            fstencode_proc.communicate()

        #os.remove(ngram_count_path)
        #os.remove(ngram_make_path)
        #os.remove(ngram_shrink_path)

        directory, filename = os.path.split(self.model_path)
        basename, _ = os.path.splitext(filename)
        model = G2PModel.empty(basename)
        model.add_meta_file(self.dictionary)
        model.add_fst_model(self.temp_directory)
        model.add_sym_path(self.temp_directory)
        os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(self.model_path)
        model.dump(basename)
        print('Saved model to {}'.format(self.model_path))

    def train(self, word_dict=None):
        input_path = os.path.join(self.temp_directory, 'input.txt')
        phones_path = os.path.join(self.temp_directory, 'phones_only.txt')
        if word_dict is None:
            word_dict = self.dictionary.actual_words
        with open(input_path, "w", encoding='utf8') as f2, \
            open(phones_path, 'w', encoding='utf8') as phonef:
            for word, v in word_dict.items():
                if re.match(r'\W', word) is not None:
                    continue
                for v2 in v:
                    f2.write(word + "\t" + " ".join(v2[0]) + "\n")
                for v2 in v:
                    phonef.write(" ".join(v2[0]) + "\n")
        subprocess.call(['ngramsymbols', phones_path, self.sym_path])
        aligner = PairNGramAligner(self.temp_directory)
        input_token_type = "utf8"
        self.output_token_type = pynini.SymbolTable.read_text(self.sym_path)

        aligner.align(input_path,
                      self.far_path,
                      self.encoder_path,
                      input_token_type,
                      self.input_epsilon,
                      self.output_token_type,
                      self.output_epsilon,
                      self.num_jobs,
                      self.random_starts,
                      self.seed,
                      self.delta,
                      self.fst_default_cache_gc,
                      self.fst_default_cache_gc_limit,
                      self.max_iters)
        self.generate_model()

    def validate(self):
        from .generator import PyniniDictionaryGenerator
        from ..models import G2PModel
        word_dict = self.dictionary.actual_words
        print(word_dict)
        validation = 0.1
        words = word_dict.keys()
        total_items = len(words)
        validation_items = int(total_items * validation)
        validation_words = random.sample(words, validation_items)
        training_dictionary = {k: v for k, v in word_dict.items()
                               #if k not in validation_words
                               }
        validation_dictionary = {k: v for k, v in word_dict.items() if k in validation_words}
        train_graphemes = set()
        for  k, v in word_dict.items():
            print(k)
            train_graphemes.update(k)
        print(train_graphemes)
        self.train(training_dictionary)

        model = G2PModel(self.model_path)
        output_path = os.path.join(self.temp_directory, 'validation.txt')
        validation_errors_path = os.path.join(self.temp_directory, 'validation_errors.txt')
        gen = PyniniDictionaryGenerator(model, validation_dictionary.keys(),
                                               temp_directory=os.path.join(self.temp_directory, 'validation'))
        gen.output(output_path)


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
        # if stderr:
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

        # if stderr:
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
                                               temp_directory=os.path.join(self.temp_directory, 'validation'))
        gen.output(output_path)
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
