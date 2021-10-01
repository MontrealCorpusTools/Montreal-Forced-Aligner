import subprocess
import os
import random
import re
import logging
import queue
import multiprocessing as mp
import operator
import functools
import time
import shutil
import traceback
import sys
import tqdm
from typing import Set, NamedTuple, Optional, Any, List, Tuple

try:
    import pynini
    from pynini import Fst, TokenType
    import pywrapfst
    from pywrapfst import convert
    G2P_DISABLED = False

except ImportError:
    pynini = None
    pywrapfst = None
    TokenType = None
    Fst = None
    convert = lambda x: x
    G2P_DISABLED = True

from ..config import TEMP_DIR
from ..models import G2PModel
from ..multiprocessing import Stopped, Counter

from ..helper import score

Labels = List[Any]

TOKEN_TYPES = ["byte", "utf8"]
DEV_NULL = open(os.devnull, "w")
INF = float("inf")
RAND_MAX = 32767

class RandomStart(NamedTuple):

    idx: int
    seed: int
    g_path: str
    p_path: str
    c_path: str
    tempdir: str
    train_opts: List[str]



def compute_validation_errors(gold_values, hypothesis_values, num_jobs=3):
    # Word-level measures.
    correct = 0
    incorrect = 0
    # Label-level measures.
    total_edits = 0
    total_length = 0
    # Since the edit distance algorithm is quadratic, let's do this with
    # multiprocessing.
    with mp.Pool(num_jobs) as pool:
        to_comp = []
        for word, hyp in hypothesis_values.items():
            g = gold_values[word][0]['pronunciation']
            hyp = [ h.split(' ') for h in hyp]
            to_comp.append((g, hyp))
        gen = pool.starmap(score, to_comp)
        for (edits, length) in gen:
            if edits == 0:
                correct += 1
            else:
                incorrect += 1
            total_edits += edits
            total_length += length
        for w, gold in gold_values.items():
            if w not in hypothesis_values:
                incorrect += 1
                gold = gold[0]['pronunciation']
                total_edits += len(gold)
                total_length += len(gold)

    return 100 * incorrect / (correct + incorrect), 100 * total_edits / total_length


class RandomStartWorker(mp.Process):
    def __init__(self, job_q, return_dict, function, counter, stopped):
        mp.Process.__init__(self)
        self.job_q = job_q
        self.return_dict = return_dict
        self.function = function
        self.counter = counter
        self.stopped = stopped

    def run(self):
        while True:
            try:
                args = self.job_q.get(timeout=1)
            except queue.Empty:
                break
            self.job_q.task_done()
            if self.stopped.stop_check():
                continue
            try:
                fst_path, likelihood = self.function(args)
                self.return_dict[fst_path] = likelihood
            except Exception as _:
                self.stopped.stop()
                self.return_dict['error'] = args, Exception(traceback.format_exception(*sys.exc_info()))
            self.counter.increment()
        return


class PairNGramAligner:
    """Produces FSA alignments for pair n-gram model training."""

    _compactor = functools.partial(
        convert, fst_type="compact_string"
    )

    def __init__(self, temp_directory):
        self.tempdir = temp_directory
        self.g_path = os.path.join(self.tempdir, "g.far")
        self.p_path = os.path.join(self.tempdir, "p.far")
        self.c_path = os.path.join(self.tempdir, "c.fst")
        self.align_path = os.path.join(self.tempdir, "align.fst")
        self.afst_path = os.path.join(self.tempdir, "afst.far")
        self.align_log_path = os.path.join(self.tempdir, 'align.log')
        self.logger = logging.getLogger('g2p_aligner')
        self.logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(self.align_log_path)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

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
            batch_size: int = 0,
            delta: float = 1 / 1024,
            lr: float = 1.0,
            max_iters: int = 50,
            fst_default_cache_gc: str = "",
            fst_default_cache_gc_limit: str = "",
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
            batch_size,
            delta,
            lr,
            max_iters,
            fst_default_cache_gc,
            fst_default_cache_gc_limit,
        )
        self._encode(far_path, encoder_path)
        self.logger.info(
            "Success! FAR path: %s; encoder path: %s", far_path, encoder_path
        )

    @staticmethod
    def _label_union(labels: Set[int], epsilon: bool) -> Fst:
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
    def _narcs(f: Fst) -> int:
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
        self.logger.info("Constructing grapheme and phoneme FARs")
        g_writer = pywrapfst.FarWriter.create(self.g_path)
        p_writer = pywrapfst.FarWriter.create(self.p_path)
        with open(tsv_path, "r") as source:
            for (linenum, line) in enumerate(source, 1):
                key = f"{linenum:08x}"
                (g, p) = line.rstrip().split("\t", 1)
                # For both G and P, we compile a FSA, store the labels, and
                # then write the compact version to the FAR.
                g_fst = pynini.accep(g, token_type=input_token_type)
                g_labels.update(g_fst.paths().ilabels())
                g_writer[key] = self._compactor(g_fst)
                p_fst = pynini.accep(p, token_type=output_token_type)
                p_labels.update(p_fst.paths().ilabels())
                p_writer[key] = self._compactor(p_fst)
        self.logger.info("Processed %s examples", f"{linenum:,d}")
        self.logger.info("Constructing covering grammar")
        self.logger.info("%d unique graphemes", len(g_labels))
        g_side = self._label_union(g_labels, input_epsilon)
        self.logger.info("%d unique phones", len(p_labels))
        p_side = self._label_union(p_labels, output_epsilon)
        # The covering grammar is given by (G x P)^*.
        covering = pynini.cross(g_side, p_side).closure().optimize()
        assert covering.num_states() == 1, "Covering grammar FST is ill-formed"
        self.logger.info(
            "Covering grammar has %s arcs",
            f"{PairNGramAligner._narcs(covering):,d}",
        )
        covering.write(self.c_path)

    @staticmethod
    def _random_start(random_start: RandomStart) -> Tuple[str, float]:
        """Performs a single random start."""
        start = time.time()
        logger = logging.getLogger('g2p_aligner')
        # Randomize channel model.
        c_path = os.path.join(
            random_start.tempdir, f"c-{random_start.seed:05d}.fst"
        )
        t_path = os.path.join(
            random_start.tempdir, f"t-{random_start.seed:05d}.fst"
        )
        likelihood_path = t_path.replace('.fst', '.like')
        if not os.path.exists(t_path):
            cmd = [
                "baumwelchrandomize",
                f"--seed={random_start.seed}",
                random_start.c_path,
                c_path,
            ]
            subprocess.check_call(cmd)
            random_end = time.time()
            logger.debug('{} randomization took {} seconds'.format(random_start.seed, random_end - start))
            # Train on randomized channel model.

            likelihood = INF
            cmd = [
                "baumwelchtrain",
                *random_start.train_opts,
                random_start.g_path,
                random_start.p_path,
                c_path,
                t_path,
            ]
            logger.debug('{} train command: {}'.format(random_start.seed, ' '.join(cmd)))
            with subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True) as proc:
                # Parses STDERR to capture the likelihood.
                for line in proc.stderr:  # type: ignore
                    line = line.rstrip()
                    match = re.match(r"INFO: Iteration \d+: (-?\d*(\.\d*)?)", line)
                    assert match, line
                    likelihood = float(match.group(1))
                with open(likelihood_path, 'w') as f:
                    f.write(str(likelihood))
            logger.debug('{} training took {} seconds'.format(random_start.seed, time.time() - random_end))
        else:
            with open(likelihood_path, 'r') as f:
                likelihood = f.read().strip()
        return t_path, likelihood

    def _alignments(
            self,
            cores: int,
            random_starts: int,
            seed: int,
            batch_size: Optional[int] = None,
            delta: Optional[float] = None,
            lr: Optional[float] = None,
            max_iters: Optional[int] = None,
            fst_default_cache_gc: str = "",
            fst_default_cache_gc_limit: str = "",
    ) -> None:
        """Trains the aligner and constructs the alignments FAR."""
        if not os.path.exists(self.align_path):
            self.logger.info("Training aligner")
            train_opts = []
            if batch_size:
                train_opts.append(f"--batch_size={batch_size}")
            if delta:
                train_opts.append(f"--delta={delta}")
            if fst_default_cache_gc:
                train_opts.append(f"--fst_default_cache_gc={fst_default_cache_gc}")
            if fst_default_cache_gc_limit:
                train_opts.append(
                    f"--fst_default_cache_gc_limit={fst_default_cache_gc_limit}"
                )
            if lr:
                train_opts.append(f"--lr={lr}")
            if max_iters:
                train_opts.append(f"--max_iters={max_iters}")
            # Constructs the actual command vectors (plus an index for logging
            # purposes).
            random.seed(seed)
            starts = [
                (
                    RandomStart(
                        idx,
                        seed,
                        self.g_path,
                        self.p_path,
                        self.c_path,
                        self.tempdir,
                        train_opts,
                    )
                )
                for (idx, seed) in enumerate(
                    random.sample(range(1, RAND_MAX), random_starts), 1
                )
            ]
            stopped = Stopped()
            num_commands = len(starts)
            if cores > len(starts):
                cores = len(starts)
            job_queue = mp.JoinableQueue(cores + 2)

            # Actually runs starts.
            self.logger.info("Random starts")
            print('Calculating alignments...')
            begin = time.time()
            last_value = 0
            ind = 0
            with tqdm.tqdm(total=num_commands) as pbar:
                while True:
                    if ind == num_commands:
                        break
                    try:
                        job_queue.put(starts[ind], False)
                    except queue.Full:
                        break
                    ind += 1
                manager = mp.Manager()
                return_dict = manager.dict()
                procs = []
                counter = Counter()
                for i in range(cores):
                    p = RandomStartWorker(job_queue, return_dict, self._random_start, counter, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    if ind == num_commands:
                        break
                    job_queue.put(starts[ind])
                    value = counter.value()
                    pbar.update(value - last_value)
                    last_value = value
                    ind += 1
                while True:
                    time.sleep(30)
                    value = counter.value()
                    if value != last_value:
                        pbar.update(value - last_value)
                        last_value = value
                    if value >= random_starts:
                        break
                job_queue.join()
                for p in procs:
                    p.join()
            if 'error' in return_dict:
                element, exc = return_dict['error']
                print(element)
                raise exc
            (best_fst, best_likelihood) = min(return_dict.items(), key=operator.itemgetter(1))
            self.logger.info("Best likelihood: %f", best_likelihood)
            self.logger.debug("Ran {} random starts in {} seconds".format(random_starts, time.time() - begin))
            # Moves best likelihood solution to the requested location.
            shutil.move(best_fst, self.align_path)
        self.logger.info("Computing alignments")
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
        self.logger.debug("Subprocess call: %s", cmd)
        subprocess.check_call(cmd)

    def _encode(self, far_path: str, encoder_path: str) -> None:
        """Encodes the alignments."""
        self.logger.info("Encoding the alignments as FSAs")
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
    def __init__(self, dictionary, model_path, train_config, temp_directory=None,
                 input_epsilon=True, output_epsilon=True, num_jobs=3,
                 verbose=False):
        super(PyniniTrainer, self).__init__()
        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = os.path.join(temp_directory, 'G2P')
        self.train_config = train_config
        self.verbose = verbose
        self.models_temp_dir = os.path.join(temp_directory, 'models', 'G2P')

        self.name, _ = os.path.splitext(os.path.basename(model_path))
        self.temp_directory = os.path.join(self.temp_directory, self.name)
        os.makedirs(self.temp_directory, exist_ok=True)
        os.makedirs(self.models_temp_dir, exist_ok=True)
        self.model_path = model_path
        self.fst_path = os.path.join(self.temp_directory, 'model.fst')
        self.far_path = os.path.join(self.temp_directory, self.name + '.far')
        self.encoder_path = os.path.join(self.temp_directory, self.name + '.enc')
        self.dictionary = dictionary
        self.input_epsilon = input_epsilon
        self.output_epsilon = output_epsilon
        self.num_jobs = num_jobs
        if not self.train_config.use_mp:
            self.num_jobs = 1
        self.fst_default_cache_gc = ''
        self.fst_default_cache_gc_limit = ''
        self.train_log_path = os.path.join(self.temp_directory, 'train.log')

        self.logger = logging.getLogger('g2p_trainer')
        self.logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(self.train_log_path)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        handler = logging.StreamHandler(sys.stdout)
        if self.verbose:
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.model_log_path = os.path.join(self.temp_directory, 'model.log')
        self.sym_path = os.path.join(self.temp_directory, 'phones.sym')
        self.output_token_type = None

    def clean_up(self):
        for name in os.listdir(self.temp_directory):
            path = os.path.join(self.temp_directory, name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif not name.endswith('.log'):
                os.remove(path)

    def generate_model(self):
        assert os.path.exists(self.far_path)
        with open(self.model_log_path, 'w', encoding='utf8') as logf:
            ngram_count_path = os.path.join(self.temp_directory, 'ngram.count')
            ngram_make_path = os.path.join(self.temp_directory, 'ngram.make')
            ngram_shrink_path = os.path.join(self.temp_directory, 'ngram.shrink')
            ngramcount_proc = subprocess.Popen(['ngramcount', "--require_symbols=false",
                                                '--order={}'.format(self.train_config.order),
                                                self.far_path, ngram_count_path],
                                               stderr=logf)
            ngramcount_proc.communicate()

            ngrammake_proc = subprocess.Popen(['ngrammake',
                                               '--method=' + self.train_config.smoothing_method,
                                               ngram_count_path, ngram_make_path],
                                              stderr=logf)
            ngrammake_proc.communicate()

            ngramshrink_proc = subprocess.Popen(['ngramshrink',
                                                 '--method=' + self.train_config.pruning_method,
                                                 '--target_number_of_ngrams={}'.format(self.train_config.model_size),
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

        os.remove(ngram_count_path)
        os.remove(ngram_make_path)
        os.remove(ngram_shrink_path)

        directory, filename = os.path.split(self.model_path)
        basename, _ = os.path.splitext(filename)
        model = G2PModel.empty(basename, root_directory=self.models_temp_dir)
        model.add_meta_file(self.dictionary, 'pynini')
        model.add_fst_model(self.temp_directory)
        model.add_sym_path(self.temp_directory)
        if directory:
            os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(self.model_path)
        model.dump(basename)
        model.clean_up()
        self.clean_up()
        print('Saved model to {}'.format(self.model_path))
        self.logger.info('Saved model to {}'.format(self.model_path))

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
                    f2.write(word + "\t" + " ".join(v2['pronunciation']) + "\n")
                for v2 in v:
                    phonef.write(" ".join(v2['pronunciation']) + "\n")
        subprocess.call(['ngramsymbols', phones_path, self.sym_path])
        os.remove(phones_path)
        aligner = PairNGramAligner(self.temp_directory)
        input_token_type = "utf8"
        self.output_token_type = pynini.SymbolTable.read_text(self.sym_path)
        begin = time.time()
        if not os.path.exists(self.far_path) or not os.path.exists(self.encoder_path):
            aligner.align(input_path,
                          self.far_path,
                          self.encoder_path,
                          input_token_type,
                          self.input_epsilon,
                          self.output_token_type,
                          self.output_epsilon,
                          self.num_jobs,
                          self.train_config.random_starts,
                          self.train_config.seed,
                          self.train_config.batch_size,
                          self.train_config.delta,
                          self.train_config.lr,
                          self.train_config.max_iterations,
                          self.fst_default_cache_gc,
                          self.fst_default_cache_gc_limit)
        self.logger.debug('Aligning {} words took {} seconds'.format(len(word_dict), time.time() - begin))
        begin = time.time()
        self.generate_model()
        self.logger.debug('Generating model for {} words took {} seconds'.format(len(word_dict), time.time() - begin))

    def validate(self):
        from .generator import PyniniDictionaryGenerator
        from ..models import G2PModel
        word_dict = self.dictionary.actual_words
        validation = 0.1
        words = word_dict.keys()
        total_items = len(words)
        validation_items = int(total_items * validation)
        validation_words = random.sample(words, validation_items)
        training_dictionary = {k: v for k, v in word_dict.items()
                               if k not in validation_words
                               }
        validation_dictionary = {k: v for k, v in word_dict.items() if k in validation_words}
        train_graphemes = set()
        for k, v in word_dict.items():
            train_graphemes.update(k)
        self.train(training_dictionary)

        model = G2PModel(self.model_path, root_directory=self.temp_directory)

        gen = PyniniDictionaryGenerator(model, validation_dictionary.keys(),
                                        temp_directory=os.path.join(self.temp_directory, 'validation'),
                                        num_jobs=self.num_jobs, num_pronunciations=self.train_config.num_pronunciations)
        output = gen.generate()
        begin = time.time()
        wer, ler = compute_validation_errors(validation_dictionary, output, num_jobs=self.num_jobs)
        print(f"WER:\t{wer:.2f}")
        print(f"LER:\t{ler:.2f}")
        self.logger.info(f"WER:\t{wer:.2f}")
        self.logger.info(f"LER:\t{ler:.2f}")
        self.logger.debug('Computation of errors for {} words took {} seconds'.format(len(validation_dictionary),
                                                                                      time.time() - begin))
