"""Classes for tokenizers"""

import csv
import functools
import logging
import multiprocessing as mp
import os
import queue
import time
import typing
from pathlib import Path

import pynini
import pywrapfst
import sqlalchemy
from praatio import textgrid
from pynini import Fst
from pynini.lib import rewrite
from pywrapfst import SymbolTable
from sqlalchemy.orm import joinedload, selectinload
from tqdm.rich import tqdm

from montreal_forced_aligner.abc import KaldiFunction, TopLevelMfaWorker
from montreal_forced_aligner.alignment.multiprocessing import construct_output_path
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusMixin
from montreal_forced_aligner.data import MfaArguments, TextgridFormats
from montreal_forced_aligner.db import File, Utterance, bulk_update
from montreal_forced_aligner.dictionary.mixins import DictionaryMixin
from montreal_forced_aligner.exceptions import PyniniGenerationError
from montreal_forced_aligner.g2p.generator import PhonetisaurusRewriter, Rewriter, RewriterWorker
from montreal_forced_aligner.helper import edit_distance, mfa_open
from montreal_forced_aligner.models import TokenizerModel
from montreal_forced_aligner.utils import Stopped, run_kaldi_function

if typing.TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from dataclassy import dataclass

__all__ = [
    "TokenizerRewriter",
    "TokenizerArguments",
    "TokenizerFunction",
    "TokenizerValidator",
    "CorpusTokenizer",
]

logger = logging.getLogger("mfa")


class TokenizerRewriter(Rewriter):
    """
    Helper object for rewriting

    Parameters
    ----------
    fst: pynini.Fst
        Tokenizer FST model
    grapheme_symbols: pynini.SymbolTable
        Grapheme symbol table
    """

    def __init__(
        self,
        fst: Fst,
        grapheme_symbols: SymbolTable,
    ):
        self.grapheme_symbols = grapheme_symbols
        self.rewrite = functools.partial(
            rewrite.top_rewrite,
            rule=fst,
            input_token_type=grapheme_symbols,
            output_token_type=grapheme_symbols,
        )

    def __call__(self, i: str) -> str:  # pragma: no cover
        """Call the rewrite function"""
        i = i.replace(" ", "")
        original = list(i)
        unks = []
        normalized = []
        for c in original:
            if self.grapheme_symbols.member(c):
                normalized.append(c)
            else:
                unks.append(c)
                normalized.append("<unk>")
        hypothesis = self.rewrite(" ".join(normalized)).split()
        unk_index = 0
        for i, w in enumerate(hypothesis):
            if w == "<unk>":
                hypothesis[i] = unks[unk_index]
                unk_index += 1
            elif w == "<space>":
                hypothesis[i] = " "
        return "".join(hypothesis)


class TokenizerPhonetisaurusRewriter(PhonetisaurusRewriter):
    """
    Helper function for rewriting

    Parameters
    ----------
    fst: pynini.Fst
        G2P FST model
    input_token_type: pynini.SymbolTable
        Grapheme symbol table
    output_token_type: pynini.SymbolTable
    num_pronunciations: int
        Number of pronunciations, default to 0.  If this is 0, thresholding is used
    threshold: float
        Threshold to use for pruning rewrite lattice, defaults to 1.5, only used if num_pronunciations is 0
    grapheme_order: int
        Maximum number of graphemes to consider single segment
    seq_sep: str
        Separator to use between grapheme symbols
    """

    def __init__(
        self,
        fst: Fst,
        input_token_type: SymbolTable,
        output_token_type: SymbolTable,
        input_order: int = 2,
        seq_sep: str = "|",
    ):
        self.fst = fst
        self.seq_sep = seq_sep
        self.input_token_type = input_token_type
        self.output_token_type = output_token_type
        self.input_order = input_order
        self.rewrite = functools.partial(
            rewrite.top_rewrite,
            rule=fst,
            input_token_type=None,
            output_token_type=output_token_type,
        )

    def __call__(self, graphemes: str) -> str:  # pragma: no cover
        """Call the rewrite function"""
        graphemes = graphemes.replace(" ", "")
        original = list(graphemes)
        unks = []
        normalized = []
        for c in original:
            if self.output_token_type.member(c):
                normalized.append(c)
            else:
                unks.append(c)
                normalized.append("<unk>")
        fst = pynini.Fst()
        one = pynini.Weight.one(fst.weight_type())
        max_state = 0
        for i in range(len(normalized)):
            start_state = fst.add_state()
            for j in range(1, self.input_order + 1):
                if i + j <= len(normalized):
                    substring = self.seq_sep.join(normalized[i : i + j])
                    ilabel = self.input_token_type.find(substring)
                    if ilabel != pynini.NO_LABEL:
                        fst.add_arc(start_state, pynini.Arc(ilabel, ilabel, one, i + j))
                    if i + j >= max_state:
                        max_state = i + j
        for _ in range(fst.num_states(), max_state + 1):
            fst.add_state()
        fst.set_start(0)
        fst.set_final(len(normalized), one)
        fst.set_input_symbols(self.input_token_type)
        fst.set_output_symbols(self.input_token_type)
        hypothesis = self.rewrite(fst).split()
        unk_index = 0
        output = []
        for i, w in enumerate(hypothesis):
            if w == "<unk>":
                output.append(unks[unk_index])
                unk_index += 1
            elif w == "<space>":
                if i > 0 and hypothesis[i - 1] == " ":
                    continue
                output.append(" ")
            else:
                output.append(w)
        return "".join(output).strip()


@dataclass
class TokenizerArguments(MfaArguments):
    rewriter: Rewriter


class TokenizerFunction(KaldiFunction):
    def __init__(self, args: TokenizerArguments):
        super().__init__(args)
        self.rewriter = args.rewriter

    def _run(self) -> typing.Generator:
        """Run the function"""
        engine = sqlalchemy.create_engine(self.db_string)
        with sqlalchemy.orm.Session(engine) as session:
            utterances = session.query(Utterance.id, Utterance.normalized_text).filter(
                Utterance.job_id == self.job_name
            )
            for u_id, text in utterances:
                tokenized_text = self.rewriter(text)
                yield u_id, tokenized_text


class CorpusTokenizer(AcousticCorpusMixin, TopLevelMfaWorker, DictionaryMixin):
    """
    Top-level worker for generating pronunciations from a corpus and a Pynini tokenizer model
    """

    model_class = TokenizerModel

    def __init__(self, tokenizer_model_path: Path = None, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer_model = TokenizerModel(
            tokenizer_model_path, root_directory=getattr(self, "workflow_directory", None)
        )

    def setup(self) -> None:
        """Set up the pronunciation generator"""
        if self.initialized:
            return
        self._load_corpus()
        self.initialize_jobs()
        super().setup()
        self._create_dummy_dictionary()
        self.normalize_text()
        self.fst = pynini.Fst.read(self.tokenizer_model.fst_path)

        if self.tokenizer_model.meta["architecture"] == "phonetisaurus":
            self.output_token_type = pywrapfst.SymbolTable.read_text(
                self.tokenizer_model.output_sym_path
            )
            self.input_token_type = pywrapfst.SymbolTable.read_text(
                self.tokenizer_model.input_sym_path
            )
            self.rewriter = TokenizerPhonetisaurusRewriter(
                self.fst,
                self.input_token_type,
                self.output_token_type,
                input_order=self.tokenizer_model.meta["input_order"],
            )
        else:
            self.grapheme_symbols = pywrapfst.SymbolTable.read_text(self.tokenizer_model.sym_path)

            self.rewriter = TokenizerRewriter(
                self.fst,
                self.grapheme_symbols,
            )
        self.initialized = True

    def export_files(self, output_directory: Path) -> None:
        """Export transcriptions"""
        with self.session() as session:
            files = session.query(File).options(
                selectinload(File.utterances),
                selectinload(File.speakers),
                joinedload(File.sound_file),
            )
            for file in files:
                utterance_count = len(file.utterances)
                if file.sound_file is not None:
                    duration = file.sound_file.duration
                else:
                    duration = max([u.end for u in file.utterances])
                if utterance_count == 0:
                    logger.debug(f"Could not find any utterances for {file.name}")
                elif (
                    utterance_count == 1
                    and file.utterances[0].begin == 0
                    and file.utterances[0].end == duration
                ):
                    output_format = "lab"
                else:
                    output_format = TextgridFormats.SHORT_TEXTGRID
                output_path = construct_output_path(
                    file.name,
                    file.relative_path,
                    output_directory,
                    output_format=output_format,
                )
                data = file.construct_transcription_tiers(original_text=True)
                if output_format == "lab":
                    for intervals in data.values():
                        with mfa_open(output_path, "w") as f:
                            f.write(intervals["text"][0].label)
                else:
                    tg = textgrid.Textgrid()
                    tg.minTimestamp = 0
                    tg.maxTimestamp = round(duration, 5)
                    for speaker in file.speakers:
                        speaker = speaker.name
                        intervals = data[speaker]["text"]
                        tier = textgrid.IntervalTier(
                            speaker,
                            [x.to_tg_interval() for x in intervals],
                            minT=0,
                            maxT=round(duration, 5),
                        )

                        tg.addTier(tier)
                    tg.save(output_path, includeBlankSpaces=True, format=output_format)

    def tokenize_arguments(self) -> typing.List[TokenizerArguments]:
        return [TokenizerArguments(j.id, self.db_string, None, self.rewriter) for j in self.jobs]

    def tokenize_utterances(self) -> None:
        """
        Tokenize utterances

        Returns
        -------
        dict[str, list[str]]
            Mappings of keys to their tokenized utterances
        """
        begin = time.time()
        if not self.initialized:
            self.setup()
        logger.info("Tokenizing utterances...")
        args = self.tokenize_arguments()
        with tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            update_mapping = []
            for utt_id, tokenized in run_kaldi_function(TokenizerFunction, args, pbar.update):
                update_mapping.append({"id": utt_id, "text": tokenized})
        with self.session() as session:
            bulk_update(session, Utterance, update_mapping)
            session.commit()

        logger.debug(f"Tokenizing utterances took {time.time() - begin:.3f} seconds")


class TokenizerValidator(CorpusTokenizer):
    def __init__(self, utterances_to_tokenize: typing.List[str] = None, **kwargs):
        super().__init__(**kwargs)
        if utterances_to_tokenize is None:
            utterances_to_tokenize = []
        self.utterances_to_tokenize = utterances_to_tokenize
        self.uer = None
        self.cer = None

    def setup(self):
        TopLevelMfaWorker.setup(self)
        if self.initialized:
            return
        self._current_workflow = "validation"
        os.makedirs(self.working_log_directory, exist_ok=True)
        self.fst = pynini.Fst.read(self.tokenizer_model.fst_path)

        if self.tokenizer_model.meta["architecture"] == "phonetisaurus":
            self.output_token_type = pywrapfst.SymbolTable.read_text(
                self.tokenizer_model.output_sym_path
            )
            self.input_token_type = pywrapfst.SymbolTable.read_text(
                self.tokenizer_model.input_sym_path
            )
            self.rewriter = TokenizerPhonetisaurusRewriter(
                self.fst,
                self.input_token_type,
                self.output_token_type,
                input_order=self.tokenizer_model.meta["input_order"],
            )
        else:
            self.grapheme_symbols = pywrapfst.SymbolTable.read_text(self.tokenizer_model.sym_path)

            self.rewriter = TokenizerRewriter(
                self.fst,
                self.grapheme_symbols,
            )
        self.initialized = True

    def tokenize_utterances(self) -> typing.Dict[str, str]:
        """
        Tokenize utterances

        Returns
        -------
        dict[str, list[str]]
            Mappings of keys to their tokenized utterances
        """
        num_utterances = len(self.utterances_to_tokenize)
        begin = time.time()
        if not self.initialized:
            self.setup()
        logger.info("Tokenizing utterances...")
        to_return = {}
        if num_utterances < 30 or GLOBAL_CONFIG.num_jobs == 1:
            with tqdm(total=num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
                for utterance in self.utterances_to_tokenize:
                    pbar.update(1)
                    result = self.rewriter(utterance)
                    to_return[utterance] = result
        else:
            stopped = Stopped()
            job_queue = mp.Queue()
            for utterance in self.utterances_to_tokenize:
                job_queue.put(utterance)
            error_dict = {}
            return_queue = mp.Queue()
            procs = []
            for _ in range(GLOBAL_CONFIG.num_jobs):
                p = RewriterWorker(
                    job_queue,
                    return_queue,
                    self.rewriter,
                    stopped,
                )
                procs.append(p)
                p.start()
            with tqdm(total=num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
                while True:
                    try:
                        utterance, result = return_queue.get(timeout=1)
                        if stopped.stop_check():
                            continue
                    except queue.Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(1)
                    if isinstance(result, Exception):
                        error_dict[utterance] = result
                        continue
                    to_return[utterance] = result

            for p in procs:
                p.join()
            if error_dict:
                raise PyniniGenerationError(error_dict)
        logger.debug(f"Processed {num_utterances} in {time.time() - begin:.3f} seconds")
        return to_return

    @property
    def data_source_identifier(self) -> str:
        """Dummy "validation" data source"""
        return "validation"

    @property
    def data_directory(self) -> Path:
        """Data directory"""
        return self.working_directory

    @property
    def evaluation_csv_path(self) -> Path:
        """Path to working directory's CSV file"""
        return self.working_directory.joinpath("pronunciation_evaluation.csv")

    def compute_validation_errors(
        self,
        gold_values: typing.Dict[str, str],
        hypothesis_values: typing.Dict[str, str],
    ):
        """
        Computes validation errors

        Parameters
        ----------
        gold_values: dict[str, set[str]]
            Gold pronunciations
        hypothesis_values: dict[str, list[str]]
            Hypothesis pronunciations
        """
        begin = time.time()
        # Word-level measures.
        correct = 0
        incorrect = 0
        # Label-level measures.
        total_edits = 0
        total_length = 0
        # Since the edit distance algorithm is quadratic, let's do this with
        # multiprocessing.
        logger.debug(f"Processing results for {len(hypothesis_values)} hypotheses")
        to_comp = []
        indices = []
        output = []
        for word, gold in gold_values.items():
            if word not in hypothesis_values:
                incorrect += 1
                gold_length = len(gold)
                total_edits += gold_length
                total_length += gold_length
                output.append(
                    {
                        "Word": word,
                        "Gold tokenization": gold,
                        "Hypothesis tokenization": "",
                        "Accuracy": 0,
                        "Error rate": 1.0,
                        "Length": gold_length,
                    }
                )
                continue
            hyp = hypothesis_values[word]
            if hyp == gold:
                correct += 1
                total_length += len(hyp)
                output.append(
                    {
                        "Word": word,
                        "Gold tokenization": gold,
                        "Hypothesis tokenization": hyp,
                        "Accuracy": 1,
                        "Error rate": 0.0,
                        "Length": len(hyp),
                    }
                )
            else:
                incorrect += 1
                indices.append(word)
                to_comp.append((gold, hyp))  # Multiple hypotheses to compare
        with mp.Pool(GLOBAL_CONFIG.num_jobs) as pool:
            gen = pool.starmap(edit_distance, to_comp)
            for i, (edits) in enumerate(gen):
                word = indices[i]
                gold = gold_values[word]
                length = len(gold)
                hyp = hypothesis_values[word]
                output.append(
                    {
                        "Word": word,
                        "Gold tokenization": gold,
                        "Hypothesis tokenization": hyp,
                        "Accuracy": 1,
                        "Error rate": edits / length,
                        "Length": length,
                    }
                )
                total_edits += edits
                total_length += length
        with mfa_open(self.evaluation_csv_path, "w") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Word",
                    "Gold tokenization",
                    "Hypothesis tokenization",
                    "Accuracy",
                    "Error rate",
                    "Length",
                ],
            )
            writer.writeheader()
            for line in output:
                writer.writerow(line)
        self.uer = 100 * incorrect / (correct + incorrect)
        self.cer = 100 * total_edits / total_length
        logger.info(f"UER:\t{self.uer:.2f}")
        logger.info(f"CER:\t{self.cer:.2f}")
        logger.debug(
            f"Computation of errors for {len(gold_values)} utterances took {time.time() - begin:.3f} seconds"
        )
