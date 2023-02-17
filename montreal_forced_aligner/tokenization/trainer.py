"""Classes for training tokenizers"""
import collections
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import pynini
import pywrapfst
import sqlalchemy

from montreal_forced_aligner.abc import MetaDict, TopLevelMfaWorker
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusMixin
from montreal_forced_aligner.data import WorkflowType
from montreal_forced_aligner.db import M2M2Job, M2MSymbol, Utterance
from montreal_forced_aligner.dictionary.mixins import DictionaryMixin
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.g2p.phonetisaurus_trainer import (
    AlignmentInitWorker,
    PhonetisaurusTrainerMixin,
)
from montreal_forced_aligner.g2p.trainer import G2PTrainer, PyniniTrainerMixin
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.models import TokenizerModel
from montreal_forced_aligner.tokenization.tokenizer import TokenizerValidator
from montreal_forced_aligner.utils import log_kaldi_errors, thirdparty_binary

__all__ = ["TokenizerTrainer"]

logger = logging.getLogger("mfa")


class TokenizerAlignmentInitWorker(AlignmentInitWorker):
    """
    Multiprocessing worker that initializes alignment FSTs for a subset of the data

    Parameters
    ----------
    job_name: int
        Integer ID for the job
    return_queue: :class:`multiprocessing.Queue`
        Queue to return data
    stopped: :class:`~montreal_forced_aligner.utils.Stopped`
        Stop check
    finished_adding: :class:`~montreal_forced_aligner.utils.Stopped`
        Check for whether the job queue is done
    args: :class:`~montreal_forced_aligner.g2p.phonetisaurus_trainer.AlignmentInitArguments`
        Arguments for initialization
    """

    def data_generator(self, session):
        grapheme_table = pywrapfst.SymbolTable.read_text(self.far_path.with_name("graphemes.syms"))
        query = session.query(Utterance.normalized_character_text).filter(
            Utterance.ignored == False, Utterance.job_id == self.job_name  # noqa
        )
        for (text,) in query:
            tokenized = [x if grapheme_table.member(x) else "<unk>" for x in text.split()]
            untokenized = [x for x in tokenized if x != "<space>"]
            yield untokenized, tokenized

    def run(self) -> None:
        """Run the function"""
        engine = sqlalchemy.create_engine(
            self.db_string,
            poolclass=sqlalchemy.NullPool,
            pool_reset_on_return=None,
            isolation_level="AUTOCOMMIT",
            logging_name=f"{type(self).__name__}_engine",
        ).execution_options(logging_token=f"{type(self).__name__}_engine")
        try:
            symbol_table = pynini.SymbolTable()
            symbol_table.add_symbol(self.eps)
            valid_output_ngrams = set()
            base_dir = os.path.dirname(self.far_path)
            with mfa_open(os.path.join(base_dir, "output_ngram.ngrams"), "r") as f:
                for line in f:
                    line = line.strip()
                    valid_output_ngrams.add(line)
            valid_input_ngrams = set()
            with mfa_open(os.path.join(base_dir, "input_ngram.ngrams"), "r") as f:
                for line in f:
                    line = line.strip()
                    valid_input_ngrams.add(line)
            count = 0
            data = {}
            with mfa_open(self.log_path, "w") as log_file, sqlalchemy.orm.Session(
                engine
            ) as session:
                far_writer = pywrapfst.FarWriter.create(self.far_path, arc_type="log")
                for current_index, (input, output) in enumerate(self.data_generator(session)):
                    if self.stopped.stop_check():
                        continue
                    try:
                        key = f"{current_index:08x}"
                        fst = pynini.Fst(arc_type="log")
                        final_state = ((len(input) + 1) * (len(output) + 1)) - 1

                        for _ in range(final_state + 1):
                            fst.add_state()
                        for i in range(len(input) + 1):
                            for j in range(len(output) + 1):
                                istate = i * (len(output) + 1) + j

                                for input_range in range(1, self.input_order + 1):
                                    for output_range in range(input_range, self.output_order + 1):
                                        if i + input_range <= len(
                                            input
                                        ) and j + output_range <= len(output):
                                            if (
                                                self.restrict
                                                and input_range > 1
                                                and output_range > 1
                                            ):
                                                continue
                                            subseq_output = output[j : j + output_range]
                                            output_string = self.seq_sep.join(subseq_output)
                                            if (
                                                output_range > 1
                                                and output_string not in valid_output_ngrams
                                            ):
                                                continue
                                            subseq_input = input[i : i + input_range]
                                            input_string = self.seq_sep.join(subseq_input)
                                            if output_range > 1:
                                                if "<space>" not in subseq_output:
                                                    continue
                                                if input_string not in output_string:
                                                    continue
                                            if (
                                                output_range == input_range
                                                and input_string != output_string
                                            ):
                                                continue
                                            if (
                                                input_range > 1
                                                and input_string not in valid_input_ngrams
                                            ):
                                                continue
                                            symbol = self.s1s2_sep.join(
                                                [input_string, output_string]
                                            )
                                            ilabel = symbol_table.find(symbol)
                                            if ilabel == pynini.NO_LABEL:
                                                ilabel = symbol_table.add_symbol(symbol)
                                            ostate = (i + input_range) * (len(output) + 1) + (
                                                j + output_range
                                            )
                                            fst.add_arc(
                                                istate,
                                                pywrapfst.Arc(
                                                    ilabel,
                                                    ilabel,
                                                    pynini.Weight(
                                                        "log", float(input_range * output_range)
                                                    ),
                                                    ostate,
                                                ),
                                            )
                        fst.set_start(0)
                        fst.set_final(final_state, pywrapfst.Weight.one(fst.weight_type()))
                        fst = pynini.connect(fst)
                        for state in fst.states():
                            for arc in fst.arcs(state):
                                sym = symbol_table.find(arc.ilabel)
                                if sym not in data:
                                    data[sym] = arc.weight
                                else:
                                    data[sym] = pynini.plus(data[sym], arc.weight)
                        if count >= self.batch_size:
                            data = {k: float(v) for k, v in data.items()}
                            self.return_queue.put((self.job_name, data, count))
                            data = {}
                            count = 0
                        log_file.flush()
                        far_writer[key] = fst
                        del fst
                        count += 1
                    except Exception as e:  # noqa
                        self.stopped.stop()
                        self.return_queue.put(e)
            if data:
                data = {k: float(v) for k, v in data.items()}
                self.return_queue.put((self.job_name, data, count))
            symbol_table.write_text(self.far_path.with_suffix(".syms"))
            return
        except Exception as e:
            self.stopped.stop()
            self.return_queue.put(e)
        finally:
            self.finished.stop()
            del far_writer


class TokenizerMixin(AcousticCorpusMixin, G2PTrainer, DictionaryMixin, TopLevelMfaWorker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_graphemes = set()
        self.uer = None
        self.cer = None
        self.deletions = False
        self.insertions = True
        self.num_training_utterances = 0
        self.num_validation_utterances = 0

    def setup(self) -> None:
        super().setup()
        self.ignore_empty_utterances = True
        if self.initialized:
            return
        try:
            self._load_corpus()
            self._create_dummy_dictionary()
            self.initialize_jobs()
            self.normalize_text()
            self.initialize_training()
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise
        self.initialized = True

    def evaluate_tokenizer(self) -> None:
        """
        Validate the tokenizer model against held out data
        """
        temp_model_path = self.working_log_directory.joinpath("tokenizer_model.zip")
        self.export_model(temp_model_path)
        temp_dir = self.working_directory.joinpath("validation")
        temp_dir.mkdir(parents=True, exist_ok=True)
        with self.session() as session:
            validation_set = {}
            query = session.query(Utterance.normalized_character_text).filter(
                Utterance.ignored == True  # noqa
            )
            for (text,) in query:
                tokenized = text.split()
                untokenized = [x for x in tokenized if x != "<space>"]
                tokenized = [x if x != "<space>" else " " for x in tokenized]
                validation_set[" ".join(untokenized)] = "".join(tokenized)
        gen = TokenizerValidator(
            tokenizer_model_path=temp_model_path,
            corpus_directory=self.corpus_directory,
            utterances_to_tokenize=list(validation_set.keys()),
        )
        output = gen.tokenize_utterances()
        with mfa_open(temp_dir.joinpath("validation_output.txt"), "w") as f:
            for (orthography, pronunciations) in output.items():
                if not pronunciations:
                    continue
                for p in pronunciations:
                    if not p:
                        continue
                    f.write(f"{orthography}\t{p}\n")
        gen.compute_validation_errors(validation_set, output)
        self.uer = gen.uer
        self.cer = gen.cer


class PhonetisaurusTokenizerTrainer(PhonetisaurusTrainerMixin, TokenizerMixin):

    alignment_init_function = TokenizerAlignmentInitWorker

    def __init__(
        self, input_order: int = 2, output_order: int = 3, oov_count_threshold: int = 5, **kwargs
    ):
        super().__init__(
            oov_count_threshold=oov_count_threshold,
            grapheme_order=input_order,
            phone_order=output_order,
            **kwargs,
        )

    @property
    def data_source_identifier(self) -> str:
        """Corpus name"""
        return self.corpus_directory.name

    @property
    def meta(self) -> MetaDict:
        """Metadata for exported tokenizer model"""
        from datetime import datetime

        from ..utils import get_mfa_version

        m = {
            "version": get_mfa_version(),
            "architecture": self.architecture,
            "train_date": str(datetime.now()),
            "evaluation": {},
            "input_order": self.input_order,
            "output_order": self.output_order,
            "oov_count_threshold": self.oov_count_threshold,
            "training": {
                "num_utterances": self.num_training_utterances,
                "num_graphemes": len(self.training_graphemes),
            },
        }

        if self.evaluation_mode:
            m["evaluation"]["num_utterances"] = self.num_validation_utterances
            m["evaluation"]["utterance_error_rate"] = self.uer
            m["evaluation"]["character_error_rate"] = self.cer
        return m

    def train(self) -> None:
        if os.path.exists(self.fst_path):
            self.finalize_training()
            return
        super().train()

    def initialize_training(self) -> None:
        """Initialize training tokenizer model"""

        self.create_new_current_workflow(WorkflowType.tokenizer_training)
        if self.fst_path.exists():
            return
        logger.info("Initializing training...")
        with self.session() as session:
            session.query(M2M2Job).delete()
            session.query(M2MSymbol).delete()
            session.commit()
            self.num_validation_utterances = 0
            self.num_training_utterances = 0
            if self.evaluation_mode:
                validation_items = int(self.num_utterances * self.validation_proportion)
                validation_utterances = (
                    sqlalchemy.select(Utterance.id)
                    .order_by(sqlalchemy.func.random())
                    .limit(validation_items)
                    .scalar_subquery()
                )
                query = (
                    sqlalchemy.update(Utterance)
                    .execution_options(synchronize_session="fetch")
                    .values(ignored=True)
                    .where(Utterance.id.in_(validation_utterances))
                )
                with session.begin_nested():
                    session.execute(query)
                    session.flush()
                session.commit()
                self.num_validation_utterances = (
                    session.query(Utterance.id).filter(Utterance.ignored == True).count()  # noqa
                )

            query = session.query(Utterance.normalized_character_text).filter(
                Utterance.ignored == False  # noqa
            )
            unk_character = "<unk>"
            self.training_graphemes.add(unk_character)
            counts = collections.Counter()
            for (text,) in query:
                counts.update(text.split())
            with mfa_open(
                self.working_directory.joinpath("input.txt"), "w"
            ) as untokenized_f, mfa_open(
                self.working_directory.joinpath("output.txt"), "w"
            ) as tokenized_f:
                for (text,) in query:
                    assert text
                    tokenized = [
                        x if counts[x] >= self.oov_count_threshold else unk_character
                        for x in text.split()
                    ]
                    untokenized = [x for x in tokenized if x != "<space>"]
                    self.num_training_utterances += 1
                    self.training_graphemes.update(tokenized)
                    untokenized_f.write(" ".join(untokenized) + "\n")
                    tokenized_f.write(" ".join(tokenized) + "\n")
            index = 1
            with mfa_open(self.working_directory.joinpath("graphemes.syms"), "w") as f:
                f.write("<eps>\t0\n")
                for g in sorted(self.training_graphemes):
                    f.write(f"{g}\t{index}\n")
                    index += 1
            self.compute_initial_ngrams()
            self.g2p_num_training_pronunciations = self.num_training_utterances

    def finalize_training(self) -> None:
        """Finalize training"""
        shutil.copyfile(self.fst_path, self.working_directory.joinpath("tokenizer.fst"))
        shutil.copyfile(self.grapheme_symbols_path, self.working_directory.joinpath("input.syms"))
        shutil.copyfile(self.phone_symbols_path, self.working_directory.joinpath("output.syms"))
        if self.evaluation_mode:
            self.evaluate_tokenizer()

    def export_model(self, output_model_path: Path) -> None:
        """
        Export tokenizer model to specified path

        Parameters
        ----------
        output_model_path: :class:`~pathlib.Path`
            Path to export model
        """
        directory = output_model_path.parent

        models_temp_dir = self.working_directory.joinpath("model_archive_temp")
        model = TokenizerModel.empty(output_model_path.stem, root_directory=models_temp_dir)
        model.add_meta_file(self)
        model.add_tokenizer_model(self.working_directory)
        model.add_graphemes_path(self.working_directory)
        if directory:
            os.makedirs(directory, exist_ok=True)
        model.dump(output_model_path)
        if not GLOBAL_CONFIG.current_profile.debug:
            model.clean_up()
        # self.clean_up()
        logger.info(f"Saved model to {output_model_path}")


class TokenizerTrainer(PyniniTrainerMixin, TokenizerMixin):
    def __init__(self, oov_count_threshold=5, **kwargs):
        super().__init__(oov_count_threshold=oov_count_threshold, **kwargs)
        self.training_graphemes = set()
        self.uer = None
        self.cer = None
        self.deletions = False
        self.insertions = True

    @property
    def meta(self) -> MetaDict:
        """Metadata for exported tokenizer model"""
        from datetime import datetime

        from ..utils import get_mfa_version

        m = {
            "version": get_mfa_version(),
            "architecture": self.architecture,
            "train_date": str(datetime.now()),
            "evaluation": {},
            "training": {
                "num_utterances": self.num_training_utterances,
                "num_graphemes": len(self.training_graphemes),
            },
        }

        if self.evaluation_mode:
            m["evaluation"]["num_utterances"] = self.num_validation_utterances
            m["evaluation"]["utterance_error_rate"] = self.uer
            m["evaluation"]["character_error_rate"] = self.cer
        return m

    @property
    def data_source_identifier(self) -> str:
        """Corpus name"""
        return self.corpus_directory.name

    @property
    def sym_path(self) -> Path:
        return self.working_directory.joinpath("graphemes.syms")

    def initialize_training(self) -> None:
        """Initialize training tokenizer model"""
        self.create_new_current_workflow(WorkflowType.tokenizer_training)
        with self.session() as session:
            self.num_validation_utterances = 0
            self.num_training_utterances = 0
            self.num_iterations = 1
            self.random_starts = 1
            self.input_token_type = self.sym_path
            self.output_token_type = self.sym_path
            if self.evaluation_mode:
                validation_items = int(self.num_utterances * self.validation_proportion)
                validation_utterances = (
                    sqlalchemy.select(Utterance.id)
                    .order_by(sqlalchemy.func.random())
                    .limit(validation_items)
                    .scalar_subquery()
                )
                query = (
                    sqlalchemy.update(Utterance)
                    .execution_options(synchronize_session="fetch")
                    .values(ignored=True)
                    .where(Utterance.id.in_(validation_utterances))
                )
                with session.begin_nested():
                    session.execute(query)
                    session.flush()
                session.commit()
                self.num_validation_utterances = (
                    session.query(Utterance.id).filter(Utterance.ignored == True).count()  # noqa
                )

            query = session.query(Utterance.normalized_character_text).filter(
                Utterance.ignored == False  # noqa
            )
            unk_character = "<unk>"
            self.training_graphemes.add(unk_character)
            counts = collections.Counter()
            for (text,) in query:
                counts.update(text.split())
            with mfa_open(self.input_path, "w") as untokenized_f, mfa_open(
                self.output_path, "w"
            ) as tokenized_f:
                for (text,) in query:
                    assert text
                    tokenized = [
                        x if counts[x] >= self.oov_count_threshold else unk_character
                        for x in text.split()
                    ]
                    untokenized = [x for x in tokenized if x != "<space>"]
                    self.num_training_utterances += 1
                    self.training_graphemes.update(tokenized)
                    untokenized_f.write(" ".join(untokenized) + "\n")
                    tokenized_f.write(" ".join(tokenized) + "\n")
            index = 1
            with mfa_open(self.sym_path, "w") as f:
                f.write("<eps>\t0\n")
                for g in sorted(self.training_graphemes):
                    f.write(f"{g}\t{index}\n")
                    index += 1

    def _lexicon_covering(self, input_path=None, output_path=None) -> None:
        """Builds covering grammar and lexicon FARs."""
        # Sets of labels for the covering grammar.
        with mfa_open(
            self.working_log_directory.joinpath("covering_grammar.log"), "w"
        ) as log_file:
            if input_path is None:
                input_path = self.input_path
            if output_path is None:
                output_path = self.output_path
            com = [
                thirdparty_binary("farcompilestrings"),
                "--fst_type=compact",
            ]
            com.append("--token_type=symbol")
            com.append(
                f"--symbols={self.sym_path}",
            )
            com.append("--unknown_symbol=<unk>")
            com.extend([input_path, self.input_far_path])
            print(" ".join(map(str, com)), file=log_file)
            subprocess.check_call(com, env=os.environ, stderr=log_file, stdout=log_file)
            com = [
                thirdparty_binary("farcompilestrings"),
                "--fst_type=compact",
                "--token_type=symbol",
                f"--symbols={self.sym_path}",
                output_path,
                self.output_far_path,
            ]
            print(" ".join(map(str, com)), file=log_file)
            subprocess.check_call(com, env=os.environ, stderr=log_file, stdout=log_file)
            cg = pywrapfst.VectorFst()
            state = cg.add_state()
            cg.set_start(state)
            labels = pywrapfst.SymbolTable.read_text(self.sym_path)
            one = pywrapfst.Weight.one(cg.weight_type())
            for i in range(labels.num_symbols()):
                if labels.find(i) == "<eps>":
                    continue
                cg.add_arc(state, pywrapfst.Arc(i, i, one, state))
            olabel = labels.find("<space>")
            cg.add_arc(state, pywrapfst.Arc(0, olabel, one, state))
            cg.set_final(state)
            assert cg.verify(), "Label acceptor is ill-formed"
            cg.write(self.cg_path)

    def train(self) -> None:
        """
        Train a tokenizer model
        """
        os.makedirs(self.working_log_directory, exist_ok=True)
        begin = time.time()
        if os.path.exists(self.far_path) and os.path.exists(self.encoder_path):
            logger.info("Alignment already done, skipping!")
        else:
            self.align_g2p()
            logger.debug(f"Aligning took {time.time() - begin:.3f} seconds")
        begin = time.time()
        self.generate_model()
        logger.debug(f"Generating model took {time.time() - begin:.3f} seconds")
        self.finalize_training()

    def finalize_training(self) -> None:
        """Finalize training"""
        shutil.copyfile(self.fst_path, self.working_directory.joinpath("tokenizer.fst"))
        if self.evaluation_mode:
            self.evaluate_tokenizer()

    def export_model(self, output_model_path: Path) -> None:
        """
        Export tokenizer model to specified path

        Parameters
        ----------
        output_model_path: :class:`~pathlib.Path`
            Path to export model
        """
        directory = output_model_path.parent

        models_temp_dir = self.working_directory.joinpath("model_archive_temp")
        model = TokenizerModel.empty(output_model_path.stem, root_directory=models_temp_dir)
        model.add_meta_file(self)
        model.add_tokenizer_model(self.working_directory)
        model.add_graphemes_path(self.working_directory)
        if directory:
            os.makedirs(directory, exist_ok=True)
        model.dump(output_model_path)
        if not GLOBAL_CONFIG.current_profile.debug:
            model.clean_up()
        # self.clean_up()
        logger.info(f"Saved model to {output_model_path}")
