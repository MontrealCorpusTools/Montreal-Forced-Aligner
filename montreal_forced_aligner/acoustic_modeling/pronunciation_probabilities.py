"""Class definitions for PronunciationProbabilityTrainer"""
import json
import logging
import os
import re
import shutil
import time
import typing
from pathlib import Path

import pynini
import pywrapfst
from _kalpy.fstext import fst_determinize_star, fst_minimize_encoded, fst_push_special
from kalpy.fstext.lexicon import G2PCompiler
from kalpy.fstext.utils import kaldi_to_pynini, pynini_to_kaldi
from kalpy.gmm.align import GmmAligner
from sqlalchemy.orm import joinedload
from tqdm.rich import tqdm

from montreal_forced_aligner import config
from montreal_forced_aligner.acoustic_modeling.base import AcousticModelTrainingMixin
from montreal_forced_aligner.alignment.multiprocessing import (
    GeneratePronunciationsArguments,
    GeneratePronunciationsFunction,
)
from montreal_forced_aligner.db import CorpusWorkflow, Dictionary, Pronunciation, Utterance, Word
from montreal_forced_aligner.g2p.trainer import PyniniTrainerMixin
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import parse_dictionary_file, run_kaldi_function

__all__ = ["PronunciationProbabilityTrainer"]

logger = logging.getLogger("mfa")
logger.write = lambda msg: logger.info(msg) if msg != "\n" else None
logger.flush = lambda: None


class PronunciationProbabilityTrainer(AcousticModelTrainingMixin, PyniniTrainerMixin):
    """
    Class for training pronunciation probabilities based off of alignment pronunciations

    Parameters
    ----------
    previous_trainer: AcousticModelTrainingMixin
        Previous trainer in the training configuration
    silence_probabilities: bool
        Flag for whether to save silence probabilities
    """

    def __init__(
        self,
        previous_trainer: typing.Optional[AcousticModelTrainingMixin] = None,
        silence_probabilities: bool = True,
        train_g2p: bool = False,
        use_phonetisaurus: bool = False,
        num_iterations: int = 10,
        model_size: int = 100000,
        **kwargs,
    ):
        self.previous_trainer = previous_trainer
        self.silence_probabilities = silence_probabilities
        self.train_g2p = train_g2p
        self.use_phonetisaurus = use_phonetisaurus
        super(PronunciationProbabilityTrainer, self).__init__(
            num_iterations=num_iterations, model_size=model_size, **kwargs
        )
        self.subset = self.previous_trainer.subset
        self.pronunciations_complete = False

    @property
    def train_type(self) -> str:
        """Training type"""
        return "pronunciation_probabilities"

    def compute_calculated_properties(self) -> None:
        """Compute calculated properties"""
        pass

    def _trainer_initialization(self) -> None:
        """Initialize trainer"""
        pass

    @property
    def exported_model_path(self) -> Path:
        """Path to exported acoustic model"""
        return self.previous_trainer.exported_model_path

    @property
    def model_path(self) -> Path:
        """Current acoustic model path"""
        return self.working_directory.joinpath("final.mdl")

    @property
    def alignment_model_path(self) -> Path:
        """Alignment model path"""
        path = self.model_path.with_suffix(".alimdl")
        if os.path.exists(path):
            return path
        return self.model_path

    @property
    def phone_symbol_table_path(self) -> Path:
        """Worker's phone symbol table"""
        return self.worker.phone_symbol_table_path

    @property
    def grapheme_symbol_table_path(self) -> Path:
        """Worker's grapheme symbol table"""
        return self.worker.grapheme_symbol_table_path

    @property
    def input_path(self) -> Path:
        """Path to temporary file to store training data"""
        return self.working_directory.joinpath(f"input_{self._data_source}.txt")

    @property
    def output_path(self) -> Path:
        """Path to temporary file to store training data"""
        return self.working_directory.joinpath(f"output_{self._data_source}.txt")

    @property
    def output_alignment_path(self) -> Path:
        """Path to temporary file to store training data"""
        return self.working_directory.joinpath(f"output_{self._data_source}_alignment.txt")

    def generate_pronunciations_arguments(self) -> typing.List[GeneratePronunciationsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.GeneratePronunciationsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.GeneratePronunciationsArguments`]
            Arguments for processing
        """
        align_options = self.align_options
        align_options.pop("boost_silence", 1.0)
        disambiguation_symbols = [self.phone_mapping[p] for p in self.disambiguation_symbols]
        aligner = GmmAligner(
            self.model_path, disambiguation_symbols=disambiguation_symbols, **align_options
        )

        return [
            GeneratePronunciationsArguments(
                j.id,
                getattr(self, "session", ""),
                self.working_log_directory.joinpath(f"generate_pronunciations.{j.id}.log"),
                aligner,
                self.lexicon_compilers,
                True,
            )
            for j in self.jobs
        ]

    def align_g2p(self, output_path=None) -> None:
        """Runs the entire alignment regimen."""
        self._lexicon_covering(output_path=output_path)
        self._alignments()
        self._encode()

    def train_g2p_lexicon(self) -> None:
        """Generate a G2P lexicon based on aligned transcripts"""
        arguments = self.generate_pronunciations_arguments()
        working_dir = super(PronunciationProbabilityTrainer, self).working_directory
        texts = {}
        with self.worker.session() as session:
            query = session.query(Utterance.id, Utterance.normalized_character_text)
            query = query.filter(Utterance.ignored == False)  # noqa
            # query = query.filter(sqlalchemy.or_(Utterance.oovs == '', Utterance.oovs == None))
            if self.subset:
                query = query.filter_by(in_subset=True)
            for utt_id, text in query:
                texts[utt_id] = text
            input_files = {
                x: open(
                    os.path.join(working_dir, f"input_{self.worker.dictionary_base_names[x]}.txt"),
                    "w",
                    encoding="utf8",
                    newline="",
                )
                for x in self.worker.dictionary_lookup.values()
            }
            output_files = {
                x: open(
                    os.path.join(
                        working_dir, f"output_{self.worker.dictionary_base_names[x]}.txt"
                    ),
                    "w",
                    encoding="utf8",
                    newline="",
                )
                for x in self.worker.dictionary_lookup.values()
            }
            output_alignment_files = {
                x: open(
                    os.path.join(
                        working_dir, f"output_{self.worker.dictionary_base_names[x]}_alignment.txt"
                    ),
                    "w",
                    encoding="utf8",
                    newline="",
                )
                for x in self.worker.dictionary_lookup.values()
            }
            with tqdm(total=self.num_current_utterances, disable=config.QUIET) as pbar:
                for dict_id, utt_id, phones in run_kaldi_function(
                    GeneratePronunciationsFunction, arguments, pbar.update
                ):
                    if utt_id not in texts or not texts[utt_id]:
                        continue

                    print(phones, file=output_alignment_files[dict_id])
                    print(
                        re.sub(r"\s+", " ", phones.replace("#1", "").replace("#2", "")).strip(),
                        file=output_files[dict_id],
                    )
                    print(texts[utt_id], file=input_files[dict_id])
            for f in input_files.values():
                f.close()
            for f in output_files.values():
                f.close()
            for f in output_alignment_files.values():
                f.close()
            self.pronunciations_complete = True
            os.makedirs(self.working_log_directory, exist_ok=True)
            dictionaries = session.query(Dictionary)
            shutil.copyfile(
                self.phone_symbol_table_path, self.working_directory.joinpath("phones.txt")
            )
            shutil.copyfile(
                self.grapheme_symbol_table_path,
                self.working_directory.joinpath("graphemes.txt"),
            )
            self.input_token_type = self.grapheme_symbol_table_path
            self.output_token_type = self.phone_symbol_table_path
            for d in dictionaries:
                logger.info(f"Training G2P for {d.name}...")
                self._data_source = self.worker.dictionary_base_names[d.id]

                begin = time.time()
                if os.path.exists(self.far_path) and os.path.exists(self.encoder_path):
                    logger.info("Alignment already done, skipping!")
                else:
                    self.align_g2p()
                    logger.debug(
                        f"Aligning utterances for {d.name} took {time.time() - begin:.3f} seconds"
                    )
                begin = time.time()
                self.generate_model()
                logger.debug(
                    f"Generating model for {d.name} took {time.time() - begin:.3f} seconds"
                )
                if d.lexicon_fst_path.exists():
                    os.rename(d.lexicon_fst_path, d.lexicon_fst_path.with_suffix(".backup"))
                os.rename(self.fst_path, d.lexicon_fst_path)

                if False and not config.DEBUG:
                    os.remove(self.output_path)
                    os.remove(self.input_far_path)
                    os.remove(self.output_far_path)
                for f in os.listdir(self.working_directory):
                    if any(f.endswith(x) for x in [".fst", ".like", ".far", ".enc"]):
                        os.remove(self.working_directory.joinpath(f))

                begin = time.time()
                self.align_g2p(self.output_alignment_path)
                logger.debug(
                    f"Aligning utterances for {d.name} took {time.time() - begin:.3f} seconds"
                )
                begin = time.time()
                self.generate_model()
                logger.debug(
                    f"Generating model for {d.name} took {time.time() - begin:.3f} seconds"
                )
                if d.align_lexicon_path.exists():
                    os.rename(d.align_lexicon_path, d.align_lexicon_path.with_suffix(".backup"))
                os.rename(self.fst_path, d.align_lexicon_path)
                if not config.DEBUG:
                    os.remove(self.output_alignment_path)
                    os.remove(self.input_path)
                    os.remove(self.input_far_path)
                    os.remove(self.output_far_path)
                    for f in os.listdir(self.working_directory):
                        if any(f.endswith(x) for x in [".fst", ".like", ".far", ".enc"]):
                            os.remove(self.working_directory.joinpath(f))
                d.use_g2p = True
                fst = pynini.Fst.read(d.lexicon_fst_path)
                align_fst = pynini.Fst.read(d.align_lexicon_path)
                grapheme_table = pywrapfst.SymbolTable.read_text(d.grapheme_symbol_table_path)
                phone_table = pywrapfst.SymbolTable.read_text(self.phone_symbol_table_path)
                self.worker.lexicon_compilers[d.id] = G2PCompiler(
                    fst,
                    grapheme_table,
                    phone_table,
                    align_fst=align_fst,
                    silence_phone=self.optional_silence_phone,
                )
                if config.DEBUG and False:
                    fst = pynini.Fst.read(d.lexicon_fst_path)
                    grapheme_table = pywrapfst.SymbolTable.read_text(
                        self.grapheme_symbol_table_path
                    )
                    phone_table = pywrapfst.SymbolTable.read_text(self.phone_symbol_table_path)
                    query = session.query(Utterance.kaldi_id, Utterance.normalized_character_text)
                    for utt_id, text in query:
                        in_fst = pynini.accep(text, token_type=grapheme_table)
                        logger.debug(f"{utt_id}: {text}")
                        lg_fst = pynini.compose(in_fst, fst, compose_filter="alt_sequence")
                        lg_fst = lg_fst.project("output").rmepsilon()
                        weight_type = lg_fst.weight_type()
                        weight_threshold = pywrapfst.Weight(weight_type, 2.0)
                        state_threshold = 256 + 2 * lg_fst.num_states()
                        lg_fst = pynini.determinize(
                            lg_fst, nstate=state_threshold, weight=weight_threshold
                        )

                        lg_fst = pynini_to_kaldi(lg_fst)
                        fst_determinize_star(lg_fst, use_log=True)
                        fst_minimize_encoded(lg_fst)
                        fst_push_special(lg_fst)
                        lg_fst = kaldi_to_pynini(lg_fst)
                        path_string = (
                            pynini.shortestpath(lg_fst).project("output").string(phone_table)
                        )
                        logger.debug(f"Output: {path_string}")
            session.commit()
            self.worker.use_g2p = True

    def export_model(self, output_model_path: Path) -> None:
        """
        Export an acoustic model to the specified path

        Parameters
        ----------
        output_model_path : str
            Path to save acoustic model
        """
        AcousticModelTrainingMixin.export_model(self, output_model_path)

    def setup(self):
        wf = self.worker.current_workflow
        previous_directory = self.previous_aligner.working_directory
        for j in self.jobs:
            for p in j.construct_path_dictionary(previous_directory, "ali", "ark").values():
                shutil.copy(p, wf.working_directory.joinpath(p.name))
            for p in j.construct_path_dictionary(previous_directory, "words", "ark").values():
                shutil.copy(p, wf.working_directory.joinpath(p.name))
        for f in ["final.mdl", "final.alimdl", "lda.mat", "tree"]:
            p = previous_directory.joinpath(f)
            if os.path.exists(p):
                shutil.copy(p, wf.working_directory.joinpath(p.name))

    def train_pronunciation_probabilities(self) -> None:
        """
        Train pronunciation probabilities based on previous alignment
        """
        wf = self.worker.current_workflow
        os.makedirs(os.path.join(wf.working_directory, "log"), exist_ok=True)
        if wf.done:
            logger.info(
                "Pronunciation probability estimation already done, loading saved probabilities..."
            )
            self.training_complete = True
            if self.train_g2p:
                self.pronunciations_complete = True
                with self.worker.session() as session:
                    dictionaries = session.query(Dictionary).all()
                    for d in dictionaries:
                        fst_path = os.path.join(
                            self.working_directory,
                            f"{self.worker.dictionary_base_names[d.id]}.fst",
                        )
                        os.rename(d.lexicon_fst_path, d.lexicon_fst_path.with_suffix(".backup"))
                        shutil.copy(fst_path, d.lexicon_fst_path)
                        d.use_g2p = True
                    session.commit()
                    self.worker.use_g2p = True
                return

            silence_prob_sum = 0
            initial_silence_prob_sum = 0
            final_silence_correction_sum = 0
            final_non_silence_correction_sum = 0

            with self.worker.session() as session:

                dictionaries = session.query(Dictionary).all()
                for d in dictionaries:
                    pronunciations = (
                        session.query(Pronunciation)
                        .join(Pronunciation.word)
                        .options(joinedload(Pronunciation.word, innerjoin=True))
                        .filter(Word.dictionary_id == d.id)
                    )
                    cache = {(x.word.word, x.pronunciation): x for x in pronunciations}
                    new_dictionary_path = self.working_directory.joinpath(f"{d.id}.dict")
                    for (
                        word,
                        pron,
                        prob,
                        silence_after_prob,
                        silence_before_correct,
                        non_silence_before_correct,
                    ) in parse_dictionary_file(new_dictionary_path):
                        if (word, " ".join(pron)) not in cache:
                            continue
                        p = cache[(word, " ".join(pron))]
                        p.probability = prob
                        p.silence_after_probability = silence_after_prob
                        p.silence_before_correction = silence_before_correct
                        p.non_silence_before_correction = non_silence_before_correct

                    silence_info_path = os.path.join(
                        self.working_directory, f"{d.id}_silence_info.json"
                    )
                    with mfa_open(silence_info_path, "r") as f:
                        data = json.load(f)
                    if self.silence_probabilities:
                        d.silence_probability = data["silence_probability"]
                        d.initial_silence_probability = data["initial_silence_probability"]
                        d.final_silence_correction = data["final_silence_correction"]
                        d.final_non_silence_correction = data["final_non_silence_correction"]
                        silence_prob_sum += d.silence_probability
                        initial_silence_prob_sum += d.initial_silence_probability
                        final_silence_correction_sum += d.final_silence_correction
                        final_non_silence_correction_sum += d.final_non_silence_correction

                if self.silence_probabilities:
                    self.worker.silence_probability = silence_prob_sum / len(dictionaries)
                    self.worker.initial_silence_probability = initial_silence_prob_sum / len(
                        dictionaries
                    )
                    self.worker.final_silence_correction = final_silence_correction_sum / len(
                        dictionaries
                    )
                    self.worker.final_non_silence_correction = (
                        final_non_silence_correction_sum / len(dictionaries)
                    )
                session.commit()
            self.worker.write_lexicon_information()
            return
        self.setup()
        if self.train_g2p:
            self.train_g2p_lexicon()
        else:
            os.makedirs(self.working_log_directory, exist_ok=True)
            self.worker.compute_pronunciation_probabilities()
            self.worker.write_lexicon_information()
            with self.worker.session() as session:
                for d in session.query(Dictionary):
                    dict_path = self.working_directory.joinpath(f"{d.id}.dict")
                    self.worker.export_trained_rules(self.working_directory)
                    self.worker.export_lexicon(
                        d.id,
                        dict_path,
                        probability=True,
                    )
                    silence_info_path = os.path.join(
                        self.working_directory, f"{d.id}_silence_info.json"
                    )
                    with mfa_open(silence_info_path, "w") as f:
                        json.dump(d.silence_probability_info, f)
        with self.session() as session:
            session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update({"done": True})
            session.commit()

    def train_iteration(self) -> None:
        """Training iteration"""
        pass
