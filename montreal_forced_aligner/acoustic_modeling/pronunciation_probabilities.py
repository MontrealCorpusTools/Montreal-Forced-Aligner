"""Class definitions for PronunciationProbabilityTrainer"""
import json
import logging
import os
import shutil
import typing
from pathlib import Path

from sqlalchemy.orm import joinedload

from montreal_forced_aligner.acoustic_modeling.base import AcousticModelTrainingMixin
from montreal_forced_aligner.db import CorpusWorkflow, Dictionary, Pronunciation, Word
from montreal_forced_aligner.g2p.trainer import PyniniTrainerMixin
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import parse_dictionary_file

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
                if not p.exists():
                    continue
                shutil.copy(p, wf.working_directory.joinpath(p.name))
            for p in j.construct_path_dictionary(previous_directory, "words", "ark").values():
                if not p.exists():
                    continue
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

            silence_prob_sum = 0

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
                        for k, v in data.items():
                            if v is None:
                                if "correction" in k:
                                    data[k] = 1.0
                                else:
                                    data[k] = 0.5
                    if self.silence_probabilities:
                        d.silence_probability = data["silence_probability"]
                        # d.initial_silence_probability = data["initial_silence_probability"]
                        # d.final_silence_correction = data["final_silence_correction"]
                        # d.final_non_silence_correction = data["final_non_silence_correction"]
                        silence_prob_sum += d.silence_probability
                        # initial_silence_prob_sum += d.initial_silence_probability
                        # final_silence_correction_sum += d.final_silence_correction
                        # final_non_silence_correction_sum += d.final_non_silence_correction

                if self.silence_probabilities:
                    self.worker.silence_probability = silence_prob_sum / len(dictionaries)
                    # self.worker.initial_silence_probability = initial_silence_prob_sum / len(
                    #    dictionaries
                    # )
                    # self.worker.final_silence_correction = final_silence_correction_sum / len(
                    #    dictionaries
                    # )
                    # self.worker.final_non_silence_correction = (
                    #    final_non_silence_correction_sum / len(dictionaries)
                    # )
                session.commit()
            self.worker.write_lexicon_information()
            return
        self.setup()
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
