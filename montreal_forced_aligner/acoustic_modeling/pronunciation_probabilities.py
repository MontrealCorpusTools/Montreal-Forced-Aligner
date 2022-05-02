"""Class definitions for PronunciationProbabilityTrainer"""
import json
import os
import shutil
import typing

from sqlalchemy.orm import joinedload

from montreal_forced_aligner.acoustic_modeling.base import AcousticModelTrainingMixin
from montreal_forced_aligner.db import Dictionary, Pronunciation, Word

__all__ = ["PronunciationProbabilityTrainer"]


class PronunciationProbabilityTrainer(AcousticModelTrainingMixin):
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
        **kwargs,
    ):
        self.previous_trainer = previous_trainer
        self.silence_probabilities = silence_probabilities
        super(PronunciationProbabilityTrainer, self).__init__(**kwargs)
        self.num_iterations = 1
        self.subset = self.previous_trainer.subset

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
    def exported_model_path(self) -> str:
        """Path to exported acoustic model"""
        return self.previous_trainer.exported_model_path

    @property
    def model_path(self) -> str:
        """Current acoustic model path"""
        return os.path.join(self.working_directory, "final.mdl")

    @property
    def alignment_model_path(self) -> str:
        """Alignment model path"""
        path = self.model_path.replace(".mdl", ".alimdl")
        if os.path.exists(path):
            return path
        return self.model_path

    @property
    def working_directory(self) -> str:
        """Training directory"""
        if self.training_complete:
            return super(PronunciationProbabilityTrainer, self).working_directory
        return os.path.join(self.worker.output_directory, self.previous_aligner.identifier)

    def train_pronunciation_probabilities(self) -> None:
        """
        Train pronunciation probabilities based on previous alignment
        """
        working_dir = super(PronunciationProbabilityTrainer, self).working_directory
        done_path = os.path.join(working_dir, "done")
        dirty_path = os.path.join(working_dir, "dirty")
        if os.path.exists(dirty_path):  # if there was an error, let's redo from scratch
            shutil.rmtree(working_dir)
        os.makedirs(working_dir, exist_ok=True)
        if os.path.exists(done_path):
            self.log_info(
                "Pronunciation probability estimation already done, loading saved probabilities..."
            )
            self.training_complete = True
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
                    new_dictionary_path = os.path.join(working_dir, f"{d.id}.dict")
                    with open(new_dictionary_path, "r", encoding="utf8") as f:
                        for line in f:
                            line = line.strip()
                            line = line.split()
                            word = line.pop(0)
                            prob = float(line.pop(0))
                            silence_after_prob = None
                            silence_before_correct = None
                            non_silence_before_correct = None
                            if self.silence_probabilities:
                                silence_after_prob = float(line.pop(0))
                                silence_before_correct = float(line.pop(0))
                                non_silence_before_correct = float(line.pop(0))
                            pron = " ".join(line)
                            p = cache[(word, pron)]
                            p.probability = prob
                            p.silence_after_probability = silence_after_prob
                            p.silence_before_correction = silence_before_correct
                            p.non_silence_before_correction = non_silence_before_correct

                    silence_info_path = os.path.join(working_dir, f"{d.id}_silence_info.json")
                    with open(silence_info_path, "r", encoding="utf8") as f:
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
        self.worker.compute_pronunciation_probabilities(self.silence_probabilities)
        self.worker.write_lexicon_information()
        self.training_complete = True
        with self.worker.session() as session:
            for d in session.query(Dictionary):
                dict_path = os.path.join(working_dir, f"{d.id}.dict")
                self.worker.export_lexicon(
                    d.id,
                    dict_path,
                    probability=True,
                    silence_probabilities=self.silence_probabilities,
                )
                silence_info_path = os.path.join(working_dir, f"{d.id}_silence_info.json")
                with open(silence_info_path, "w", encoding="utf8") as f:
                    json.dump(d.silence_probability_info, f)
        with open(done_path, "w"):
            pass

    def train_iteration(self) -> None:
        """Training iteration"""
        pass
