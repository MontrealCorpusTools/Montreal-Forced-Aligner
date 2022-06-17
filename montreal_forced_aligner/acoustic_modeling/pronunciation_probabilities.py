"""Class definitions for PronunciationProbabilityTrainer"""
import json
import multiprocessing as mp
import os
import shutil
import time
import typing
from queue import Empty

import tqdm
from sqlalchemy.orm import joinedload

from montreal_forced_aligner.acoustic_modeling.base import AcousticModelTrainingMixin
from montreal_forced_aligner.alignment.multiprocessing import GeneratePronunciationsFunction
from montreal_forced_aligner.db import Dictionary, Pronunciation, Utterance, Word
from montreal_forced_aligner.g2p.trainer import PyniniTrainerMixin
from montreal_forced_aligner.utils import KaldiProcessWorker, Stopped

__all__ = ["PronunciationProbabilityTrainer"]


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
        num_iterations: int = 10,
        model_size: int = 100000,
        **kwargs,
    ):
        self.previous_trainer = previous_trainer
        self.silence_probabilities = silence_probabilities
        self.train_g2p = train_g2p
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
        if self.pronunciations_complete:
            return super(PronunciationProbabilityTrainer, self).working_directory
        return self.previous_aligner.working_directory

    @property
    def num_jobs(self) -> int:
        """Number of jobs from the root worker"""
        return self.worker.num_jobs

    @property
    def phone_symbol_table_path(self) -> str:
        """Worker's phone symbol table"""
        return self.worker.phone_symbol_table_path

    @property
    def grapheme_symbol_table_path(self) -> str:
        """Worker's grapheme symbol table"""
        return self.worker.grapheme_symbol_table_path

    @property
    def input_path(self) -> str:
        """Path to temporary file to store training data"""
        return os.path.join(self.working_directory, f"input_{self._data_source}.txt")

    @property
    def output_path(self) -> str:
        """Path to temporary file to store training data"""
        return os.path.join(self.working_directory, f"output_{self._data_source}.txt")

    def train_g2p_lexicon(self) -> None:
        """Generate a G2P lexicon based on aligned transcripts"""
        arguments = self.worker.generate_pronunciations_arguments()
        working_dir = super(PronunciationProbabilityTrainer, self).working_directory
        texts = {}
        with self.worker.session() as session:
            query = session.query(Utterance.id, Utterance.normalized_character_text)
            query = query.filter(Utterance.ignored == False)  # noqa
            initial_brackets = "".join(x[0] for x in self.worker.brackets)
            query = query.filter(~Utterance.oovs.regexp_match(f"(^| )[^{initial_brackets}]"))
            if self.subset:
                query = query.filter_by(in_subset=True)
            for utt_id, text in query:
                texts[utt_id] = text
            input_files = {
                x: open(
                    os.path.join(working_dir, f"input_{self.worker.dictionary_base_names[x]}.txt"),
                    "w",
                    encoding="utf8",
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
                )
                for x in self.worker.dictionary_lookup.values()
            }
            with tqdm.tqdm(
                total=self.num_current_utterances, disable=getattr(self, "quiet", False)
            ) as pbar:
                if self.use_mp:
                    error_dict = {}
                    return_queue = mp.Queue()
                    stopped = Stopped()
                    procs = []
                    for i, args in enumerate(arguments):
                        args.for_g2p = True
                        function = GeneratePronunciationsFunction(args)
                        p = KaldiProcessWorker(i, return_queue, function, stopped)
                        procs.append(p)
                        p.start()
                    while True:
                        try:
                            result = return_queue.get(timeout=1)
                            if isinstance(result, Exception):
                                error_dict[getattr(result, "job_name", 0)] = result
                                continue
                            if stopped.stop_check():
                                continue
                        except Empty:
                            for proc in procs:
                                if not proc.finished.stop_check():
                                    break
                            else:
                                break
                            continue
                        dict_id, utt_id, phones = result
                        utt_id = int(utt_id.split("-")[-1])
                        pbar.update(1)
                        if utt_id not in texts or not texts[utt_id]:
                            continue
                        print(phones, file=output_files[dict_id])
                        print(f"<s> {texts[utt_id]} </s>", file=input_files[dict_id])

                    for p in procs:
                        p.join()
                    if error_dict:
                        for v in error_dict.values():
                            raise v
                else:
                    self.log_debug("Not using multiprocessing...")
                    for args in arguments:
                        function = GeneratePronunciationsFunction(args)
                        for dict_id, utt_id, phones in function.run():
                            print(phones, file=output_files[dict_id])
                            print(f"<s> {texts[utt_id]} </s>", file=input_files[dict_id])
                            pbar.update(1)
            for f in input_files.values():
                f.close()
            for f in output_files.values():
                f.close()
            self.pronunciations_complete = True
            os.makedirs(self.working_log_directory, exist_ok=True)
            dictionaries = session.query(Dictionary)
            shutil.copyfile(
                self.phone_symbol_table_path, os.path.join(self.working_directory, "phones.txt")
            )
            shutil.copyfile(
                self.grapheme_symbol_table_path,
                os.path.join(self.working_directory, "graphemes.txt"),
            )
            self.input_token_type = self.grapheme_symbol_table_path
            self.output_token_type = self.phone_symbol_table_path
            for d in dictionaries:
                self.log_info(f"Training G2P for {d.name}...")
                self._data_source = self.worker.dictionary_base_names[d.id]
                begin = time.time()
                if os.path.exists(self.far_path) and os.path.exists(self.encoder_path):
                    self.log_info("Alignment already done, skipping!")
                else:
                    self.align_g2p()
                    self.log_debug(
                        f"Aligning utterances for {d.name} took {time.time() - begin} seconds"
                    )
                begin = time.time()
                self.generate_model()
                self.log_debug(f"Generating model for {d.name} took {time.time() - begin} seconds")
                os.rename(d.lexicon_fst_path, d.lexicon_fst_path + ".backup")
                shutil.copy(self.fst_path, d.lexicon_fst_path)
                d.use_g2p = True
            session.commit()
            self.worker.use_g2p = True

    def export_model(self, output_model_path: str) -> None:
        """
        Export an acoustic model to the specified path

        Parameters
        ----------
        output_model_path : str
            Path to save acoustic model
        """
        AcousticModelTrainingMixin.export_model(self, output_model_path)

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
            if self.train_g2p:
                self.pronunciations_complete = True
                with self.worker.session() as session:
                    dictionaries = session.query(Dictionary).all()
                    for d in dictionaries:
                        fst_path = os.path.join(
                            self.working_directory,
                            f"{self.worker.dictionary_base_names[d.id]}.fst",
                        )
                        os.rename(d.lexicon_fst_path, d.lexicon_fst_path + ".backup")
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
        if self.train_g2p:
            self.train_g2p_lexicon()
        else:
            self.worker.compute_pronunciation_probabilities(self.silence_probabilities)
            self.worker.write_lexicon_information()
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
        self.training_complete = True
        with open(done_path, "w"):
            pass

    def train_iteration(self) -> None:
        """Training iteration"""
        pass
