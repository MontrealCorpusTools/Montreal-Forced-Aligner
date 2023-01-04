"""Classes for validating dictionaries"""
import logging
import os
import shutil
import typing

from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.data import WorkflowType
from montreal_forced_aligner.g2p.generator import PyniniValidator
from montreal_forced_aligner.g2p.trainer import PyniniTrainer

logger = logging.getLogger("mfa")


class DictionaryValidator(PyniniTrainer):
    """
    Mixin class for performing validation on a corpus

    Parameters
    ----------
    g2p_model_path: str, optional
        Path to pretrained G2P model
    g2p_threshold: float, optional
        Threshold for pruning pronunciations, defaults to 1.5, which returns the optimal pronunciations and those with scores less than 1.5 times
        the optimal pronunciation's score. Increase
        to allow for more suboptimal pronunciations

    See Also
    --------
    :class:`~montreal_forced_aligner.alignment.base.CorpusAligner`
        For corpus, dictionary, and alignment parameters

    Attributes
    ----------
    printer: TerminalPrinter
        Printer for output messages
    """

    def __init__(
        self,
        g2p_model_path: typing.Optional[str] = None,
        g2p_threshold: float = 1.5,
        **kwargs,
    ):
        kwargs["clean"] = True
        super().__init__(**kwargs)
        self.g2p_model_path = g2p_model_path
        self.g2p_threshold = g2p_threshold

    def setup(self) -> None:
        """Set up the dictionary validator"""
        if self.initialized:
            return
        self.initialize_database()
        self.dictionary_setup()
        self.write_lexicon_information()
        if self.g2p_model_path is None:
            self.create_new_current_workflow(WorkflowType.train_g2p)
            logger.info("Not using a pretrained G2P model, training from the dictionary...")
            self.initialize_training()
            self.train()
            self.g2p_model_path = os.path.join(self.working_log_directory, "g2p_model.zip")
            self.export_model(self.g2p_model_path)
            self.create_new_current_workflow(WorkflowType.g2p)
        else:
            self.create_new_current_workflow(WorkflowType.g2p)
            self.initialize_training()
        self.initialized = True

    def validate(self, output_path: typing.Optional[str] = None) -> None:
        """
        Validate the dictionary

        Parameters
        ----------
        output_path: str, optional
            Path to save scored CSV
        """
        self.setup()

        gen = PyniniValidator(
            g2p_model_path=self.g2p_model_path,
            word_list=list(self.g2p_training_dictionary.keys()),
            temporary_directory=os.path.join(self.working_directory, "validation"),
            num_jobs=GLOBAL_CONFIG.num_jobs,
            num_pronunciations=self.num_pronunciations,
        )
        gen.evaluate_g2p_model(self.g2p_training_dictionary)
        if output_path is not None:
            shutil.copyfile(gen.evaluation_csv_path, output_path)
            logger.info(f"Wrote scores to {output_path}")
