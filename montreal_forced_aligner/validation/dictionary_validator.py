import csv
import os
import typing

from montreal_forced_aligner.exceptions import G2PError
from montreal_forced_aligner.g2p.trainer import PyniniTrainer
from montreal_forced_aligner.models import G2PModel


class DictionaryValidator(PyniniTrainer):
    """
    Mixin class for performing validation on a corpus

    Parameters
    ----------
    ignore_acoustics: bool
        Flag for whether feature generation and training/alignment should be skipped
    test_transcriptions: bool
        Flag for whether utterance transcriptions should be tested with a unigram language model
    target_num_ngrams: int
        Target number of ngrams from speaker models to use

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
        score_threshold: float = 1.0,
        g2p_threshold: float = 1.5,
        **kwargs,
    ):
        kwargs["clean"] = True
        super().__init__(**kwargs)
        self.g2p_model = None
        if g2p_model_path:
            self.g2p_model = G2PModel(g2p_model_path)
        self.score_threshold = score_threshold
        self.g2p_threshold = g2p_threshold
        # self.printer = TerminalPrinter(print_function=self.log_info)

    @property
    def workflow_identifier(self) -> str:
        """Identifier for validation"""
        return "validate_dictionary"

    def setup(self):
        if self.initialized:
            return
        self.initialize_database()
        self.dictionary_setup()
        self.write_lexicon_information()
        if self.g2p_model is None:
            self.log_info("Not using a pretrained G2P model, training from the dictionary...")
            self.initialize_training()
            self.train()
        else:
            self.log_info("Setting up using pretrained G2P model...")
            self.g2p_model.export_fst_model(self.working_directory)
            if self.g2p_model.meta["architecture"] == "phonetisaurus":
                raise G2PError(
                    "Previously trained Phonetisaurus models from 1.1 and earlier are not currently supported. "
                    "Please retrain your model using 2.0+"
                )
            self._fst_path = self.g2p_model.fst_path
            self._sym_path = self.g2p_model.sym_path
            self.initialize_training()
        self.initialized = True

    def validate(self, output_path=None):
        self.setup()
        scores = self.score_pronunciations(self.score_threshold)
        if not output_path:
            output_path = os.path.join(self.working_directory, "pronunciation_scores.csv")
        with open(output_path, "w", encoding="utf8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["Word", "Pronunciation", "Absolute_score", "Relative_score"]
            )
            writer.writeheader()
            for (w, pron), (absolute_score, relative_score) in scores.items():
                writer.writerow(
                    {
                        "Word": w,
                        "Pronunciation": pron,
                        "Absolute_score": absolute_score,
                        "Relative_score": relative_score,
                    }
                )
        self.log_info(f"Wrote scores to {output_path}")
