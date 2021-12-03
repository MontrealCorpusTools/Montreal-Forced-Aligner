from montreal_forced_aligner.abc import MfaWorker, TrainerMixin
from montreal_forced_aligner.acoustic_modeling import SatTrainer, TrainableAligner
from montreal_forced_aligner.alignment import AlignMixin


def test_typing(sick_corpus, sick_dict, temp_dir):
    am_trainer = TrainableAligner(
        corpus_directory=sick_corpus, dictionary_path=sick_dict, temporary_directory=temp_dir
    )
    trainer = SatTrainer(identifier="sat", worker=am_trainer)
    assert type(trainer).__name__ == "SatTrainer"
    assert isinstance(trainer, TrainerMixin)
    assert isinstance(trainer, AlignMixin)
    assert isinstance(trainer, MfaWorker)
    assert isinstance(am_trainer, MfaWorker)
