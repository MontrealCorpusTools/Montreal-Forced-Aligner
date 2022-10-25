from montreal_forced_aligner.abc import MfaWorker, TrainerMixin
from montreal_forced_aligner.acoustic_modeling import SatTrainer, TrainableAligner
from montreal_forced_aligner.alignment import AlignMixin


def test_typing(basic_corpus_dir, basic_dict_path, temp_dir):
    am_trainer = TrainableAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
    )
    trainer = SatTrainer(identifier="sat", worker=am_trainer)
    assert type(trainer).__name__ == "SatTrainer"
    assert isinstance(trainer, TrainerMixin)
    assert isinstance(trainer, AlignMixin)
    assert isinstance(trainer, MfaWorker)
    assert isinstance(am_trainer, MfaWorker)
