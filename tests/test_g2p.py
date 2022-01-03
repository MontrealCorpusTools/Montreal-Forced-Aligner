import os

import pytest

from montreal_forced_aligner.dictionary.pronunciation import PronunciationDictionary
from montreal_forced_aligner.g2p.generator import (
    PyniniCorpusGenerator,
    PyniniWordListGenerator,
    clean_up_word,
)
from montreal_forced_aligner.g2p.trainer import G2P_DISABLED, PyniniTrainer
from montreal_forced_aligner.models import G2PModel
from montreal_forced_aligner.utils import get_mfa_version


def test_clean_up_word():
    original_word = "+abc"
    w, m = clean_up_word(original_word, {"a", "b", "c"})
    assert w == "abc"
    assert m == {"+"}


def test_check_bracketed(sick_dict):
    """Checks if the brackets are removed correctly and handling an empty string works"""
    word_set = ["uh", "(the)", "sick", "<corpus>", "[a]", "{cold}", ""]
    expected_result = ["uh", "sick", ""]
    dictionary_config = PronunciationDictionary(dictionary_path=sick_dict)
    assert [x for x in word_set if not dictionary_config.check_bracketed(x)] == expected_result


def test_training(sick_dict_path, sick_g2p_model_path, temp_dir):
    if G2P_DISABLED:
        pytest.skip("No Pynini found")
    trainer = PyniniTrainer(
        dictionary_path=sick_dict_path,
        temporary_directory=temp_dir,
        random_starts=1,
        num_iterations=5,
        evaluate=True,
    )
    trainer.setup()

    trainer.train()
    trainer.export_model(sick_g2p_model_path)
    model = G2PModel(sick_g2p_model_path, root_directory=temp_dir)
    assert model.meta["version"] == get_mfa_version()
    assert model.meta["architecture"] == "pynini"
    assert model.meta["phones"] == trainer.non_silence_phones
    assert model.meta["graphemes"] == trainer.graphemes
    trainer.cleanup()


def test_generator(sick_g2p_model_path, sick_corpus, g2p_sick_output, temp_dir):
    if G2P_DISABLED:
        pytest.skip("No Pynini found")
    output_directory = os.path.join(temp_dir, "g2p_tests")
    gen = PyniniCorpusGenerator(
        g2p_model_path=sick_g2p_model_path,
        corpus_directory=sick_corpus,
        temporary_directory=output_directory,
    )

    gen.setup()
    assert not gen.g2p_model.validate(gen.corpus_word_set)
    assert gen.g2p_model.validate([x for x in gen.corpus_word_set if not gen.check_bracketed(x)])

    gen.export_pronunciations(g2p_sick_output)
    assert os.path.exists(g2p_sick_output)
    gen.cleanup()


def test_generator_pretrained(english_g2p_model, temp_dir):
    if G2P_DISABLED:
        pytest.skip("No Pynini found")
    words = ["petted", "petted-patted", "pedal"]
    output_directory = os.path.join(temp_dir, "g2p_tests")
    word_list_path = os.path.join(output_directory, "word_list.txt")
    os.makedirs(output_directory, exist_ok=True)
    with open(word_list_path, "w", encoding="utf8") as f:
        for w in words:
            f.write(w + "\n")
    gen = PyniniWordListGenerator(
        g2p_model_path="english_g2p", word_list_path=word_list_path, num_pronunciations=3
    )
    gen.setup()
    results = gen.generate_pronunciations()
    print(results)
    assert len(results["petted"]) == 3
    gen.cleanup()
