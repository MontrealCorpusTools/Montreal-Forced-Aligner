import os
import shutil

from montreal_forced_aligner.dictionary import MultispeakerDictionary
from montreal_forced_aligner.g2p.generator import (
    PyniniCorpusGenerator,
    PyniniWordListGenerator,
    clean_up_word,
)
from montreal_forced_aligner.g2p.trainer import PyniniTrainer
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.models import G2PModel
from montreal_forced_aligner.utils import get_mfa_version


def test_clean_up_word():
    original_word = "+abc"
    w, m = clean_up_word(original_word, {"a", "b", "c"})
    assert w == "abc"
    assert m == {"+"}


def test_check_bracketed(basic_dict_path, db_setup):
    """Checks if the brackets are removed correctly and handling an empty string works"""
    word_set = ["uh", "(the)", "sick", "<corpus>", "[a]", "{cold}", ""]
    expected_result = ["uh", "sick", ""]
    dictionary_config = MultispeakerDictionary(dictionary_path=basic_dict_path)
    assert [x for x in word_set if not dictionary_config.check_bracketed(x)] == expected_result


def test_training(basic_dict_path, basic_g2p_model_path, temp_dir, global_config, db_setup):
    output_directory = os.path.join(temp_dir, "g2p_tests", "train")
    global_config.temporary_directory = output_directory
    trainer = PyniniTrainer(
        dictionary_path=basic_dict_path,
        random_starts=1,
        num_iterations=5,
        evaluate=True,
    )
    trainer.setup()

    trainer.train()
    trainer.export_model(basic_g2p_model_path)
    model = G2PModel(basic_g2p_model_path, root_directory=temp_dir)
    assert model.meta["version"] == get_mfa_version()
    assert model.meta["architecture"] == "pynini"
    assert model.meta["phones"] == trainer.non_silence_phones
    assert model.meta["graphemes"] == trainer.g2p_training_graphemes
    trainer.cleanup()


def test_generator(
    basic_g2p_model_path, basic_corpus_dir, g2p_basic_output, temp_dir, global_config, db_setup
):
    output_directory = os.path.join(temp_dir, "g2p_tests", "gen")
    global_config.temporary_directory = output_directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    global_config.clean = True
    gen = PyniniCorpusGenerator(
        g2p_model_path=basic_g2p_model_path,
        corpus_directory=basic_corpus_dir,
    )

    gen.setup()
    print(gen.corpus_word_set)
    assert not gen.g2p_model.validate(gen.corpus_word_set)
    assert gen.g2p_model.validate([x for x in gen.corpus_word_set if not gen.check_bracketed(x)])

    gen.export_pronunciations(g2p_basic_output)
    assert os.path.exists(g2p_basic_output)
    gen.cleanup()


def test_generator_pretrained(english_g2p_model, temp_dir, global_config, db_setup):
    words = ["petted", "petted-patted", "pedal"]
    output_directory = os.path.join(temp_dir, "g2p_tests")
    global_config.temporary_directory = output_directory
    word_list_path = os.path.join(output_directory, "word_list.txt")
    os.makedirs(output_directory, exist_ok=True)
    with mfa_open(word_list_path, "w") as f:
        for w in words:
            f.write(w + "\n")
    gen = PyniniWordListGenerator(
        g2p_model_path=english_g2p_model, word_list_path=word_list_path, num_pronunciations=3
    )
    gen.setup()
    results = gen.generate_pronunciations()
    assert len(results["petted"]) == 3
    gen.cleanup()
