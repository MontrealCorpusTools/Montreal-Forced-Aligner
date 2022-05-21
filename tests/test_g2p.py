import os

import pytest

from montreal_forced_aligner.dictionary import MultispeakerDictionary
from montreal_forced_aligner.g2p.generator import (
    PyniniCorpusGenerator,
    PyniniWordListGenerator,
    clean_up_word,
    scored_match,
)
from montreal_forced_aligner.g2p.trainer import G2P_DISABLED, PyniniTrainer, pynini
from montreal_forced_aligner.models import G2PModel
from montreal_forced_aligner.utils import get_mfa_version


def test_clean_up_word():
    original_word = "+abc"
    w, m = clean_up_word(original_word, {"a", "b", "c"})
    assert w == "abc"
    assert m == {"+"}


def test_check_bracketed(basic_dict_path):
    """Checks if the brackets are removed correctly and handling an empty string works"""
    word_set = ["uh", "(the)", "sick", "<corpus>", "[a]", "{cold}", ""]
    expected_result = ["uh", "sick", ""]
    dictionary_config = MultispeakerDictionary(dictionary_path=basic_dict_path)
    assert [x for x in word_set if not dictionary_config.check_bracketed(x)] == expected_result


def test_scored_match(english_g2p_model, temp_dir):
    if G2P_DISABLED:
        pytest.skip("No Pynini found")
    g2p_model = G2PModel(english_g2p_model)
    pron_scores = {
        ("read", "R IY1 D"): 12.3903465,
        ("read", "R EH1 D"): 12.2733212,
        ("theatres", "TH IY1 AH0 T ER0 Z"): 12.3603706,
        ("theatres", "DH IY1"): 39.7762604,
        ("the", "DH"): 11.7815981,
        ("the", "TH"): 12.3016987,
        ("the", "DH AH0"): 16.9708042,
        ("the", "DH IY1"): 16.2190933,
        ("the", "DH IY0"): 15.3279858,
        ("them", "DH EH0 M"): 14.7657909,
        ("them", "DH AH0 M"): 11.9653378,
        ("them", "DH EY1"): 28.7747478,
    }
    fst = pynini.Fst.read(g2p_model.fst_path)
    for (word, pron), absolute_reference in pron_scores.items():
        print(word, pron, absolute_reference)
        absolute_score = scored_match(
            word,
            pron,
            fst,
            input_token_type="utf8",
            output_token_type=pynini.SymbolTable.read_text(g2p_model.sym_path),
        )
        assert absolute_score == absolute_reference


def test_training(basic_dict_path, basic_g2p_model_path, temp_dir):
    if G2P_DISABLED:
        pytest.skip("No Pynini found")
    trainer = PyniniTrainer(
        dictionary_path=basic_dict_path,
        temporary_directory=temp_dir,
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


def test_generator(basic_g2p_model_path, basic_corpus_dir, g2p_basic_output, temp_dir):
    if G2P_DISABLED:
        pytest.skip("No Pynini found")
    output_directory = os.path.join(temp_dir, "g2p_tests")
    gen = PyniniCorpusGenerator(
        g2p_model_path=basic_g2p_model_path,
        corpus_directory=basic_corpus_dir,
        temporary_directory=output_directory,
    )

    gen.setup()
    assert not gen.g2p_model.validate(gen.corpus_word_set)
    assert gen.g2p_model.validate([x for x in gen.corpus_word_set if not gen.check_bracketed(x)])

    gen.export_pronunciations(g2p_basic_output)
    assert os.path.exists(g2p_basic_output)
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
        g2p_model_path=english_g2p_model, word_list_path=word_list_path, num_pronunciations=3
    )
    gen.setup()
    results = gen.generate_pronunciations()
    print(results)
    assert len(results["petted"]) == 3
    gen.cleanup()
