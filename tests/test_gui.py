import os

from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpus


def test_save_text_lab(
    basic_dict_path,
    basic_corpus_dir,
    generated_dir,
):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    corpus = AcousticCorpus(
        corpus_directory=basic_corpus_dir,
        dictionary_path=basic_dict_path,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus._load_corpus()
    corpus.files["acoustic_corpus"].save()


def test_flac_tg(basic_dict_path, flac_tg_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    corpus = AcousticCorpus(
        corpus_directory=flac_tg_corpus_dir,
        dictionary_path=basic_dict_path,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus._load_corpus()
    corpus.files["61-70968-0000"].save()
