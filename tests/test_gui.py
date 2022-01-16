import os

from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpus


def test_save_text_lab(
    basic_corpus_dir,
    generated_dir,
):
    output_directory = os.path.join(generated_dir, "gui_tests")
    corpus = AcousticCorpus(
        corpus_directory=basic_corpus_dir,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus._load_corpus()
    corpus.files["michael_acoustic_corpus"].save()


def test_file_properties(
    stereo_corpus_dir,
    generated_dir,
):
    output_directory = os.path.join(generated_dir, "gui_tests")
    corpus = AcousticCorpus(
        corpus_directory=stereo_corpus_dir,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus._load_corpus()
    assert corpus.files["michaelandsickmichael"].num_channels == 2
    assert corpus.files["michaelandsickmichael"].num_speakers == 2
    assert corpus.files["michaelandsickmichael"].num_utterances == 7
    x, y = corpus.files["michaelandsickmichael"].normalized_waveform()
    assert y.shape[0] == 2


def test_flac_tg(flac_tg_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "gui_tests")
    corpus = AcousticCorpus(
        corpus_directory=flac_tg_corpus_dir,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus._load_corpus()
    corpus.files["61-70968-0000"].save()
