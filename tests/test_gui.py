import os

from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpus


def test_save_text_lab(basic_corpus_dir, generated_dir, global_config, db_setup):
    output_directory = os.path.join(generated_dir, "gui_tests")
    global_config.temporary_directory = output_directory
    corpus = AcousticCorpus(
        corpus_directory=basic_corpus_dir,
    )
    corpus._load_corpus()
    corpus.get_file(name="acoustic_corpus").save(corpus.corpus_directory)


def test_file_properties(
    stereo_corpus_dir,
    generated_dir,
    global_config,
    db_setup,
):
    output_directory = os.path.join(generated_dir, "gui_tests")
    global_config.temporary_directory = output_directory
    corpus = AcousticCorpus(
        corpus_directory=stereo_corpus_dir,
    )
    corpus._load_corpus()
    file = corpus.get_file(name="michaelandsickmichael")
    assert file.sound_file.num_channels == 2
    assert file.num_speakers == 2
    assert file.num_utterances == 7
    x, y = file.sound_file.normalized_waveform()
    assert y.shape[0] == 2


def test_flac_tg(flac_tg_corpus_dir, generated_dir, global_config, db_setup):
    output_directory = os.path.join(generated_dir, "gui_tests")
    global_config.temporary_directory = output_directory
    corpus = AcousticCorpus(
        corpus_directory=flac_tg_corpus_dir,
    )
    corpus._load_corpus()
    corpus.get_file(name="61-70968-0000").save(corpus.corpus_directory)
