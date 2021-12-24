import os
import shutil

from montreal_forced_aligner.alignment import PretrainedAligner


def test_align_sick(
    english_dictionary, english_acoustic_model, basic_corpus_dir, temp_dir, test_align_config
):
    temp = os.path.join(temp_dir, "align_export_temp")
    a = PretrainedAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=english_dictionary,
        acoustic_model_path=english_acoustic_model,
        temporary_directory=temp,
        debug=True,
        verbose=True,
        **test_align_config
    )
    a.align()
    export_directory = os.path.join(temp_dir, "test_align_export")
    shutil.rmtree(export_directory, ignore_errors=True)
    os.makedirs(export_directory, exist_ok=True)
    assert "AY_S" not in a.phone_mapping
    assert "AY_S" not in a.default_dictionary.phone_mapping
    assert "AY_S" not in a.default_dictionary.reversed_phone_mapping.values()
    a.export_files(export_directory)
    assert os.path.exists(os.path.join(export_directory, "michael", "acoustic_corpus.TextGrid"))
