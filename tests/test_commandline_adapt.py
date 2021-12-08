import os

from montreal_forced_aligner.command_line.adapt import run_adapt_model
from montreal_forced_aligner.command_line.mfa import parser


def test_adapt_basic(
    basic_corpus_dir,
    sick_dict_path,
    generated_dir,
    english_dictionary,
    temp_dir,
    test_align_config,
    english_acoustic_model,
):
    adapted_model_path = os.path.join(generated_dir, "basic_adapted.zip")
    command = [
        "adapt",
        basic_corpus_dir,
        english_dictionary,
        english_acoustic_model,
        adapted_model_path,
        "-t",
        temp_dir,
        "--clean",
        "--debug",
    ]
    args, unknown = parser.parse_known_args(command)
    run_adapt_model(args, unknown)
    assert os.path.exists(adapted_model_path)


# @pytest.mark.skip(reason='Optimization')
def test_adapt_multilingual(
    multilingual_ipa_corpus_dir,
    ipa_speaker_dict_path,
    generated_dir,
    temp_dir,
    basic_align_config_path,
    english_acoustic_model,
    english_ipa_acoustic_model,
):
    adapted_model_path = os.path.join(generated_dir, "multilingual_adapted.zip")
    output_path = os.path.join(generated_dir, "multilingual_output")
    command = [
        "adapt",
        multilingual_ipa_corpus_dir,
        ipa_speaker_dict_path,
        english_ipa_acoustic_model,
        adapted_model_path,
        output_path,
        "-t",
        temp_dir,
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
    ]
    args, unknown = parser.parse_known_args(command)
    run_adapt_model(args, unknown)
    assert os.path.exists(adapted_model_path)
