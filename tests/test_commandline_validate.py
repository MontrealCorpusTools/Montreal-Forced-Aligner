from montreal_forced_aligner.command_line.mfa import parser
from montreal_forced_aligner.command_line.validate import run_validate_corpus


def test_validate_corpus(
    multilingual_ipa_tg_corpus_dir, english_ipa_acoustic_model, english_us_ipa_dictionary, temp_dir
):

    command = [
        "validate",
        multilingual_ipa_tg_corpus_dir,
        english_us_ipa_dictionary,
        english_ipa_acoustic_model,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--debug",
        "--disable_mp",
        "--test_transcriptions",
    ]
    args, unknown = parser.parse_known_args(command)
    run_validate_corpus(args)


def test_validate_training_corpus(
    multilingual_ipa_tg_corpus_dir,
    english_ipa_acoustic_model,
    english_dictionary,
    temp_dir,
    mono_train_config_path,
):

    command = [
        "validate",
        multilingual_ipa_tg_corpus_dir,
        english_dictionary,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--debug",
        "--config_path",
        mono_train_config_path,
    ]
    args, unknown = parser.parse_known_args(command)
    run_validate_corpus(args)


def test_validate_missing_phones(
    multilingual_ipa_tg_corpus_dir,
    german_prosodylab_acoustic_model,
    german_prosodylab_dictionary,
    temp_dir,
):

    command = [
        "validate",
        multilingual_ipa_tg_corpus_dir,
        german_prosodylab_dictionary,
        german_prosodylab_acoustic_model,
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--debug",
        "--ignore_acoustics",
    ]
    args, unknown = parser.parse_known_args(command)
    run_validate_corpus(args)
