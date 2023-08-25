import click.testing

from montreal_forced_aligner.command_line.mfa import mfa_cli


def test_validate_corpus(
    multilingual_ipa_tg_corpus_dir,
    english_mfa_acoustic_model,
    english_us_mfa_dictionary,
    temp_dir,
    db_setup,
):
    command = [
        "validate",
        multilingual_ipa_tg_corpus_dir,
        english_us_mfa_dictionary,
        "--acoustic_model_path",
        english_mfa_acoustic_model,
        "-q",
        "-s",
        "4",
        "--oov_count_threshold",
        "0",
        "--clean",
        "--no_use_mp",
        "--test_transcriptions",
        "--phone_confidence",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value


def test_validate_training_corpus(
    multilingual_ipa_tg_corpus_dir,
    english_dictionary,
    temp_dir,
    mono_train_config_path,
    db_setup,
):

    command = [
        "validate",
        multilingual_ipa_tg_corpus_dir,
        english_dictionary,
        "-q",
        "--clean",
        "--no_debug",
        "--config_path",
        mono_train_config_path,
        "--test_transcriptions",
        "--phone_confidence",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value


def test_validate_xsampa(
    xsampa_corpus_dir,
    xsampa_dict_path,
    temp_dir,
    xsampa_train_config_path,
    db_setup,
):

    command = [
        "validate",
        xsampa_corpus_dir,
        xsampa_dict_path,
        "-q",
        "--clean",
        "--ignore_acoustics",
        "--config_path",
        xsampa_train_config_path,
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value


def test_validate_dictionary(
    english_us_mfa_g2p_model,
    english_us_mfa_dictionary_subset,
    temp_dir,
    db_setup,
):

    command = [
        "validate_dictionary",
        english_us_mfa_dictionary_subset,
        "--g2p_model_path",
        english_us_mfa_g2p_model,
        "-j",
        "1",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value


def test_validate_dictionary_train(
    basic_dict_path,
    temp_dir,
    db_setup,
):

    command = [
        "validate_dictionary",
        basic_dict_path,
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner(mix_stderr=False).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
