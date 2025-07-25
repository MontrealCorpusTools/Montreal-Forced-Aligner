import click.testing

from montreal_forced_aligner.command_line.mfa import mfa_cli


def test_validate_corpus(
    multilingual_ipa_tg_corpus_dir,
    english_mfa_acoustic_model,
    english_us_mfa_dictionary,
    temp_dir,
    generated_dir,
    db_setup,
):
    output_path = generated_dir.joinpath("find_oovs_output")
    command = [
        "find_oovs",
        multilingual_ipa_tg_corpus_dir,
        english_us_mfa_dictionary,
        output_path,
        "-q",
        "-s",
        "4",
        "--oov_count_threshold",
        "0",
        "--clean",
        "--no_use_mp",
    ]
    command = [str(x) for x in command]
    result = click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=True)
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value
    assert output_path.joinpath(f"oovs_found_{english_us_mfa_dictionary}.txt")
    assert output_path.joinpath(f"oov_counts_{english_us_mfa_dictionary}.txt")
    assert output_path.joinpath("utterance_oovs.txt")
