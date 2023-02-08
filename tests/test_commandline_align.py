import os

import click.testing
from praatio import textgrid as tgio

from montreal_forced_aligner.command_line.mfa import mfa_cli


def assert_export_exist(old_directory, new_directory):
    for root, dirs, files in os.walk(old_directory):
        new_root = root.replace(old_directory, new_directory)
        for d in dirs:
            assert os.path.exists(os.path.join(new_root, d))
        for f in files:
            if not f.endswith(".wav"):
                continue
            new_f = f.replace(".wav", ".TextGrid")
            assert os.path.exists(os.path.join(new_root, new_f))


def test_align_no_speaker_adaptation(
    basic_corpus_dir,
    generated_dir,
    english_dictionary,
    temp_dir,
    english_acoustic_model,
):
    output_directory = os.path.join(generated_dir, "basic_output")
    command = [
        "align",
        basic_corpus_dir,
        english_dictionary,
        "english_us_arpa",
        output_directory,
        "-t",
        os.path.join(temp_dir, "test_align_no_speaker_adaptation"),
        "-q",
        "--clean",
        "--debug",
        "--verbose",
        "--uses_speaker_adaptation",
        "False",
    ]
    click.testing.CliRunner().invoke(mfa_cli, command, catch_exceptions=False)
    assert os.path.exists(output_directory)


def test_align_single_speaker(
    basic_corpus_dir,
    generated_dir,
    english_dictionary,
    temp_dir,
    basic_align_config_path,
    english_acoustic_model,
):
    output_directory = os.path.join(generated_dir, "basic_align_output")
    command = [
        "align",
        basic_corpus_dir,
        english_dictionary,
        english_acoustic_model,
        output_directory,
        "-t",
        os.path.join(temp_dir, "test_align_single_speaker"),
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
        "--single_speaker",
    ]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value

    assert os.path.exists(output_directory)

    output_paths = [
        os.path.join(output_directory, "michael", "acoustic corpus.TextGrid"),
        os.path.join(output_directory, "michael", "acoustic_corpus.TextGrid"),
        os.path.join(output_directory, "sickmichael", "cold corpus.TextGrid"),
        os.path.join(output_directory, "sickmichael", "cold_corpus.TextGrid"),
        os.path.join(output_directory, "sickmichael", "cold corpus3.TextGrid"),
        os.path.join(output_directory, "sickmichael", "cold_corpus3.TextGrid"),
    ]

    for path in output_paths:
        assert os.path.exists(path)


def test_align_duplicated(
    duplicated_name_corpus_dir,
    generated_dir,
    english_dictionary,
    temp_dir,
    basic_align_config_path,
    english_acoustic_model,
):
    output_directory = os.path.join(generated_dir, "duplicated_align_output")
    command = [
        "align",
        duplicated_name_corpus_dir,
        english_dictionary,
        english_acoustic_model,
        output_directory,
        "-t",
        os.path.join(temp_dir, "test_align_duplicated"),
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--no_debug",
    ]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value

    assert os.path.exists(output_directory)

    output_paths = [
        os.path.join(output_directory, "michael", "recording_0.TextGrid"),
        os.path.join(output_directory, "sickmichael", "recording_0.TextGrid"),
        os.path.join(output_directory, "sickmichael", "recording_1.TextGrid"),
    ]

    for path in output_paths:
        assert os.path.exists(path)


def test_align_multilingual(
    multilingual_ipa_corpus_dir,
    english_uk_mfa_dictionary,
    generated_dir,
    temp_dir,
    basic_align_config_path,
    english_mfa_acoustic_model,
):

    command = [
        "align",
        multilingual_ipa_corpus_dir,
        english_uk_mfa_dictionary,
        english_mfa_acoustic_model,
        os.path.join(generated_dir, "multilingual"),
        "-t",
        os.path.join(temp_dir, "test_align_multilingual"),
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
        "--output_format",
        "short_textgrid",
    ]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value


def test_align_multilingual_speaker_dict(
    multilingual_ipa_corpus_dir,
    mfa_speaker_dict_path,
    generated_dir,
    temp_dir,
    basic_align_config_path,
    english_mfa_acoustic_model,
):

    command = [
        "align",
        multilingual_ipa_corpus_dir,
        mfa_speaker_dict_path,
        english_mfa_acoustic_model,
        os.path.join(generated_dir, "multilingual_speaker_dict"),
        "-t",
        os.path.join(temp_dir, "test_align_multilingual_speaker_dict"),
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
        "--output_format",
        "json",
    ]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value


def test_align_multilingual_tg_speaker_dict(
    multilingual_ipa_tg_corpus_dir,
    mfa_speaker_dict_path,
    generated_dir,
    temp_dir,
    basic_align_config_path,
    english_mfa_acoustic_model,
):

    command = [
        "align",
        multilingual_ipa_tg_corpus_dir,
        mfa_speaker_dict_path,
        english_mfa_acoustic_model,
        os.path.join(generated_dir, "multilingual_speaker_dict_tg"),
        "-t",
        os.path.join(temp_dir, "test_align_multilingual_tg_speaker_dict"),
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
        "--include_original_text",
    ]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value


def test_align_evaluation(
    basic_corpus_dir,
    english_us_mfa_dictionary,
    generated_dir,
    temp_dir,
    basic_align_config_path,
    english_mfa_acoustic_model,
    basic_reference_dir,
    eval_mapping_path,
):

    command = [
        "align",
        basic_corpus_dir,
        english_us_mfa_dictionary,
        english_mfa_acoustic_model,
        os.path.join(generated_dir, "align_eval_output"),
        "-t",
        os.path.join(temp_dir, "test_align_evaluation"),
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
        "--no_use_mp",
        "--fine_tune",
        "--phone_confidence",
        "--reference_directory",
        basic_reference_dir,
        "--custom_mapping_path",
        eval_mapping_path,
    ]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value


def test_align_split(
    basic_split_dir,
    english_us_mfa_dictionary,
    generated_dir,
    temp_dir,
    basic_align_config_path,
    english_acoustic_model,
    english_mfa_acoustic_model,
):
    audio_dir, text_dir = basic_split_dir
    command = [
        "align",
        text_dir,
        english_us_mfa_dictionary,
        english_mfa_acoustic_model,
        os.path.join(generated_dir, "multilingual"),
        "-t",
        os.path.join(temp_dir, "test_align_split"),
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
        "--output_format",
        "json",
        "--audio_directory",
        audio_dir,
    ]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value


def test_align_stereo(
    stereo_corpus_dir,
    generated_dir,
    english_dictionary,
    temp_dir,
    basic_align_config_path,
    english_acoustic_model,
):
    output_dir = os.path.join(generated_dir, "stereo_output")
    command = [
        "align",
        stereo_corpus_dir,
        english_dictionary,
        english_acoustic_model,
        output_dir,
        "-t",
        os.path.join(temp_dir, "test_align_stereo"),
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
    ]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value

    tg = tgio.openTextgrid(
        os.path.join(output_dir, "michaelandsickmichael.TextGrid"), includeEmptyIntervals=False
    )
    assert len(tg.tierNames) == 4


def test_align_mp3s(
    mp3_corpus_dir,
    generated_dir,
    english_dictionary,
    temp_dir,
    basic_align_config_path,
    english_acoustic_model,
):
    output_dir = os.path.join(generated_dir, "mp3_output")
    command = [
        "align",
        mp3_corpus_dir,
        english_dictionary,
        english_acoustic_model,
        output_dir,
        "-t",
        os.path.join(temp_dir, "test_align_mp3s"),
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
    ]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value

    tg = tgio.openTextgrid(
        os.path.join(output_dir, "common_voice_en_22058267.TextGrid"), includeEmptyIntervals=False
    )
    assert len(tg.tierNames) == 2


def test_align_opus(
    opus_corpus_dir,
    generated_dir,
    english_dictionary,
    temp_dir,
    basic_align_config_path,
    english_acoustic_model,
):
    output_dir = os.path.join(generated_dir, "opus_output")
    command = [
        "align",
        opus_corpus_dir,
        english_dictionary,
        english_acoustic_model,
        output_dir,
        "-t",
        os.path.join(temp_dir, "test_align_opus"),
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
    ]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value

    tg = tgio.openTextgrid(
        os.path.join(output_dir, "13697_11991_000000.TextGrid"), includeEmptyIntervals=False
    )
    assert len(tg.tierNames) == 2


def test_swedish_cv(
    swedish_dir,
    generated_dir,
    swedish_cv_dictionary,
    temp_dir,
    basic_align_config_path,
    swedish_cv_acoustic_model,
):
    output_dir = os.path.join(generated_dir, "swedish_cv_output")
    command = [
        "align",
        swedish_dir,
        swedish_cv_dictionary,
        swedish_cv_acoustic_model,
        output_dir,
        "-t",
        os.path.join(temp_dir, "test_swedish_cv"),
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
    ]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value

    output_speaker_dir = os.path.join(output_dir, "se10x016")
    assert os.path.exists(output_speaker_dir)
    for file in [
        "se10x016-08071999-1334_u0016001",
        "se10x016-08071999-1334_u0016002",
        "se10x016-08071999-1334_u0016003",
        "se10x016-08071999-1334_u0016004",
    ]:
        tg_path = os.path.join(output_speaker_dir, file + ".TextGrid")
        assert os.path.exists(tg_path)
        tg = tgio.openTextgrid(tg_path, includeEmptyIntervals=False)
        assert len(tg.tierNames) == 2


def test_swedish_mfa(
    swedish_dir,
    generated_dir,
    swedish_cv_dictionary,
    temp_dir,
    basic_align_config_path,
    swedish_cv_acoustic_model,
):
    output_dir = os.path.join(generated_dir, "swedish_mfa_output")
    command = [
        "align",
        swedish_dir,
        swedish_cv_dictionary,
        swedish_cv_acoustic_model,
        output_dir,
        "-t",
        os.path.join(temp_dir, "test_swedish_mfa"),
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
    ]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value

    output_speaker_dir = os.path.join(output_dir, "se10x016")
    assert os.path.exists(output_speaker_dir)
    for file in [
        "se10x016-08071999-1334_u0016001",
        "se10x016-08071999-1334_u0016002",
        "se10x016-08071999-1334_u0016003",
        "se10x016-08071999-1334_u0016004",
    ]:
        tg_path = os.path.join(output_speaker_dir, file + ".TextGrid")
        assert os.path.exists(tg_path)
        tg = tgio.openTextgrid(tg_path, includeEmptyIntervals=False)
        assert len(tg.tierNames) == 2


def test_acoustic_g2p_model(
    basic_corpus_dir,
    acoustic_model_dir,
    dict_dir,
    generated_dir,
    temp_dir,
    basic_align_config_path,
):
    model_path = os.path.join(acoustic_model_dir, "acoustic_g2p_output_model.zip")
    dict_path = os.path.join(dict_dir, "acoustic_g2p_dictionary.yaml")
    output_directory = os.path.join(generated_dir, "acoustic_g2p_output")
    command = [
        "align",
        basic_corpus_dir,
        dict_path,
        model_path,
        output_directory,
        "-t",
        os.path.join(temp_dir, "test_acoustic_g2p_model"),
        "--config_path",
        basic_align_config_path,
        "--clean",
        "--debug",
    ]
    result = click.testing.CliRunner(mix_stderr=False, echo_stdin=True).invoke(
        mfa_cli, command, catch_exceptions=True
    )
    print(result.stdout)
    print(result.stderr)
    if result.exception:
        print(result.exc_info)
        raise result.exception
    assert not result.return_value

    assert os.path.exists(output_directory)

    output_paths = [
        os.path.join(output_directory, "michael", "acoustic corpus.TextGrid"),
        os.path.join(output_directory, "michael", "acoustic_corpus.TextGrid"),
        os.path.join(output_directory, "sickmichael", "cold corpus.TextGrid"),
        os.path.join(output_directory, "sickmichael", "cold_corpus.TextGrid"),
        os.path.join(output_directory, "sickmichael", "cold corpus3.TextGrid"),
        os.path.join(output_directory, "sickmichael", "cold_corpus3.TextGrid"),
    ]

    for path in output_paths:
        assert os.path.exists(path)
