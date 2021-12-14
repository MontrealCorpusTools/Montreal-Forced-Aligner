import os

from praatio import textgrid as tgio

from montreal_forced_aligner.alignment.pretrained import PretrainedAligner
from montreal_forced_aligner.command_line.align import run_align_corpus
from montreal_forced_aligner.command_line.mfa import parser


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


def test_align_arguments(
    basic_corpus_dir,
    sick_dict_path,
    generated_dir,
    english_dictionary,
    temp_dir,
    english_acoustic_model,
):

    command = [
        "align",
        basic_corpus_dir,
        english_dictionary,
        "english",
        os.path.join(generated_dir, "basic_output"),
        "-t",
        temp_dir,
        "-q",
        "--clean",
        "--debug",
        "--uses_speaker_adaptation",
        "False",
    ]
    args, unknown_args = parser.parse_known_args(command)
    params = PretrainedAligner.parse_parameters(args=args, unknown_args=unknown_args)
    assert not params["uses_speaker_adaptation"]


# @pytest.mark.skip(reason='Optimization')
def test_align_basic(
    basic_corpus_dir,
    sick_dict_path,
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
        "english",
        output_directory,
        "-t",
        temp_dir,
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
    ]
    args, unknown = parser.parse_known_args(command)
    run_align_corpus(args, unknown)

    assert os.path.exists(output_directory)

    output_paths = [
        os.path.join(output_directory, "michael", "acoustic corpus.TextGrid"),
        os.path.join(output_directory, "michael", "acoustic_corpus.TextGrid"),
        os.path.join(output_directory, "sickmichael", "cold corpus.TextGrid"),
        os.path.join(output_directory, "sickmichael", "cold_corpus.TextGrid"),
        os.path.join(output_directory, "sickmichael", "cold corpus3.TextGrid"),
        os.path.join(output_directory, "sickmichael", "cold_corpus3.TextGrid"),
    ]

    mod_times = {}
    for path in output_paths:
        assert os.path.exists(path)
        mod_times[path] = os.stat(path).st_mtime

    align_temp_dir = os.path.join(temp_dir, "basic_pretrained_aligner", "pretrained_aligner")
    assert os.path.exists(align_temp_dir)

    backup_textgrid_dir = os.path.join(align_temp_dir, "textgrids")
    assert not os.listdir(backup_textgrid_dir)

    command = [
        "align",
        basic_corpus_dir,
        english_dictionary,
        "english",
        output_directory,
        "-t",
        temp_dir,
        "--config_path",
        basic_align_config_path,
        "-q",
        "--debug",
        "--disable_mp",
    ]
    args, unknown = parser.parse_known_args(command)

    run_align_corpus(args, unknown)
    assert os.listdir(backup_textgrid_dir)

    for path in output_paths:
        assert os.path.exists(path)
        assert mod_times[path] == os.stat(path).st_mtime

    command = [
        "align",
        basic_corpus_dir,
        english_dictionary,
        "english",
        output_directory,
        "-t",
        temp_dir,
        "--config_path",
        basic_align_config_path,
        "-q",
        "--disable_textgrid_cleanup",
        "--clean",
        "--overwrite",
    ]
    args, unknown = parser.parse_known_args(command)

    run_align_corpus(args, unknown)
    assert not os.path.exists(backup_textgrid_dir) or not os.listdir(backup_textgrid_dir)
    for path in output_paths:
        assert os.path.exists(path)
        assert mod_times[path] != os.stat(path).st_mtime


def test_align_multilingual(
    multilingual_ipa_corpus_dir,
    english_uk_ipa_dictionary,
    generated_dir,
    temp_dir,
    basic_align_config_path,
    english_acoustic_model,
    english_ipa_acoustic_model,
):

    command = [
        "align",
        multilingual_ipa_corpus_dir,
        english_uk_ipa_dictionary,
        english_ipa_acoustic_model,
        os.path.join(generated_dir, "multilingual"),
        "-t",
        temp_dir,
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
    ]
    args, unknown = parser.parse_known_args(command)
    run_align_corpus(args, unknown)


def test_align_multilingual_speaker_dict(
    multilingual_ipa_corpus_dir,
    ipa_speaker_dict_path,
    generated_dir,
    temp_dir,
    basic_align_config_path,
    english_ipa_acoustic_model,
):

    command = [
        "align",
        multilingual_ipa_corpus_dir,
        ipa_speaker_dict_path,
        english_ipa_acoustic_model,
        os.path.join(generated_dir, "multilingual_speaker_dict"),
        "-t",
        temp_dir,
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
    ]
    args, unknown = parser.parse_known_args(command)
    run_align_corpus(args, unknown)


def test_align_multilingual_tg_speaker_dict(
    multilingual_ipa_tg_corpus_dir,
    ipa_speaker_dict_path,
    generated_dir,
    temp_dir,
    basic_align_config_path,
    english_ipa_acoustic_model,
):

    command = [
        "align",
        multilingual_ipa_tg_corpus_dir,
        ipa_speaker_dict_path,
        english_ipa_acoustic_model,
        os.path.join(generated_dir, "multilingual_speaker_dict_tg"),
        "-t",
        temp_dir,
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
    ]
    args, unknown = parser.parse_known_args(command)
    run_align_corpus(args, unknown)


def test_align_split(
    basic_split_dir,
    english_us_ipa_dictionary,
    generated_dir,
    temp_dir,
    basic_align_config_path,
    english_acoustic_model,
    english_ipa_acoustic_model,
):
    audio_dir, text_dir = basic_split_dir
    command = [
        "align",
        text_dir,
        english_us_ipa_dictionary,
        english_ipa_acoustic_model,
        os.path.join(generated_dir, "multilingual"),
        "-t",
        temp_dir,
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
        "--audio_directory",
        audio_dir,
    ]
    args, unknown = parser.parse_known_args(command)
    run_align_corpus(args, unknown)


def test_align_stereo(
    stereo_corpus_dir,
    sick_dict_path,
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
        "english",
        output_dir,
        "-t",
        temp_dir,
        "--config_path",
        basic_align_config_path,
        "-q",
        "--clean",
        "--debug",
    ]
    args, unknown = parser.parse_known_args(command)
    run_align_corpus(args, unknown)

    tg = tgio.openTextgrid(
        os.path.join(output_dir, "michaelandsickmichael.TextGrid"), includeEmptyIntervals=False
    )
    assert len(tg.tierNameList) == 4
