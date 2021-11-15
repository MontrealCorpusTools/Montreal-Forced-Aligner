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
