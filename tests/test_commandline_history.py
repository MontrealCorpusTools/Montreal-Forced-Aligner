from montreal_forced_aligner.command_line.mfa import parser, print_history


def test_mfa_history(
    multilingual_ipa_tg_corpus_dir, english_ipa_acoustic_model, english_us_ipa_dictionary, temp_dir
):

    command = ["history", "--depth", "60"]
    args, unknown = parser.parse_known_args(command)
    print_history(args)

    command = ["history"]
    args, unknown = parser.parse_known_args(command)
    print_history(args)


def test_mfa_history_verbose(
    multilingual_ipa_tg_corpus_dir, english_ipa_acoustic_model, english_us_ipa_dictionary, temp_dir
):

    command = ["history", "-v", "--depth", "60"]
    args, unknown = parser.parse_known_args(command)
    print_history(args)

    command = ["history", "-v"]
    args, unknown = parser.parse_known_args(command)
    print_history(args)
