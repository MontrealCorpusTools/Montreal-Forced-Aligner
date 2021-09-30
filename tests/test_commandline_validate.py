
from montreal_forced_aligner.command_line.validate import run_validate_corpus
from montreal_forced_aligner.command_line.mfa import parser


def test_validate_corpus(large_prosodylab_format_directory, large_dataset_dictionary, temp_dir):

    command = ['validate', large_prosodylab_format_directory, large_dataset_dictionary, 'english',
               '-t', temp_dir, '-q', '--clean', '--debug', '--disable_mp', '--test_transcriptions', '-j', '0']
    args, unknown = parser.parse_known_args(command)
    run_validate_corpus(args)
