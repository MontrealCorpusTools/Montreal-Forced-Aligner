
from aligner.command_line.validate import run_validate_corpus


class ValidatorDummyArgs(object):
    def __init__(self):
        self.temp_directory = None
        self.test_transcriptions = False
        self.num_jobs = 0
        self.speaker_characters = 0
        self.ignore_acoustics = False


def test_validate_corpus(large_prosodylab_format_directory, large_dataset_dictionary,temp_dir):
    args = ValidatorDummyArgs()
    args.num_jobs = 2
    args.corpus_directory = large_prosodylab_format_directory
    args.dictionary_path = large_dataset_dictionary
    args.temp_directory = temp_dir
    args.test_transcriptions = True
    run_validate_corpus(args)
