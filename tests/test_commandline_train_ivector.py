import os

from montreal_forced_aligner.command_line.train_ivector_extractor import run_train_ivector_extractor


class DummyArgs(object):
    def __init__(self):
        self.corpus_directory = ''
        self.dictionary_path = ''
        self.speaker_characters = 0
        self.num_jobs = 0
        self.verbose = False
        self.clean = True
        self.fast = True
        self.debug = False
        self.temp_directory = None
        self.config_path = ''
        self.output_model_path = ''


# @pytest.mark.skip(reason='Optimization')
def test_basic_ivector(basic_corpus_dir, sick_dict_path, generated_dir, large_dataset_dictionary, temp_dir,
                       basic_train_ivector_config, english_acoustic_model, ivector_output_model_path):
    args = DummyArgs()
    args.corpus_directory = basic_corpus_dir
    args.quiet = True
    args.clean = True
    args.temp_directory = temp_dir
    args.dictionary_path = large_dataset_dictionary
    args.output_directory = os.path.join(generated_dir, 'basic_output')
    args.output_model_path = ivector_output_model_path
    args.config_path = basic_train_ivector_config
    run_train_ivector_extractor(args)
    assert os.path.exists(args.output_model_path)
