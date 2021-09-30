import os

from montreal_forced_aligner.command_line.train_ivector_extractor import run_train_ivector_extractor
from montreal_forced_aligner.command_line.mfa import parser


# @pytest.mark.skip(reason='Optimization')
def test_basic_ivector(basic_corpus_dir, generated_dir, large_dataset_dictionary, temp_dir,
                       train_ivector_config, english_acoustic_model, ivector_output_model_path):
    command = ['train_ivector', basic_corpus_dir, large_dataset_dictionary, 'english', ivector_output_model_path,
               '-t', temp_dir, '--config_path', train_ivector_config, '-q', '--clean', '--debug']
    args, unknown = parser.parse_known_args(command)
    run_train_ivector_extractor(args)
    assert os.path.exists(args.output_model_path)
