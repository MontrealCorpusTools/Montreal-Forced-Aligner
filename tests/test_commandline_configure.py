import os

from montreal_forced_aligner.config import update_global_config, load_global_config, generate_config_path, TEMP_DIR
from montreal_forced_aligner.command_line.mfa import create_parser


def test_configure(temp_dir, basic_corpus_dir, sick_dict_path, generated_dir, large_dataset_dictionary,
                     basic_align_config, english_acoustic_model):
    parser = create_parser()
    path = generate_config_path()
    if os.path.exists(path):
        os.remove(path)
    GLOBAL_CONFIG = load_global_config()
    assert GLOBAL_CONFIG == {
        'clean': False,
        'verbose': False,
        'debug': False,
        'overwrite': False,
        'terminal_colors': True,
        'terminal_width': 120,
        'cleanup_textgrids': True,
        'num_jobs': 3,
        'use_mp': True,
        'temp_directory': TEMP_DIR
    }
    command = ['configure', '--always_clean', '-t', temp_dir, '-j', '10', '--disable_mp', '--always_verbose']
    args, unknown = parser.parse_known_args(command)
    update_global_config(args)
    assert os.path.exists(path)
    GLOBAL_CONFIG = load_global_config()
    assert GLOBAL_CONFIG == {
        'clean': True,
        'verbose': True,
        'debug': False,
        'overwrite': False,
        'terminal_colors': True,
        'terminal_width': 120,
        'cleanup_textgrids': True,
        'num_jobs': 10,
        'use_mp': False,
        'temp_directory': temp_dir
    }
    command = ['configure', '--never_clean', '--enable_mp', '--never_verbose']
    args, unknown = parser.parse_known_args(command)
    update_global_config(args)
    assert os.path.exists(path)
    GLOBAL_CONFIG = load_global_config()
    assert GLOBAL_CONFIG == {
        'clean': False,
        'verbose': False,
        'debug': False,
        'overwrite': False,
        'terminal_colors': True,
        'terminal_width': 120,
        'cleanup_textgrids': True,
        'num_jobs': 10,
        'use_mp': True,
        'temp_directory': temp_dir
    }
    parser = create_parser()

    command = ['align', basic_corpus_dir, sick_dict_path, 'english', os.path.join(generated_dir, 'basic_output'),
               '-t', TEMP_DIR, '-c', basic_align_config, '-q', '--clean', '-d']
    args, unknown = parser.parse_known_args(command)
    assert args.num_jobs == 10
    assert args.temp_directory == TEMP_DIR
    assert args.clean
    assert not args.disable_mp
    if os.path.exists(path):
        os.remove(path)




