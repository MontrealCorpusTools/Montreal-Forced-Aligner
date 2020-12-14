import os

from montreal_forced_aligner.command_line.align import DummyArgs

from montreal_forced_aligner.command_line.train_and_align import run_train_corpus


# @pytest.mark.skip(reason='Optimization')
def test_train_and_align_basic(basic_corpus_dir, sick_dict_path, generated_dir, temp_dir,
                               mono_train_config_path, textgrid_output_model_path):
    if os.path.exists(textgrid_output_model_path):
        os.remove(textgrid_output_model_path)
    args = DummyArgs()
    args.corpus_directory = basic_corpus_dir
    args.dictionary_path = sick_dict_path
    args.output_directory = os.path.join(generated_dir, 'basic_output')
    args.quiet = True
    args.clean = True
    args.temp_directory = temp_dir
    args.config_path = mono_train_config_path
    args.output_model_path = textgrid_output_model_path

    args.corpus_directory = basic_corpus_dir
    run_train_corpus(args)
    assert os.path.exists(args.output_model_path)
