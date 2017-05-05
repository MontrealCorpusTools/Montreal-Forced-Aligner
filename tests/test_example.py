import pytest
import os
from aligner.dictmaker.makedict import DictMaker 
from aligner.command_line.train_and_align import align_corpus as train_and_align_corpus
from aligner.g2p_trainer.train import Trainer

        
class DummyArgs(object):
    def __init__(self):
        self.speaker_characters = 0
        self.num_jobs = 0
        self.verbose = False
        self.clean = True
        self.no_speaker_adaptation = False
        self.debug = False
        self.errors = False



def test_example_CH(dict_model_path, dict_input_directory, dict_output_path):
    D = DictMaker(dict_model_path, dict_input_directory, dict_output_path)
    assert(os.path.exists(dict_output_path))
    # accuracy = compare(os.path.join(path, "test_output"), "/Volumes/data/corpora/GP_for_MFA/CH/dict/CH_dictionary.txt")[0]
    # print(accuracy)
    # assert(accuracy > .95)


def test_example_CH_chars(dict_model_path_char, dict_input_directory_char, dict_output_path_char):
    D = DictMaker(dict_model_path_char, dict_input_directory_char, dict_output_path_char)
    assert(os.path.exists(dict_output_path_char))

# def test_full_CH(dict_model_path, dict_input_directory, dict_output_path):
#     path_to_model = os.path.split(Trainer(dict_model_path, dict_output_path, False).get_path_to_model())[0]
#     D = DictMaker(path_to_model, dict_input_directory, dict_output_path)
#     assert(os.path.exists(dict_output_path))

# def test_example_alignment(dict_input_directory,
#                     dict_output_path, example_output_directory,
#                     example_output_model_path):



#     args = DummyArgs()
#     args.num_jobs = 1
#     args.fast = False
#     train_and_align_corpus(dict_input_directory, dict_output_path,  example_output_directory, '',
#             example_output_model_path, args, skip_input=False)