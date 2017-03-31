import pytest
import os
from aligner.dictmaker.tools.compare import compare
from aligner.dictmaker.makedict import DictMaker 
from aligner.command_line.generate_dict import generate_dict


        



def test_example_CH():
    D = DictMaker("CH", "/Users/elias/MontrealForcedAligner/examples/example_labs", "test_output")

    # accuracy = compare(os.path.join(path, "test_output"), "/Volumes/data/corpora/GP_for_MFA/CH/dict/CH_dictionary.txt")[0]
    # print(accuracy)
    # assert(accuracy > .95)
