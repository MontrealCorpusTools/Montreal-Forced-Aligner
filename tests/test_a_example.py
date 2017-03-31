import pytest
import os
from aligner.dictmaker.makedict import DictMaker 


        



def test_example_CH():
    D = DictMaker("CH", "/Users/elias/Montreal-Forced-Aligner/examples/example_labs", "/Users/elias/Montreal-Forced-Aligner/examples/test_output")

    # accuracy = compare(os.path.join(path, "test_output"), "/Volumes/data/corpora/GP_for_MFA/CH/dict/CH_dictionary.txt")[0]
    # print(accuracy)
    # assert(accuracy > .95)
    assert(1==2)