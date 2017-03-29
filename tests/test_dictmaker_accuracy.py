import pytest
import os
from aligner.dictmaker.tools.compare import compare
from aligner.dictmaker.makedict import DictMaker 


class ArgStandIn(object):
    """placeholder namespace"""
    def __init__(self, language, input_dir, outfile, KO=None):
        super(ArgStandIn, self).__init__()
        self.language = language
        self.input_dir = input_dir
        self.outfile = outfile
        self.KO = KO
        
def get_path():
        return os.path.dirname(__file__)


# def test_accuracy_AR():
#     path = get_path()
#     args = ArgStandIn("AR", "/Volumes/data/corpora/GP_for_MFA/AR/files", os.path.join(path, "test_output"))
#     D = DictMaker(args)

#     accuracy = compare(os.path.join(path, "test_output"), "/Volumes/data/corpora/GP_for_MFA/AR/dict/AR_dictionary.txt")[0]
#     print(accuracy)
#     assert(accuracy > .95)

# def test_accuracy_BG():
#     path = get_path()
#     args = ArgStandIn("BG", "/Volumes/data/corpora/GP_for_MFA/BG/files", os.path.join(path, "test_output"))
#     D = DictMaker(args)

#     accuracy = compare(os.path.join(path, "test_output"), "/Volumes/data/corpora/GP_for_MFA/BG/dict/BG_dictionary.txt")[0]
#     print(accuracy)
#     assert(accuracy > .95)

# def test_accuracy_CR():
#     path = get_path()
#     args = ArgStandIn("CR_no_kaer", "/Volumes/data/corpora/GP_for_MFA/CR/files", os.path.join(path, "test_output"))
#     D = DictMaker(args)

#     accuracy = compare(os.path.join(path, "test_output"), "/Volumes/data/corpora/GP_for_MFA/CR/dict/CR_no_kaer.txt")[0]
#     print(accuracy)
#     assert(accuracy > .95)


# def test_accuracy_SP():
#     path = get_path()
#     args = ArgStandIn("SP", "/Volumes/data/corpora/GP_for_MFA/SP/files", os.path.join(path, "test_output"))
#     D = DictMaker(args)

#     accuracy = compare(os.path.join(path, "test_output"), "/Volumes/data/corpora/GP_for_MFA/SP/dict/lexicon_nosil.txt")[0]
#     print(accuracy)
#     assert(accuracy > .95)

def test_accuracy_KO():
    path = get_path()
    args = ArgStandIn("KO", "/Volumes/data/corpora/GP_for_MFA/KO/files", os.path.join(path, "test_output"), KO=True)
    D = DictMaker(args)

    accuracy = compare(os.path.join(path, "test_output"), "/Volumes/data/corpora/GP_for_MFA/KO/dict/decomposed_dict.txt")[0]
    print(accuracy)
    assert(accuracy > .95)

def test_accuracy_PL():
    path = get_path()
    args = ArgStandIn("PL", "/Volumes/data/corpora/GP_for_MFA/PL/files", os.path.join(path, "test_output"))
    D = DictMaker(args)

    accuracy = compare(os.path.join(path, "test_output"), "/Volumes/data/corpora/GP_for_MFA/PL/dict/lexicon_nosil.txt")[0]
    print(accuracy)
    assert(accuracy > .95)


def test_accuracy_RU():
    path = get_path()
    args = ArgStandIn("RU", "/Volumes/data/corpora/GP_for_MFA/RU/files", os.path.join(path, "test_output"))
    D = DictMaker(args)

    accuracy = compare(os.path.join(path, "test_output"), "/Volumes/data/corpora/GP_for_MFA/RU/dict/lexicon_nosil.txt")[0]
    print(accuracy)
    assert(accuracy > .95)

def test_accuracy_SA():
    path = get_path()
    args = ArgStandIn("SA", "/Volumes/data/corpora/GP_for_MFA/SA/files", os.path.join(path, "test_output"))
    D = DictMaker(args)

    accuracy = compare(os.path.join(path, "test_output"), "/Volumes/data/corpora/GP_for_MFA/SA/dict/lexicon_nosil.txt")[0]
    print(accuracy)
    assert(accuracy > .95)

def test_accuracy_UA():
    path = get_path()
    args = ArgStandIn("UA", "/Volumes/data/corpora/GP_for_MFA/UA/files", os.path.join(path, "test_output"))
    D = DictMaker(args)

    accuracy = compare(os.path.join(path, "test_output"), "/Volumes/data/corpora/GP_for_MFA/UA/dict/lexicon_nosil.txt")[0]
    print(accuracy)
    assert(accuracy > .95)

def test_accuracy_CH():
    path = get_path()
    args = ArgStandIn("CH", "/Volumes/data/corpora/GP_for_MFA/CH/files", os.path.join(path, "test_output"))
    D = DictMaker(args)

    accuracy = compare(os.path.join(path, "test_output"), "/Volumes/data/corpora/GP_for_MFA/CH/dict/CH_dictionary.txt")[0]
    print(accuracy)
    assert(accuracy > .95)

def test_accuracy_VN():
    path = get_path()
    args = ArgStandIn("VN", "/Volumes/data/corpora/GP_for_MFA/VN/files", os.path.join(path, "test_output"))
    D = DictMaker(args)

    accuracy = compare(os.path.join(path, "test_output"), "/Volumes/data/corpora/GP_for_MFA/VN/dict/lexicon_nosil.txt")[0]
    print(accuracy)
    assert(accuracy > .95)

def test_accuracy_CZ():
    path = get_path()
    args = ArgStandIn("CZ", "/Volumes/data/corpora/GP_for_MFA/CZ/files", os.path.join(path, "test_output"))
    D = DictMaker(args)

    accuracy = compare(os.path.join(path, "test_output"), "/Volumes/data/corpora/GP_for_MFA/CZ/dict/CZ_dictionary.txt")[0]
    print(accuracy)
    assert(accuracy > .95)


