import os
import pytest

from aligner.dictionary import Dictionary

def ListLines(path):
    lines = []
    thefile = open(path)
    text = thefile.readlines()
    for line in text:
        stripped = line.strip()
        if stripped != '':
            lines.append(stripped)
    return lines

def test_basic(basic_dict_path, generated_dir):
    d = Dictionary(basic_dict_path, os.path.join(generated_dir, 'basic'))
    x = d.write()

@pytest.mark.xfail
def test_basic_topo(generated_dir, basic_topo_path):
    oldfile = ListLines(os.path.join(generated_dir, 'basic/dictionary/topo'))
    newfile = ListLines(basic_topo_path)
    assert (len(oldfile) == len(newfile))
    for num in range(0, len(newfile)):
        assert (oldfile[num]==newfile[num])

@pytest.mark.xfail
def test_basic_graphemes(basic_dict_path, basic_topo_path):
    oldfile = ListLines(basic_dict_path)
    newfile = ListLines(basic_graphemes_path)
    assert (len(oldfile) == len(newfile))
    for num in range(0, len(newfile)):
        assert (oldfile[num]==newfile[num])

@pytest.mark.xfail
def test_basic_phone_map(basic_dict_path, basic_phone_map_path):
    oldfile = ListLines(basic_dict_path)
    newfile = ListLines(basic_phone_map_path)
    assert (len(oldfile) == len(newfile))
    for num in range(0, len(newfile)):
        assert (oldfile[num]==newfile[num])

@pytest.mark.xfail
def test_basic_phones(basic_dict_path, basic_phones_path):
    oldfile = ListLines(basic_dict_path)
    newfile = ListLines(basic_phones_path)
    assert (len(oldfile) == len(newfile))
    for num in range(0, len(newfile)):
        assert (oldfile[num]==newfile[num])

@pytest.mark.xfail
def test_basic_words(basic_dict_path, basic_words_path):
    oldfile = ListLines(basic_dict_path)
    newfile = ListLines(basic_words_path)
    assert (len(oldfile) == len(newfile))
    for num in range(0, len(newfile)):
        assert (oldfile[num]==newfile[num])

@pytest.mark.xfail
def test_basic_rootsint(basic_dict_path, basic_rootsint_path):
    oldfile = ListLines(basic_dict_path)
    newfile = ListLines(basic_rootsint_path)
    assert (len(oldfile) == len(newfile))
    for num in range(0, len(newfile)):
        assert (oldfile[num]==newfile[num])

@pytest.mark.xfail
def test_basic_rootstxt(basic_dict_path, basic_rootstxt_path):
    oldfile = ListLines(basic_dict_path)
    newfile = ListLines(basic_rootstxt_path)
    assert (len(oldfile) == len(newfile))
    for num in range(0, len(newfile)):
        assert (oldfile[num]==newfile[num])

@pytest.mark.xfail
def test_basic_setsint(basic_dict_path, basic_setsint_path):
    oldfile = ListLines(basic_dict_path)
    newfile = ListLines(basic_setsint_path)
    assert (len(oldfile) == len(newfile))
    for num in range(0, len(newfile)):
        assert (oldfile[num]==newfile[num])

@pytest.mark.xfail
def test_basic_setstxt(basic_dict_path, basic_setstxt_path):
    oldfile = ListLines(basic_dict_path)
    newfile = ListLines(basic_setstxt_path)
    assert (len(oldfile) == len(newfile))
    for num in range(0, len(newfile)):
        assert (oldfile[num]==newfile[num])

@pytest.mark.xfail
def test_basic_word_boundaryint(basic_dict_path, basic_word_boundaryint_path):
    oldfile = ListLines(basic_dict_path)
    newfile = ListLines(basic_word_boundaryint_path)
    assert (len(oldfile) == len(newfile))
    for num in range(0, len(newfile)):
        assert (oldfile[num]==newfile[num])

@pytest.mark.xfail
def test_basic_word_boundarytxt(basic_dict_path, basic_word_boundarytxt_path):
    oldfile = ListLines(basic_dict_path)
    newfile = ListLines(basic_word_boundarytxt_path)
    assert (len(oldfile) == len(newfile))
    for num in range(0, len(newfile)):
        assert (oldfile[num]==newfile[num])


