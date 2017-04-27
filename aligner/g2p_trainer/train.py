
import subprocess
import os
import tempfile
from aligner.dictionary import Dictionary

from pathlib import Path

"""
TODO: make all non .fst files be generated in temp directory, only .fst goes in final dir 
"""

class Trainer(object):
    """Train a g2p model from a pronunciation dictionary

    Parameters
    ----------
    language: str
        the path and language code 
    input_dict : str
        path to the pronunciation dictionary

    """
    def __init__(self, path, input_dict, **kwargs):
        super(Trainer, self).__init__()

        self.kwargs = kwargs
        if "KO" in kwargs.keys():
            self.KO = kwargs["KO"]
        else:
            self.KO = None
        if "CH_chars" in kwargs.keys():
            self.CH_chars = kwargs["CH_chars"]
        else:
            self.CH_chars = None
        self.path = path
        if not self.KO:
            self.input_dict = input_dict
        else:
            self.input_dict = self.decompose(input_dict)
        generated_dict = Dictionary(input_dict, "")
        self.dictionary = {k: v[0] for k,v in generated_dict.words.items()}
        self.path_to_model = self.train(self.dictionary)

    def train(self, dictionary):
        os.environ["LANGUAGE"] = self.path

        temp_lang_dir = tempfile.gettempdir()

        os.environ["TEMP_LANGUAGE"] = temp_lang_dir

        training_file = os.path.join(str(Path(__file__).parent), 'train.txt')

        os.environ["TRAINING_FILE"] = training_file


        os.environ["PATH_TO_PHON"] = self.get_path_to_phonetisaurus()

        with open(training_file,"w") as f2:
            [f2.write(k.strip() + "\t" + " ".join(list(v[0])) + "\n") for k,v in dictionary.items()]

        stderr = tempfile.mkstemp()[1]

        with open(stderr, "w") as f3:
            if self.CH_chars == "True":
                proc = subprocess.Popen(os.path.join(os.path.split(__file__)[0], "execute_ch.sh"), stderr=f3, shell=True).wait()
            else:
                proc = subprocess.Popen(os.path.join(os.path.split(__file__)[0], "execute.sh"), shell=True, stderr = f3).wait()
        
        return os.path.join(self.path, "full.fst")


    def decompose(self, dictionary):
        from jamo import h2j, j2hcj

        decomp_name = os.path.join(os.path.split(dictionary)[0] , "decomposed_" + os.path.split(dictionary)[1])

        with open(dictionary, errors="ignore") as f1:
            lines = f1.readlines()

        with open(decomp_name,"w") as f2:
            for line in lines:
                word, pron = line.split("\t")[0],line.split("\t")[1]
                decomposed_word = j2hcj(h2j(word))
                f2.write(decomposed_word+ "\t"+ pron)

    def get_path_to_model(self):
        return self.path_to_model



    def get_path_to_phonetisaurus(self):
        """
        returns a path to the phonetisaurus-g2pfst binary
        """
        path_to_file = os.path.dirname(__file__)
        path_to_phon = os.path.join(str(Path(path_to_file).parent.parent), 'thirdparty' )

        return path_to_phon
