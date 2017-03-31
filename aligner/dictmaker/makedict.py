import os 
import re
import subprocess
import sys
import tempfile

from pathlib import Path
from aligner.corpus import Corpus


class DictMaker(object):
    """loads arguments and creates a Dictionary from a g2pfst model

    Parameters
    ----------
        args : arguments from command line: language, input_dir, [outfile]
    """
    def __init__(self, language, input_dir,  outfile, KO=None):
        super(DictMaker, self).__init__()
        self.KO = KO
        
        self.language = language

        self.input_dir = input_dir
        corpus = Corpus(input_dir, input_dir, 0)
        TEMP_FILE = tempfile.mkstemp()
        self.wordlist = TEMP_FILE[1]

        with open(TEMP_FILE[1], 'w') as f1:
            words = corpus.word_set
            print(words)
            for word in words:
                f1.write(word.strip() + '\n')

        self.path_to_models = self.get_path_to_models()
        self.path_to_phon = self.get_path_to_phonetisaurus()

        self.outfile = outfile
        if self.outfile == None:
            self.outfile = "_".join([self.language, 'dict.txt'])

        self.execute()

    def execute(self):
        print("writing file 1")
        with open(self.outfile,"w") as f3:
            print("{} --model={} --wordlist={}".format(self.path_to_phon, 
                os.path.join(self.path_to_models, "full.fst"), self.wordlist))
            result = subprocess.Popen("{} --model={} --wordlist={}".format(self.path_to_phon, 
                os.path.join(self.path_to_models, "full.fst"), self.wordlist), stdout=f3, shell=True).wait()
            print("result " , result)
        print("reading file 1")
        with open(self.outfile) as f4:
            lines = f4.readlines()
        print("writing file 2")
        with open(self.outfile, "w") as f5:
            for line in lines:
                splitline = line.split("\t")
                print(splitline, "splitline")
                f5.write(splitline[0] + "\t" + splitline[2])

    def get_path_to_models(self):
        path_to_file = os.path.dirname(__file__)
        path_to_models = os.path.join(str(Path(path_to_file).parent.parent), 'dict_models', self.language)

        return path_to_models

    def get_path_to_phonetisaurus(self):
        path_to_file = os.path.dirname(__file__)
        path_to_phon = os.path.join(str(Path(path_to_file).parent.parent), 'thirdparty' , 'phonetisaurus-g2pfst')

        return path_to_phon