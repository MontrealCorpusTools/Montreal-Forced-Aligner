import os 
import re
import subprocess
import sys

from pathlib import Path
from aligner.dictmaker.inspect import BaseInspector

class DictMaker(object):
    """loads arguments and creates a Dictionary from a g2pfst model

    Parameters
    ----------
        args : arguments from command line: language, input_dir, [outfile]
    """
    def __init__(self, args):
        super(DictMaker, self).__init__()
        self.KO = args.KO
        self.args = args
        
        self.language = args.language
        self.input_dir = args.input_dir

        self.path_to_models = self.get_path_to_models()
        self.path_to_phon = self.get_path_to_phonetisaurus()

        BI =  BaseInspector(self.input_dir, self.KO)
        self.wordlist = BI.output_path

        self.outfile = args.outfile
        if self.outfile == None:
            self.outfile = "_".join([self.language, 'dict.txt'])





        self.execute()

    def execute(self):
        with open(self.outfile,"w") as f3:
            result = subprocess.Popen("{} --model={} --wordlist={}".format(self.path_to_phon, 
                os.path.join(self.path_to_models, "full.fst"), self.wordlist), stdout=f3, shell=True).wait()

        with open(self.outfile) as f4:
            lines = f4.readlines()

        with open(self.outfile, "w") as f5:
            for line in lines:
                splitline = line.split("\t")
                f5.write(splitline[0] + "\t" + splitline[2])

    def get_path_to_models(self):
        path_to_file = os.path.dirname(__file__)
        path_to_models = os.path.join(str(Path(path_to_file).parent.parent), 'dict_models', self.language)

        return path_to_models

    def get_path_to_phonetisaurus(self):
        path_to_file = os.path.dirname(__file__)
        path_to_phon = os.path.join(str(Path(path_to_file).parent.parent), 'thirdparty' , 'phonetisaurus-g2pfst')

        return path_to_phon