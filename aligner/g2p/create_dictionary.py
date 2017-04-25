import os 
import re
import subprocess
import sys
import tempfile

from aligner.corpus import Corpus


class DictMaker(object):
    """creates a Dictionary from a g2pfst model

    Parameters
    ----------
        path_to_models: str
            path to the models
        input dir: str
            location of .lab files
        outfile: str
            destination for the dictionary
        korean:bool
            default to False, to be used if using a Korean corpus in Hangul
    """
    def __init__(self, g2p_model, input_dir,  outfile, korean=False):
        super(DictMaker, self).__init__()
        self.korean = korean
        
        self.g2p_model = g2p_model

        self.input_dir = input_dir
        corpus = Corpus(input_dir, input_dir, 0)
        TEMP_FILE = tempfile.mkstemp()
        self.wordlist = TEMP_FILE[1]

        with open(TEMP_FILE[1], 'w') as f1:
            words = corpus.word_set
            for word in words:
                f1.write(word.strip() + '\n')

        self.outfile = outfile
        if self.outfile == None:
            self.outfile = os.path.join(self.path_to_models, 'generated_dict.txt')

        self.stderr = tempfile.mkstemp()[1]
        self.execute()

    def execute(self):
        """
        runs the phonetisaurus-g2pfst binary with the language and all the words in the corpus
        """

        with open(self.outfile,"w") as f3:
            with open(self.stderr,'w') as f4:
                result = subprocess.Popen("{} --model={} --wordlist={}".format(self.path_to_phon, 
                    os.path.join(self.path_to_models, "full.fst"), self.wordlist), stdout=f3, stderr=f4, shell=True).wait()

        with open(self.stderr) as f3:
            syms = []
            sym_count =0
            lineregex = re.compile("Symbol: '.*' not found in input symbols table")
            symbolregex = re.compile("(?<=').(?=')")
            lines = f3.readlines()
            for line in lines:
                if lineregex.match(line) is not None:
                    syms.append(symbolregex.match(line).group(0) if symbolregex.match(line) is not None else "")
                    sym_count +=1 

            print("There were {} unmatched symbols in your transcriptions.".format(sym_count))
        with open("unknown_syms","w") as f5:
            for sym in syms:
                if sym != "":
                    f5.write(sym + "\n")


        with open(self.outfile) as f4:
            lines = f4.readlines()

        print(lines)
        with open(self.outfile, "w") as f5:
            for line in lines:
                splitline = line.split("\t")
                f5.write(splitline[0] + "\t" + splitline[2])