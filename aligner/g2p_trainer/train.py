
import subprocess
import os
import tempfile
from aligner.dictionary import Dictionary



class Trainer(object):
    """Train a g2p model from a pronunciation dictionary

    Parameters
    ----------
    language: str
        the path and language code 
    input_dict : str
        path to the pronunciation dictionary

    """
    def __init__(self, language, input_dict):
        super(Validator, self).__init__()
        self.language = language
        self.input_dict = input_dict
        print("getting dict")
        generated_dict = Dictionary(input_dict, "")
        self.dictionary = {k: v[0] for k,v in generated_dict.words.items()}
        print("got dict")
        return self.train(self.dictionary)

    def train(self, dictionary):
        os.environ["LANGUAGE"] = self.language

        training_file = tempfile.mkstemp()[0]
        os.environ["TRAINING_FILE"] = training_file

        with open(training_file,"w") as f2:
            [f2.write(k.strip() + "\t" + v.strip() + "\n") for k,v in dictionary.items()]

        proc = subprocess.Popen("./execute.sh", shell=True).wait()
        
        return os.path.join(self.language, "full.fst")


