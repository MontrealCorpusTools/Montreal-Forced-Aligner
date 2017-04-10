
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
    def __init__(self, path, input_dict, KO = False):
        super(Trainer, self).__init__()
        self.path = path
        if not KO:
            self.input_dict = input_dict
        else:
            self.input_dict = self.decompose(input_dict)
        generated_dict = Dictionary(input_dict, "")
        self.dictionary = {k: v[0] for k,v in generated_dict.words.items()}
        self.path_to_model = self.train(self.dictionary)

    def train(self, dictionary):
        os.environ["LANGUAGE"] = self.path

        training_file = tempfile.mkstemp()[1]
        os.environ["TRAINING_FILE"] = training_file

        with open(training_file,"w") as f2:
            [f2.write(k.strip() + "\t" + " ".join(list(v[0])) + "\n") for k,v in dictionary.items()]

        proc = subprocess.Popen("./execute.sh", shell=True).wait()
        
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

