import os
import re
from aligner.dictmaker.exceptions import EmptyDirectoryException

class BaseInspector(object):
    """Base class for inspector object, instantiated with a path to parent directory of TextGrid files"""
    def __init__(self, path, KO = None):
        super(BaseInspector, self).__init__()
        self.path = path
        self.KO = KO
        if KO is not None:

            self.KO = True

        self.files = self.get_files()
        self.words = self.get_words()
        self.output_path = os.path.join(os.path.dirname(__file__), "wordlist.lst")
        self.write_words(self.output_path)

        

    # def get_file_type(self):
        


    def get_files(self):
        return_files = []
        lab_regex = re.compile(".*\.(lab)|(LAB)")
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if lab_regex.match(file) is not None:
                    return_files.append(os.path.join(root,file))
                

        if len(return_files) == 0:
            raise EmptyDirectoryException
        return return_files

    def get_words(self):    
        words = set()
        punct_regex = re.compile("[.,;:?()!@#$%%^&*+\"]")
        for file in self.files:
            with open(file) as f1:
                content = f1.read()
            splitcontent = [re.sub(punct_regex, "", x) for x in re.split("\s", content)]
            if self.KO:
                from jamo import h2j, j2hcj
                splitcontent = [j2hcj(h2j(word)) for word in splitcontent]

            words |= set(splitcontent)
        return words

    def write_words(self, path):
        with open(path, "w") as f1:
            for word in self.words:
                f1.write(str(word) + "\n")


