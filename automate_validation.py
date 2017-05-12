import re
import os
import subprocess


GP_DIR = '/Volumes/data/corpora/GP_for_MFA/'

OUTPUT_DIR = '/Users/elias/MFA_testing/results'

exclude = re.compile("(AR)|(CH)|(BG)|(CZ)|(PL)|(RU)|(SA)|(UA)|(VN)|(SP)|(FR)|(CR)|(GE)|(PO)|(TU)")

for root, dirs, files in os.walk(GP_DIR):
    for filename in files:
        if re.match("(lexicon_nosil.txt)|(.*_dictionary.txt)", filename) is not None and exclude.search(root) is None:

            window_size = 4
            path_to_file = os.path.join(root, filename) 
            print(path_to_file)
            subprocess.call(['python3','-m','aligner.command_line.train_g2p', '--validate', 
                '--window_size=' + str(window_size), path_to_file, os.path.join(OUTPUT_DIR, 
                    filename.split(".")[0] + ".zip")])
            break




