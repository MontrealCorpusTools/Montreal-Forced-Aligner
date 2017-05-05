import os
import subprocess
import re
"""python3 -m aligner.command_line.train_g2p /Volumes/data/corpora/GP_for_MFA/FR/dict/FR_dictionary.txt 
/Users/elias/MFA_testing/results/GP_FR/model/FR.zip"""


for root, dirs, files in os.walk('/Volumes/data/corpora/GP_for_MFA/'):
    for filename in files:
        if re.match("(.*dictionary\.txt)|(lexicon_nosil\.txt)", filename) is not None:
            subprocess.call(['python3', '-m', 'aligner.command_line.train_g2p', 
                os.path.join(root, filename), os.path.join('/Users/elias/MFA_testing/', 
                os.path.split(os.path.split(root)[0])[1])])

            break

