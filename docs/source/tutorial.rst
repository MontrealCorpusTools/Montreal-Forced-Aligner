.. Montreal Forced Aligner documentation master file, created by
   sphinx-quickstart on Wed Jun 15 13:27:38 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tutorial
===================================================

Things you need before you can align:

1. Every .wav sound file you are aligning must have a corresponding .lab file which contains the text transcription of that .wav file.  The .wav and .lab files must have the same name. For example, if you have givrep_1027_2_1.wav, its transcription should be in givrep_1027_2_1.lab (which is just a text file with the .lab extension). If you have transcriptions in a tab-separated text file (or an Excel file which can be saved as one), you can generate .lab files from it using the relabel function of relabel_clean.py. The relabel_clean.py script is currently in the prosodylab.alignertools repository on GitHub.

2. These .lab files must be in the same format as the words in the dictionary (i.e. all capitalized for our dictionaries), and should ideally contain no punctuation.  (The aligner deals with punctuation for you.)  If your .lab files aren't in the correct format, you can use our relabel_clean.py script to clean your .lab files - this puts them into the correct format to work with our dictionaries.

3. You also need a pronunciation dictionary for the language you're aligning.  Our dictionaries for English and French are provided with the old Prosodylab Aligner (French is in prosodylab.alignermodels).  You can also write your own dictionary or download others.

Steps to align:

1. Open terminal, and change directory to montreal-forced-aligner.

2. type ./montreal-forced-aligner followed by the arguments described above in Usage.  (On Mac/Unix, to save time typing out the path, you can drag a folder from Finder into Terminal and it will put the full path to that folder into your command.)
    A template command:
    ./montreal-forced-aligner -s [#] [corpus-folder] [dictionary] [output-folder]
    This command will train a new model and align the files in [corpus-folder] using the file [dictionary], and save the output TextGrids to [output-folder].  It will take the first [#] characters of the file name to be the speaker ID number.
    
    An example command: 
    ./montreal-forced-aligner -s 7 ~/2_French_training ~/French/fr-QuEu.dict ~/2_French_training -f -v
    This command will train a new model and align the files in ~/2_French_training using the dictionary file ~/French/fr-QuEu.dict, and save the output TextGrids to ~/2_French_training.  It will take the first 7 characters of the file name to be the speaker ID number.  It will be fast (do half as many training iterations) and verbose (output more info to Terminal during training).

3. Once the aligner finishes, the resulting TextGrids will be in the specified output directory.  Training can take a couple hours for large datasets.


