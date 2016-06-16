.. Montreal Forced Aligner documentation master file, created by
   sphinx-quickstart on Wed Jun 15 13:27:38 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Common errors (and how to fix them)
===================================================


1. FileNotFoundError: [Errno 2] No such file or directory: '~/Documents/MFA/[corpus name]/train/split3/cmvndeltafeats.0':

    This happens when you do not delete the old folder for your training data in the MFA folder after a crash during training.  For example, if you were aligning ~/2_French_training, there would be a folder /Documents/MFA/2_French_training that the aligner created which you need to delete before trying to align again.

2. KeyError: 
    Error when you have files in the folder you're aligning (or its subfolders: i.e. old .lab files in a subfolder) that are missing something crucial such as a corresponding .lab/.wav file.

3. <unk> words in the output TextGrids for words that are in the dictionary:
    This is because your .lab files are not in the same format as the dictionary (i.e. all caps).  This can be fixed by running relabel_clean.py to clean your files if you're using our dictionary.  (relabel_clean.py is in prosodylab.alignertools on GitHub)


