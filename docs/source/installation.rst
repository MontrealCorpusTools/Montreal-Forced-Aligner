
.. _`Montreal Forced Aligner releases`: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases

.. _`Kaldi GitHub repository`: https://github.com/kaldi-asr/kaldi

.. _`OpenGrm NGram library`: http://opengrm.org/NGramLibrary

.. _`Phonetisaurus`: https://github.com/AdolfVonKleist/Phonetisaurus

.. _installation:

************
Installation
************

All releases for the Montreal Forced Aligner are available on
`Montreal Forced Aligner releases`_.

Mac
===

.. note::
   Montreal Forced Aligner works on Mac OS X 10.11+

1. Download the zip archive for Mac and unzip the folder to any directory
2. Open a terminal window
3. Navigate to the ``montreal-forced-aligner`` folder (``cd /path/to/montreal-forced-aligner``)
4. Test the commands ``bin/mfa_align`` and ``bin/mfa_train_and_align`` (these should print usage messages about the commands)

Windows
=======

.. note::
   Montreal Forced Aligner has only been tested on Windows 10, but should
   work on Windows 7 and 8

1. Download the zip archive for Windows and unzip the folder to any directory
2. Open a command window (open the Start menu and search for ``cmd``)
3. Navigate to the ``montreal-forced-aligner`` folder (``cd C:\path\to\montreal-forced-aligner``;
   you can copy its path by holding Shift and right clicking on the folder, then
   selecting "Copy as path" and pasting it into the command prompt)
4. Test the commands ``bin\mfa_align`` and ``bin\mfa_train_and_align`` (these should print usage messages about the commands)

Linux
=====

The Linux distributions were built on Ubuntu 16.04, and so may not work on
machines that have older versions of Linux system packages.  If these instructions
do not work, then the executables will have to be built from source.

1. Download the tar.gz archive for Linux and untar the folder to any directory
2. Open a terminal window
3. Navigate to the ``montreal-forced-aligner`` folder (``cd /path/to/montreal-forced-aligner``)
4. Test the commands ``bin/mfa_align`` and ``bin/mfa_train_and_align`` (these should print usage messages about the commands)

.. note::

   The `gfortran` system package is required.

Building from source
====================

NB: These instructions require Python 3 (you may have to replace
instances of ``python`` and ``pip`` with ``python3`` and ``pip3`` if Python 3 is
not your default Python) and assume Linux in the commands.

1. Get kaldi compiled and working: `Kaldi GitHub repository`_
2. Download the source zip from the releases page
3. Open a terminal and go to the unzipped folder (``cd /path/to/Montreal-Forced-Aligner/thirdparty``)
4. Run the ``thirdparty/kaldi_binaries.py`` script, pointing it to the Kaldi repo (``python thirdparty/kaldi_binaries.py /path/to/kaldi``)
5. Run ``pip install -r requirements.txt`` to install the requirements for the aligner
6. Run the build script via ``python freezing/freeze.py``. There will now be a ``montreal-forced-aligner`` folder in the ``dist`` folder. This folder should contain a ``bin`` folder with the two executables ``mfa_align`` and ``mfa_train_and_align`` that should be used for alignment.

If you would like to use the G2P capabilities, you will also have to build the `OpenGrm NGram library`_ and `Phonetisaurus`_
and then run the ``thirdparty/opengrm_ngram_binaries.py`` and ``thirdparty/phonetisaurus_binaries.py`` pointing at the respective root directories.

There are some additional notes for compiling the various third party binaries available at
https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/tree/master/thirdparty.

Files created when using the Montreal Forced Aligner
====================================================

The aligner will save data and logs for the models it trains in a new folder,
``Documents/MFA`` (which it creates in your user's home directory).  If a model for a corpus already
exists in MFA, it will use any existing models if you try to align it again.
(If this is not desired, delete or move the old model folder.)  You can specify your own temporary directory by using the ``-t``
flag when calling the executable.

