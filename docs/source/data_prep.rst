
.. _`MFA-reorganization-scripts repository`: https://github.com/MontrealCorpusTools/MFA-reorganization-scripts
.. _data_prep:

****************
Data preparation
****************

Prior to running the aligner, make sure the following are set up:

1. A pronunciation dictionary for your language should specify the pronunciations of orthographic transcriptions.

2. The sound files to align.

3. Orthographic annotations in .lab files for individual sound files (Prosodylab-aligner format)
   or in TextGrid intervals for longer sound files (TextGrid format).

The sound files and the orthographic annotations should be contained in one directory structured as follows::

	+-- corpus_directory
	|	--- recording1.wav
	|	--- recording1.TextGrid
	|	--- recording2.wav
	|	--- recording2.TextGrid
	|	--- ...


.. note::

   A collection of preprocessing scripts to get various corpora of other formats is available in the
   `MFA-reorganization-scripts repository`_.

For details on how to organize each of these three components, see below.

.. toctree::
   :maxdepth: 3

   dictionary.rst
   sound_files.rst
   data_format.rst
