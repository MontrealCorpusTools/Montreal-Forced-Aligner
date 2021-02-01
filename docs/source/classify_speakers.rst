.. _classify_speakers:

**********************
Speaker classification
**********************

The Montreal Forced Aligner can use trained ivector models (see :ref:`train_ivector` for more information about training
these models) to classify or cluster utterances according to speakers.

Steps to classify speakers:


1. Provided the steps in :ref:`installation` have been completed and you are in the same Conda/virtual environment that
   MFA was installed in.
2. Run the following command, substituting the arguments with your own paths:

  .. code-block:: bash

     mfa classify_speakers corpus_directory ivector_extractor_path output_directory

If the input uses TextGrids, the output TextGrids will have utterances sorted into tiers by each identified speaker. At
the moment, there is no way to retrain the classifier based on new data.

If the input corpus directory does not have TextGrids associated with them, then the speaker classifier will output
speaker directories with a text file that contains all the utterances that were classified.

Options available:

.. option:: -h
               --help

  Display help message for the command

.. option:: -t DIRECTORY
               --temp_directory DIRECTORY

   Temporary directory root to use for aligning, default is ``~/Documents/MFA``

.. option:: -j NUMBER
               --num_jobs NUMBER

  Number of jobs to use; defaults to 3, set higher if you have more
  processors available and would like to process faster

.. option:: -s NUMBER
               --num_speakers NUMBER

  Number of speakers to return.  If ``--cluster`` is present, this specifies the number of clusters.  Otherwise,
  MFA will sort speakers according to the first pass classification and then takes the top X speakers, and reclassify
  the utterances to only use those speakers.

.. option:: --cluster

  MFA will perform clustering of utterance ivectors into the number of speakers specified by ``--num_speakers``

.. option:: -v
               --verbose

  The aligner will print out more information if present

.. option:: -d
               --debug

  The aligner will run in debug mode

.. option:: -c
               --clean

  Forces removal of temporary files in ``~/Documents/MFA``