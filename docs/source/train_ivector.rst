.. _train_ivector:

*****************************
Training an ivector extractor
*****************************

The Montreal Forced Aligner can train ivector extractors using an acoustic model for generating alignments.  As part
of this training process, a classifier is built in that can be used as part of :ref:`classify_speakers`.

Steps to train ivector extractor:

1. Provided the steps in :ref:`installation` have been completed and you are in the same Conda/virtual environment that
   MFA was installed in.
2. Run the following command, substituting the arguments with your own paths:

  .. code-block:: bash

     mfa train_ivector corpus_directory dictionary_path acoustic_model_path output_model_path


Options available:

.. option:: -h
               --help

  Display help message for the command

.. option:: --config_path PATH

   Path to a YAML config file that will specify the training configuration. See
   :ref:`configuration_ivector` for more details.

.. option:: -s NUMBER
               --speaker_characters NUMBER

   Number of characters to use to identify speakers; if not specified,
   the aligner assumes that the directory name is the identifier for the
   speaker.  Additionally, it accepts the value ``prosodylab`` to use the second field of a ``_`` delimited file name,
   following the convention of labelling production data in the ProsodyLab at McGill.

.. option:: -t DIRECTORY
               --temp_directory DIRECTORY

   Temporary directory root to use for aligning, default is ``~/Documents/MFA``

.. option:: -j NUMBER
               --num_jobs NUMBER

  Number of jobs to use; defaults to 3, set higher if you have more
  processors available and would like to process faster

.. option:: -v
               --verbose

  The aligner will print out more information if present

.. option:: -d
               --debug

  The aligner will run in debug mode

.. option:: -c
               --clean

  Forces removal of temporary files in ``~/Documents/MFA``
