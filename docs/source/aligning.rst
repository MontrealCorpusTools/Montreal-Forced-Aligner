.. _aligning:

.. _`LibriSpeech corpus`: http://www.openslr.org/12/

*******************
Running the aligner
*******************

.. _pretrained_alignment:

Align using pretrained models
-----------------------------

The Montreal Forced Aligner comes with :ref:`pretrained_acoustic` for several languages.

Steps to align:

1. Provided the steps in :ref:`installation` have been completed and you are in the same Conda/virtual environment that
   MFA was installed in.

2. Run the following command, substituting the arguments with your own paths:

  .. code-block:: bash

     mfa align corpus_directory dictionary_path acoustic_model_path output_directory


.. warning::

   Aligned TextGrids will overwrite any existing TextGrids (with the same name as the wav files) in the output directory.
   The aligner will throw an error if the corpus directory is specified as the output directory (to prevent overwriting
   any input TextGrids).

.. note::

   ``acoustic_model_path`` can also be a language that has been pretrained by MFA developers.  For instance, to use
   the pretrained English model, first download it via :code:`mfa download acoustic english`.  A list of available
   acoustic models will be provided if you run :code:`mfa download acoustic`.  See :ref:`pretrained_models` for more details.

.. note::
   On Mac/Unix, to save time typing out the path, you
   can drag a folder from Finder into Terminal and it will put the full
   path to that folder into your command.

   On Windows, you can hold Shift and right-click on a folder/file. Select
   "Copy as path..." and paste it into the command window.

Once the aligner finishes, the resulting TextGrids will be in the
specified output directory.

Options available:

.. option:: -h
               --help

  Display help message for the command

.. option:: --config_path PATH

   Path to a YAML config file that will specify the alignment configuration. See
   :ref:`align_config` for more details.

.. option:: -s NUMBER
               --speaker_characters NUMBER

   Number of characters to use to identify speakers; if not specified,
   the aligner assumes that the directory name is the identifier for the
   speaker.  Additionally, it accepts the value ``prosodylab`` to use the second field of a ``_`` delimited file name,
   following the convention of labelling production data in the ProsodyLab at McGill.

.. option:: -a PATH
               --audio_directory PATH

  Path to directory containing audio files if they are in a different root directory than the transcription files.
  Useful when you have multiple steps of a transcription/alignment pipeline

.. option:: -t DIRECTORY
               --temp_directory DIRECTORY

   Temporary directory root to use for aligning, default is ``~/Documents/MFA``

.. option:: -j NUMBER
               --num_jobs NUMBER

  Number of jobs to use; defaults to 3, set higher if you have more
  processors available and would like to align faster

.. option:: -v
               --verbose

  The aligner will print out more information if present

.. option:: -d
               --debug

  The aligner will run in debug mode

.. option:: -c
               --clean

  Forces removal of temporary files in ``~/Documents/MFA``
  prior to aligning.  This is good to use when aligning a new dataset,
  but it shares a name with a previously aligned dataset.  Cleaning automatically happens if the previous alignment
  run had an error.

.. _trained_alignment:

Align using only the data set
-----------------------------

Steps to align:

1. Provided the steps in :ref:`installation` have been completed and you are in the same Conda/virtual environment that
   MFA was installed in.

2. Run the following command, substituting the arguments with your own paths:

  .. code-block:: bash

     mfa train corpus_directory dictionary_path output_directory

.. warning::

   Aligned TextGrids will overwrite any existing TextGrids (with the same name as the wav files) in the output directory.
   The aligner will throw an error if the corpus directory is specified as the output directory (to prevent overwriting
   any input TextGrids).


Once the aligner finishes, the resulting TextGrids will be in the
specified output directory.  Training can take several hours for large datasets.

Options available:

.. option:: -h
               --help

  Display help message for the command

.. option:: --config_path PATH

   Path to a YAML config file that will specify either the alignment options or the training configuration. see
   :ref:`training_config` for more details.

.. option:: -o PATH
               --output_model_path PATH

  Path to a zip file to save the results' acoustic models
  from training to use in future aligning

.. option:: -a PATH
               --audio_directory PATH

  Path to directory containing audio files if they are in a different root directory than the transcription files.
  Useful when you have multiple steps of a transcription/alignment pipeline

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
  processors available and would like to align faster

.. option:: -v
               --verbose

  The aligner will print out more information if present

.. option:: -d
               --debug

  The aligner will run in debug mode

.. option:: -c
               --clean

  Forces removal of temporary files in ``~/Documents/MFA``
  prior to aligning.  This is good to use when aligning a new dataset,
  but it shares a name with a previously aligned dataset.  Cleaning automatically happens if the previous alignment
  run had an error.

.. _adapting_model:

Adapt pretrained models to new dataset
--------------------------------------

A recent 2.0 functionality for MFA is to adapt pretrained models to a new dataset.  MFA will first align the dataset using the pretrained model, and then perform a couple of rounds of speaker-adaptation training.

Steps to align:

1. Provided the steps in :ref:`installation` have been completed and you are in the same Conda/virtual environment that
   MFA was installed in.

2. Run the following command, substituting the arguments with your own paths:

  .. code-block:: bash

     mfa adapt corpus_directory dictionary_path acoustic_model_path output_model_path

3. Using the new model, run the alignment

  .. code-block:: bash

     mfa align corpus_directory dictionary_path output_model_path output_path


.. note::

   ``acoustic_model_path`` can also be a language that has been pretrained by MFA developers.  For instance, to use
   the pretrained English model, first download it via :code:`mfa download acoustic english`.  A list of available
   acoustic models will be provided if you run :code:`mfa download acoustic`.  See :ref:`pretrained_models` for more details.

Once the aligner finishes, the resulting TextGrids will be in the
specified output directory.

Options available:

.. option:: -h
               --help

  Display help message for the command

.. option:: --config_path PATH

   Path to a YAML config file that will specify the alignment configuration. See
   :ref:`align_config` for more details.

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
  processors available and would like to align faster

.. option:: -v
               --verbose

  The aligner will print out more information if present

.. option:: -d
               --debug

  The aligner will run in debug mode

.. option:: -c
               --clean

  Forces removal of temporary files in ``~/Documents/MFA``
  prior to aligning.  This is good to use when aligning a new dataset,
  but it shares a name with a previously aligned dataset.  Cleaning automatically happens if the previous alignment
  run had an error.
