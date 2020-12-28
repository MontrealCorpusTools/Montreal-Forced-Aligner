.. _transcribing:

***********************
Running the transcriber
***********************

Steps to transcribe:

1. Provided the steps in :ref:`installation` have been completed and you are in the same Conda/virtual environment that
   MFA was installed in.

2. Run the following command, substituting the arguments with your own paths:

  .. code-block:: bash

     mfa transcribe corpus_directory dictionary_path acoustic_model_path language_model_path output_directory


.. note::

   ``acoustic_model_path`` can also be a language that has been pretrained by MFA developers.  For instance, to use
   the pretrained English model, first download it via :code:`mfa download acoustic english`.  A list of available
   acoustic models will be provided if you run :code:`mfa download acoustic`.  See :ref:`pretrained_models` for more details.

.. note::

   ``language_model_path`` should specify an ARPA-format ngram language model.  Note that the text should be all lower case
   in the arpa file, as MFA uses all lower case.

Options available:

.. option:: -h
               --help

  Display help message for the command

.. option:: --config_path PATH

   Path to a YAML config file that will specify the transcription configuration. See
   :ref:`transcribe_config` for more details.

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