.. _create_segments:

***************
Create segments
***************

The Montreal Forced Aligner can use Voice Activity Detection (VAD) capabilities from Kaldi to generate segments from
a longer sound file.

Steps to create segments:


1. Provided the steps in :ref:`installation` have been completed and you are in the same Conda/virtual environment that
   MFA was installed in.
2. Run the following command, substituting the arguments with your own paths:

  .. code-block:: bash

     mfa create_segments corpus_directory output_directory


.. note::

   The default configuration for VAD uses configuration values based on quiet speech. The algorithm is based on energy,
   so if your recordings are more noisy, you may need to adjust the configuration.  See :ref:`configuration_segments`
   for more information on changing these parameters.


Options available:

.. option:: -h
               --help

  Display help message for the command

.. option:: --config_path PATH

   Path to a YAML config file that will specify the alignment configuration. See
   :ref:`align_config` for more details.

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
