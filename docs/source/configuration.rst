
.. _configuration:

*************
Configuration
*************

Global configuration for MFA can be updated via the ``mfa configure`` subcommand. Once the command is called with a flag, it will set a default value for any future runs (though, you can overwrite most settings when you call other commands).

Options available:

.. option:: -t
               --temp_directory

   Set the default temporary directory

.. option:: -j
               --num_jobs

   Set the number of processes to use by default

.. option:: --always_clean

   Always remove files from previous runs by default

.. option:: --never_clean

   Don't remove files from previous runs by default

.. option:: --always_verbose

   Default to verbose output (outputs debug messages)

.. option:: --never_verbose

   Default to non-verbose output

   Default to verbose output (outputs debug messages)

.. option:: --always_debug

   Default to running debugging steps

.. option:: --never_debug

   Default to not running debugging steps

.. option:: --always_overwrite

   Always overwrite output files

.. option:: --never_overwrite

   Never overwrite output files (if file already exists, the output will be saved in the temp directory)

.. option:: --disable_mp

   Disable all multiprocessing (not recommended as it will usually increase processing times)

.. option:: --enable_mp

   Enable multiprocessing (recommended and enabled by default)

.. option:: --disable_textgrid_cleanup

   Disable postprocessing of TextGrids that cleans up silences and recombines compound words and clitics

.. option:: --enable_textgrid_cleanup

   Enable postprocessing of TextGrids that cleans up silences and recombines compound words and clitics

.. option:: -h
               --help

   Display help message for the command

.. toctree::
   :maxdepth: 2

   configuration_align.rst
   configuration_transcription.rst
   configuration_lm.rst
   configuration_segment.rst
   configuration_ivector.rst
   configuration_g2p.rst