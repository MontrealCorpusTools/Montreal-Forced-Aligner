
.. _configuration:

*************
Configuration
*************

Global configuration for MFA can be updated via the ``mfa configure`` subcommand. Once the command is called with a flag, it will set a default value for any future runs (though, you can overwrite most settings when you call other commands).

Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:parser
   :prog: mfa
   :start_command: configure

Configuring specific commands
=============================

.. toctree::
   :maxdepth: 1

   dictionary.rst
   align.rst
   transcription.rst
   lm.rst
   segment.rst
   ivector.rst
   g2p.rst
