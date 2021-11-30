.. _create_segments:

Create segments ``(mfa create_segments)``
=========================================

The Montreal Forced Aligner can use Voice Activity Detection (VAD) capabilities from Kaldi to generate segments from
a longer sound file.

.. note::

   The default configuration for VAD uses configuration values based on quiet speech. The algorithm is based on energy,
   so if your recordings are more noisy, you may need to adjust the configuration.  See :ref:`configuration_segmentation`
   for more information on changing these parameters.


Command reference
-----------------

.. autoprogram:: montreal_forced_aligner.command_line.mfa:create_parser()
   :prog: mfa
   :start_command: create_segments

Configuration reference
-----------------------

- :ref:`configuration_segmentation`

API reference
-------------

- :ref:`segmentation_api`
