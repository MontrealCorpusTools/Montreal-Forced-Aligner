
.. _create_segments:

Create segments ``(mfa create_segments)``
=========================================

The Montreal Forced Aligner can use Voice Activity Detection (VAD) capabilities from :xref:`speechbrain` to generate segments from
a longer sound file.

.. note::

   On Windows, if you get an ``OSError/WinError 1314`` during the run, follow `these instructions <https://www.scivision.dev/windows-symbolic-link-permission-enable/>`_ to enable symbolic link creation permissions.

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.create_segments:create_segments_cli
   :prog: mfa create_segments
   :nested: full


Configuration reference
-----------------------

- :ref:`configuration_segmentation`

API reference
-------------

- :ref:`segmentation_api`
