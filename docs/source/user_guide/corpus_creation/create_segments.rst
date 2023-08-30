
.. _create_segments:

Segment transcribed files ``(mfa segment)``
===========================================

The Montreal Forced Aligner can use Voice Activity Detection (VAD) capabilities from :xref:`speechbrain` to generate segments from
a longer sound file, while attempting to segment transcripts as well.  If you do not have transcripts, see :ref:`create_segments_vad`.

.. note::

   On Windows, if you get an ``OSError/WinError 1314`` during the run, follow `these instructions <https://www.scivision.dev/windows-symbolic-link-permission-enable/>`_ to enable symbolic link creation permissions.

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.create_segments:create_segments_cli
   :prog: mfa segment
   :nested: full


Configuration reference
-----------------------

- :ref:`configuration_segmentation`

API reference
-------------

- :ref:`segmentation_api`

.. _create_segments_vad:

Segment untranscribed files ``(mfa segment_vad)``
=================================================

The Montreal Forced Aligner can use Voice Activity Detection (VAD) capabilities from :xref:`speechbrain` or energy based VAD to generate segments from
a longer sound file.  This command does not split transcripts, instead assigning a default label of "speech" to all identified speech segments.  If you would like to preserve transcripts for each segment, see :ref:`create_segments`.

.. note::

   On Windows, if you get an ``OSError/WinError 1314`` during the run, follow `these instructions <https://www.scivision.dev/windows-symbolic-link-permission-enable/>`_ to enable symbolic link creation permissions.

Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.create_segments:create_segments_vad_cli
   :prog: mfa segment_vad
   :nested: full


Configuration reference
-----------------------

- :ref:`configuration_segmentation`

API reference
-------------

- :ref:`segmentation_api`
