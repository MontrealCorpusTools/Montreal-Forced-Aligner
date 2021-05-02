
.. _sound_files:

***********
Sound files
***********

The default format for sound files in Kaldi is ``.wav``.  However, if you have :code:`sox` available on your machine,
MFA will use it to convert ``.flac``, ``.ogg`` and ``.aiff`` files to WAV for Kaldi to process.

.. note::

   Sound files will be ignored if there is no ``.lab`` or ``.TextGrid`` with the same name as the sound file. The validation
   utility (:ref:`validating_data`) will print a warning message when this happens and log all such files.

Sampling rate
=============

Feature generation for MFA uses a consistent frequency range (20-7800 Hz).  Files that are higher or lower sampling rate
than 16 kHz will be up- or down-sampled by default to 16 kHz during the feature generation procedure, which may produce artifacts for
upsampled files.  You can modify this sample rate as part of configuring features (see :ref:`feature_config` for more details).

.. note::

   The validation utility (:ref:`validating_data`) will note any ignored files, and the list of such files will be available in
   a log file.

Bit depth
=========

Kaldi can only process 16-bit WAV files.  Higher bit depths (24 and 32 bit) are getting more common for recording, so
MFA will automatically convert higher bit depths if you have :code:`sox` available on your machine.

Duration
========

In general, audio segments (sound files for Prosodylab-aligner format or intervals
for the TextGrid format) should be less than 30 seconds for best performance
(the shorter the faster).  We recommend using breaks like breaths
or silent pauses (i.e., not associated with a stop closure) to separate the audio segments.  For longer segments,
setting the beam and retry beam higher than their defaults will allow them to be aligned.  The default beam/retry beam is very
conservative 10/40, so something like 400/1000 will allow for much longer sequences to be aligned.  See :ref:`configuration`
for more details.


