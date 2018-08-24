.. _sound_files:

***********
Sound files
***********

Currently, the only supported format for sound files is ``.wav``.  Sound files
must adhere to the following conditions:

.. note::

   Sound files will be ignored if there is no ``.lab`` or ``.TextGrid`` with the same name as the sound file. The validation
   utility (:ref:`validating_data`) will print a warning message when this happens and log all such files.

Sampling rate
=============

Sound files can be aligned regardless of their sampling rate so long as
it is more than 16 kHz. Sound files to be aligned do not have to be all
a single sampling rate, but each speaker must have a consistent sampling
rate across their files.

.. note::

   Sound files with sampling rate lower 16 kHz will currently be ignored by the aligner. The features that the aligner
   extracts are MFCCs derived from spectra from 0 to 7.8 kHz, so sound files must have Nyquist frequencies higher than 7.8 kHz.
   There is no significant difference between the features derived from a sound file sampled at 16 kHz and one sampled at 44.1 kHz.
   The validation utility (:ref:`validating_data`) will note any ignored files, and the list of such files will be available in
   a log file.

Duration
========

In general, audio segments (sound files for Prosodylab-aligner format or intervals
for the TextGrid format) should be less than 30 seconds for best performance
(the shorter the faster).  We recommend using breaks like breaths
or silent pauses (i.e., not associated with a stop closure) to separate the audio segments.  For longer segments,
setting the beam and retry beam higher than their defaults will allow them to be aligned.  The default beam/retry beam is very
conservative 10/40, so something like 400/1000 will allow for much longer sequences to be aligned.  See :ref:`configuration`
for more details.


