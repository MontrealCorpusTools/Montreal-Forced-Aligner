.. _data_prep:

****************
Data preparation
****************

Prior to running the aligner, two aspects have to be set up.  First, a
pronunciation dictionary for your language should specify the pronunciations
of orthographic transcriptions.  Second, the sound files to aligned using
those pronunciations should have orthographic transcriptions for small
segments of audio.  These segments of audio can be either in individual
wav files or specified as intervals in a TextGrid for a longer sound file.
In general, segments should be less than ~ 30 seconds for better performance,
and the shorter the segments the better.  We recommend breaks like breaths
or silent pauses (i.e., not associated with a stop closure) to separate
the audio segments.


.. toctree::
   :maxdepth: 3

   dictionary.rst
   data_format.rst
