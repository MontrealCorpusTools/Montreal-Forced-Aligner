.. _data_prep:

****************
Data preparation
****************

Prior to running the aligner, make sure the following are set up:

1. A pronunciation dictionary for your language should specify the pronunciations
of orthographic transcriptions.  

2. The sound files to align using
those pronunciations should have orthographic transcriptions for small
segments of audio.  These audio segments can either be in individual
.wav files or specified as intervals in a TextGrid for a longer sound file.
In general, audio segments should be less than ~ 30 seconds for best performance
(the shorter the better).  We recommend using breaks like breaths
or silent pauses (i.e., not associated with a stop closure) to separate the audio segments.


.. toctree::
   :maxdepth: 3

   dictionary.rst
   data_format.rst
