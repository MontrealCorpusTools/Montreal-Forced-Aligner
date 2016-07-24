.. _sound_files:

***********
Sound files
***********

The currently only support format for sound files is ``.wav``.  Sound files
must adhear to the following conditions:

Sampling rate
=============

Sound files can be aligned regardless of their sampling rate so long as
it is more than 16 kHz. Sound files to be aligned do not have to be all
a single sampling rate, but each speaker must have a consistent sampling
rate across their files.

Duration
========

In general, audio segments (sound files for Prosodylab-aligner format or intervals
for the TextGrid format) should be less than ~ 30 seconds for best performance
(the shorter the faster).  We recommend using breaks like breaths
or silent pauses (i.e., not associated with a stop closure) to separate the audio segments.


