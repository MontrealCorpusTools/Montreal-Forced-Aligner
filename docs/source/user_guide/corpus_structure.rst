

.. _corpus_structure:

****************************
Corpus formats and structure
****************************

Prior to running the aligner, make sure the following are set up:

1. A pronunciation dictionary for your language should specify the pronunciations of orthographic transcriptions.

2. The sound files to align.

3. Orthographic annotations in .lab files for individual sound files (:ref:`prosodylab_format`)
   or in TextGrid intervals for longer sound files (:ref:`textgrid_format`).

The sound files and the orthographic annotations should be contained in one directory structured as follows::

    +-- textgrid_corpus_directory
    |   --- recording1.wav
    |   --- recording1.TextGrid
    |   --- recording2.wav
    |   --- recording2.TextGrid
    |   --- ...

    +-- prosodylab_corpus_directory
    |   +-- speaker1
    |       --- recording1.wav
    |       --- recording1.lab
    |       --- recording2.wav
    |       --- recording2.lab
    |   +-- speaker2
    |       --- recording3.wav
    |       --- recording3.lab
    |   --- ...



.. note::

   A collection of preprocessing scripts to get various corpora of other formats is available in the :xref:`mfa_reorg_scripts` and :xref:`corpus_creation_scripts`.

Transcription file formats
==========================

In addition to the sections below about file format, see :ref:`text_normalization` for details on how the transcription text is normalized for dictionary look up, and :ref:`configuration_dictionary` for how this normalization can be customized.

.. _prosodylab_format:

Prosodylab-aligner format
-------------------------

Every audio file you are aligning must have a corresponding .lab
file containing the text transcription of that audio file.  The audio and
transcription files must have the same name. For example, if you have ``givrep_1027_2_1.wav``,
its transcription should be in ``givrep_1027_2_1.lab`` (which is just a
text file with the .lab extension).

.. note:: If you have transcriptions in a
   tab-separated text file (or an Excel file, which can be saved as one),
   you can generate .lab files from it using the relabel function of relabel_clean.py.
   The relabel_clean.py script is currently in the prosodylab.alignertools repository on GitHub.

If no ``.lab`` file is found, then the aligner will look for any matching ``.txt`` files and use those.

In terms of directory structure, the default configuration assumes that
files are separated into subdirectories based on their speaker (with one
speaker per file).

An alternative way to specify which speaker says which
segment is to use the ``-s`` flag with some number of characters of the file name as the speaker identifier.

The output from aligning this format of data will be TextGrids that have a tier
for words and a tier for phones.

.. _textgrid_format:

TextGrid format
---------------

The other main format that is supported is long sound files accompanied
by TextGrids that specify orthographic transcriptions for short intervals
of speech.


    .. figure:: ../_static/librispeech_textgrid.png
        :align: center
        :alt: Input TextGrid in Praat with intervals for each utterance and a single tier for a speaker

If the ``-s`` flag is specified, the tier names will not be used as speaker names, and instead the first X characters
specified by the flag will be used as the speaker name.

By default, each tier corresponds to a speaker (speaker "237" in the above example), so it is possible to
align speech for multiple speakers per sound file using this format.


    .. figure:: ../_static/multiple_speakers_textgrid.png
        :align: center
        :alt: Input TextGrid in Praat with intervals for each utterance and tiers for each speaker

Stereo files are supported as well, where it assumes that if there are
multiple talkers, the first half of speaker tiers are associated with the first
channel, and the second half of speaker tiers are associated with the second channel.

The output from aligning will be a TextGrid with word and phone tiers for
each speaker.

    .. figure:: ../_static/multiple_speakers_output_textgrid.png
        :align: center
        :alt: TextGrid in Praat following alignment with interval tiers for each speaker's words and phones

.. note::

   Intervals in the TextGrid less than 100 milliseconds will not be aligned.

Sound files
===========

The default format for sound files in Kaldi is ``.wav``.  However, if MFA is installed via conda, you should have :code:`sox` and/or :code:`ffmpeg` available which will pipe sound files of various formats to Kaldi in wav format.  Running :code:`sox` by itself will a list of formats that it supports. Of interest to speech researchers, the version on conda-forge supports non-standard :code:`wav` formats, :code:`aiff`, :code:`flac`, :code:`ogg`, and :code:`vorbis`.

.. note::

   ``.mp3`` files on Windows are converted to wav via ``ffmpeg`` rather than ``sox``.

   Likewise, :code:`opus` files can be processed using ``ffmpeg`` on all platforms

   Note that formats other than ``.wav`` have extra processing to convert them to ``.wav`` format before processing, particularly on Windows where ``ffmpeg`` is relied upon over ``sox``.  See :ref:`wav_conversion` for more details.

Sampling rate
-------------

Feature generation for MFA uses a consistent frequency range (20-7800 Hz).  Files that are higher or lower sampling rate than 16 kHz will be up- or down-sampled by default to 16 kHz during the feature generation procedure, which may produce artifacts for upsampled files.  You can modify this default sample rate as part of configuring features (see :ref:`feature_config` for more details).

Bit depth
---------

Kaldi can only process 16-bit WAV files.  Higher bit depths (24 and 32 bit) are getting more common for recording, so MFA will automatically convert higher bit depths via :code:`sox` or :code:`ffmpeg`.

Duration
--------

In general, audio segments (sound files for Prosodylab-aligner format or intervals for the TextGrid format) should be less than 30 seconds for best performance (the shorter the faster).  We recommend using breaks like breaths or silent pauses (i.e., not associated with a stop closure) to separate the audio segments.  For longer segments, setting the beam and retry beam higher than their defaults will allow them to be aligned.  The default beam/retry beam is very conservative 10/40, so something like 400/1000 will allow for much longer sequences to be aligned.  Though also note that the higher the beam value, the slower alignment will be as well.  See :ref:`configuration_global` for more details.
