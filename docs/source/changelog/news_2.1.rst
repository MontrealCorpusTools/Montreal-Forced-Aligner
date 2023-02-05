
.. _whats_new_2_1:

What's new in 2.1
=================

Version 2.1 of the Montreal Forced Aligner changes the command line interface to use :xref:`click`, transition fully to use postgresql as the backend, expand functionality for segmentation and speaker diarization, and support the latest alpha release of Anchor, along with other improvements like fine-tuning alignments and generating phone-level confidences.  See :ref:`changelog_2.1` for a more specific changes.

.. _2_1_click:

Click
-----

MFA 2.1 uses :xref:`click` instead of Python's default argparse.  The primary benefit for this is in better validation of arguments and the ability to prompt the user if they forget an argument like :code:`dictionary_path`.

.. _2_1_postgresql:

Dependency on postgresql
------------------------

In 2.0, the default database backend was SQLite, which allowed for rapid development, but it lacks more advanced functionality in other database backends.  PostgreSQL was supported as well, but it requires a persistent running server rather a single database file for SQLite.  As more advanced functionality has been shifted to SQL over Python to speed up querying and processes, it made sense to drop SQLite support in favor of pure PostgreSQL.

.. _2_1_segmentation:

Segmentation with SpeechBrain
-----------------------------

In addition to the simple energy-based VAD used in Kaldi, MFA is capable of using :xref:`speechbrain`'s VAD model to generate better segmentation of long audio files. See :ref:`create_segments` for more information.

.. note::
   SpeechBrain is not installed by default with ``conda install montreal-forced-aligner``, so please refer to :ref:`installation` for more details.

.. _2_1_speaker_diarization:

Speaker diarization
-------------------

Speaker diarization has been overhauled to make use of pretrained ivector models or :xref:`speechbrain`'s `https://speechbrain.readthedocs.io/en/latest/API/speechbrain.pretrained.interfaces.html#speechbrain.pretrained.interfaces.SpeakerRecognition <SpeakerRecognition model>`_.  Additionally, with the :ref:`2_1_postgresql` in MFA 2.1, we have integrated with :xref:`pgvector`, which allows for storage and querying of utterance and speaker ivectors.

.. note::
   SpeechBrain is not installed by default with ``conda install montreal-forced-aligner``, so please refer to :ref:`installation` for more details.

.. _2_1_anchor_gui:

Anchor annotator GUI
--------------------

See also :ref:`anchor` for more information on using the annotation GUI.
