
.. _transcribing:

Transcribe audio files ``(mfa transcribe)``
===========================================

MFA has some limited ability to use its acoustic and language models for performing transcription.  The intent of this functionality is largely to aid in offline corpus construction, and not as an online capability like most ASR systems.

.. seealso::

   See :ref:`train_acoustic_model` and :ref:`training_lm` details on training MFA models to use in transcription.

Unlike alignment, transcription does not require transcribed audio files (except when running in :ref:`transcription_evaluation`, but instead will use the combination of acoustic model, language model, and pronunciation dictionary to create a decoding lattice and find the best path through it. When training a language model for transcription, it is recommended to train one on text/speech transcripts that are in the same domain to minimize errors.

.. warning::

   The technology that MFA uses is several years out of date, and as such if you have other options available such as :xref:`coqui` or other production systems for :abbr:`STT (Speech to Text)`, we recommend using those.  The transcription capabilities are more here for completeness.

.. _transcription_evaluation:

Evaluation mode
---------------

Transcriptions can be compared to a gold-standard references by transcribing a corpus in the same format as for alignment (i.e., each sound file has a corresponding TextGrid or lab file).  Transcript will proceed as above, and then the resulting transcripts will be aligned with the gold transcriptions using the :mod:`Bio.pairwise2` alignment algorithm. From the aligned transcripts, Word Error Rate and Character Error Rate will be calculated for each utterance as follows:

.. math::

   Error \: rate = \frac{insertions + deletions + (2 * substitutions)} {length_{ref}}


Command reference
-----------------

.. click:: montreal_forced_aligner.command_line.transcribe:transcribe_corpus_cli
   :prog: mfa transcribe
   :nested: full

Configuration reference
-----------------------

- :ref:`transcribe_config`

API reference
-------------

- :ref:`transcription_api`
