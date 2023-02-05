.. _corpus_creation:

*************************
Corpus creation utilities
*************************

MFA now contains several command line utilities for helping to create corpora from scratch.  The main workflow is as follows:

1. If the corpus made up of long sound file that need segmenting, :ref:`segment the audio files using VAD <create_segments>`
2. If the corpus does not contain transcriptions, :ref:`transcribe utterances using existing acoustic models,
   language models, and dictionaries <transcribing>`
3. Use the :ref:`Anchor annotator tool <anchor>` to manually correct error in transcription
4. As necessary, bootstrap better transcriptions:

   1. :ref:`Train language model  <training_lm>` with updated transcriptions
   2. :ref:`Add pronunciation and silence probabilities to the dictionary <training_dictionary>`

.. toctree::
   :hidden:

   create_segments
   train_ivector
   diarize_speakers
   transcribing
   training_lm
   training_dictionary
   anchor
