.. _corpus_creation:

*************************
Corpus creation utilities
*************************

MFA now contains several command line utilities for helping to create corpora from scratch.  The main workflow is as follows:

1. If the corpus made up of long sound file that need segmenting, :ref:`create_segments`
2. If the corpus does not contain transcriptions, transcribe utterances using existing acoustic models,
   language models, and dictionaries (:ref:`transcribing`)
3. Use the annotator tool to fix up any errors (:ref:`anchor`)
4. As necessary, bootstrap better transcriptions:

   1. Retrain language model with new fixed transcriptions (:ref:`training_lm`)
   2. Train dictionary pronunciation probabilities (:ref:`training_dictionary`)

.. toctree::
   :maxdepth: 3

   create_segments.rst
   train_ivector.rst
   classify_speakers.rst
   transcribing.rst
   training_lm.rst
   training_dictionary.rst
   anchor.rst
