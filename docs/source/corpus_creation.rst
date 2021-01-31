.. _corpus_creation:

***************
Corpus creation
***************

MFA now contains several command line utilities for helping to create corpora from scratch.  The main workflow is as
follows:

1. If the corpus made up of long sound file that need segmenting, create segments
2. If the corpus does not contain transcriptions, transcribe utterances using existing acoustic models,
   language models, and dictionaries
3. Use the annotator tool to fix up any errors
4. As necessary, boot strap better transcriptions:

   1. Retrain language model with new fixed transcriptions
   2. Train dictionary pronunciation probabilities

.. toctree::
   :maxdepth: 3

   g2p_dictionary_generating.rst
   g2p_model_training.rst