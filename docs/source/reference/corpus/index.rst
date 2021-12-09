
.. _corpus_api:

Corpora
=======

.. currentmodule::  montreal_forced_aligner.corpus.acoustic_corpus

.. autosummary::
   :toctree: generated/

   AcousticCorpus

.. currentmodule::  montreal_forced_aligner.corpus.text_corpus

.. autosummary::
   :toctree: generated/

   TextCorpus

.. currentmodule::  montreal_forced_aligner.corpus.classes

.. autosummary::
   :toctree: generated/

   Speaker -- Class for collecting metadata about speakers in corpora
   File -- Class for representing sound file/transcription file pairs in corpora
   Utterance -- Class for collecting information about utterances

Helper classes and functions
============================

Collections
-----------

.. currentmodule::  montreal_forced_aligner.corpus.classes

.. autosummary::
   :toctree: generated/

   Collection
   SpeakerCollection
   FileCollection
   UtteranceCollection

Multiprocessing
---------------

.. currentmodule::  montreal_forced_aligner.corpus.multiprocessing

.. autosummary::
   :toctree: generated/

   Job
   CorpusProcessWorker

Mixins
------

.. currentmodule:: montreal_forced_aligner.corpus.base

.. autosummary::
   :toctree: generated/

   CorpusMixin

.. currentmodule:: montreal_forced_aligner.corpus.acoustic_corpus

.. autosummary::
   :toctree: generated/

   AcousticCorpusMixin
   AcousticCorpusPronunciationMixin

.. currentmodule:: montreal_forced_aligner.corpus.ivector_corpus

.. autosummary::
   :toctree: generated/

   IvectorCorpusMixin

.. currentmodule:: montreal_forced_aligner.corpus.text_corpus

.. autosummary::
   :toctree: generated/

   TextCorpusMixin
   DictionaryTextCorpusMixin

Features
--------

.. currentmodule:: montreal_forced_aligner.corpus.features

.. autosummary::
   :toctree: generated/

   FeatureConfigMixin
   mfcc_func
   MfccArguments
   calc_fmllr_func
   CalcFmllrArguments
   IvectorConfigMixin
   VadConfigMixin
   compute_vad_func
   VadArguments
   VadArguments

Ivector
-------

.. currentmodule:: montreal_forced_aligner.corpus.features

.. autosummary::
   :toctree: generated/

   extract_ivectors_func
   ExtractIvectorsArguments
