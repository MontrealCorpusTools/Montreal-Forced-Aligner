
.. _database_api:

Database
========

MFA uses a SQLite database to cache information during training/alignment runs.  An issue with training larger corpora was running into memory bottlenecks as all the information in the corpus was stored in memory, and fMLLR estimations in later stages would crash.  Additionally, there was always a trade off between storing results for use in other applications like :xref:`anchor` or providing diagnostic information to users, and ensuring that the core MFA workflows were as memory/time efficient as possible.  Offloading to a database frees up some memory, and makes some computations more efficient, and should be optimized enough to not slow down regular processing.

.. currentmodule::  montreal_forced_aligner.db

.. autosummary::
   :toctree: generated/

   Dictionary
   Dialect
   Word
   Pronunciation
   Phone
   Grapheme
   File
   TextFile
   SoundFile
   Speaker
   Utterance
   WordInterval
   PhoneInterval
   CorpusWorkflow
   PhonologicalRule
   RuleApplication
   Job
   M2MSymbol
   M2M2Job
   Word2Job
