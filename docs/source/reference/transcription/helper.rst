Helper functions
================

Mixins
------

.. currentmodule:: montreal_forced_aligner.transcription.transcriber

.. autosummary::
  :toctree: generated/

  TranscriberMixin

Decoding graph
--------------

.. currentmodule:: montreal_forced_aligner.transcription.multiprocessing

.. autosummary::
   :toctree: generated/

   CreateHclgFunction
   CreateHclgArguments
   compose_hclg
   compose_clg
   compose_lg
   compose_g
   compose_g_carpa


Speaker-independent transcription
---------------------------------

.. currentmodule:: montreal_forced_aligner.transcription.multiprocessing

.. autosummary::
   :toctree: generated/

   DecodeFunction
   DecodeArguments
   LmRescoreFunction
   LmRescoreArguments
   CarpaLmRescoreFunction
   CarpaLmRescoreArguments

Speaker-adapted transcription
-----------------------------

.. currentmodule:: montreal_forced_aligner.transcription.multiprocessing

.. autosummary::
   :toctree: generated/

   InitialFmllrFunction
   InitialFmllrArguments
   LatGenFmllrFunction
   LatGenFmllrArguments
   FmllrRescoreFunction
   FmllrRescoreArguments
   FinalFmllrFunction
   FinalFmllrArguments
