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

   create_hclg_func
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

   decode_func
   DecodeArguments
   lm_rescore_func
   LmRescoreArguments
   carpa_lm_rescore_func
   CarpaLmRescoreArguments
   score_func
   ScoreArguments

Speaker-adapted transcription
-----------------------------

.. currentmodule:: montreal_forced_aligner.transcription.multiprocessing

.. autosummary::
   :toctree: generated/

   initial_fmllr_func
   InitialFmllrArguments
   lat_gen_fmllr_func
   LatGenFmllrArguments
   fmllr_rescore_func
   FmllrRescoreArguments
   final_fmllr_est_func
   FinalFmllrArguments
