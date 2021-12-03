
Helper functionality
====================

Mixins
------

.. currentmodule:: montreal_forced_aligner.acoustic_modeling.base

.. autosummary::
   :toctree: generated/

   AcousticModelTrainingMixin -- Basic mixin


Multiprocessing workers and functions
-------------------------------------

.. currentmodule:: montreal_forced_aligner.acoustic_modeling.base

.. autosummary::
   :toctree: generated/

   acc_stats_func
   compute_alignment_improvement_func
   compare_alignments


.. currentmodule:: montreal_forced_aligner.acoustic_modeling.monophone

.. autosummary::
   :toctree: generated/

   mono_align_equal_func


.. currentmodule:: montreal_forced_aligner.acoustic_modeling.triphone

.. autosummary::
   :toctree: generated/

   tree_stats_func
   convert_alignments_func


.. currentmodule:: montreal_forced_aligner.acoustic_modeling.lda

.. autosummary::
   :toctree: generated/

   lda_acc_stats_func
   calc_lda_mllt_func


.. currentmodule:: montreal_forced_aligner.acoustic_modeling.sat

.. autosummary::
   :toctree: generated/

   acc_stats_two_feats_func

Multiprocessing argument classes
--------------------------------

.. currentmodule:: montreal_forced_aligner.acoustic_modeling.base

.. autosummary::
   :toctree: generated/

   AccStatsArguments
   AlignmentImprovementArguments


.. currentmodule:: montreal_forced_aligner.acoustic_modeling.monophone

.. autosummary::
   :toctree: generated/

   MonoAlignEqualArguments


.. currentmodule:: montreal_forced_aligner.acoustic_modeling.triphone

.. autosummary::
   :toctree: generated/

   TreeStatsArguments
   ConvertAlignmentsArguments

.. currentmodule:: montreal_forced_aligner.acoustic_modeling.lda

.. autosummary::
   :toctree: generated/

   LdaAccStatsArguments
   CalcLdaMlltArguments


.. currentmodule:: montreal_forced_aligner.acoustic_modeling.sat

.. autosummary::
   :toctree: generated/

   AccStatsTwoFeatsArguments
