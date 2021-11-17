Alignment
=========

Basic
-----

.. currentmodule:: montreal_forced_aligner.multiprocessing.alignment

.. autosummary::
   :toctree: generated/

   acc_stats
   acc_stats_func
   align
   align_func
   mono_align_equal
   mono_align_equal_func
   tree_stats
   tree_stats_func
   compile_train_graphs
   compile_train_graphs_func
   convert_alignments
   convert_alignments_func

LDA training
------------

.. currentmodule:: montreal_forced_aligner.multiprocessing.alignment

.. autosummary::
   :toctree: generated/

   calc_lda_mllt
   calc_lda_mllt_func
   lda_acc_stats
   lda_acc_stats_func

Speaker adapted models
----------------------

.. currentmodule:: montreal_forced_aligner.multiprocessing.alignment

.. autosummary::
   :toctree: generated/

   calc_fmllr
   calc_fmllr_func
   create_align_model
   acc_stats_two_feats_func

Acoustic model adaptation
-------------------------

.. currentmodule:: montreal_forced_aligner.multiprocessing.alignment

.. autosummary::
   :toctree: generated/

   train_map
   map_acc_stats_func


TextGrid Export
---------------

.. currentmodule:: montreal_forced_aligner.multiprocessing.alignment

.. autosummary::
   :toctree: generated/

   ctms_to_textgrids_mp
   convert_ali_to_textgrids
   ali_to_ctm_func
   PhoneCtmProcessWorker
   CleanupWordCtmProcessWorker
   NoCleanupWordCtmProcessWorker
   CombineProcessWorker
   ExportPreparationProcessWorker
   ExportTextGridProcessWorker

Pronunciation probabilities
---------------------------

.. currentmodule:: montreal_forced_aligner.multiprocessing.pronunciations

.. autosummary::
   :toctree: generated/

   generate_pronunciations
   generate_pronunciations_func

Validation
----------

.. currentmodule:: montreal_forced_aligner.multiprocessing.alignment

.. autosummary::
   :toctree: generated/

   compile_information
   compile_information_func
   compute_alignment_improvement
   compute_alignment_improvement_func
   compare_alignments
   parse_iteration_alignments
   compile_utterance_train_graphs_func
   test_utterances_func
