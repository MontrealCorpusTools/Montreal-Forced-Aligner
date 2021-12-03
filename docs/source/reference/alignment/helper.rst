
Helper functionality
====================

Mixins
------

.. currentmodule:: montreal_forced_aligner.alignment.mixins

.. autosummary::
   :toctree: generated/

   AlignMixin -- Alignment mixin

Multiprocessing workers and functions
-------------------------------------

.. currentmodule:: montreal_forced_aligner.alignment.adapting

.. autosummary::
   :toctree: generated/

   map_acc_stats_func

.. currentmodule:: montreal_forced_aligner.alignment.multiprocessing

.. autosummary::
   :toctree: generated/

   align_func
   compile_train_graphs_func
   compile_information_func
   ali_to_ctm_func
   PhoneCtmProcessWorker
   CleanupWordCtmProcessWorker
   NoCleanupWordCtmProcessWorker
   CombineProcessWorker
   ExportPreparationProcessWorker
   ExportTextGridProcessWorker


Multiprocessing argument classes
--------------------------------

.. currentmodule:: montreal_forced_aligner.alignment.adapting

.. autosummary::
   :toctree: generated/

   MapAccStatsArguments

.. currentmodule:: montreal_forced_aligner.alignment.multiprocessing

.. autosummary::
   :toctree: generated/

   AlignArguments
   compile_train_graphs_func
   CompileTrainGraphsArguments
   compile_information_func
   CompileInformationArguments
   ali_to_ctm_func
   AliToCtmArguments
   PhoneCtmProcessWorker
   PhoneCtmArguments
   CleanupWordCtmProcessWorker
   CleanupWordCtmArguments
   NoCleanupWordCtmProcessWorker
   NoCleanupWordCtmArguments
   CombineProcessWorker
   CombineCtmArguments
   ExportPreparationProcessWorker
   ExportTextGridProcessWorker
   ExportTextGridArguments
