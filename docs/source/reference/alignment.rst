
Alignment classes
=================

.. automodule:: montreal_forced_aligner.alignment

    .. autosummary::
       :toctree: generated/

       CorpusAligner -- Base aligner
       AdaptingAligner -- Adapting an acoustic model to new data
       PretrainedAligner -- Pretrained aligner
       DictionaryTrainer -- Train pronunciation probabilities from alignments

Mixins
------

.. automodule:: montreal_forced_aligner.alignment

    .. autosummary::
       :toctree: generated/

       AlignMixin -- Alignment mixin

Helper
------

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
   AlignArguments
   AliToCtmArguments
   CompileTrainGraphsArguments
   CompileInformationArguments
   PhoneCtmArguments
   CombineCtmArguments
   CleanupWordCtmArguments
   NoCleanupWordCtmArguments
   ExportTextGridArguments
   MapAccStatsArguments
   GeneratePronunciationsArguments
