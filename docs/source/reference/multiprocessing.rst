Multiprocessing helper functions
================================

.. automodule:: montreal_forced_aligner.multiprocessing.alignment

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
       compile_information
       compile_information_func
       convert_alignments
       convert_alignments_func
       convert_ali_to_textgrids
       compute_alignment_improvement
       compute_alignment_improvement_func
       compare_alignments
       PhoneCtmProcessWorker
       CleanupWordCtmProcessWorker
       NoCleanupWordCtmProcessWorker
       CombineProcessWorker
       ExportPreparationProcessWorker
       ExportTextGridProcessWorker
       calc_fmllr
       calc_fmllr_func
       calc_lda_mllt
       calc_lda_mllt_func
       create_align_model
       ctms_to_textgrids_mp
       lda_acc_stats
       lda_acc_stats_func
       train_map
       acc_stats_two_feats_func
       map_acc_stats_func
       parse_iteration_alignments
       ali_to_ctm_func
       compile_utterance_train_graphs_func
       test_utterances_func

.. automodule:: montreal_forced_aligner.multiprocessing.classes

    .. autosummary::
       :toctree: generated/

       Job
       AlignArguments
       VadArguments
       SegmentVadArguments
       CreateHclgArguments
       AccGlobalStatsArguments
       AccStatsArguments
       AccIvectorStatsArguments
       AccStatsTwoFeatsArguments
       AliToCtmArguments
       MfccArguments
       ScoreArguments
       DecodeArguments
       PhoneCtmArguments
       CombineCtmArguments
       CleanupWordCtmArguments
       NoCleanupWordCtmArguments
       LmRescoreArguments
       AlignmentImprovementArguments
       ConvertAlignmentsArguments
       CalcFmllrArguments
       CalcLdaMlltArguments
       GmmGselectArguments
       FinalFmllrArguments
       LatGenFmllrArguments
       FmllrRescoreArguments
       TreeStatsArguments
       LdaAccStatsArguments
       MapAccStatsArguments
       GaussToPostArguments
       InitialFmllrArguments
       ExtractIvectorsArguments
       ExportTextGridArguments
       CompileTrainGraphsArguments
       CompileInformationArguments
       CompileUtteranceTrainGraphsArguments
       MonoAlignEqualArguments
       TestUtterancesArguments
       CarpaLmRescoreArguments
       GeneratePronunciationsArguments

.. automodule:: montreal_forced_aligner.multiprocessing.corpus

    .. autosummary::
       :toctree: generated/

       CorpusProcessWorker

.. automodule:: montreal_forced_aligner.multiprocessing.features

    .. autosummary::
       :toctree: generated/

       mfcc
       mfcc_func
       calc_cmvn
       compute_vad
       compute_vad_func

.. automodule:: montreal_forced_aligner.multiprocessing.helper

    .. autosummary::
       :toctree: generated/

       Counter
       Stopped
       ProcessWorker
       run_mp
       run_non_mp

.. automodule:: montreal_forced_aligner.multiprocessing.ivector

    .. autosummary::
       :toctree: generated/

       gmm_gselect
       gmm_gselect_func
       gauss_to_post
       gauss_to_post_func
       acc_global_stats
       acc_global_stats_func
       acc_ivector_stats
       acc_ivector_stats_func
       extract_ivectors
       extract_ivectors_func
       segment_vad
       segment_vad_func
       get_initial_segmentation
       merge_segments

.. automodule:: montreal_forced_aligner.multiprocessing.pronunciations

    .. autosummary::
       :toctree: generated/

       generate_pronunciations
       generate_pronunciations_func

.. automodule:: montreal_forced_aligner.multiprocessing.transcription

    .. autosummary::
       :toctree: generated/

       create_hclgs
       create_hclg_func
       compose_hclg
       compose_clg
       compose_lg
       compose_g
       compose_g_carpa
       transcribe
       decode_func
       transcribe_fmllr
       initial_fmllr_func
       lat_gen_fmllr_func
       fmllr_rescore_func
       final_fmllr_est_func
       score_transcriptions
       score_func
       lm_rescore_func
       carpa_lm_rescore_func
