import os
import shutil

from montreal_forced_aligner.alignment import PretrainedAligner
from montreal_forced_aligner.data import WordType, WorkflowType
from montreal_forced_aligner.db import (
    File,
    Phone,
    PhoneInterval,
    Utterance,
    Word,
    WordInterval,
    bulk_update,
)
from montreal_forced_aligner.helper import align_words


def test_align_sick(
    english_dictionary,
    english_acoustic_model,
    basic_corpus_dir,
    temp_dir,
    test_align_config,
    db_setup,
):
    a = PretrainedAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=english_dictionary,
        acoustic_model_path=english_acoustic_model,
        oov_count_threshold=1,
        dither=0,
        **test_align_config,
    )
    a.align()
    assert a.dither == 0
    assert a.mfcc_options["dither"] == 0
    export_directory = os.path.join(temp_dir, "test_align_export")
    shutil.rmtree(export_directory, ignore_errors=True)
    a.export_files(export_directory)
    with a.session() as session:
        word_interval_count = (
            session.query(WordInterval)
            .join(WordInterval.word)
            .filter(Word.word_type != WordType.silence)
            .count()
        )
        assert word_interval_count == 370
    assert "AY1" in a.phone_mapping
    assert os.path.exists(os.path.join(export_directory, "michael", "acoustic_corpus.TextGrid"))
    a.cleanup()
    a.clean_working_directory()


def test_align_sick_mfa(
    english_us_mfa_dictionary,
    english_mfa_acoustic_model,
    basic_corpus_dir,
    temp_dir,
    test_align_config,
    db_setup,
):
    a = PretrainedAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=english_us_mfa_dictionary,
        acoustic_model_path=english_mfa_acoustic_model,
        oov_count_threshold=1,
        **test_align_config,
    )
    a.align()
    export_directory = os.path.join(temp_dir, "test_align_mfa_export")
    shutil.rmtree(export_directory, ignore_errors=True)
    a.export_files(export_directory)
    with a.session() as session:
        word_interval_count = (
            session.query(WordInterval)
            .join(WordInterval.word)
            .filter(Word.word_type != WordType.silence)
            .count()
        )
        assert word_interval_count == 374
    assert os.path.exists(os.path.join(export_directory, "michael", "acoustic_corpus.TextGrid"))
    a.cleanup()
    a.clean_working_directory()


def test_align_one(
    english_dictionary,
    english_acoustic_model,
    basic_corpus_dir,
    temp_dir,
    test_align_config,
    db_setup,
):
    a = PretrainedAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=english_dictionary,
        acoustic_model_path=english_acoustic_model,
        debug=True,
        verbose=True,
        clean=True,
        **test_align_config,
    )
    a.initialize_database()
    a.create_new_current_workflow(WorkflowType.online_alignment)
    a.setup()
    with a.session() as session:
        utterance = session.get(Utterance, 3)
        assert utterance.alignment_log_likelihood is None
        assert utterance.features is not None
        assert len(utterance.phone_intervals) == 0
        a.align_one_utterance(utterance, session)

    with a.session() as session:
        utterance = session.get(Utterance, 3)
        assert utterance.alignment_log_likelihood is not None
        assert len(utterance.phone_intervals) > 0

    with a.session() as session:
        session.query(Utterance).update({"features": None, "alignment_log_likelihood": None})
        session.query(PhoneInterval).delete()
        session.query(WordInterval).delete()
        session.commit()

    with a.session() as session:
        utterance = session.get(Utterance, 3)
        assert utterance.alignment_log_likelihood is None
        assert utterance.features is None
        assert len(utterance.phone_intervals) == 0
        a.align_one_utterance(utterance, session)

    with a.session() as session:
        utterance = session.get(Utterance, 3)
        assert utterance.alignment_log_likelihood is not None
        assert utterance.features is None
        assert len(utterance.phone_intervals) > 0
    a.cleanup()
    a.clean_working_directory()


def test_no_silence(
    english_us_mfa_reduced_dict,
    english_mfa_acoustic_model,
    pronunciation_variation_corpus,
    temp_dir,
    test_align_config,
    db_setup,
):
    a = PretrainedAligner(
        corpus_directory=pronunciation_variation_corpus,
        dictionary_path=english_us_mfa_reduced_dict,
        acoustic_model_path=english_mfa_acoustic_model,
        debug=True,
        verbose=True,
        clean=True,
        silence_probability=0.0,
        **test_align_config,
    )
    a.initialize_database()
    a.create_new_current_workflow(WorkflowType.online_alignment)
    a.setup()
    with a.session() as session:
        utterance = (
            session.query(Utterance)
            .join(Utterance.file)
            .filter(File.name == "mfa_erpause")
            .first()
        )
        assert utterance.alignment_log_likelihood is None
        assert utterance.features is not None
        assert len(utterance.phone_intervals) == 0
        print(a.silence_probability)
        print(a.lexicon_compilers[1].silence_probability)
        a.lexicon_compilers[1]._fst.set_input_symbols(a.lexicon_compilers[1].phone_table)
        a.lexicon_compilers[1]._fst.set_output_symbols(a.lexicon_compilers[1].word_table)
        print(a.lexicon_compilers[1]._fst)
        a.align_one_utterance(utterance, session)

    with a.session() as session:
        utterance = (
            session.query(Utterance)
            .join(Utterance.file)
            .filter(File.name == "mfa_erpause")
            .first()
        )
        silence_count = (
            session.query(PhoneInterval)
            .join(PhoneInterval.phone)
            .filter(Phone.phone == a.optional_silence_phone)
            .count()
        )
        assert silence_count == 0
        assert utterance.alignment_log_likelihood is not None
        assert len(utterance.phone_intervals) > 0
        assert (
            len(
                [x for x in utterance.phone_intervals if x.phone.phone != a.optional_silence_phone]
            )
            > 0
        )

    a.cleanup()
    a.clean_working_directory()


def test_transcript_verification(
    filler_insertion_corpus,
    english_us_mfa_reduced_dict,
    english_mfa_acoustic_model,
    temp_dir,
    db_setup,
    reference_transcripts,
):
    a = PretrainedAligner(
        corpus_directory=filler_insertion_corpus,
        dictionary_path=english_us_mfa_reduced_dict,
        acoustic_model_path=english_mfa_acoustic_model,
        boost_silence=3.0,
        acoustic_scale=0.0833,
        self_loop_scale=1.0,
        transition_scale=1.0,
        use_cutoff_model=True,
        uses_speaker_adaptation=False,
    )
    a.initialize_database()
    a.create_new_current_workflow(WorkflowType.transcript_verification)
    a.setup()
    with a.session() as session:
        update_mappings = []
        counts = {"uh": 10, "um": 10, "hm": 1}
        for w in session.query(Word).filter(Word.word.in_(["uh", "um", "hm"])):
            update_mappings.append(
                {"id": w.id, "word_type": WordType.interjection, "count": counts[w.word]}
            )
        bulk_update(session, Word, update_mappings)
        session.commit()
    a.verify_transcripts()
    export_directory = os.path.join(temp_dir, "test_transcript_verification_export")
    shutil.rmtree(export_directory, ignore_errors=True)
    a.export_files(export_directory)
    successes = []
    with a.session() as session:
        utterances = session.query(Utterance).all()
        for utterance in utterances:
            if utterance.file_name in {"mfa_breaths"}:
                continue
            print("FILE:", utterance.file_name)
            print("REFERENCE:", reference_transcripts[utterance.file_name])
            print("ORIGINAL: ", utterance.normalized_text)
            word_intervals = (
                session.query(WordInterval)
                .join(WordInterval.word)
                .filter(
                    WordInterval.utterance_id == utterance.id,
                    Word.word_type != WordType.silence,
                    WordInterval.end - WordInterval.begin > 0.03,
                )
                .order_by(WordInterval.begin)
            )
            generated = " ".join(x.word.word for x in word_intervals)
            extra_duration, wer, aligned_duration = align_words(
                utterance.normalized_text.split(),
                [x.as_ctm() for x in word_intervals],
                "<eps>",
                debug=True,
            )

            print("FILE:", utterance.file_name)
            print("LOG LIKELIHOOD:", utterance.alignment_log_likelihood)
            print("DURATION DEVIATION:", utterance.duration_deviation)
            if reference_transcripts[utterance.file_name] != generated:
                print("VERIFIED: ", generated)
                print(wer, extra_duration)
            else:
                successes.append(utterance.file_name)
            if reference_transcripts[utterance.file_name] == utterance.normalized_text:
                assert utterance.duration_deviation == 0
                assert utterance.word_error_rate == 0
            else:
                assert utterance.duration_deviation > 0
                assert utterance.word_error_rate > 0

        print(f"Successful: {successes} of {len(utterances)}")
