import os
import shutil

from montreal_forced_aligner.alignment import PretrainedAligner
from montreal_forced_aligner.db import PhoneInterval, Utterance, WordInterval, WorkflowType


def test_align_sick(
    english_dictionary,
    english_acoustic_model,
    basic_corpus_dir,
    temp_dir,
    test_align_config,
    db_setup,
):
    temp_dir = os.path.join(temp_dir, "align_corpus_cli")
    shutil.rmtree(temp_dir, ignore_errors=True)
    a = PretrainedAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=english_dictionary,
        acoustic_model_path=english_acoustic_model,
        oov_count_threshold=1,
        temporary_directory=temp_dir,
        **test_align_config
    )
    a.align()
    export_directory = os.path.join(temp_dir, "test_align_export")
    shutil.rmtree(export_directory, ignore_errors=True)
    assert "AY_S" in a.phone_mapping
    a.export_files(export_directory)
    assert os.path.exists(os.path.join(export_directory, "michael", "acoustic_corpus.TextGrid"))


def test_align_one(
    english_dictionary,
    english_acoustic_model,
    basic_corpus_dir,
    temp_dir,
    test_align_config,
    db_setup,
):
    temp = os.path.join(temp_dir, "align_one_temp")
    a = PretrainedAligner(
        corpus_directory=basic_corpus_dir,
        dictionary_path=english_dictionary,
        acoustic_model_path=english_acoustic_model,
        temporary_directory=temp,
        debug=True,
        verbose=True,
        clean=True,
        **test_align_config
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
