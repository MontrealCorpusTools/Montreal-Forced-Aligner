import pytest

from montreal_forced_aligner.diarization.speaker_diarizer import FOUND_SPEECHBRAIN
from montreal_forced_aligner.vad.segmenter import TranscriptionSegmenter


def test_segment_transcript(
    basic_corpus_dir,
    english_mfa_acoustic_model,
    english_us_mfa_reduced_dict,
    generated_dir,
    temp_dir,
    basic_segment_config_path,
    db_setup,
):
    if not FOUND_SPEECHBRAIN:
        pytest.skip("SpeechBrain not installed")
    segmenter = TranscriptionSegmenter(
        corpus_directory=basic_corpus_dir,
        dictionary_path=english_us_mfa_reduced_dict,
        acoustic_model_path=english_mfa_acoustic_model,
        speechbrain=True,
        en_activation_th=0.4,
        en_deactivation_th=0.4,
    )
    segmenter.setup()
    new_utterances = segmenter.segment_transcript(1)
    assert len(new_utterances) > 0
    segmenter.cleanup()
