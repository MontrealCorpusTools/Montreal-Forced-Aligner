import os
import shutil
import unicodedata

from montreal_forced_aligner import config
from montreal_forced_aligner.corpus.acoustic_corpus import (
    AcousticCorpus,
    AcousticCorpusWithPronunciations,
)
from montreal_forced_aligner.corpus.classes import FileData, UtteranceData
from montreal_forced_aligner.corpus.helper import get_wav_info
from montreal_forced_aligner.corpus.text_corpus import DictionaryTextCorpus, TextCorpus
from montreal_forced_aligner.data import TextFileType, WordType
from montreal_forced_aligner.db import Word


def test_mp3(mp3_test_path):
    info = get_wav_info(str(mp3_test_path))
    assert info.duration > 0


def test_opus(opus_test_path):
    info = get_wav_info(str(opus_test_path))
    assert info.duration > 0


def test_add(basic_corpus_dir, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests")
    config.TEMPORARY_DIRECTORY = output_directory
    corpus = AcousticCorpus(
        corpus_directory=basic_corpus_dir,
    )
    print(corpus.db_string)
    corpus.load_corpus()
    with corpus.session() as session:
        new_speaker = "new_speaker"
        corpus.add_speaker(new_speaker, session)
        new_file_name = "new_file"
        new_file = FileData(
            text_path=os.path.join(basic_corpus_dir, "michael", "acoustic_corpus.lab"),
            text_type=TextFileType.NONE,
            name=new_file_name,
            relative_path="",
            wav_path=None,
        )
        new_utterance = UtteranceData(new_speaker, new_file_name, 0, 1, text="blah blah")
        assert len(corpus.get_utterances(file=new_file_name, speaker=new_speaker)) == 0

        corpus.add_file(new_file, session)
        session.commit()
        corpus.add_utterance(new_utterance, session)
        session.commit()

    utts = corpus.get_utterances(file=new_file_name, speaker=new_speaker)
    assert len(utts) == 1
    assert utts[0].text == "blah blah"
    corpus.delete_utterance(utts[0].id)
    assert len(corpus.get_utterances(file=new_file_name, speaker=new_speaker)) == 0
    corpus.cleanup_connections()


def test_basic_txt(basic_corpus_txt_dir, basic_dict_path, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    config.TEMPORARY_DIRECTORY = output_directory
    corpus = AcousticCorpus(
        corpus_directory=basic_corpus_txt_dir, use_pitch=True, use_voicing=True
    )
    corpus.load_corpus()

    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 45
    corpus.cleanup_connections()


def test_acoustic_from_temp(basic_corpus_txt_dir, basic_dict_path, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests_acoustic_from_temp")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    config.TEMPORARY_DIRECTORY = output_directory
    config.CLEAN = True
    corpus = AcousticCorpus(
        corpus_directory=basic_corpus_txt_dir,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    corpus.cleanup_connections()
    config.CLEAN = False

    new_corpus = AcousticCorpus(
        corpus_directory=basic_corpus_txt_dir,
    )
    new_corpus.load_corpus()
    assert len(new_corpus.no_transcription_files) == 0
    assert new_corpus.get_feat_dim() == 39
    config.CLEAN = True
    new_corpus.cleanup_connections()


def test_text_corpus_from_temp(basic_corpus_txt_dir, basic_dict_path, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests_text_from_temp")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    config.TEMPORARY_DIRECTORY = output_directory
    config.CLEAN = True
    corpus = TextCorpus(
        corpus_directory=basic_corpus_txt_dir,
    )
    corpus.load_corpus()
    assert corpus.num_utterances > 0
    corpus.cleanup_connections()
    config.CLEAN = False

    new_corpus = TextCorpus(
        corpus_directory=basic_corpus_txt_dir,
    )
    new_corpus.load_corpus()
    assert new_corpus.num_utterances > 0
    config.CLEAN = True
    new_corpus.cleanup_connections()


def test_extra(basic_dict_path, extra_corpus_dir, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests", "extra")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    config.TEMPORARY_DIRECTORY = output_directory

    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=extra_corpus_dir,
        dictionary_path=basic_dict_path,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    corpus.cleanup_connections()


def test_stereo(basic_dict_path, stereo_corpus_dir, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests", "stereo")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    config.TEMPORARY_DIRECTORY = output_directory

    corpus = AcousticCorpus(
        corpus_directory=stereo_corpus_dir,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert corpus.get_file(name="michaelandsickmichael").num_channels == 2
    corpus.cleanup_connections()


def test_stereo_short_tg(basic_dict_path, stereo_corpus_short_tg_dir, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests", "stereo_short")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    config.TEMPORARY_DIRECTORY = output_directory

    corpus = AcousticCorpus(
        corpus_directory=stereo_corpus_short_tg_dir,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert corpus.get_file(name="michaelandsickmichael").num_channels == 2
    corpus.cleanup_connections()


def test_audio_directory(basic_dict_path, basic_split_dir, generated_dir, db_setup):
    audio_dir, text_dir = basic_split_dir
    output_directory = generated_dir.joinpath("corpus_tests", "audio_dir")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    config.TEMPORARY_DIRECTORY = output_directory

    corpus = AcousticCorpus(
        corpus_directory=text_dir,
        audio_directory=audio_dir,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert corpus.num_files > 0
    corpus.cleanup_connections()


def test_flac(basic_dict_path, flac_corpus_dir, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests", "flac")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    config.TEMPORARY_DIRECTORY = output_directory
    config.USE_MP = False

    corpus = AcousticCorpus(
        corpus_directory=flac_corpus_dir,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    corpus.cleanup_connections()


def test_flac_mp(basic_dict_path, flac_corpus_dir, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests", "flac_mp")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    config.USE_MP = True
    config.TEMPORARY_DIRECTORY = output_directory

    corpus = AcousticCorpus(
        corpus_directory=flac_corpus_dir,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert corpus.num_files > 0
    corpus.cleanup_connections()


def test_flac_tg(basic_dict_path, flac_tg_corpus_dir, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests", "flac_no_mp")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    config.TEMPORARY_DIRECTORY = output_directory
    config.USE_MP = False

    corpus = AcousticCorpus(
        corpus_directory=flac_tg_corpus_dir,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert corpus.num_files > 0
    corpus.cleanup_connections()


def test_flac_tg_mp(basic_dict_path, flac_tg_corpus_dir, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests", "flac_tg_mp")
    config.TEMPORARY_DIRECTORY = output_directory
    config.USE_MP = True
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=flac_tg_corpus_dir,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert corpus.num_files > 0
    corpus.cleanup_connections()


def test_24bit_wav(transcribe_corpus_24bit_dir, basic_dict_path, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests", "24bit")
    config.TEMPORARY_DIRECTORY = output_directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=transcribe_corpus_24bit_dir,
    )
    corpus.transcriptions_required = False
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 2
    assert corpus.get_feat_dim() == 39
    assert corpus.num_files > 0
    corpus.cleanup_connections()


def test_short_segments(shortsegments_corpus_dir, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests", "short_segments")
    config.TEMPORARY_DIRECTORY = output_directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=shortsegments_corpus_dir,
    )
    corpus.load_corpus()
    assert corpus.num_utterances == 3
    assert len([x for x in corpus.utterances() if not x.ignored]) == 2
    assert len([x for x in corpus.utterances() if x.features is not None]) == 2
    assert len([x for x in corpus.utterances() if x.ignored]) == 1
    assert len([x for x in corpus.utterances() if x.features is None]) == 1
    corpus.cleanup_connections()


def test_speaker_groupings(
    multilingual_ipa_corpus_dir, generated_dir, english_us_mfa_dictionary, db_setup
):
    output_directory = generated_dir.joinpath("corpus_tests", "speaker_groupings")
    config.TEMPORARY_DIRECTORY = output_directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=multilingual_ipa_corpus_dir,
        dictionary_path=english_us_mfa_dictionary,
    )
    corpus.load_corpus()
    with corpus.session() as session:
        files = corpus.files(session)
        print(files)
        assert files.count() > 0
        for _, _, file_listing in os.walk(multilingual_ipa_corpus_dir):
            for f in file_listing:
                name, ext = os.path.splitext(f)
                for file in files:
                    if name == file.name:
                        break
                else:
                    raise Exception(f"File {name} not loaded")
    corpus.cleanup_connections()
    del corpus
    new_corpus = AcousticCorpusWithPronunciations(
        corpus_directory=multilingual_ipa_corpus_dir,
        dictionary_path=english_us_mfa_dictionary,
    )
    new_corpus.load_corpus()
    files = new_corpus.files()
    assert files.count() > 0
    for _, _, file_listing in os.walk(multilingual_ipa_corpus_dir):
        for f in file_listing:
            name, ext = os.path.splitext(f)
            for file in files:
                if name == file.name:
                    break
            else:
                raise Exception(f"File {name} not loaded")
    new_corpus.cleanup_connections()


def test_subset(multilingual_ipa_corpus_dir, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests", "subset")
    config.TEMPORARY_DIRECTORY = output_directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=multilingual_ipa_corpus_dir,
    )
    corpus.load_corpus()
    sd = corpus.split_directory

    s = corpus.subset_directory(5)
    assert os.path.exists(sd)
    assert os.path.exists(s)
    corpus.cleanup_connections()


def test_weird_words(weird_words_dir, generated_dir, basic_dict_path, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests", "weird_words")
    config.TEMPORARY_DIRECTORY = output_directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=weird_words_dir,
        dictionary_path=basic_dict_path,
    )
    corpus.load_corpus()
    with corpus.session() as session:
        w = (
            session.query(Word)
            .filter(Word.dictionary_id == corpus._default_dictionary_id)
            .filter(Word.word == "i’m")
            .first()
        )
        assert w is None
        w = (
            session.query(Word)
            .filter(Word.dictionary_id == corpus._default_dictionary_id)
            .filter(Word.word == "’m")
            .first()
        )
        assert w is None
        w = (
            session.query(Word)
            .filter(Word.dictionary_id == corpus._default_dictionary_id)
            .filter(Word.word == "i'm")
            .first()
        )
        pronunciations = [x.pronunciation for x in w.pronunciations]
        assert "ay m ih" in pronunciations
        assert "ay m" in pronunciations
        w = (
            session.query(Word)
            .filter(Word.dictionary_id == corpus._default_dictionary_id)
            .filter(Word.word == "'m")
            .first()
        )
        pronunciations = [x.pronunciation for x in w.pronunciations]
        assert "m" in pronunciations
        print(corpus.utterances())
        weird_words = corpus.get_utterances(file="weird_words")[0]
        print(weird_words.text)
        print(weird_words.oovs)
        oovs = [
            x[0]
            for x in session.query(Word.word).filter(
                Word.word_type == WordType.oov, Word.dictionary_id == corpus._default_dictionary_id
            )
        ]
        print(oovs)
        assert all(
            x in oovs
            for x in {
                "ajfish",
                "asds-asda",
                "sdasd",
            }
        )
    assert (
        weird_words.text
        == "i’m talking-ajfish me-really [me-really] [me'really] [me_??_really] asds-asda sdasd-me <s> </s>"
    )
    print(weird_words.normalized_text.split())
    assert weird_words.normalized_text.split() == [
        "i'm",
        "talking",
        "ajfish",
        "me",
        "really",
        "[bracketed]",
        "[bracketed]",
        "[bracketed]",
        "asds-asda",
        "sdasd",
        "me",
        "<unk>",
        "<unk>",
    ]
    print(oovs)
    assert "'m" not in oovs
    corpus.cleanup_connections()


def test_punctuated(punctuated_dir, generated_dir, english_us_mfa_dictionary, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests", "punctuated")
    config.TEMPORARY_DIRECTORY = output_directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=punctuated_dir,
        dictionary_path=english_us_mfa_dictionary,
    )
    corpus.load_corpus()
    print(corpus.files())
    print(corpus.utterances())

    punctuated = corpus.get_utterances(file="punctuated")[0]
    assert (
        punctuated.text
        == "oh yes, they - they, you know, they love her' and so' 'something 'i mean... ‘you The village name is Anglo Saxon in origin, and means 'Myrsa's woodland'."
    )
    assert (
        punctuated.normalized_text
        == "oh yes they they you know they love her and so something i mean you the village name is anglo saxon in origin and means myrsa 's woodland"
    )
    corpus.cleanup_connections()


def test_alternate_punctuation(
    punctuated_dir,
    generated_dir,
    basic_dict_path,
    different_punctuation_config_path,
    db_setup,
):
    from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner

    output_directory = generated_dir.joinpath("corpus_tests", "alternate")
    config.TEMPORARY_DIRECTORY = output_directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    params, skipped = AcousticCorpusWithPronunciations.extract_relevant_parameters(
        TrainableAligner.parse_parameters(different_punctuation_config_path)
    )
    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=punctuated_dir,
        dictionary_path=basic_dict_path,
        **params,
    )
    corpus.load_corpus()
    punctuated = corpus.get_utterances(file="punctuated")[0]
    assert (
        punctuated.text
        == "oh yes, they - they, you know, they love her' and so' 'something 'i mean... ‘you The village name is Anglo Saxon in origin, and means 'Myrsa's woodland'."
    )
    corpus.cleanup_connections()


def test_no_punctuation(
    punctuated_dir,
    generated_dir,
    basic_dict_path,
    no_punctuation_config_path,
    db_setup,
):
    from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner

    output_directory = generated_dir.joinpath("corpus_tests", "no_punctuation")
    config.TEMPORARY_DIRECTORY = output_directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    params, skipped = AcousticCorpusWithPronunciations.extract_relevant_parameters(
        TrainableAligner.parse_parameters(no_punctuation_config_path)
    )
    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=punctuated_dir,
        dictionary_path=basic_dict_path,
        **params,
    )
    assert not corpus.punctuation
    assert not corpus.compound_markers
    assert not corpus.clitic_markers
    corpus.load_corpus()
    punctuated = corpus.get_utterances(file="punctuated")[0]
    print(corpus.punctuation)
    print(corpus.word_break_markers)
    assert (
        punctuated.text
        == "oh yes, they - they, you know, they love her' and so' 'something 'i mean... ‘you The village name is Anglo Saxon in origin, and means 'Myrsa's woodland'."
    )
    assert (
        punctuated.normalized_text
        == "oh yes, they - they, you know, they love her' and so' 'something 'i mean... ‘you the village name is anglo saxon in origin, and means 'myrsa's woodland'."
    )
    weird_words = corpus.get_utterances(file="weird_words")[0]
    assert (
        weird_words.text
        == "i’m talking-ajfish me-really [me-really] [me'really] [me_??_really] asds-asda sdasd-me <s> </s>"
    )
    print(weird_words.normalized_text)
    assert weird_words.normalized_text.split() == [
        "i’m",
        "talking-ajfish",
        "me-really",
        "[bracketed]",
        "[bracketed]",
        "[bracketed]",
        "asds-asda",
        "sdasd-me",
        "<unk>",
        "<unk>",
    ]
    corpus.cleanup_connections()


def test_xsampa_corpus(
    xsampa_corpus_dir,
    xsampa_dict_path,
    generated_dir,
    different_punctuation_config_path,
    db_setup,
):
    from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner

    output_directory = generated_dir.joinpath("corpus_tests", "xsampa")
    config.TEMPORARY_DIRECTORY = output_directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    params, skipped = AcousticCorpusWithPronunciations.extract_relevant_parameters(
        TrainableAligner.parse_parameters(different_punctuation_config_path)
    )
    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=xsampa_corpus_dir,
        dictionary_path=xsampa_dict_path,
        **params,
    )
    corpus.load_corpus()
    xsampa = corpus.get_utterances(file="xsampa")[0]
    assert (
        xsampa.text
        == r"@bUr\tOU {bstr\{kt {bSaIr\ Abr\utseIzi {br\@geItIN @bor\n {b3kr\Ambi {bI5s@`n Ar\g thr\Ip@5eI Ar\dvAr\k"
    )
    corpus.cleanup_connections()


def test_japanese(japanese_dir, japanese_dict_path, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests", "japanese")
    config.TEMPORARY_DIRECTORY = output_directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = DictionaryTextCorpus(
        corpus_directory=japanese_dir, dictionary_path=japanese_dict_path
    )
    corpus.load_corpus()
    print(corpus.files())
    print(corpus.utterances())

    punctuated = corpus.get_utterances(file="日本語")[0]
    assert punctuated.text == unicodedata.normalize("NFKC", "「はい」、。！ 『何 でしょう』")
    assert punctuated.normalized_text == unicodedata.normalize("NFKC", "はい 何 でしょう")
    corpus.cleanup_connections()


def test_devanagari(devanagari_dir, hindi_dict_path, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests", "devanagari")
    config.TEMPORARY_DIRECTORY = output_directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = DictionaryTextCorpus(corpus_directory=devanagari_dir, dictionary_path=hindi_dict_path)
    corpus.load_corpus()
    print(corpus.files())
    print(corpus.utterances())

    punctuated = corpus.get_utterances(file="devanagari")[0]
    assert punctuated.text == unicodedata.normalize("NFKC", "हैंः हूं हौंसला")
    assert punctuated.normalized_text == unicodedata.normalize("NFKC", "हैंः हूं हौंसला")
    corpus.cleanup_connections()


def test_french_clitics(french_clitics_dir, frclitics_dict_path, generated_dir, db_setup):
    output_directory = generated_dir.joinpath("corpus_tests", "french_clitics")
    config.TEMPORARY_DIRECTORY = output_directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = DictionaryTextCorpus(
        corpus_directory=french_clitics_dir, dictionary_path=frclitics_dict_path
    )
    corpus.load_corpus()

    punctuated = corpus.get_utterances(file="french_clitics")[0]
    assert punctuated.text == unicodedata.normalize(
        "NFKC",
        "aujourd aujourd'hui m'appelle purple-people-eater vingt-six m'm'appelle c'est m'c'est m'appele m'ving-sic flying'purple-people-eater",
    )
    assert punctuated.normalized_text == unicodedata.normalize(
        "NFKC",
        "aujourd aujourd'hui m' appelle purple-people-eater vingt six m' m' appelle c'est m' c'est m' appele m' ving sic flying'purple-people-eater",
    )
    corpus.cleanup_connections()
