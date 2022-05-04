import os
import shutil

from montreal_forced_aligner.corpus.acoustic_corpus import (
    AcousticCorpus,
    AcousticCorpusWithPronunciations,
)
from montreal_forced_aligner.corpus.classes import FileData, UtteranceData
from montreal_forced_aligner.corpus.helper import get_wav_info
from montreal_forced_aligner.corpus.text_corpus import TextCorpus
from montreal_forced_aligner.data import TextFileType
from montreal_forced_aligner.db import OovWord, Word


def test_mp3(mp3_test_path):
    info = get_wav_info(mp3_test_path)
    assert info.sox_string
    assert info.duration > 0


def test_opus(opus_test_path):
    info = get_wav_info(opus_test_path)
    assert info.sox_string
    assert info.duration > 0


def test_speaker_word_set(
    multilingual_ipa_tg_corpus_dir, multispeaker_dictionary_config_path, temp_dir
):
    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=multilingual_ipa_tg_corpus_dir,
        dictionary_path=multispeaker_dictionary_config_path,
        temporary_directory=temp_dir,
    )
    corpus.load_corpus()
    sanitize = corpus.sanitize_function
    split, san = sanitize.get_functions_for_speaker("speaker_one")
    sp = san.split_clitics("chad-like")
    assert len(sp) > 1
    assert san.oov_word not in sp


def test_add(basic_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    corpus = AcousticCorpus(
        corpus_directory=basic_corpus_dir,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus._load_corpus()
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
        corpus.add_utterance(new_utterance, session)
        session.commit()

    utts = corpus.get_utterances(file=new_file_name, speaker=new_speaker)
    assert len(utts) == 1
    assert utts[0].text == "blah blah"
    print(utts[0].id)
    corpus.delete_utterance(utts[0].id)
    assert len(corpus.get_utterances(file=new_file_name, speaker=new_speaker)) == 0


def test_basic_txt(basic_corpus_txt_dir, basic_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    corpus = AcousticCorpus(
        corpus_directory=basic_corpus_txt_dir,
        use_mp=False,
        temporary_directory=output_directory,
        use_pitch=True,
    )
    corpus.load_corpus()

    print(corpus.no_transcription_files)
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 48


def test_acoustic_from_temp(basic_corpus_txt_dir, basic_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    corpus = AcousticCorpus(
        corpus_directory=basic_corpus_txt_dir,
        use_mp=False,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39

    new_corpus = AcousticCorpus(
        corpus_directory=basic_corpus_txt_dir,
        use_mp=False,
        temporary_directory=output_directory,
    )
    new_corpus.load_corpus()
    assert len(new_corpus.no_transcription_files) == 0
    assert new_corpus.get_feat_dim() == 39


def test_text_corpus_from_temp(basic_corpus_txt_dir, basic_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    corpus = TextCorpus(
        corpus_directory=basic_corpus_txt_dir,
        dictionary_path=basic_dict_path,
        use_mp=False,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert corpus.num_utterances > 0


def test_extra(basic_dict_path, extra_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=extra_corpus_dir,
        dictionary_path=basic_dict_path,
        use_mp=False,
        num_jobs=2,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39


def test_stereo(basic_dict_path, stereo_corpus_dir, generated_dir):

    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=stereo_corpus_dir,
        use_mp=False,
        num_jobs=1,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert corpus.get_file(name="michaelandsickmichael").num_channels == 2


def test_stereo_short_tg(basic_dict_path, stereo_corpus_short_tg_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=stereo_corpus_short_tg_dir,
        use_mp=False,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert corpus.get_file(name="michaelandsickmichael").num_channels == 2


def test_flac(basic_dict_path, flac_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=flac_corpus_dir,
        use_mp=False,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39


def test_audio_directory(basic_dict_path, basic_split_dir, generated_dir):
    audio_dir, text_dir = basic_split_dir
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=text_dir,
        use_mp=False,
        audio_directory=audio_dir,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert corpus.num_files > 0

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    corpus = AcousticCorpus(
        corpus_directory=text_dir,
        use_mp=True,
        audio_directory=audio_dir,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert corpus.num_files > 0


def test_flac_mp(basic_dict_path, flac_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=flac_corpus_dir,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert corpus.num_files > 0


def test_flac_tg(basic_dict_path, flac_tg_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=flac_tg_corpus_dir,
        use_mp=False,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert corpus.num_files > 0


def test_flac_tg_mp(basic_dict_path, flac_tg_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=flac_tg_corpus_dir,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 0
    assert corpus.get_feat_dim() == 39
    assert corpus.num_files > 0


def test_24bit_wav(transcribe_corpus_24bit_dir, basic_dict_path, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=transcribe_corpus_24bit_dir,
        use_mp=False,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert len(corpus.no_transcription_files) == 2
    assert corpus.get_feat_dim() == 39
    assert corpus.num_files > 0


def test_short_segments(shortsegments_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=shortsegments_corpus_dir,
        use_mp=False,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    assert corpus.num_utterances == 3
    assert len([x for x in corpus.utterances() if not x.ignored]) == 2
    assert len([x for x in corpus.utterances() if x.features is not None]) == 2
    assert len([x for x in corpus.utterances() if x.ignored]) == 1
    assert len([x for x in corpus.utterances() if x.features is None]) == 1


def test_speaker_groupings(multilingual_ipa_corpus_dir, generated_dir, english_us_mfa_dictionary):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=multilingual_ipa_corpus_dir,
        dictionary_path=english_us_mfa_dictionary,
        use_mp=True,
        temporary_directory=output_directory,
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
    del corpus
    shutil.rmtree(output_directory)
    new_corpus = AcousticCorpusWithPronunciations(
        corpus_directory=multilingual_ipa_corpus_dir,
        dictionary_path=english_us_mfa_dictionary,
        num_jobs=1,
        use_mp=True,
        temporary_directory=output_directory,
    )
    new_corpus.load_corpus()
    files = new_corpus.files()
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


def test_subset(multilingual_ipa_corpus_dir, generated_dir):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpus(
        corpus_directory=multilingual_ipa_corpus_dir,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    sd = corpus.split_directory

    s = corpus.subset_directory(5)
    assert os.path.exists(sd)
    assert os.path.exists(s)


def test_weird_words(weird_words_dir, generated_dir, basic_dict_path):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=weird_words_dir,
        dictionary_path=basic_dict_path,
        use_mp=True,
        temporary_directory=output_directory,
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
            for x in session.query(OovWord.word).filter(
                OovWord.dictionary_id == corpus._default_dictionary_id
            )
        ]
        print(oovs)
        assert all(
            x in oovs
            for x in {
                "ajfish",
                "asds-asda",
                "sdasd",
                "[me_really]",
                "[me____really]",
                "[me_really]",
                "<s>",
                "<_s>",
            }
        )
    assert (
        weird_words.text
        == "i’m talking-ajfish me-really [me-really] [me'really] [me_??_really] asds-asda sdasd-me <s> </s>"
    )
    assert weird_words.normalized_text.split() == [
        "i'm",
        "talking",
        "ajfish",
        "me",
        "really",
        "[me_really]",
        "[me_really]",
        "[me____really]",
        "asds-asda",
        "sdasd",
        "me",
        "<s>",
        "<_s>",
    ]
    assert weird_words.normalized_text_int.split()[-1] == str(
        corpus.word_mapping(corpus._default_dictionary_id)[corpus.bracketed_word]
    )
    print(oovs)
    assert "'m" not in oovs


def test_punctuated(punctuated_dir, generated_dir, basic_dict_path):
    output_directory = os.path.join(generated_dir, "corpus_tests")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)

    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=punctuated_dir,
        dictionary_path=basic_dict_path,
        use_mp=True,
        temporary_directory=output_directory,
    )
    corpus.load_corpus()
    print(corpus.files())
    print(corpus.utterances())

    punctuated = corpus.get_utterances(file="punctuated")[0]
    assert (
        punctuated.text == "oh yes, they - they, you know, they love her' and so' 'i mean... ‘you"
    )
    assert (
        punctuated.normalized_text == "oh yes they they you know they love her' and so i mean 'you"
    )


def test_alternate_punctuation(
    punctuated_dir, generated_dir, basic_dict_path, different_punctuation_config_path
):
    from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner

    output_directory = os.path.join(generated_dir, "corpus_tests", "alternate")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    params, skipped = AcousticCorpusWithPronunciations.extract_relevant_parameters(
        TrainableAligner.parse_parameters(different_punctuation_config_path)
    )
    params["use_mp"] = True
    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=punctuated_dir,
        dictionary_path=basic_dict_path,
        temporary_directory=output_directory,
        **params,
    )
    corpus.load_corpus()
    punctuated = corpus.get_utterances(file="punctuated")[0]
    assert (
        punctuated.text == "oh yes, they - they, you know, they love her' and so' 'i mean... ‘you"
    )


def test_no_punctuation(
    punctuated_dir, generated_dir, basic_dict_path, no_punctuation_config_path
):
    from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner

    output_directory = os.path.join(generated_dir, "corpus_tests", "no_punctuation")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    params, skipped = AcousticCorpusWithPronunciations.extract_relevant_parameters(
        TrainableAligner.parse_parameters(no_punctuation_config_path)
    )
    params["use_mp"] = True
    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=punctuated_dir,
        dictionary_path=basic_dict_path,
        temporary_directory=output_directory,
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
        punctuated.text == "oh yes, they - they, you know, they love her' and so' 'i mean... ‘you"
    )
    assert punctuated.normalized_text.split() == [
        "oh",
        "yes,",
        "they",
        "-",
        "they,",
        "you",
        "know,",
        "they",
        "love",
        "her'",
        "and",
        "so'",
        "'i",
        "mean...",
        "‘you",
    ]
    weird_words = corpus.get_utterances(file="weird_words")[0]
    assert (
        weird_words.text
        == "i’m talking-ajfish me-really [me-really] [me'really] [me_??_really] asds-asda sdasd-me <s> </s>"
    )
    assert weird_words.normalized_text.split() == [
        "i’m",
        "talking-ajfish",
        "me-really",
        "[me-really]",
        "[me'really]",
        "[me_??_really]",
        "asds-asda",
        "sdasd-me",
        "<s>",
        "</s>",
    ]


def test_xsampa_corpus(
    xsampa_corpus_dir, xsampa_dict_path, generated_dir, different_punctuation_config_path
):
    from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner

    output_directory = os.path.join(generated_dir, "corpus_tests", "xsampa")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory, ignore_errors=True)
    params, skipped = AcousticCorpusWithPronunciations.extract_relevant_parameters(
        TrainableAligner.parse_parameters(different_punctuation_config_path)
    )
    params["use_mp"] = True
    corpus = AcousticCorpusWithPronunciations(
        corpus_directory=xsampa_corpus_dir,
        dictionary_path=xsampa_dict_path,
        temporary_directory=output_directory,
        **params,
    )
    print(corpus.quote_markers)
    corpus.load_corpus()
    xsampa = corpus.get_utterances(file="xsampa")[0]
    assert (
        xsampa.text
        == r"@bUr\tOU {bstr\{kt {bSaIr\ Abr\utseIzi {br\@geItIN @bor\n {b3kr\Ambi {bI5s@`n Ar\g thr\Ip@5eI Ar\dvAr\k"
    )
