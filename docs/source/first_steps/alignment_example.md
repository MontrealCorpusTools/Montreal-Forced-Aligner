
(alignment_example)=
# Example: Aligning a demo corpus

```{note}
See also our [Google Colab notebook](https://colab.research.google.com/drive/1kqaSSyx-DEVAxrSmoWhJTNXtEsVI15yf?usp=sharing) for running this example without installing or downloading anything locally.
There is also [NTT123's Jupyter notebook](https://gist.github.com/NTT123/12264d15afad861cb897f7a20a01762e) for running the alignment example with a custom LibriSpeech dataset, created by [NTT123](https://github.com/NTT123).
```

## Set up

```{important}
Ensure you have installed MFA via {ref}`installation`.
```

::::{tab-set}

:::{tab-item} English
:sync: english

1. Ensure you have downloaded the pretrained model via {code}`mfa model download acoustic english_mfa`
2. Ensure you have downloaded the pretrained US English dictionary via {code}`mfa model download dictionary english_us_mfa`
3. Download the [English LibriSpeech demo corpus](https://github.com/MontrealCorpusTools/librispeech-demo/archive/refs/tags/v1.0.0.tar.gz) and extract it to somewhere on your computer

:::

:::{tab-item} Japanese
:sync: japanese

1. Ensure you have downloaded the pretrained model via {code}`mfa model download acoustic japanese_mfa`
2. Ensure you have downloaded the pretrained Japanese dictionary via {code}`mfa model download dictionary japanese_mfa`
3. Download the [Japanese JVS demo corpus](https://github.com/MontrealCorpusTools/japanese-jvs-demo/archive/refs/tags/v1.0.0.tar.gz) and extract it to somewhere on your computer
4. Install Japanese-specific dependencies via {code}`conda install -c conda-forge spacy sudachipy sudachidict-core`

:::


:::{tab-item} Mandarin
:sync: mandarin

1. Ensure you have downloaded the pretrained model via {code}`mfa model download acoustic mandarin_mfa`
2. Ensure you have downloaded the pretrained China Mandarin dictionary via {code}`mfa model download dictionary mandarin_china_mfa`
3. Download the [Mandarin THCHS-30 demo corpus](https://github.com/MontrealCorpusTools/mandarin-thchs-30-demo/archive/refs/tags/v1.0.0.tar.gz) and extract it to somewhere on your computer
4. Install Mandarin-specific dependencies via {code}`pip install spacy-pkuseg dragonmapper hanziconv`

:::


::::

```{important}
This example assumes you have a directory named ``mfa_data`` in your home directory in which the demo corpus was extracted.
```

## Alignment

### Aligning using pre-trained models

In the same environment that you've installed MFA, enter the following command into the terminal:


:::::{tab-set}

::::{tab-item} English
:sync: english


:::{code-block} bash
mfa align ~/mfa_data/librispeech-demo-1.0.0 english_us_mfa english_mfa ~/mfa_data/aligned_librispeech_demo --clean
:::

::::

::::{tab-item} Japanese
:sync: japanese


:::{code-block} bash
mfa align ~/mfa_data/japanese-jvs-demo-1.0.0 japanese_mfa japanese_mfa ~/mfa_data/aligned_jvs_demo --clean
:::

::::

::::{tab-item} Mandarin
:sync: mandarin


:::{code-block} bash
mfa align ~/mfa_data/mandarin-thchs-30-demo-1.0.0 mandarin_china_mfa mandarin_mfa ~/mfa_data/aligned_thchs_30_demo --clean
:::

::::

:::::

Once the files are aligned we can take a look at the alignment_analysis.csv file in the output directory to see if there are any glaring issues in alignment.  This file is sorted initially by the `phone_duration_deviation` column, which is the maximum z-scored duration for phones in the utterance.  High values indication much longer or shorter phones than we would expect given the phone, i.e., a {ipa_inline}`[ɾ]` lasting 100ms is very unlikely given the usual duration is typically around 10-20ms.


:::::{tab-set}

::::{tab-item} English
:sync: english

:::{csv-table} ~/mfa/data/aligned_librispeech_demo/alignment_analysis.csv
:header-rows: 1
:stub-columns: 1

"file","begin","end","speaker","overall_log_likelihood","speech_log_likelihood","phone_duration_deviation","snr"
26-495-0029,0.0,15.33,26,-54.456238788323546,-57.6560760433391,15.243577592668007,6.645144699335612
19-227-0027,0.0,15.645,19,-44.529997004792335,-46.942441727063674,13.854204008671575,12.191697302549363
19-198-0010,0.0,12.6,19,-43.50238715277778,-45.62727763510158,9.378276520846718,13.180338896436085
26-495-0004,0.0,14.055,26,-50.670524759957324,-50.98836160475208,9.132307439395934,11.955992448065652
19-227-0016,0.0,12.685,19,-42.93412012411348,-46.04780329319469,8.374659305119293,12.57889123443367

:::

::::

::::{tab-item} Japanese
:sync: japanese

:::{csv-table} ~/mfa/data/aligned_jvs_demo/alignment_analysis.csv
:header-rows: 1
:stub-columns: 1

"file","begin","end","speaker","overall_log_likelihood","speech_log_likelihood","phone_duration_deviation","snr"
jvs004_nonpara30_BASIC5000_1571,0.0,4.724208333333333,jvs004,-44.60921775688559,-53.64337515830994,11.069623295539825,20.73555387405615
jvs001_parallel100_VOICEACTRESS100_001,0.0,8.621041666666667,jvs001,-41.720485607598604,-45.22640653756949,6.211908079327884,27.302904438762084
jvs003_nonpara30_BASIC5000_2550,0.0,10.909041666666667,jvs003,-40.970862454170486,-45.1975002500746,5.892864020481638,23.07655949682694
jvs004_nonpara30_BASIC5000_1560,0.0,9.180166666666667,jvs004,-46.13757829520697,-50.26279640197754,5.803162260964776,21.536162380195314
jvs003_nonpara30_BASIC5000_0440,0.0,9.649583333333334,jvs003,-41.17138924870466,-45.75857627223915,4.7608877931967415,23.593413391753792
:::

The above table shows the five files with highest phone duration deviation, which we can then take a look more closely and see what's causing the alignment issues.

For example, looking at `jvs004_nonpara30_BASIC5000_1571` in a program like [Praat](https://www.fon.hum.uva.nl/praat/), we can see the issue is due to an out of vocabulary item "1泊".  The original transcript is:

:::{code}

税、その他全て込みだと、１泊いくらですか。

:::

Which is tokenized by MFA using sudachipy to the following sequence of words:

:::{code}

税 その 他 全て込み だ と １泊 いくら です か。

:::

Every word other than "１泊" is present in `japanese_mfa` dictionary, but the lack of "１泊" causes a small section of the previous word "と" to be aligned as "spn", and then the following word "いくら" takes up the actual time span of "いっぱく" in addition to the time span corresponding to "いくら", since both "いっぱく" and "いくら" start with {ipa_inline}`[i]` followed by a stop closure.

There are a couple of ways to fix this issue.  The most straight-forward way would be to add a dictionary entry for "１泊" as {ipa_inline}`i pː a k ɯ`  or change the input transcript to use "一泊" instead of "１泊", as the correct pronunciation is present for "一泊".

However, this fix will only affect this one particular word/utterance, and there are likely many more out of vocabulary items.  The Japanese alignment has the capability of using the katakana generated by the sudachipy morphological parser and use the katakana forms as input to grapheme-to-phoneme models on the fly, so for the purposes of this example, we'll skip over adding words to the Japanese dictionary and just focus on using G2P models at alignment time.

::::

::::{tab-item} Mandarin
:sync: mandarin

:::{csv-table} ~/mfa/data/aligned_thchs_30_demo/alignment_analysis.csv
:header-rows: 1
:stub-columns: 1

"file","begin","end","speaker","overall_log_likelihood","speech_log_likelihood","phone_duration_deviation","snr"
A4_8,0.0,12.4375,A4,-49.248455084405144,-55.64855964978536,6.169737157239693,8.524069008855722
C12_515,0.0,11.69,C12,-49.446983265611635,-51.683270083533394,5.940588177352259,10.706482170359841
A4_25,0.0,10.5,A4,-48.68171130952381,-50.71304063911898,5.613372337759242,6.7809069072769255
B34_251,0.0,9.25,B34,-40.79677787162162,-41.75823864792333,5.540030677004922,8.962213255170383
A4_6,0.0,9.625,A4,-50.510534300363446,-53.57253624076274,5.429628766470327,5.500242213510484
:::


::::

:::::



### Adding words to the dictionary

First we'll need the pretrained G2P model.  These are installed via the {code}`mfa model download` command:

:::::{tab-set}

::::{tab-item} English
:sync: english


:::{code-block} bash
mfa model download g2p english_us_mfa
:::

You should be able to run {code}`mfa model inspect g2p english_us_mfa` and it will output information about the {code}`english_us_mfa` G2P model.

::::

::::{tab-item} Japanese
:sync: japanese


:::{code-block} bash
mfa model download g2p japanese_mfa
:::

You should be able to run {code}`mfa model inspect g2p japanese_mfa` and it will output information about the {code}`japanese_mfa` G2P model.

::::

::::{tab-item} Mandarin
:sync: mandarin


:::{code-block} bash
mfa model download g2p mandarin_china_mfa
:::

You should be able to run {code}`mfa model inspect g2p mandarin_china_mfa` and it will output information about the {code}`mandarin_china_mfa` G2P model.

::::

:::::

Depending on your use case, you might have a list of words to run G2P over, or just a corpus of sound and transcription files.  The {code}`mfa g2p` command can process either:

:::::{tab-set}

::::{tab-item} English
:sync: english


:::{code-block} bash
mfa g2p ~/mfa_data/librispeech-demo-1.0.0 english_us_mfa ~/mfa_data/g2pped_oovs.txt --dictionary_path english_us_mfa --clean
:::

::::

::::{tab-item} Japanese
:sync: japanese


For Japanese, G2P functionality is done as part of alignment by specifying {code}`--g2p_model_path`.

::::

::::{tab-item} Mandarin
:sync: mandarin


:::{code-block} bash
mfa g2p ~/mfa_data/mandarin-thchs-30-demo-1.0.0 mandarin_china_mfa ~/mfa_data/g2pped_oovs.txt --dictionary_path mandarin_china_mfa --clean
:::

::::

:::::

Running the above will output a text file in the format that MFA uses ({ref}`dictionary_format`) with all the OOV words (ignoring bracketed words like {ipa_inline}`<cutoff>`).  I recommend looking over the pronunciations generated and make sure that they look sensible.  For languages where the orthography is not transparent, it may be helpful to include {code}`--num_pronunciations 3` so that more pronunciations are generated than just the most likely one. For more details on running G2P, see {ref}`g2p_dictionary_generating`.

Once you have looked over the dictionary, you can save the new pronunciations via:

:::::{tab-set}

::::{tab-item} English
:sync: english


:::{code-block} bash
mfa model add_words english_us_mfa ~/mfa_data/g2pped_oovs.txt
:::

::::

::::{tab-item} Japanese
:sync: japanese

For Japanese, G2P functionality is done as part of alignment by specifying {code}`--g2p_model_path`.

::::

::::{tab-item} Mandarin
:sync: mandarin


:::{code-block} bash
mfa model add_words mandarin_china_mfa ~/mfa_data/g2pped_oovs.txt
:::

::::

:::::

The new pronunciations will be available when you use the dictionary identifier in an MFA command, i.e. the modified command from {ref}`first_steps_align_pretrained`:

:::::{tab-set}

::::{tab-item} English
:sync: english


:::{code-block} bash
mfa align ~/mfa_data/librispeech-demo-1.0.0 english_us_mfa english_mfa ~/mfa_data/aligned_librispeech_demo_no_oovs --clean
:::

::::

::::{tab-item} Japanese
:sync: japanese


:::{code-block} bash
mfa align ~/mfa_data/japanese-jvs-demo-1.0.0 japanese_mfa japanese_mfa ~/mfa_data/aligned_jvs_demo_no_oovs --g2p_model_path japanese_mfa --clean
:::

::::

::::{tab-item} Mandarin
:sync: mandarin


:::{code-block} bash
mfa align ~/mfa_data/mandarin-thchs-30-demo-1.0.0 mandarin_china_mfa mandarin_mfa ~/mfa_data/aligned_mandarin_demo_no_oovs --clean
:::

::::

:::::

Now let's see if adding words to the dictionary helped improve the alignment issues we were seeing before.

:::::{tab-set}

::::{tab-item} English
:sync: english

:::{csv-table} ~/mfa/data/aligned_librispeech_demo_no_oovs/alignment_analysis.csv
:header-rows: 1
:stub-columns: 1

"file","begin","end","speaker","overall_log_likelihood","speech_log_likelihood","phone_duration_deviation","snr"
26-495-0029,0.0,15.33,26,-54.456238788323546,-57.6560760433391,15.243577592668007,6.645144699335612
19-227-0027,0.0,15.645,19,-44.529997004792335,-46.942441727063674,13.854204008671575,12.191697302549363
19-198-0010,0.0,12.6,19,-43.50238715277778,-45.62727763510158,9.378276520846718,13.180338896436085
26-495-0004,0.0,14.055,26,-50.670524759957324,-50.98836160475208,9.132307439395934,11.955992448065652
19-227-0016,0.0,12.685,19,-42.93412012411348,-46.04780329319469,8.374659305119293,12.57889123443367

:::

::::

::::{tab-item} Japanese
:sync: japanese

:::{csv-table} ~/mfa/data/aligned_jvs_demo_no_oovs/alignment_analysis.csv
:header-rows: 1
:stub-columns: 1

"file","begin","end","speaker","overall_log_likelihood","speech_log_likelihood","phone_duration_deviation","snr"
jvs001_parallel100_VOICEACTRESS100_001,0.0,8.621041666666667,jvs001,-41.84382250580047,-45.2327318925124,6.20348375405817,27.293943985523807
jvs003_nonpara30_BASIC5000_2550,0.0,10.909041666666667,jvs003,-40.983222101283225,-45.076715024312335,5.908918038417016,23.07655949682694
jvs002_nonpara30_BASIC5000_0114,0.0,4.245291666666667,jvs002,-39.14056525735294,-42.082562075720894,5.185645170149811,20.482769181663702
jvs003_nonpara30_BASIC5000_0440,0.0,9.649583333333334,jvs003,-41.123874676165805,-45.70696529871981,4.736230315134372,23.619350634943906
jvs001_parallel100_VOICEACTRESS100_004,0.0,6.523541666666667,jvs001,-44.13319593558282,-47.53735481118256,4.546602511572727,26.105708678065128
:::

We can see now that `jvs004_nonpara30_BASIC5000_1571` no longer has the top phone duration deviation. Looking at  in a program like [Praat](https://www.fon.hum.uva.nl/praat/), we can see that "1泊" and "いくら" are now properly aligned at the word level, and the phones in "いくら" are properly aligned.  

However, it is important to note that the pronunciation generated for "1泊"  is not correct.  The pronunciation generated is not {ipa_inline}`i pː a k ɯ` as it should be, but is instead {ipa_inline}`i tɕ i h a k ɯ`.  The source of this error is due to the sudachipy parse gives the pronunciation form of "1泊" as "イチハク"　and not "イッパク".  However, there is enough of the correct pronunciation to ensure that this error does not affect the alignment of surrounding words.

::::

::::{tab-item} Mandarin
:sync: mandarin


:::{csv-table} ~/mfa/data/aligned_thchs_30_demo_no_oovs/alignment_analysis.csv
:header-rows: 1
:stub-columns: 1

"file","begin","end","speaker","overall_log_likelihood","speech_log_likelihood","phone_duration_deviation","snr"
A4_8,0.0,12.4375,A4,-49.248455084405144,-55.64855964978536,6.169737157239693,8.524069008855722
C12_515,0.0,11.69,C12,-49.446983265611635,-51.683270083533394,5.940588177352259,10.706482170359841
A4_25,0.0,10.5,A4,-48.68171130952381,-50.71304063911898,5.613372337759242,6.7809069072769255
B34_251,0.0,9.25,B34,-40.79677787162162,-41.75823864792333,5.540030677004922,8.962213255170383
A4_6,0.0,9.625,A4,-50.510534300363446,-53.57253624076274,5.429628766470327,5.500242213510484
:::

::::

:::::

```{seealso}
* {ref}`first_steps_g2p_oovs`
```

### Adapting the acoustic model

In general, adapting a pretrained acoustic model to your specific data will improve alignments.

We can adapt our pretrained model via the {code}`mfa adapt` command:

:::::{tab-set}

::::{tab-item} English
:sync: english


:::{code-block} bash
mfa adapt ~/mfa_data/librispeech-demo-1.0.0 english_us_mfa english_mfa ~/mfa_data/english_mfa_adapted.zip --clean
:::

We can now use the adapted model to align the librispeech-demo corpus.  Note the change from ``english_mfa`` to ``~/mfa_data/english_mfa_adapted.zip`` below.

:::{code-block} bash
mfa align ~/mfa_data/librispeech-demo-1.0.0 english_us_mfa ~/mfa_data/english_mfa_adapted.zip ~/mfa_data/aligned_librispeech_demo_adapted --clean
:::

::::

::::{tab-item} Japanese
:sync: japanese


:::{code-block} bash
mfa adapt ~/mfa_data/japanese-jvs-demo-1.0.0 japanese_mfa japanese_mfa ~/mfa_data/japanese_mfa_adapted.zip --g2p_model_path japanese_mfa --clean
:::

We can now use the adapted model to align the japanese-jvs-demo corpus.  Note the change from ``japanese_mfa`` to ``~/mfa_data/japanese_mfa_adapted.zip`` below.

:::{code-block} bash
mfa align ~/mfa_data/japanese-jvs-demo-1.0.0 japanese_mfa ~/mfa_data/japanese_mfa_adapted.zip ~/mfa_data/aligned_jvs_demo_adapted --g2p_model_path japanese_mfa --clean
:::

::::

::::{tab-item} Mandarin
:sync: mandarin


:::{code-block} bash
mfa adapt ~/mfa_data/mandarin-thchs-30-demo-1.0.0 mandarin_china_mfa mandarin_mfa ~/mfa_data/mandarin_mfa_adapted.zip --clean
:::

We can now use the adapted model to align the mandarin-thchs-30-demo corpus.  Note the change from ``mandarin_mfa`` to ``~/mfa_data/mandarin_mfa_adapted.zip`` below.

:::{code-block} bash
mfa align ~/mfa_data/mandarin-thchs-30-demo-1.0.0 mandarin_china_mfa ~/mfa_data/mandarin_mfa_adapted.zip ~/mfa_data/aligned_thchs_30_demo_adapted --clean
:::

::::

:::::


```{seealso}
* {ref}`first_steps_adapt_pretrained`
```