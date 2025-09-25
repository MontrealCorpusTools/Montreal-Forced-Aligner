
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
3. Download the [Japanese JVS demo corpus](https://github.com/MontrealCorpusTools/japanaese-jvs-demo/archive/refs/tags/v1.0.0.tar.gz) and extract it to somewhere on your computer
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
mfa align ~/mfa_data/japanese-jvs-demo-1.0.0 japanese_mfa japanese_mfa ~/mfa_data/aligned_jva_demo_no_oovs --g2p_model_path japanese_mfa --clean
:::

::::

::::{tab-item} Mandarin
:sync: mandarin


:::{code-block} bash
mfa align ~/mfa_data/mandarin-thchs-30-demo-1.0.0 mandarin_china_mfa mandarin_mfa ~/mfa_data/aligned_mandarin_demo_no_oovs --clean
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