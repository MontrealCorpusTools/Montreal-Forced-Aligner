
(remapping_example)=
# Example: Adapting a model to a new language 


## Set up

```{important}
Ensure you have installed MFA via {ref}`installation`.  For comparing alignments to reference alignments from aligning via native language models, ensure you have completed the initital alignment for demo corpus in {ref}`alignment_example`.

You can see a more fully worked example of this with scripts for analyzing German, Czech, and Mandarin applied to an English corpus in the [mfa-adaptation GitHub repository](https://github.com/mmcauliffe/mfa-adaptation).
```

::::{tab-set}

:::{tab-item} English
:sync: english

For English, we will align the demo English corpus with the Mandarin pretrained acoustic model, and remap the English dictionary into the phone set that the Mandarin acoustic model uses.

1. Ensure you have downloaded the pretrained Mandarin model via {code}`mfa model download acoustic mandarin_mfa`
2. Ensure you have downloaded the pretrained US English dictionary via {code}`mfa model download dictionary english_us_mfa`
3. Download the [English LibriSpeech demo corpus](https://github.com/MontrealCorpusTools/librispeech-demo/archive/refs/tags/v1.0.0.tar.gz) and extract it to somewhere on your computer

:::

:::{tab-item} Japanese
:sync: japanese

For Japanese, we will align the demo Japanese corpus with the English pretrained acoustic model, and remap the Japanese dictionary into the phone set that the English acoustic model uses.

1. Ensure you have downloaded the pretrained English model via {code}`mfa model download acoustic english_mfa`
2. Ensure you have downloaded the pretrained Japanese dictionary via {code}`mfa model download dictionary japanese_mfa`
3. Download the [Japanese JVS demo corpus](https://github.com/MontrealCorpusTools/japanese-jvs-demo/archive/refs/tags/v1.0.0.tar.gz) and extract it to somewhere on your computer
4. Install Japanese-specific dependencies via {code}`conda install -c conda-forge spacy sudachipy sudachidict-core`

:::


:::{tab-item} Mandarin
:sync: mandarin

For Mandarin, we will align the demo Mandarin corpus with the English pretrained acoustic model, and remap the Mandarin dictionary into the phone set that the English acoustic model uses.

1. Ensure you have downloaded the pretrained model via {code}`mfa model download acoustic english_mfa`
2. Ensure you have downloaded the pretrained China Mandarin dictionary via {code}`mfa model download dictionary mandarin_china_mfa`
3. Download the [Mandarin THCHS-30 demo corpus](https://github.com/MontrealCorpusTools/mandarin-thchs-30-demo/archive/refs/tags/v1.0.0.tar.gz) and extract it to somewhere on your computer
4. Install Mandarin-specific dependencies via {code}`pip install spacy-pkuseg dragonmapper hanziconv`

:::


::::

```{important}
This example assumes you have a directory named ``mfa_data`` in your home directory in which the demo corpus was extracted.
```

## Remapping the dictionary


:::::{tab-set}

::::{tab-item} English
:sync: english

First, download and save the contents of [english_to_mandarin_phone_mapping.yaml](https://raw.githubusercontent.com/mmcauliffe/mfa-adaptation/refs/heads/main/data/dictionary_mappings/english_to_mandarin_phone_mapping.yaml) to `~/mfa_data/english_to_mandarin_phone_mapping.yaml`.  This is a file that maps phones in the English MFA phone set to phones in the Japanese MFA phone set, which we can use to create a new dictionary of English words with Mandarin MFA pronunciations.

:::{code-block} bash
mfa remap dictionary english_us_mfa mandarin_mfa ~/mfa_data/english_to_mandarin_phone_mapping.yaml ~/mfa_data/english_mandarin.dict
:::

If you open up `~/mfa_data/english_mandarin.dict` in a text editor, you'll now see pronunciations for English forms using Mandarin MFA phones.  For example, any {ipa_inline}`ʒ` phones now have {ipa_inline}`ʐ` instead, as that's the closest phone in the Mandarin MFA phone set.

::::

::::{tab-item} Japanese
:sync: japanese


First, download and save the contents of [japanese_to_english_phone_mapping.yaml](https://raw.githubusercontent.com/mmcauliffe/mfa-adaptation/refs/heads/main/data/dictionary_mappings/japanese_to_english_phone_mapping.yaml) to `~/mfa_data/japanese_to_english_phone_mapping.yaml`.  This is a file that maps phones in the Japanese MFA phone set to phones in the English MFA phone set, which we can use to create a new dictionary of Japanese words with English MFA pronunciations.

:::{code-block} bash
mfa remap dictionary japanese_mfa english_mfa ~/mfa_data/japanese_to_english_phone_mapping.yaml ~/mfa_data/japanese_english.dict
:::

If you open up `~/mfa_data/japanese_english.dict` in a text editor, you'll now see pronunciations for Japanese forms using English MFA phones.  For example, any {ipa_inline}`tɕ` phones now have {ipa_inline}`tʃ` instead, as that's the closest phone in the English MFA phone set.

::::

::::{tab-item} Mandarin
:sync: mandarin


First, download and save the contents of [mandarin_to_english_phone_mapping.yaml](https://raw.githubusercontent.com/mmcauliffe/mfa-adaptation/refs/heads/main/data/dictionary_mappings/mandarin_to_english_phone_mapping.yaml) to `~/mfa_data/mandarin_to_english_phone_mapping.yaml`.  This is a file that maps phones in the Mandarin MFA phone set to phones in the English MFA phone set, which we can use to create a new dictionary of Mandarin words with English MFA pronunciations.

:::{code-block} bash
mfa remap dictionary mandarin_china_mfa english_mfa ~/mfa_data/mandarin_to_english_phone_mapping.yaml ~/mfa_data/mandarin_english.dict
:::

If you open up `~/mfa_data/mandarin_english.dict` in a text editor, you'll now see pronunciations for Mandarin forms using English MFA phones.  For example, any {ipa_inline}`tɕ` phones now have {ipa_inline}`tʃ` instead, as that's the closest phone in the English MFA phone set.

::::

:::::



## Alignment

### Aligning using pre-trained models


:::::{tab-set}

::::{tab-item} English
:sync: english


:::{code-block} bash
mfa align ~/mfa_data/librispeech-demo-1.0.0 ~/mfa_data/english_mandarin.dict english_mfa ~/mfa_data/aligned_librispeech_demo --clean
:::

::::

::::{tab-item} Japanese
:sync: japanese


First, download and save the contents of [english_to_japanese_phone_mapping.yaml](https://raw.githubusercontent.com/mmcauliffe/mfa-adaptation/refs/heads/main/data/evaluation_mappings/english_to_japanese_phone_mapping.yaml) to `~/mfa_data/english_to_japanese_phone_mapping.yaml`.  This file is similar to the previously downloaded `~/mfa_data/japanese_to_english_phone_mapping.yaml` except it maps phones in the opposite direction.  This mapping says for every Japanese phone, what is an acceptable phone that counts as a "matching phone", allowing the overlap scoring algorithm to more correctly penalize issues in alignment.

If you have not aligned the Japanese demo corpus as the first step in {ref}`alignment_example`, you will have to omit the `--reference_directory` and `--custom_mapping_path` of the following command.

:::{code-block} bash
mfa align ~/mfa_data/japanese-jvs-demo-1.0.0 ~/mfa_data/japanese_english.dict english_mfa ~/mfa_data/english_adapted/english_japanese_remapped_aligned --clean --reference_directory ~/mfa_data/aligned_jvs_demo --custom_mapping_path ~/mfa_data/english_to_japanese_phone_mapping.yaml --language japanese
:::

```{note}

The `--language japanese` flag must be included to ensure that the Japanese text is properly tokenized by the Japanese morphological parser. When aligning using the Japanese MFA model, the language is set to Japanese by default, but we must override it here when using the English MFA model.
```

The end output will give:

```{code}

INFO     Evaluating alignments...
INFO     Exporting evaluation...
INFO     Average overlap score: 0.010834011956534382
INFO     Average phone error rate: 0.02820097244732577
```

Which reports a mean phone boundary error of 10.8 ms (Average overlap score), and an average PER of 2.8% (percent of insertions, deletions and substitutions).

::::

::::{tab-item} Mandarin
:sync: mandarin


First, download and save the contents of [english_to_mandarin_phone_mapping.yaml](https://raw.githubusercontent.com/mmcauliffe/mfa-adaptation/refs/heads/main/data/evaluation_mappings/english_to_mandarin_phone_mapping.yaml) to `~/mfa_data/english_to_mandarin_phone_mapping.yaml`.  This file is similar to the previously downloaded `~/mfa_data/mandarin_to_english_phone_mapping.yaml` except it maps phones in the opposite direction.  This mapping says for every Mandarin phone, what is an acceptable phone that counts as a "matching phone", allowing the overlap scoring algorithm to more correctly penalize issues in alignment.

If you have not aligned the Mandarin demo corpus as the first step in {ref}`alignment_example`, you will have to omit the `--reference_directory` and `--custom_mapping_path` of the following command.

:::{code-block} bash
mfa align ~/mfa_data/mandarin-thchs-30-demo-1.0.0 ~/mfa_data/mandarin_english.dict english_mfa ~/mfa_data/english_adapted/english_mandarin_remapped_aligned --clean --reference_directory ~/mfa_data/aligned_thchs_30_demo --custom_mapping_path ~/mfa_data/english_to_mandarin_phone_mapping.yaml --language chinese
:::

::::

:::::

Once the files are aligned we can take a look at the alignment_analysis.csv file in the output directory to see if there are any glaring issues in alignment.  This file is sorted initially by the `phone_duration_deviation` column, which is the maximum z-scored duration for phones in the utterance.  High values indication much longer or shorter phones than we would expect given the phone, i.e., a {ipa_inline}`[ɾ]` lasting 100ms is very unlikely given the usual duration is typically around 10-20ms.

Additionally, there is are two files from the alignment evaluation triggered by having `--reference_directory` specified.  As we're comparing alignments to reference alignments, we can look at a confusion matrix in `alignment_reference_confusions.csv` and find utterances with high errors by looking at `alignment_reference_evaluation.csv` and sorting on the `alignment_score` column.

### Adapting the acoustic model

In general, adapting a pretrained acoustic model to your specific data will improve alignments, but this is particularly so when using pretrained model that was trained on a different language than what you're aligning. 

We can adapt our pretrained model via the {code}`mfa adapt` command:

:::::{tab-set}

::::{tab-item} English
:sync: english


```{warning}

Under construction
```

::::

::::{tab-item} Japanese
:sync: japanese


:::{code-block} bash
mfa adapt ~/mfa_data/japanese-jvs-demo-1.0.0 ~/mfa_data/japanese_english.dict english_mfa ~/mfa_data/english_adapted/english_adapted.zip --clean --language japanese
:::

We can now use the adapted model to align the japanese-jvs-demo corpus.  Note the change from ``english_mfa`` to ``~/mfa_data/english_adapted/english_adapted.zip`` below.

:::{code-block} bash
mfa align ~/mfa_data/japanese-jvs-demo-1.0.0 ~/mfa_data/japanese_english.dict ~/mfa_data/english_adapted/english_adapted.zip ~/mfa_data/english_adapted/english_remapped_aligned_adapted --clean --reference_directory ~/mfa_data/aligned_jvs_demo --custom_mapping_path ~/mfa_data/english_to_japanese_phone_mapping.yaml --language japanese
:::

The end output will give:

```{code}

INFO     Evaluating alignments...
INFO     Exporting evaluation...
INFO     Average overlap score: 0.010524732208295882
INFO     Average phone error rate: 0.026904376012965966
```

Which reports a mean phone boundary error of 10.5 ms, improving on the previous 10.8 ms error aligning by default, and an average PER of 2.7%, improving from 2.8%.  So adaptation gives some modest gains for making the alignments generated from English MFA more similar to those generated by the Japanese MFA model.  The benefit for adaptation is going to be a function of the size of the dataset, and the demo corpus here is pretty small, so only a little bit of improvement is to be expected and observed.

::::

::::{tab-item} Mandarin
:sync: mandarin


```{warning}

Under construction
```

::::

:::::


```{seealso}
* {ref}`first_steps_adapt_pretrained`
```