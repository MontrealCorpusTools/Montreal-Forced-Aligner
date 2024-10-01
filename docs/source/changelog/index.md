
# News

## Roadmap

```{warning}
   Please bear in mind that all plans below are tentative and subject to change.
```

### Version 3.1

* Persistent server for sending audio/text files to
  * May not be necessary/prioritized with {ref}`align_one` command
* Update tokenization to use spacy tokenizers instead of custom specification
  * Should be more robust than MFA's custom rules
  * Some languages are more finely tokenized than others (i.e., Japanese and Korean tokens are largely morphemes, while the English one doesn't do morpheme analysis), but the ideal would be some morphologically-aware G2P of phonological words
* Add option for training SpeechBrain ASR model on phone strings of MFA models
  * Should allow for better single-pass alignment and faster with GPUs
* Release Anchor compatible with the latest versions of MFA

### Future

* Retrain existing acoustic models with new phone groups and rules features
* Begin work on expanding to new languages
    * Japanese (in progress)
    * Arabic
    * Tamil
* Localize documentation
    * I'll initially do a pass at localizing the documentation to Japanese and see if I can crowd source other languages (and fixing my initial Japanese pass)
* Update pitch feature calculation to use speaker-adjusted min and max f0 ranges

* Moving away from Kaldi-based dependencies
  * Kaldi is not being actively developed and I don't have much of a desire to depend on it long term
  * Most actively developed ASR toolkits and libraries are based around neural networks
    * I'm not the biggest fan of using these for alignment, as most of the research is geared towards improving end-to-end signal to orthographic text models that don't have intermediate representations of phones
    * That said, if alignment were the task that was being optimized for rather than some "word error rate" style metric, then alignment performance could improve significantly
      * One particular direction would be towards sample-based or waveform-based alignment rather than frame-based
        * Frame-based methods are time-smeared, so providing an exact time for voicing onset or stop closure is murky
        * Phoneticians use spectrograms for gross boundaries, but more accurate manual alignments are determined based on the waveform
      * Perhaps combining a model that performs language-independent boundary insertion combined with per-language models to combine resulting segments might perform better ({ipa_inline}`[a]` + {ipa_inline}`[j]` becomes {ipa_inline}`[aj]` in English, but not in other languages like Japanese, Spanish, or Portuguese, etc)
    * Additionally, neural networks might allow for better modeling of phone symbols, so embedding {ipa_inline}`[pʲ]` could result in a more compositional "voiceless bilabial stop plus palatalization"
  * Other options for toolkits to support MFA are
    * [SpeechBrain](https://speechbrain.github.io/)
    * Custom PyTorch code
    * Custom tensorflow code
* Update dictionary model format to move away from the current plain-text lexicons to a more robust compressed format
  * With extra meta data and capabilities in the form of phonological rules and phone groupings, it makes more sense to package those with the lexicon rather than the acoustic model
  * Another option would be to package up the lexicon (and maybe G2P models) with the acoustic model into a complete MFA model
  * As part of any update, I would expand the {ref}`MFA model CLI <pretrained_models>` with functionality for adding new pronunciations to internal lexicons
    * Something like {code}`mfa model update /path/to/g2pped_file.txt`

```{toctree}
:hidden:
:maxdepth: 1

changelog_3.2.rst
changelog_3.1.rst
news_3.0.rst
changelog_3.0.rst
changelog_2.2.rst
news_2.1.rst
changelog_2.1.rst
news_2.0.rst
changelog_2.0.rst
changelog_2.0_pre_release.rst
news_1.1.rst
changelog_1.0.rst
```
