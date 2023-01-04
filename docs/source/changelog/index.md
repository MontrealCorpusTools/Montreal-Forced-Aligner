
.. _news:


# News

## Roadmap

```{warning}
   Please bear in mind that all plans below are tentative and subject to change.
```

### Version 2.1

* Generalize phone group specification for user specification
    * Phone groups are allophonic variation, common neutralization etc.
      * English {ipa_inline}`[ɾ]` is grouped with {ipa_inline}`[t]` and {ipa_inline}`[d]`, but Spanish {ipa_inline}`[ɾ]` is grouped with {ipa_inline}`[r]`, {ipa_inline}`[ð]` is grouped with {ipa_inline}`[d]`
* Update to use PyKaldi for interfacing with Kaldi components rather than relying on piped CLI commands to Kaldi binaries
  * This change should also allow for more nnet3 functionality to be available (i.e., for segmentation and speaker diarization below).  The {code}`nnet3` scripts rely on python code in the Kaldi {code}`egs/wsj/s5/steps` folder that is not currently exported as part of the Kaldi feedstock on conda forge.
* Update segmentation functionality
  * At the moment, only a simple speech activity detection (SAD) algorithm is implemented that uses amplitude of the signal and thresholds for speech vs non-speech
  * For 2.1, I plan to implement new SAD training capability as well as release a pretrained SAD model trained across all the current training data for every language with a pretrained acoustic model
* Update speaker diarization functionality
  * Support x-vector models as well as ivector models
  * Properly implement and train PLDA models for diarization
* Update dictionary model format to move away from the current plain-text lexicons to a more robust compressed format
  * With extra meta data and capabilities in the form of phonological rules and phone groupings, it makes more sense to package those with the lexicon rather than the acoustic model
  * Another option would be to package up the lexicon (and maybe G2P models) with the acoustic model into a complete MFA model
  * As part of any update, I would expand the {ref}`MFA model CLI <pretrained_models>` with functionality for adding new pronunciations to internal lexicons
    * Something like {code}`mfa model update /path/to/g2pped_file.txt`

Not tied to 2.1, but in the near-ish term I would like to:

* Retrain existing acoustic models with new phone groups and rules features
* Begin work on expanding to new languages
    * Japanese (in progress)
    * Arabic
    * Tamil
* Localize documentation
    * I'll initially do a pass at localizing the documentation to Japanese and see if I can crowd source other languages (and fixing my initial Japanese pass)
* Finally release Anchor compatible with the latest versions of MFA
* Update pitch feature calculation to use speaker-adjusted min and max f0 ranges

### Future

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

```{toctree}
:hidden:
:maxdepth: 1

news_2.1.rst
changelog_2.1.rst
news_2.0.rst
changelog_2.0.rst
changelog_2.0_pre_release.rst
news_1.1.rst
changelog_1.0.rst
```
