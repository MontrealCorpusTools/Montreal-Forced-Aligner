

(concept_speaker_adaptation)=
# Speaker adaptation

There are two ways that speaker information affects alignment directly through feature transforms.  The first is via calculating Cepstral Mean and Variance Normalization (CMVN) on the MFCC features.  While CMVN includes variance normalization in the name, by default MFA just normalizes the means of the coefficients.  This transform helps control a bit for the combination of speaker/microphone response/room response, all these things that can affect the frequency slope that the MFCCs are calculated from.

MFA performs two stages for typical {ref}`pretrained_alignment`. The first pass uses a speaker independent model and speaker independent features (CMVN-transformed MFCCs).  This first pass of alignments should generate alignments for at least some utterances for any given speaker. The first pass alignments are then used to calculate a transform that maps a given speaker's features into a common feature space, using the alignments to say (at a broad level), ok, for a given phone, we have these features for this speaker, but these other features for another speaker, they should be transformed this other way.  The end result is that we should have more consistent features for each phone.

It's a little more complicated than that because the aligned TextGrids you see at the end are not how the alignments are represented internally.  Instead for each utterance we have a list of integer IDs for each frame, and these integer IDs are transition IDs in the HMM-GMM model. These transition IDs map to probability distribution functions (PDFs) over features, and so it's these PDFs that are used to generate a mapping between two speakers to create per speaker transforms to common space.

It's similar in some ways to vowel normalization, since both space transforms. However, most vowel space normalization techniques don't have the same supervision from phone labels that speaker adaptation in MFA does, but the end goal is to transform speaker-dependent spaces into a common one and then use a model trained on those features to generate a final alignment.

So with that grounding, there are a couple of principles for how/whether speaker adaptation will improvement alignments:

1. If an utterance was aligned on the first pass, its alignments won't likely change much in the second pass.  This is particularly true for smaller corpora where that utterance is going to have a larger impact on the speaker transform calculation than it would in larger corpora.  In general, alignments between the two passes are mostly likely to just differ a bit in terms of assigning transitional frames to one phone or another.
2. Speaker adaptation is heavily input data dependent, it's basically a way to bootstrap utterances that don't align from utterances that did align (though it can also have negative impacts for utterances that are atypical of a speaker, falsetto, excited, yelling, etc if most of their speech is not that style).  Related to this, size of the input data matters both in the number of speakers and number of utterances per speaker, because you need some alignments for each speaker on the first pass and the more the better, and a variety of speakers is generally helpful to get to the "common" feature space more accurately.

So basically, if you align more files at a time, you should get better speaker adapted alignments than if you're repeatedly aligning a few files at a time (which is why {ref}`align_one` does not do speaker adaptation, because there's just not enough data to get meaningful improvements).

```{seealso}

* [Kaldi docs on fMLLR](https://kaldi-asr.org/doc/transform.html)
```
