
(probabilistic_lexicons)=
# Probabilistic lexicons

(pronunciation_probabilities)=
## Pronunciation probabilities

Pronunciation probabilities are estimated based on the counts of a specific pronunciations normalized by the count of the most frequent pronunciation. Counts are estimated using add-one smoothing.

:::{math}
p(w.p_{i} | w) = \frac{c(w.p_{i} | w)}{max_{1\le i \le N_{w}}c(w.p_{i} | w)}
:::

The reason for using max normalization is to not penalize words with many pronunciations. Even though the probabilities no longer sum to 1, the log of the probabilities is used as summed costs in the {term}`lexicon FST`, so summing to 1 within a word is not problematic.

If a word is not seen in the training data, pronunciation probabilities are not estimated for its pronunciations.


(silence_probability)=
## Silence probability and correction factors

Words differ in their likelihood to appear before or after silence. In English, a word like "the" is more likely to appear after silence than a word like "us". An pronoun in the accusative case like "us" is not grammatical at the start of a sentence or phrase, whereas "the" starts sentences and phrases regularly.  That is not to say that a speaker would not pause before saying "us" for paralinguistic effect or due to a disfluency or simple pause, it's just less likely to occur after silence than "the".

By the same token, silence following "the" is also less likely than for "us" due to syntax, but pauses are more likely to follow some pronunciations of "the" than others.  For instance, if a speaker produces a full vowel variant like {ipa_inline}`[ð i]`, a pause is more likely to follow than a reduced variant like {ipa_inline}`[ð ə]`.  The reduced variant will be more likely overall, but it often occurs in running connected speech at normal speech rates. The full vowel variant is more likely to occur in less connected speech, such as when the speaker is planning upcoming speech or speaking more slowly.  Accounting for the likelihood of silence before and after a variant allows the model to output a variant that is less likely overall, but more likely given the context.

However, when we take into account more context, it is not just the single word that determines the likelihood of silence, but also the preceding/following words.  The difficulty in estimating the overall likelihood of silence is that the lexicon FST is predicated on each word being independent and composable with any preceding/following word. Thus, for each word, we estimate a probability of silence following (independent of the following words), and two correction factors for silence and non-silence before.  The two correction factors take into account the general likelihood of silence following each of the preceding words and gives two factors that represent "is silence or not silence more likely than we would expect given the previous word". These factors are only an approximation, however they do help in alignment.

:::{note}
Probabilities of multi-word sequences are the domain of the {term}`grammar FST`, please refer [grammar FST concept section](grammar_fst)`.
:::

MFA uses three variables to capture the probabilities of silence before and after a pronunciation. The most straightforward is ``probability of silence following``, which is calculated as the count of instances where the word was followed by silence divided by the overall count of that pronunciation, with a smoothing factor. Reproducing equation 3 of [Chen et al (2015)](https://www.danielpovey.com/files/2015_interspeech_silprob.pdf):

:::{math}
   P(s_{r} | w.p) = \frac{C(w.p \: s) + \lambda_{2}P(s)}{C(w.p) + \lambda_{2}}
:::

Given that we're using a lexicon where words are assumed to be completely independent, modelling the silence before the pronunciation is a little tricky.  The approach used in [Chen et al (2015)](https://www.danielpovey.com/files/2015_interspeech_silprob.pdf) is to estimate two correction factors for silence and non-silence before the pronunciation.  These correction factors capture that for a given pronunciation, it is more or less likely than average to have silence.  The factors are estimated as follows, reproducing equations 4-6 from [Chen et al (2015)](https://www.danielpovey.com/files/2015_interspeech_silprob.pdf):


:::{math}
F(s_{l} | w.p) = \frac{C(s \: w.p) + \lambda_{3}}{\tilde{C}(s \: w.p) + \lambda_{3}}

F(ns_{l} | w.p) = \frac{C(ns \: w.p) + \lambda_{3}}{\tilde{C}(ns \: w.p) + \lambda_{3}}

\tilde{C}(s \: w.p) = \sum_{v} C(v \: w.p) P(s_r|v)
:::

The estimate count {math}`\tilde{C}` represents a "mean" count of silence or non-silence preceding a given pronunciation, taking into account the likelihood of silence from the preceding pronunciation.  The correction factors are weights on the FST transitions from silence and non-silence state.

Consider the following {term}`FST` with  three pronunciations of "lot" from the [English US MFA dictionary](https://mfa-models.readthedocs.io/en/latest/dictionary/English/English%20%28US%29%20MFA%20dictionary%20v2_0_0a.html#English%20(US)%20MFA%20dictionary%20v2_0_0a).


:::{figure} ../../_static/lot.svg
:align: center

{term}`FST` for three pronunciations of "lot" in the English US dictionary
:::



## Example

As an example, consider the following English and Japanese sentences:

::::{tab-set}

:::{tab-item} English
:sync: english
The red fox has read many books, but there's always more to read.
Normalized:
the red fox has read many books but there 's always more to read
:::

:::{tab-item} Japanese
:sync: japanese
アカギツネさんは本を読んだことがたくさんありますけれども、読むべき本はまだまだいっぱい残っています。
Normalized:
アカギツネ さん は 本 を 読んだ こと が たくさん あり ます けれども 読む べき 本 は まだまだ いっぱい 残って い ます
:::

::::

For each of the above sentences (please pardon my Japanese), I recorded a normal speaking rate version and a fast speaking rate version.  The two speech rates induce variation in pronunciation, as well as different pause placement.  We'll then walk through the calculations that result in the final trained lexicon.

:::::{tab-set}

::::{tab-item} English
:sync: english

:::{raw} html

 <div class="align-center">
 <audio controls="controls">
 <source src="../../_static/sound_files/english_slow.wav" type="audio/wav">
 Your browser does not support the <code>audio</code> element.</audio>
 </div>
:::

:::{figure} ../../_static/sound_files/english_slow.svg
:align: center
Waveform, spectrogram, and aligned labels for the slow reading of the English text
:::

:::{raw} html
<div class="align-center">
<audio controls="controls">
<source src="../../_static/sound_files/english_fast.wav" type="audio/wav">
Your browser does not support the <code>audio</code> element.</audio>
</div>
:::

:::{figure} ../../_static/sound_files/english_fast.svg
:align: center
Waveform, spectrogram, and aligned labels for the fast reading of the English text
:::

::::

::::{tab-item} Japanese
:sync: japanese

:::{raw} html
<div class="align-center">
<audio controls="controls">
<source src="../../_static/sound_files/japanese_slow.wav" type="audio/wav">
Your browser does not support the <code>audio</code> element.
</audio>
</div>
:::

:::{figure} ../../_static/sound_files/japanese_slow.svg
:align: center
Waveform, spectrogram, and aligned labels for the slow reading of the Japanese text
:::

:::{raw} html
<div class="align-center">
<audio controls="controls">
<source src="../../_static/sound_files/japanese_fast.wav" type="audio/wav">
Your browser does not support the <code>audio</code> element.
</audio>
</div>
:::

:::{figure} ../../_static/sound_files/japanese_fast.svg
:align: center

Waveform, spectrogram, and aligned labels for the fast reading of the Japanese text
:::

::::

:::::

For alignment, we use the following pronunciation dictionaries, taking pronunciation variants from the [English US MFA dictionary](https://mfa-models.readthedocs.io/en/latest/dictionary/English/English%20%28US%29%20MFA%20dictionary%20v2_0_0a.html#English%20(US)%20MFA%20dictionary%20v2_0_0a) and the [Japanese MFA dictionary](https://mfa-models.readthedocs.io/en/latest/dictionary/Japanese/Japanese%20MFA%20dictionary%20v2_0_0.html#Japanese%20MFA%20dictionary%20v2_0_0).

:::::{tab-set}

::::{tab-item} English
:sync: english

In addition to lexical variants for the present and past tense of "read", function words have several variants listed. The genitive marker "'s" has variants to account for stem-final voicing ({ipa_inline}`[s]` and {ipa_inline}`[z]`) and stem-final alveolar obstruents ({ipa_inline}`[ɪ z]`). The negative conjuction "but" has variants for the pronunciation of the vowel and final {ipa_inline}`/t/` as {ipa_inline}`[ʔ]` or {ipa_inline}`[ɾ]`. Likewise, the preposition "to" has variants for the initial {ipa_inline}`/t/` and vowel reductions.  The definite determiner "the" and distal demonstrative "there" have variants for stopping {ipa_inline}`/ð/` to {ipa_inline}`[d̪]`, along with reductions for vowels.

:::{csv-table} English US pronunciation dictionary
:widths: 30,70
:header-rows: 1
:stub-columns: 1
:class: table-striped table-bordered dataTable table
"Word","Pronunciation"
"'s","s"
"'s","z"
"'s","ɪ z"
"always","ɒː ɫ w ej z"
"always","ɑː ɫ w ej z"
"always","ɒː w ej z"
"always","ɑː w ej z"
"books","b ʊ k s"
"but","b ɐ t"
"but","b ɐ ʔ"
"but","b ə ɾ"
"fox","f ɑː k s"
"has","h æ s"
"has","h æ z"
"many","m ɛ ɲ i"
"more","m ɒː ɹ"
"read","ɹ iː d"
"read","ɹ ɛ d"
"red","ɹ ɛ d"
"the","d̪ iː"
"the","d̪ iː ʔ"
"the","d̪ ə"
"the","iː"
"the","iː ʔ"
"the","l ə"
"the","n ə"
"the","s ə"
"the","ð iː"
"the","ð iː ʔ"
"the","ð ə"
"the","ə"
"there","d̪ ɚ"
"there","d̪ ɛ ɹ"
"there","ð ɚ"
"there","ð ɛ ɹ"
"to","t ə"
"to","tʰ ʉː"
"to","tʰ ʊ"
"to","ɾ ə"
:::

::::

::::{tab-item} Japanese
:sync: japanese

The main pronunciation variants are in the topic particle "は", the object particle "を", the adjective "たくさん", and the "but" conjunction "けれども". The particles are always pronounced as {ipa_inline}`[w a]` and {ipa_inline}`[o]` and never as their hiragana readings {ipa_inline}`[h a]` and {ipa_inline}`[w o]`, respectively.  For "ました", I've included various levels of devoicing for {ipa_inline}`/i/` between the voiceless obstruents from full voiced {ipa_inline}`[i]`, to devoiced {ipa_inline}`[i̥]` to deleted.

:::{csv-table} Japanese pronunciation dictionary
:widths: 30, 70
:header-rows: 1
:stub-columns: 1
:class: table-striped table-bordered
"Word","Pronunciation"
"アカギツネ","a k a ɟ i ts ɨ n e"
"さん","s a ɴ"
"は","h a"
"は","w a"
"本","h o ɴ"
"を","o"
"を","w o"
"読んだ","j o n d a"
"こと","k o t o"
"が","ɡ a"
"あり","a ɾʲ i"
"ます","m a s ɨ"
"ます","m a s ɨ̥"
"ます","m a s"
"たくさん","t a k ɯ̥ s a ɴ"
"たくさん","t a k s a ɴ"
"たくさん","t a k ɯ s a ɴ"
"けれども","k e ɾ e d o m o"
"けれども","k e d o m o"
"けれども","k e d o"
:::

::::

:::::
The basic steps to calculating pronunciation and silence probabilities is as follows:

1. Generate word-pronunciation pairs (along with silence labels) from the alignment lattices
2. Use these pairs as input to [calculating pronunciation probability](pronunciation_probabilities) and [calculating silence probability](silence_probability).  See the results table below for a walk-through of results for various words across the two reading passage styles.

:::::{tab-set}

::::{tab-item} English
  :sync: english


:::{csv-table} Trained English US pronunciation dictionary
:widths: 10,18,18,18,18,18
:header-rows: 1
:stub-columns: 1
:class: table-striped table-bordered
"Word", "Pronunciation probability", "Probability of silence after", "Correction for silence before", "Correction for non-silence before","Pronunciation"
"'s",0.33,0.18,1.0,1.0,"s"
"'s",0.99,0.09,0.92,1.05,"z"
"'s",0.33,0.18,1.0,1.0,"ɪ z"
"always",0.99,0.09,0.92,1.05,"ɒː ɫ w ej z"
"always",0.33,0.18,1.0,1.0,"ɑː ɫ w ej z"
"always",0.33,0.18,1.0,1.0,"ɒː w ej z"
"always",0.33,0.18,1.0,1.0,"ɑː w ej z"
"books",0.99,0.34,0.92,1.05,"b ʊ k s"
"but",0.99,0.46,1.28,0.75,"b ɐ t"
"but",0.99,0.12,0.85,1.13,"b ɐ ʔ"
"but",0.5,0.18,1.0,1.0,"b ə ɾ"
"fox",0.99,0.09,0.92,1.05,"f ɑː k s"
"has",0.33,0.18,1.0,1.0,"h æ s"
"has",0.99,0.09,0.92,1.05,"h æ z"
"many",0.99,0.09,0.92,1.05,"m ɛ ɲ i"
"many",0.33,0.18,1.0,1.0,"mʲ ɪ ɲ i"
"more",0.99,0.09,0.92,1.05,"m ɒː ɹ"
"read",0.99,0.59,0.92,1.05,"ɹ iː d"
"read",0.99,0.09,0.92,1.05,"ɹ ɛ d"
"red",0.99,0.09,0.89,1.06,"ɹ ɛ d"
"the",0.5,0.18,1.0,1.0,"d̪ iː"
"the",0.5,0.18,1.0,1.0,"d̪ iː ʔ"
"the",0.5,0.18,1.0,1.0,"d̪ ə"
"the",0.5,0.18,1.0,1.0,"iː"
"the",0.5,0.18,1.0,1.0,"iː ʔ"
"the",0.5,0.18,1.0,1.0,"l ə"
"the",0.5,0.18,1.0,1.0,"n ə"
"the",0.5,0.18,1.0,1.0,"s ə"
"the",0.99,0.12,1.49,0.67,"ð iː"
"the",0.5,0.18,1.0,1.0,"ð iː ʔ"
"the",0.99,0.12,1.49,0.67,"ð ə"
"the",0.5,0.18,1.0,1.0,"ə"
"there",0.33,0.18,1.0,1.0,"d̪ ɚ"
"there",0.33,0.18,1.0,1.0,"d̪ ɛ ɹ"
"there",0.33,0.18,1.0,1.0,"ð ɚ"
"there",0.99,0.09,1.37,0.65,"ð ɛ ɹ"
"to",0.99,0.09,0.92,1.05,"t ə"
"to",0.33,0.18,1.0,1.0,"tʰ ʉː"
"to",0.33,0.18,1.0,1.0,"tʰ ʊ"
"to",0.33,0.18,1.0,1.0,"ɾ ə"
:::

**Pronunciation probabilities**

Using the alignments above for the two speech rates, the word "red" has 0.99 pronunciation probability as that's the only pronunciation variant.  The word "read" pronounced as {ipa_inline}`[ɹ ɛ d]` has 0.99 probability, as will the pronunciation as {ipa_inline}`[ɹ iː d]`, as they both appeared once in the sentence (and twice across the two speech rates), but note that it is not 0.5, as the probabilities are max-normalized.  Both full and reduced forms of "but" ({ipa_inline}`[b ɐ t]` and {ipa_inline}`[b ɐ ʔ]`) have pronunciation probability of 0.99, as they each occur once across the passages.

:::{note}

I'm not sure why the {ipa_inline}`[b ɐ ʔ]` variant is chosen over the {ipa_inline}`[b ə ɾ]`, this will require future investigation to figure out a root cause.
:::

All other words will have one pronunciation with 0.99, if they have one realized pronunciation, unrealized pronunciations will have a smoothed probability close to 0, based on the number of pronunciations.

:::{note}

 "Unrealized pronunciations" refer to pronunciation variants that are not represented in training data, i.e., for the word "to", only the {ipa_inline}`[t ə]` was used, so {ipa_inline}`[tʰ ʉː]`, {ipa_inline}`[tʰ ʊ]`, and {ipa_inline}`[ɾ ə]` are unrealized.
:::

**Probabilities of having silence following**

The word "books" has a probability of silence following at 0.34, as it only occurs before silence in the slower speech rate sentence. You might expect it to have a silence probability of 0.5, but recall from the equation of {math}`P(s_{r} | w.p)`, the smoothing factor is influenced by the overall rate of silence following words, which is quite low for the sentences with connected speech.

The pronunciation of "read" as {ipa_inline}`[ɹ iː d]` has a higher probability of silence following of 0.59, as both instances of that pronunciation are followed by silence at the end of the sentence.  The pronunciation of "read" as {ipa_inline}`[ɹ ɛ d]` will have a probability of following silence of 0.09, as the only instances are in the middle of speech in the first clause.

**Probabilities of having silence preceding**

Both pronunciations present of word "the" ({ipa_inline}`[ð iː]` and {ipa_inline}`[ð ə]`) have a silence preceding correction factor (1.49) greater than the non-silence correction factor (0.67), as it only appears after silence in both speech rates.  With the non-silence correction factor below 1, the cost in the FST of transitioning out of the non-silence state will be much higher than transitioning out of the silence state. When the silence correction factor is greater than 1, the pronunciation is more likely following silence than you would expect given all the previous words, which will reduce the cost of transitioning out of the silence state.

The fuller form of the word "but" ({ipa_inline}`[b ɐ t]`) has a silence preceding correction factor (1.28) greater than the non-silence correction factor (0.75), so the full form will have lower cost transitioning out of the silence state and than the non-silence state. On the other hand, the more reduced form {ipa_inline}`[b ɐ ʔ]` has the opposite patten, with a silence before correction factor (0.85) greater than the non-silence correction factor (1.13), so the reduced form will have a lower cost transitioning out of the non-silence state than the silence state.

::::

::::{tab-item} Japanese
  :sync: japanese

:::{warning}

The Japanese walk-through of the pronunciation probability results is still under construction.
:::

::::

:::::

The resulting trained dictionary can then be used as a dictionary for [alignment](pretrained_alignment) or [transcription](transcribing).
