
# Phone groups

When training an acoustic model, MFA begins by training a monophone model, where each phone is context-independent. Consider an English ARPABET model as an example. A {ipa_inline}`[T]` is modeled the same regardless of:
* Whether it's word initial
* Whether it follows an {ipa_inline}`[S]`
* Whether it's the onset of a stressed syllable
* Whether it's the onset of an unstressed syllable
* Whether it's word final
* Whether it's followed by {ipa_inline}`[R]`

For each of these cases, the acoustic model will proceed through the same HMM states with the same GMM PDFs (Probability Distribution Functions).



:::::{tab-set}

::::{tab-item} Full utterance

The truck righted itself just before it tipped over onto it's top and came to a full stop.

:::{raw} html

 <div class="align-center">
 <audio controls="controls">
 <source src="../../_static/sound_files/english_t.wav" type="audio/wav">
 Your browser does not support the <code>audio</code> element.</audio>
 </div>
:::

:::{figure} ../../_static/sound_files/english_t.svg
:align: center
Waveform, spectrogram, and aligned labels for the full reading of the English text
:::

::::

::::{tab-item} truck

:::{figure} ../../_static/sound_files/english_t_truck.svg
:align: center
Waveform, spectrogram, and aligned labels for the word "truck", realized as {ipa_inline}`[tʃ]`
:::

::::

::::{tab-item} righted

:::{figure} ../../_static/sound_files/english_t_righted.svg
:align: center
Waveform, spectrogram, and aligned labels for the word "righted", realized as {ipa_inline}`[ɾ]`
:::

::::

::::{tab-item} itself

:::{figure} ../../_static/sound_files/english_t_itself.svg
:align: center
Waveform, spectrogram, and aligned labels for the word "itself", realized as {ipa_inline}`[t̚]`
:::

::::

::::{tab-item} just

:::{figure} ../../_static/sound_files/english_t_just.svg
:align: center
Waveform, spectrogram, and aligned labels for the word "just"
:::

::::

::::{tab-item} it

:::{figure} ../../_static/sound_files/english_t_it.svg
:align: center
Waveform, spectrogram, and aligned labels for the word "it", realized as {ipa_inline}`[t̚]`
:::

::::

::::{tab-item} tipped

:::{figure} ../../_static/sound_files/english_t_tipped.svg
:align: center
Waveform, spectrogram, and aligned labels for the word "tipped", realized as {ipa_inline}`[tʰ]`
:::

::::

::::{tab-item} onto

:::{figure} ../../_static/sound_files/english_t_onto.svg
:align: center
Waveform, spectrogram, and aligned labels for the word "onto", realized as {ipa_inline}`[tʰ]`
:::

::::

::::{tab-item} it's

:::{figure} ../../_static/sound_files/english_t_it's.svg
:align: center
Waveform, spectrogram, and aligned labels for the word "it's", realized as {ipa_inline}`[t]`
:::

::::

::::{tab-item} top

:::{figure} ../../_static/sound_files/english_t_top.svg
:align: center
Waveform, spectrogram, and aligned labels for the word "top", realized as {ipa_inline}`[tʰ]`
:::

::::

::::{tab-item} to

:::{figure} ../../_static/sound_files/english_t_to.svg
:align: center
Waveform, spectrogram, and aligned labels for the word "to", realized as {ipa_inline}`[tʰ]`
:::

::::

::::{tab-item} stop

:::{figure} ../../_static/sound_files/english_t_stop.svg
:align: center
Waveform, spectrogram, and aligned labels for the word "stop", realized as {ipa_inline}`[t]`
:::

::::

:::::

Given the range of acoustic realizations of {ipa_inline}`[T]` for the utterance above, modeling all occurrences as the same sequence of three HMM states doesn't make a ton of sense.  One aspect of the MFA ARPA model that adds some accounting for this variation is the use of position dependent phones, so rather than a single {ipa_inline}`[T]`, you actually have {ipa_inline}`[T_B]` (at the beginnings of words), {ipa_inline}`[T_E]` (at the ends of words), {ipa_inline}`[T_I]` (in the middle of words), and {ipa_inline}`[T_S]` (word consists of just {ipa_inline}`[T_S]`, doesn't really apply for {ipa_inline}`[T]`, but is more relevant for vowels like {ipa_inline}`[AY1_S]`).  So final realizations won't be modelled the same as initial realizations or those in the middle of words, each of which will have its own HMM states and GMM PDFs.  This carries its own drawback, as sometimes a final or intermediate {ipa_inline}`[T]` is realized the same as an initial {ipa_inline}`[T]` (i.e. {ipa_inline}`[tʰ]`), but there's no pooling across the positions, so {ipa_inline}`[T_E]` and {ipa_inline}`[T_I]` HMM-GMMs do not contain any learned stats from the {ipa_inline}`[T_S]`.

Moving on from monophones which by definition cannot account well for coarticulation and contextual variability, the next stage of MFA training uses triphones.  Triphones are essentially strings of three phones to represent a phone. So for a word like stop, the monophone string would be {ipa_inline}`[S T AA1 P]`, but the corresponding triphone string would be {ipa_inline}`[_/S/T S/T/AA1 T/AA1/P AA1/P/_]`, where the original {ipa_inline}`[T]` is no longer the same as all other instances of {ipa_inline}`[T]`, but instead is only the same as {ipa_inline}`[T]` preceded by {ipa_inline}`[S]` and followed by {ipa_inline}`[AA1]`.  As a result of taking the preceding and following context into account, you now have a ton of different phone symbols that are each modeled differently and have different amounts of data.  A triphone like {ipa_inline}`[S/T/AA1]` might be decently common, but one like {ipa_inline}`[S/T/AA2]` would not have much data given the rarity of {ipa_inline}`[AA2]` in transcriptions.  However, we'd really like to pool the data across these and other triphones as the key aspect for modeling the {ipa_inline}`[T]` in this case is that it is preceded by {ipa_inline}`[S]`, and followed by a vowel, not so much what quality the vowel has.

So instead of taking each triphone string as a separate phone, these triphones are clustered to make a decision tree based on the previous and following contexts.  These decision trees should learn that if a {ipa_inline}`[T]` is preceded by {ipa_inline}`[S]`, then use PDFs related to the unaspirated {ipa_inline}`[t]` realization, if it's at the beginning of a word followed by a vowel, use the PDFs related to {ipa_inline}`[tʰ]` realization, etc. By clustering PDFs into similar ones and making decision trees based on context, we can side step the sparsity issue related to blowing up the inventory of sounds with trigrams, and we can explicitly include groups of phones together that should be modeled in the same way.

These phone groups specify what phone symbols should use the same decision trees in modeling.  For position dependent phone modeling, it naturally follows that we should put all positions under the same root, so {ipa_inline}`[T_B]`, {ipa_inline}`[T_E]`, {ipa_inline}`[T_I]` and {ipa_inline}`[T_S]` can benefit from data associated with other positions, while still having some bias towards particular realizations (as the decision tree takes the central symbol into account as well as the preceding and following).

In MFA 2.1, you can now specify what phones should be grouped together, rather than specifying arbitrary phone sets like ``IPA`` or ``ARPA`` as in MFA 2.0.  There are baseline versions of these phone groups available in [mfa-models/config/acoustic/phone_groups](https://github.com/MontrealCorpusTools/mfa-models/tree/main/config/acoustic/phone_groups).  The [English US ARPA phone group](https://github.com/MontrealCorpusTools/mfa-models/blob/main/config/acoustic/phone_groups/english_arpa.yaml) gives the same phone groups that were used in training the [English (US) ARPA 2.0 models](https://mfa-models.readthedocs.io/en/latest/acoustic/English/English%20%28US%29%20ARPA%20acoustic%20model%20v2_0_0a.html#English%20(US)%20ARPA%20acoustic%20model%20v2_0_0a), while the MFA phone set ones are a bit more subject to change as I iterate on them.

A general rule of thumb that I follow is to keep phonetically similar-ish phones in the same group, so for [English MFA phone group](https://github.com/MontrealCorpusTools/mfa-models/blob/main/config/acoustic/phone_groups/english_mfa.yaml), I've added phonetic variants to dictionary and specified [phonological rules](phonological_rules.md) for adding more variation to it, but most of these share phone groups with their root phone.  So variants like {ipa_inline}`[t tʰ tʲ tʷ]` are grouped together, but less similar variants like {ipa_inline}`[ɾ]` and {ipa_inline}`[ʔ]` have their own phone groups (shown in the excerpt below).  Similar dialectal variants variants like {ipa_inline}`[ow əw o]` are grouped together as well.

:::{code} yaml
-
  - t
  - tʷ
  - tʰ
  - tʲ
-
  - d
  - dʲ
-
  - ɾ
  - ɾʲ
-
  - ʔ
:::

The default phone groups without any custom yaml file or phone set type specified is to treat each phone as its own phone group. Regardless of how phone groups are set up, if ``position_dependent_phones`` is specified, then each phone's phone group will contain all the various positional phone variants.
