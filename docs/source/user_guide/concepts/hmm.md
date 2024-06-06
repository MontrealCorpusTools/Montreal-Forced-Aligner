

(hmm)=
# Hidden Markov Models

```{warning}

Still under construction, I hope to fill these sections out as I have time.
```


## Standard topology

### Kaldi topology reference

```{seealso}

* https://kaldi-asr.org/doc/hmm.html
* https://kaldi-asr.org/doc/tree_internals.html
* https://kaldi-asr.org/doc/tree_externals.html
```

### MFA topology

MFA uses a variable 3-state topology for modeling phones.  Each state has a likelihood to transition to the final state in addition to the next state.  What this is means is that each phone has a minimum duration of 10ms (corresponding to the default time step for MFCC generation), rather than 30ms for a more standard 3-state HMM.  Having a shorter minimum duration reduces alignment errors from short or dropped phones, i.e., American English flaps or schwas, or accommodate for dictionary errors (though these should still be fixed).

#### Customizing topologies

Custom numbers of states can be specified via a topology configuration file. The configuration file should list per-phone minimum and maximum states, as below.

```{code}yaml
tʃ:
  - min_states: 3
  - max_states: 5
ɾ:
  - min_states: 1
  - max_states: 1
```

In the above example, the {ipa_inline}`[tʃ]` phone will have a variable topology with a minimum 3 states before terminating, but optional 5 states to cover additional transitions for the complex articulation. Conversely, the {ipa_inline}`[ɾ]` phone is a very short articulation and so having both minimum and maximum set to 1 state ensures that additional states are not used to model the phone.

```{seealso}
* [Example configuration files](https://github.com/MontrealCorpusTools/mfa-models/tree/main/config/acoustic/topologies)
```

## Clustering phones

In a monophone model, each phone is modeled the same regardless of the surrounding phonological context. Consider the {ipa_inline}`[P]` in the words {ipa_inline}`paid [P EY1 D]` and {ipa_inline}`spade [S P EY1 D]` in English. The actual pronunciation of the {ipa_inline}`[P]` in paid will be an aspirated {ipa_inline}`[pʰ]` but the pronunciation of {ipa_inline}`[P]` following {ipa_inline}`[S]` is an unaspirated {ipa_inline}`[p]`.

To more accurately model these phonological variants, we use triphone models. Under the hood, each phone gets transformed into a sequence of three phones, including the phone and its preceding and following phones. So the representation for "paid" and "spade" becomes {ipa_inline}`[#/P/EY1 P/EY1/D EY1/D/#]` and {ipa_inline}`[#/S/P S/P/EY1 P/EY1/D EY1/D/#]`. At this level, we have made it so that the {ipa_inline}`[P]` phones have two different labels, so each can be modeled differently.

However, representing phones this way results in a massive explosion of the number of phones, with not as many corresponding occurrences. If there is not much data for particular phones, modeling them appropriately becomes challenging. The solution to this data sparsity issue is to cluster the resulting states based on their similarity. For the triphone {ipa_inline}`[P/EY1/D]`, triphones like {ipa_inline}`[P/EY1/T]`, {ipa_inline}`[B/EY1/D]`,{ipa_inline}`[M/EY1/D]`,{ipa_inline}`[B/EY1/T]`, and {ipa_inline}`[M/EY1/T]` will all have similar acoustics, as they're {ipa_inline}`[EY1]` vowels with bilabial stops preceding and oral coronal stops following. The triphone {ipa_inline}`[P/EY1/N]` and others with following nasals will likely not be similar enough due to regressive nasalization in English.

As a result of the phone clustering, the number of PDFs being modeled is reduced to a more manageable number with less data sparsity issues.

```{note}
By default Kaldi and earlier versions MFA included silence phones with the nonsilence phones, due to the idea for instance that stops have a closure state to them and so that is similar to silence. However, having silence states be clustered with nonsilence states has led to gross alignment errors with less clean data, so MFA 3.1 and later removes all instances of the silence phone being clustered with nonsilence phones. The OOV phone is still clustered with both silence and nonsilence however, and OOVs can cover multiple words.
```