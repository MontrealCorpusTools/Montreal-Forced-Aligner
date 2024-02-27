

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

MFA uses a variable 5-state topology for modeling phones.  Each state has a likelihood to transition to the final state in addition to the next state.  What this is means is that each phone has a minimum duration of 10ms (corresponding to the default time step for MFCC generation), rather than 30ms for a more standard 3-state HMM.  Having a shorter minimum duration reduces alignment errors from short or dropped phones, i.e., American English flaps or schwas, or accommodate for dictionary errors (though these should still be fixed).
