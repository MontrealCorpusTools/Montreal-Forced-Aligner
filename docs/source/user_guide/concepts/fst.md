
(fst)=
# Finite State Transducers

```{warning}

Still under construction, I hope to fill these sections out as I have time.
```

```{seealso}

* [OpenFst Quick Tour](https://www.openfst.org/twiki/bin/view/FST/FstQuickTour)
```

(acceptor)=
## Acceptors

(wfst)=

## Weighted Finite State Transducers


(lexicon_fst)=
# Lexicon FSTs

MFA compiles input pronunciation dictionaries to a Weighted Finite State Transducer ({term}`WFST`), with phones as input symbols and words as output symbols.  During alignment, the {term}`lexicon FST` is composed with a linear acceptor created from the


(grammar_fst)=

# Grammar FSTs


(g2p_fst)=
# G2P FSTs

```{seealso}

* [Pynini documentation](https://www.openfst.org/twiki/bin/view/GRM/Pynini)
* [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus)
```
