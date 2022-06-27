
Glossary
========

.. glossary::
   :sorted:

   Acoustic model
   Acoustic models
   GMM-HMM
        Acoustic models calculate how likely a phone is given the acoustic features (and previous and following states).  The architecture used in MFA is Gaussian mixture models with hidden markov models (GMM-HMM).  The GMM component models the distributions of acoustic features per phone (well, really many distributions that map to phones in a many-to-many mapping), and then the HMM component tracks the transition probabilities between states.  State of the art approaches to acoustic modeling used deep neural networks, either in a hybrid DNN-HMM framework, or more recently, doing away with phone labels entirely to just model acoustics to words or subwords.

   Language model
   Language models
        Language models calculate how likely a string of words is to occur, given the data they were trained on.  They are typically generated over large text corpora.  The architecture used in MFA is that of an N-Gram model (typically trigram), with a window of N-1 previous words that predict the current word.  State of the art methods are typically RNN or transformer based approaches.

   Pronunciation dictionary
   Pronunciation dictionaries
        Pronunciation dictionaries are used to map words to phones that are aligned.  The phone set used in the dictionary must match that of the :term:`acoustic model` used, since the acoustic model will not be able to estimate probabilities for a phone label if it wasn't trained on it.  :term:`G2P models` can be used to generate pronunciation dictionaries.

   Grapheme-to-Phoneme
   G2P model
   G2P models
        G2P models generate sequences of phones based on an orthographic representation.  Typically, the more transparent the orthography, the better the pronunciations generated.  The architecture used in MFA is that of a weight Finite State Transducer (wFST), based on :xref:`pynini`.  More state of the art approaches use DNNs in a sequence-to-sequence task to get better performance, either RNNs or transformers.

   TextGrid
   TextGrids
        File format that can be used to mark time aligned utterances, and is the output format for alignments in MFA.  See :xref:`praat` for more details about TextGrids and their use in phonetics.

   MFCCs
        :abbr:`Mel-frequency cepstrum coefficients (MFCCs)` are the industry standard for acoustic features.  The process involves windowing the acoustic waveform, scaling the frequencies into the Mel space (an auditory representation that gives more weight to lower frequencies over higher frequencies), and then performs a :abbr:`discrete cosine transform (DCT)` on the values in each filter bank to get orthogonal coefficients.  There was a trend around 2015-2018 to use acoustic features that were more raw (i.e., not transformed to the Mel space, or the waveform directly), but in general most recent state of the art systems still use MFCC features.

   WFST
   FST
      A :abbr:`Finite State Transducer (FST)` is a graph formalism that can transform a sequence of arbitrary input symbols into arbitrary output symbols.  A :abbr:`Weighted Finite State Transducer (WFST)` is an FST that has costs associated with its various paths, so a single best output string can be selected.  Training graphs are WFSTs of the lexicon WFST composed with linear acceptors of the transcription text.  For transcription, lexicons are composed with language models as well.  MFA's :term:`G2P models` are WFSTs trained using a pair ngram algorithm or the many to many Phonetisaurus algortithm.

   lexicon FST
      A :term:`WFST` constructed from a pronunciation dictionary that can be composed with :term:`grammar FST` and HMM-GMM acoustic model to align and transcribe speech.

   grammar FST
      A :term:`WFST` compiled from a language model that represents how likely a word is given the previous words (ngram model), or a linear acceptor from a known utterance transcription where there is only one path through the words in the transcript for use in alignment.

   Pronunciation probabilities
        Pronunciation probabilities in dictionaries allow for certain spoken forms to be more likely, rather than just assigning equal weight to all pronunciation variants.

   Ivectors
   Ivector extractor
   Ivector extractors
        Ivectors are generated based off acoustic features like MFCCs, but are trained alongside a universal background model to be a representation of a speaker.
