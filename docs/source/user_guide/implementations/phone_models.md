
(phone_models)=
# Phone model alignments

With the `--use_phone_model` flag, an ngram language model for phones will be constructed and used to generate phone transcripts with alignments.  The phone language model uses bigrams and higher orders (up to 4), with no unigrams included to speed up transcription (and because the phonotactics of languages highly constrain the possible sequences of phones).  The phone language model is trained on phone transcriptions extracted from alignments and includes silence and OOV phones.

The phone transcription additionally uses speaker-adaptation transforms from the regular alignment as well to speed up transcription.  From the phone transcription lattices, we extract phone-level alignments along with confidence score using {kaldi_src}`lattice-to-ctm-conf`.

The alignments extracted from phone transcriptions are compared to the baseline alignments using the procedure outlined in {ref}`alignment_evaluation` above.
