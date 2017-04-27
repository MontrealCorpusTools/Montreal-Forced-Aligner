
$PATH_TO_PHON/phonetisaurus-align --input=$TRAINING_FILE --ofile=$TEMP_LANGUAGE/full.corpus;
ngramsymbols < $TEMP_LANGUAGE/full.corpus > $TEMP_LANGUAGE/full.syms;
farcompilestrings --symbols=$TEMP_LANGUAGE/full.syms --keep_symbols=1  $TEMP_LANGUAGE/full.corpus > $TEMP_LANGUAGE/full.far;
ngramcount --order=7 $TEMP_LANGUAGE/full.far > $TEMP_LANGUAGE/full.cnts;
ngrammake --method=kneser_ney $TEMP_LANGUAGE/full.cnts $TEMP_LANGUAGE/full.mod;
ngramprint --ARPA $TEMP_LANGUAGE/full.mod > $TEMP_LANGUAGE/full.arpa;
$PATH_TO_PHON/phonetisaurus-arpa2wfst --lm=$TEMP_LANGUAGE/full.arpa --ofile=$LANGUAGE/full.fst;
rm -f $TRAINING_FILE