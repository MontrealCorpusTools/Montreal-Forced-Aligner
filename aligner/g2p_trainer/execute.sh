
mkdir $LANGUAGE
../phonetisaurus-align --input=$TRAINING_FILE --ofile=$LANGUAGE/full.corpus;
ngramsymbols < $LANGUAGE/full.corpus > $LANGUAGE/full.syms;
farcompilestrings --symbols=$LANGUAGE/full.syms --keep_symbols=1  $LANGUAGE/full.corpus > $LANGUAGE/full.far;
ngramcount --order=7 $LANGUAGE/full.far > $LANGUAGE/full.cnts;
ngrammake --method=kneser_ney $LANGUAGE/full.cnts $LANGUAGE/full.mod;
ngramprint --ARPA $LANGUAGE/full.mod > $LANGUAGE/full.arpa;
../phonetisaurus-arpa2wfst --lm=$LANGUAGE/full.arpa --ofile=$LANGUAGE/full.fst;

