.. _g2p:

*************************
Grapheme-to-Phoneme (G2P)
*************************

There are many cases where a language's orthography is transparent, and creating an exhaustive list of all words in a corpus
is doable by rule rather than just listing.  For these cases, we offer pretrained grapheme-to-phoneme (G2P) models, as
well as a way to train new G2P models.

Currently, the way unknown symbols are handled is not perfect.  If an unknown symbol is found, it is skipped and the pronunciation
for the rest of the orthography is generated.  Please be careful when using this system for languages with logographic writing
systems such as Chinese or Japanese where unknown symbols are likely given the number of distinct characters, and be sure to
always check the resulting dictionary carefully before potentially propagating errors into the alignment.

.. toctree::
   :maxdepth: 3

   g2p_dictionary_generating.rst
   g2p_model_training.rst