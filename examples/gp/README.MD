These are notes from Michael McAuliffe (michael.e.mcauliffe@gmail.com)
about issues and things that need to be fixed in the GlobalPhone corpora
in order for successful alignment to be possible.

Issues with GlobalPhone corpora:

* See appendix A of https://www.era.lib.ed.ac.uk/bitstream/handle/1842/5773/Lal2011.pdf

French:

* Pronunciation dictionary has a newline character in the entry for "causer"

Korean:

* Transcript for speaker 34 lacks entries (rmn file only has the final
few entries)

* Rename Korean ".word" file to ".txt"

Portuguese:

* Rename Portuguese ".utf8.allwords" to ".txt"

Vietnamese:

* Pronunciation dictionary has tone as separate segments in transcriptions

Japanese:

* Initial entries lack phones, changed to:

  * {#ARTIK#} {{M_+QK WB}}
  * {#ATMEN#} {{M_+QK WB}}
  * {#MUELL#} {{M_+QK WB}}
  * {#N_ARTIK#} {{M_+QK WB}}
  * {#SCHMATZEN#} {{M_+QK WB}}

Arabic:

* AR037.rmn remove spaces from beginnings of lines 161 and 163

Croatian:

* Remove all periods from abbreviations in dictionary (e.g., mr., dr., sv., etc)
