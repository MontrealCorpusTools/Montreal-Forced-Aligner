.. _remap:

Utilities for remapping files to new phone set ``(mfa remap)``
==============================================================


.. _remap_dictionary:

Remap a dictionary to new phone set ``(mfa remap dictionary)``
--------------------------------------------------------------

If you have a mismatch in the phone sets used in your dictionary file and acoustic model, you can use this command to command to generate pronunciations with the new phone set.

Command reference
`````````````````

.. click:: montreal_forced_aligner.command_line.remap:remap_dictionary_cli
   :prog: mfa remap dictionary
   :nested: full

Configuration reference
```````````````````````

- :ref:`configuration_global`

API reference
`````````````

- :class:`~montreal_forced_aligner.dictionary.remapper.DictionaryRemapper`

.. _remap_alignments:

Remap aligned TextGrids to new phone set ``(mfa remap alignments)``
-------------------------------------------------------------------

This command will remap phones in any phone tier to new phones based on a phone mapping yaml file.

Phone mapping yaml files for remapping alignments are more strict than for :ref:`remap_dictionary` or :ref:`alignment_evaluation`.  For those usages, it is possible to have phones map to more than one option, and extra pronunciations will be generated/allowed.  For alignment, as there must be one label interval, the phone mapping yaml files must likewise not have any variation in what phones they map to.  If additional phones are supplied as variations, only the first phone will be used in remapping, and the rest ignored.

Mapping files should be of the format ``SOURCE_PHONE: TARGET PHONE``.  As an example, consider the case of a file aligned using ``english_mfa`` acoustic model and wanting to generate ARPA labels from the alignments, see below:

.. code-block:: yaml

   aj: AY1
   aw: AW1
   b: B
   bʲ: B
   c: K
   cʰ: K
   cʷ: K
   d: D
   dʒ: JH
   dʲ: D
   ej: EY1
   f: F
   fʲ: F
   h: HH
   i: IY0
   iː: IY1
   ...

Command reference
`````````````````

.. click:: montreal_forced_aligner.command_line.remap:remap_alignments_cli
   :prog: mfa remap alignments
   :nested: full

Configuration reference
```````````````````````

- :ref:`configuration_global`

API reference
`````````````

- :class:`~montreal_forced_aligner.corpus.remapper.AlignmentRemapper`
