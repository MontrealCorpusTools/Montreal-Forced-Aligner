
(phonological_rules=)
# Phonological rules

MFA 2.1 has the ability to specify phonological rules as a separate input from the dictionary itself.  The idea here is that we can over-generate a really large lexicon without having to manually specify variants of commonly applying rules.  This lexicon is then paired down to just the attested forms.

Rules for languages with MFA 2.1 models can be found in [mfa-models/config/acoustic/rules](https://github.com/MontrealCorpusTools/mfa-models/tree/main/config/acoustic/rules), though not all languages have been refreshed for 2.1.

Rules are specified via yaml dictionaries like the following example of cot-caught merger from the [English MFA phonological rules file](https://github.com/MontrealCorpusTools/mfa-models/blob/main/config/acoustic/rules/english_mfa.yaml):

:::{code} yaml
rules:
  - following_context: $  # caught-cot merger
    preceding_context: ''
    replacement: ɑː
    segment: ɒː
  - following_context: '[^ɹ]'  # caught-cot merger
    preceding_context: ''
    replacement: ɑː
    segment: ɒː
  - following_context: $  # caught-cot merger
    preceding_context: ''
    replacement: ɑ
    segment: ɒ
  - following_context: '[^ɹ]'  # caught-cot merger
    preceding_context: ''
    replacement: ɑ
    segment: ɒ
:::

For this rule, I've specified 4 rules for long/short variants and for the following context. Long/short vowels are both present in the dictionary and are correlated with stress, but note that long/short variants are modeled as part of the same [phone group](phone_groups.md).  The following context involves whether the vowel occurs at the end of the word (``following_context: $``) or if it is in the middle of the word not followed by a rhotic (``following_context: '[^ɹ]'``), as "stark" and "stork" have distinct pronunciations in r-ful dialects with the merger.

These rules are compiled to regular expressions and used to replace the ``segment`` with the ``replacement``.  For deletions, the replacement field is empty and for insertions, the segment field is empty.  Additionally, both the segment and replacement fields can be sequences of segments or regular expressions themselves.  Some more complex examples:

:::{code} yaml
rules:
  - following_context: ''  # deleting d after n
    preceding_context: 'n'
    replacement: ''
    segment: d
  - following_context: '[^ʊɔɝaɛeoæɐɪəɚɑʉɒi].*'  # syllabic l
    preceding_context: ''
    replacement: ɫ̩
    segment: ə ɫ
  - following_context: ''  # schwa deletion
    preceding_context: ''
    replacement: ɹ ə
    segment: 'ə ɹ ə'
  - following_context: ''
    preceding_context: ''
    replacement: dʒ
    segment: d[ʲʷ]? ɹ
  - following_context: $  # ask metathesis
    preceding_context: ''
    replacement: k s
    segment: s k
:::
