
(alignment_evaluation)=
# Evaluating alignments

Alignments can be compared to a gold-standard reference set by specifying the `--reference_directory` below. MFA will load all TextGrids and parse them as if they were exported by MFA (i.e., phone and speaker tiers per speaker).  The phone intervals will be aligned using the {mod}`Bio.pairwise2` alignment algorithm. If the reference TextGrids use a different phone set, then a custom mapping yaml file can be specified via the `--custom_mapping_path`.  As an example, the Buckeye reference alignments used in [Update on Montreal Forced Aligner performance](https://memcauliffe.com/update-on-montreal-forced-aligner-performance.html) use its own ARPA-based phone set that removes stress integers, is lower case, and has syllabic sonorants.  To map alignments generated with the `english` model and dictionary that use standard ARPA, a yaml file like the following allows for a better alignment of reference phones to aligned phones.

:::yaml
N: [en, n]
M: [em, m]
L: [el, l]
AA0: aa
AE0: ae
AH0: ah
AO0: ao
AW0: aw
:::

Using the above file, both {ipa_inline}`en` and {ipa_inline}`n` phones in the Buckeye corpus will not be penalized when matched with {ipa_inline}`N` phones output by MFA.

In addition to any custom mapping, phone boundaries are used in the cost function for the {mod}`Bio.pairwise2` alignment algorithm as follows:

:::{math}
Overlap \: cost = -1 * \biggl(\lvert begin_{aligned} - begin_{ref} \rvert + \lvert end_{aligned} - end_{ref} \rvert + \begin{cases}
        0, & label_{1} = label_{2} \\
        2, & otherwise
        \end{cases}\biggr)
:::

The two metrics calculated for each utterance are overlap score and phone error rate.  Overlap score is calculated similarly to the above cost function for each phone (excluding phones that are aligned to silence or were inserted/deleted) and averaged over the utterance:

:::{math}
Alignment \: score = \frac{Overlap \: cost}{2}
:::

Phone error rate is calculated as:

:::{math}
Phone \: error \: rate = \frac{insertions + deletions + (2 * substitutions)} {length_{ref}}
:::
