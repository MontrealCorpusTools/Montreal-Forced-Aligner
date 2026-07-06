#### Details for {{ dialect | default("[Dialect]", true)  }} dictionary and G2P model

- **Source:** {{ source | default("wikipron", true)}}
- **Orthography:** {{ orthography | default("[More Information Needed]", true)}}
- **Phone set:** {{ phone_set | default("[More Information Needed]", true)}}
- **Words:** {{ "{:,}".format(num_words) | default("[More Information Needed]", true)}}
* **Phones:** {{ "{:,}".format(num_phones) | default("[More Information Needed]", true)}}
* **Graphemes:** {{ "{:,}".format(num_graphemes) | default("[More Information Needed]", true)}}

##### IPA chart

###### Consonants

{{ consonant_chart | default("[More Information Needed]", true)}}

###### Vowels

{{ vowel_chart | default("[More Information Needed]", true)}}
