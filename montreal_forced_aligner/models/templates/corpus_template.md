#### {{ name | default("Corpus details", true)}}

- **Source:** {{ link | default("[More Information Needed]", true)}}
- **License:** {{ license | default("[More Information Needed]", true)}}
- **Dialects:** {{ dialects|join(', ') | default("N/A", true)}}
- **Number of hours:** {{ "{:,.2f}".format(num_hours) | default("[More Information Needed]", true)}}
- **Number of utterances:** {{ "{:,}".format(num_utterances) | default("[More Information Needed]", true)}}
- **Number of speakers:** {{ "{:,}".format(num_speakers) | default("[More Information Needed]", true)}}
  - **Female speakers:** {{ "{:,}".format(num_female) | default("0", true)}}
  - **Male speakers:** {{ "{:,}".format(num_male) | default("0", true)}}
  - **Unknown speakers:** {{ "{:,}".format(num_unspecified) | default("0", true)}}
