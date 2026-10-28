# Spanish transcript-correction study (historical CLI comparison corrected)

**Correction, 2026-10-28:** The original CLI wrapper did not replace already-imported
exporter aliases. All four historical CLI runs used the base exporter, so their
base/PR labels do not establish a CLI comparison. The stored TextGrids and input
text corrections remain historical results. Direct exporter tests and the
controlled in-memory reproduction below are unaffected. See the
[verified natural public-corpus reproductions](../pr979_public_corpora/README.md)
for corrected comparisons, call-time traces, and new figures. The reproduction
script now uses that verified wrapper; the historical outputs have not been
retroactively relabeled as a valid comparison.

Two real Spanish MLS clips, checked by listening, were aligned with original
and corrected source transcripts. Each variant was run twice; because of the
alias-replacement error described above, both runs used the base exporter.

The [historical figure](comparison.png) illustrates the transcript-correction
experiment; its base/PR CLI labels are not valid evidence for this fix.

## Results on unmodified alignment data

All four full alignment runs completed successfully and wrote eight TextGrids.
The two nominally different exports are equal after reading all nonempty tiers,
but both CLI variants actually used the base function. These clips do not naturally reproduce a missing
file group in the SQL streams. Transcript mistakes alone do not demonstrate
the export bug fixed by this PR.

| MLS clip | Source transcript | User-confirmed correction | Word intervals |
|---|---|---|---|
| `10206_10282_000001` | Ends at `ni judá afligirá` | Append `á ephraim` | 23 → 25 |
| `8882_10139_000000` | Begins with `pestañeado quedóse` | Remove only `pestañeado` | 24 → 23 |

For the first clip, the source transcript stretches `afligirá` across
9.93–11.36 s. With corrected text, the intervals are `afligirá` 9.93–10.71 s,
`á` 10.71–10.74 s and `ephraim` 10.74–11.36 s.
For the second clip, the source's extra initial word is exported as `<unk>`
at 0.00–0.03 s; that interval disappears after correction. `quedóse` remains
at 0.43–0.93 s. These historical changes come from correcting the input text; they do not
establish any acoustic improvement from the PR.
The reported timings are model outputs, not manually verified word boundaries.

## Controlled reproduction on the same real intervals

To exercise the export bug, an **in-memory copy** of the original-text alignment
database was made and all 85 phone rows belonging to the first clip were removed.
Word and utterance rows were kept. This is an intentionally injected missing-group
condition, not a naturally observed failure in these two clips.

- Base exporter: `KeyError: 'mls_es_10206'`, no TextGrid produced. The word-only
  file's speaker has not been initialized by the phone stream.
- PR exporter: both TextGrids produced; every word remains in its own file and
  the first file has an empty phone tier. Exported word tiers match the
  unmodified per-file baseline.
- The original SQLite database and source WAVs were left unchanged.

The accompanying regression test covers a word-only file followed by a phone
and word file from a different speaker.

## Reproduction

Runtime: installed MFA 3.4.1 with the selected export function replaced only
inside each child process. This is an isolated comparison of export code,
not a full installation comparison between releases.

- Base function: commit `93411fc0df3bc770dfd34c4282dc136fdf6e8dcb`.
- PR function: commit `8545fd38f47b81f8a1743c8c9f8ab7b323e853eb`.
- Spanish MFA acoustic model and dictionary; hashes in `environment.json`.
- Historical runs: CPU, two jobs, SQLite, long TextGrid, original-text tier enabled.
- Corrected reproduction script: one job and --no_use_mp, matching the new public-corpus method.
- Original text came directly from the existing MLS Spanish `transcripts.txt`.
- Only the two user-confirmed lexical changes were applied in the corrected
  copies. No ASR-generated replacement transcript was used.

Obtain MLS clips with IDs listed in `samples.json` from the Spanish MLS corpus
(OpenSLR resource 94), prepare mono 16 kHz WAVs named `<id>.wav`, and run this
inside the same MFA environment with the repository history available:

```bash
python reproduce.py --repo /path/to/Montreal-Forced-Aligner \
  --audio-dir /path/to/selected/wavs \
  --dictionary /path/to/spanish_mfa.dict \
  --acoustic-model /path/to/spanish_mfa.zip \
  --output /path/to/new/results
python analyze.py /path/to/new/results
```

`comparison.json` records the complete parsed word/phone intervals and controlled
reproduction. `runs/` contains eight actual output TextGrids; `controlled/pr/`
contains the two successful controlled-case outputs. Audio and SQLite databases
are not bundled. This is a two-clip case study, not a corpus-wide quality benchmark.

## Regression checks

`python -m pytest --noconftest tests/test_textgrid_export.py -q`: **4 passed**.
The new separate-speaker regression was also run against the base export function
and failed with the expected `KeyError: 'first_speaker'`. The same test passes
with the PR export function. `--noconftest` is used because these self-contained
tests need no project fixtures and the environment lacks the optional `mock`
package imported by the repository's global conftest. The full suite was not run.

Repository copies of TextGrids have trailing formatting whitespace normalized;
all parsed nonempty tier entries were verified unchanged.
