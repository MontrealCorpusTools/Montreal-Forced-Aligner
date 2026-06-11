from __future__ import annotations

import csv
import json
import typing
from pathlib import Path

import sqlalchemy

from montreal_forced_aligner.data import PhoneSetType
from montreal_forced_aligner.db import Dictionary, File, Utterance
from montreal_forced_aligner.helper import mfa_open

if typing.TYPE_CHECKING:
    from montreal_forced_aligner.corpus import AcousticCorpusPronunciationMixin

MFA_CITATION_TEMPLATE = (
    "@techreport{{{id},\n\tauthor={{{extra_authors}McAuliffe, Michael and Gunter, Kaylynn and Wagner, Michael and Sonderegger, Morgan}},"
    "\n\ttitle={{{title}}},"
    "\n\taddress={{\\url{{https://huggingface.co/MontrealCorpusTools//{language}_mfa}}}},"
    "\n\tyear={{{year}}},\n\tmonth={{{month}}},"
    "\n}}"
)

MFA_MAINTAINER = "[Montreal Corpus Tools](https://huggingface.co/MontrealCorpusTools)"

CORPUS_DETAIL_TEMPLATE = """
   * {link}:
     * **Hours:** `{num_hours:.2f}`
     * **Speakers:** `{num_speakers:,}`
       * **Female:** `{num_female:,}`
       * **Male:** `{num_male:,}`
       * **Unspecified:** `{num_unspecified:,}`
     * **Utterances:** `{num_utterances:,}`"""

G2P_TRAINING_TEMPLATE = """
* **Words:** `{num_words:,}`
* **Phones:** `{num_phones:,}`
* **Graphemes:** `{num_graphemes:,}`"""

G2P_EVALUATION_TEMPLATE = """
* **Words:** `{num_words:,}`
* **WER:** `{word_error_rate:.2f}%`
* **PER:** `{phone_error_rate:.2f}%`"""

LICENSES = {
    "CC-0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC BY 3.0": "https://creativecommons.org/licenses/by/3.0/",
    "CC BY-SA-NC 3.0": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
    "CC BY-NC-SA 4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "CC BY-NC-SA 3.0": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
    "CC BY-NC 4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC BY-NC-ND 4.0": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "CC BY-NC 2.0": "https://creativecommons.org/licenses/by-nc/2.0/",
    "CC BY-NC-ND 3.0": "https://creativecommons.org/licenses/by-nc-nd/3.0/",
    "Microsoft Research Data License": "https://msropendata-web-api.azurewebsites.net/licenses/2f933be3-284d-500b-7ea3-2aa2fd0f1bb2/view",
    "Apache 2.0": "https://www.apache.org/licenses/LICENSE-2.0",
    "O-UDA v1.0": "https://msropendata-web-api.azurewebsites.net/licenses/f1f352a6-243f-4905-8e00-389edbca9e83/view",
    "MIT": "https://opensource.org/licenses/MIT",
    "Public domain in the USA": "https://creativecommons.org/share-your-work/public-domain/cc0/",
    "M-AILABS License": "https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/",
    "ELRA": "https://www.elra.info/en/services-around-lrs/distribution/licensing/",
    "Buckeye License": "https://buckeyecorpus.osu.edu/php/registration.php",
    "LDC License": "https://www.ldc.upenn.edu/data-management/using/licensing",
    "LaboroTV Non-commercial": "https://laboro.ai/activity/column/engineer/eg-laboro-tv-corpus-jp/",
}


DEFAULT_METADATA = {
    "direct_use": "This model is intended to be used for forced alignment of speech varieties that it was trained on.",
    "out_of_scope_use": "This model cannot provide accurate assessments of goodness of pronunciations or provide transcripts.",
    "bias_risks_limitations": "This model will perform best on the variety of speech that it was trained on (dialect/language/demographics).",
    "bias_recommendations": "When using this model on a variety that it was not trained on, better results can be attained by adapting the "
    "model to the data to be aligned first.",
    "get_started_code": "To get started, follow the instructions for [installing MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html). "
    "To align files using this model, use the [mfa align](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html) command.",
    "model_type": "Montreal Forced Aligner model",
    "license": "cc-by-4.0",
    "software": "This model was trained via the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/).",
    "model_specs": "HMM-GMM model",
}


def create_corpus_information(corpus: AcousticCorpusPronunciationMixin, multiple_corpora=False):
    corpus_data = {}
    training_root_directory = Path(corpus.corpus_directory)
    if not multiple_corpora:
        for subdir in training_root_directory.iterdir():
            if (subdir / "corpus_data.json").exists():
                multiple_corpora = True
                break
    if not multiple_corpora:
        corpus_name = training_root_directory.name
        corpus_data_path = training_root_directory / "corpus_data.json"
        if corpus_data_path.exists():
            with mfa_open(corpus_data_path, "r") as f:
                data = json.load(f)
        else:
            data = {
                "name": corpus_name,
                "link": "",
                "dialects": "",
                "license": "",
                "citation": "",
                "version": "",
            }
        with corpus.session() as session:
            total_duration = (
                session.query(sqlalchemy.func.sum(Utterance.duration))
                .filter(Utterance.text != "")
                .first()[0]
                / 3600
            )
        data["num_utterances"] = corpus.num_utterances
        data["num_files"] = corpus.num_files
        data["num_speakers"] = corpus.num_speakers
        data["num_hours"] = total_duration
        speaker_info_path = training_root_directory / "speaker_info.tsv"
        if speaker_info_path.exists():
            num_female = 0
            num_male = 0
            num_unspecified = 0
            with mfa_open(speaker_info_path, "r") as f:
                for line in f:
                    line = line.split()
                    if line[2].lower().startswith("f"):
                        num_female += 1
                    elif line[2].lower().startswith("m"):
                        num_male += 1
                    else:
                        num_unspecified += 1
            data["num_female"] = num_female
            data["num_male"] = num_male
            data["num_unspecified"] = num_unspecified
        with mfa_open(corpus_data_path, "w") as f:
            json.dump(data, f)
        corpus_data[corpus_name] = data
    else:
        for corpus_directory in training_root_directory.iterdir():
            if not corpus_directory.is_dir():
                continue
            corpus_name = corpus_directory.name

            corpus_data_path = corpus_directory / "corpus_data.json"
            if corpus_data_path.exists():
                with mfa_open(corpus_data_path, "r") as f:
                    data = json.load(f)
            else:
                data = {
                    "name": corpus_name,
                    "link": "",
                    "dialects": [],
                    "license": "",
                    "citation": "",
                    "version": "",
                }
            with corpus.session() as session:
                file_query = session.query(File.id).filter(
                    File.relative_path.like(f"{corpus_name}%")
                )
                utterance_query = session.query(Utterance.id).filter(
                    Utterance.file_id.in_(file_query.subquery())
                )
                speaker_query = (
                    session.query(Utterance.speaker_id)
                    .filter(Utterance.file_id.in_(file_query.subquery()))
                    .distinct()
                )
                total_duration = (
                    session.query(sqlalchemy.func.sum(Utterance.duration))
                    .filter(Utterance.text != "")
                    .filter(Utterance.file_id.in_(file_query.subquery()))
                    .first()[0]
                    / 3600
                )
                data["num_utterances"] = utterance_query.count()
                data["num_files"] = file_query.count()
                data["num_speakers"] = speaker_query.count()
                data["num_hours"] = total_duration
            speaker_info_path = corpus_directory / "speaker_info.tsv"
            if speaker_info_path.exists():
                num_female = 0
                num_male = 0
                num_unspecified = 0
                with mfa_open(speaker_info_path, "r") as f:
                    reader = csv.reader(f, delimiter="\t")
                    for line in reader:
                        if line[2].lower().startswith("f"):
                            num_female += 1
                        elif line[2].lower().startswith("m"):
                            num_male += 1
                        else:
                            num_unspecified += 1
                data["num_female"] = num_female
                data["num_male"] = num_male
                data["num_unspecified"] = num_unspecified
            with mfa_open(corpus_data_path, "w") as f:
                json.dump(data, f)
            if data["license"] in LICENSES:
                data["license"] = f"[{data['license']}]({LICENSES[data['license']]})"
            corpus_data[corpus_name] = data
    return corpus_data


def check_phone(phone, feature_set, phone_set_type):
    if phone_set_type is PhoneSetType.ARPA:
        return phone in feature_set
    else:
        return any(x in phone for x in feature_set)


def format_ipa_cell(phone_data: dict[str, list[str]], extra_data) -> typing.Tuple[str, str]:
    cell_content = ""
    extra = ""
    for phone_class, v in phone_data.items():
        if not v:
            continue
        if cell_content:
            cell_content += "    "
        for phone in v:
            if cell_content:
                cell_content += " "
            # cell_content += f"[{phone}](#{phone})"
            cell_content += f"{phone}"
            if phone in extra_data:
                extra += f"#### {phone}\n\n"
                extra += f"* Occurrences: {extra_data[phone]['Occurrences']}\n"
                extra += "* Examples:\n"
                count = 0
                for example_w, pron in extra_data[phone]["Examples"].items():
                    extra += f"  * {example_w}: {pron}\n"
                    if count > 5:
                        break
    return cell_content, extra


def analyze_dictionary(dictionary: Dictionary, phone_set_type: PhoneSetType = None):
    if phone_set_type:
        dictionary.phone_set_type = phone_set_type
    dictionary_mapping = dictionary.ipa_chart_data
    extra_data = {}
    places = [
        "labial",
        "labiodental",
        "dental",
        "alveolar",
        "alveopalatal",
        "retroflex",
        "palatal",
        "velar",
        "uvular",
        "pharyngeal",
        "epiglottal",
        "glottal",
    ]
    columns = []
    for p in places:
        if p in dictionary_mapping:
            columns.append(p)
    sub_manners = [
        "tense",
        "aspirated",
        "implosive",
        "ejective",
        "unreleased",
        "prenasalized",
        "labialized",
        "palatalized",
    ]
    phone_examples = ""
    rows = []
    plotted = set()
    for manner in [
        "nasal",
        "stop",
        "affricate",
        "sibilant",
        "fricative",
        "approximant",
        "tap",
        "trill",
        "lateral_fricative",
        "lateral",
        "lateral_tap",
    ]:
        if manner not in dictionary_mapping:
            continue
        realized_submanner_rows = {}
        for x in sub_manners:
            if dictionary_mapping[manner] & dictionary_mapping[x]:
                realized_submanner_rows[x] = [x.title()]
        row_title = "**" + manner.replace("_", " ").title() + "**"
        row = [row_title]
        for place in columns:
            cell_set = dictionary_mapping[manner] & dictionary_mapping[place]
            base_set = dictionary_mapping[manner] & dictionary_mapping[place]
            for x in sub_manners:
                cell_set -= dictionary_mapping[x]
                base_set -= dictionary_mapping[x]
            voiced_set = base_set & dictionary_mapping["voiced"]
            voiceless_set = base_set & dictionary_mapping["voiceless"]
            other_set = base_set - dictionary_mapping["voiceless"] - dictionary_mapping["voiced"]
            plotted.update(voiceless_set)
            plotted.update(voiced_set)
            plotted.update(other_set)
            cell_data = {
                "voiceless": sorted(voiceless_set),
                "voiced": sorted(voiced_set),
                "other": sorted(other_set),
            }
            cell_contents, e = format_ipa_cell(cell_data, extra_data)
            phone_examples += e
            row.append(cell_contents)
        rows.append(row)
        if realized_submanner_rows:
            for place in columns:
                for sub_manner in realized_submanner_rows.keys():
                    cell_set = (
                        dictionary_mapping[manner]
                        & dictionary_mapping[place]
                        & dictionary_mapping[sub_manner]
                    )
                    for s in realized_submanner_rows.keys():
                        if s == sub_manner:
                            continue
                        cell_set -= dictionary_mapping[s]
                    voiced_set = cell_set & dictionary_mapping["voiced"]
                    voiceless_set = cell_set & dictionary_mapping["voiceless"]
                    other_set = (
                        cell_set - dictionary_mapping["voiceless"] - dictionary_mapping["voiced"]
                    )
                    plotted.update(voiceless_set)
                    plotted.update(voiced_set)
                    plotted.update(other_set)
                    cell_data = {
                        "voiceless": sorted(voiceless_set),
                        "voiced": sorted(voiced_set),
                        "other": sorted(other_set),
                    }
                    cell_contents, e = format_ipa_cell(cell_data, extra_data)
                    phone_examples += e
                    realized_submanner_rows[sub_manner].append(cell_contents)
            rows.extend(realized_submanner_rows.values())
    row_headers = ["Manner"]
    columns = row_headers + columns
    consonants = {"header": columns, "rows": rows}

    oral_rows = []
    nasal_rows = []
    headers = ["front", "near-front", "central", "near-back", "back"]
    has_nasal = False
    for height in ["close", "close-mid", "open-mid", "open"]:
        for on in ["nasalized", "oral"]:
            main_row = ["**" + height.title() + "**"]
            lax_row = [""]
            for column in headers:
                cell_set = dictionary_mapping[height] & dictionary_mapping[column]
                if on in dictionary_mapping:  # nasalized
                    cell_set &= dictionary_mapping["nasalized"]
                    if cell_set and not has_nasal:
                        has_nasal = True
                else:
                    cell_set -= dictionary_mapping["nasalized"]
                if height == "close" and column in {"front", "back"}:
                    lax_set = set()
                    tense_set = cell_set - dictionary_mapping["lax"]
                elif height == "close" and column in {"near-front", "near-back"}:
                    tense_set = set()
                    lax_set = cell_set & dictionary_mapping["lax"]
                else:
                    tense_set = cell_set - dictionary_mapping["lax"]
                    lax_set = cell_set & dictionary_mapping["lax"]

                tense_rounded = tense_set & dictionary_mapping["rounded"]
                tense_unrounded = tense_set & dictionary_mapping["unrounded"]
                cell_data = {
                    "unrounded": sorted(tense_unrounded),
                    "rounded": sorted(tense_rounded),
                }
                plotted.update(tense_unrounded)
                plotted.update(tense_rounded)
                tense_cell_contents, e = format_ipa_cell(cell_data, extra_data)
                phone_examples += e

                lax_rounded = lax_set & dictionary_mapping["rounded"]
                lax_unrounded = lax_set & dictionary_mapping["unrounded"]
                plotted.update(lax_rounded)
                plotted.update(lax_unrounded)
                cell_data = {
                    "unrounded": sorted(lax_unrounded),
                    "rounded": sorted(lax_rounded),
                }
                lax_cell_contents, e = format_ipa_cell(cell_data, extra_data)
                phone_examples += e

                main_row.append(tense_cell_contents)
                lax_row.append(lax_cell_contents)
            if on in dictionary_mapping:  # nasalized
                nasal_rows.append(main_row)
                if height != "open":
                    nasal_rows.append(lax_row)
            else:
                oral_rows.append(main_row)
                if height != "open":
                    oral_rows.append(lax_row)

    headers = [""] + [x.title() for x in headers]
    if not has_nasal:
        nasal_rows = None

    header_row_string = " | ".join(x.title() for x in consonants["header"])
    alignment_row_string = " | ".join(":----:" for _ in consonants["header"])
    row_strings = "|\n| ".join(" | ".join(x) for x in consonants["rows"])
    consonant_chart = f"| {header_row_string} |\n"
    consonant_chart += f"| {alignment_row_string} |\n"
    consonant_chart += f"| {row_strings} |"
    vowels = {
        "oral_rows": oral_rows,
        "nasal_rows": nasal_rows,
        "header": headers,
    }
    header_row_string = " | ".join(vowels["header"])
    alignment_row_string = " | ".join(":----:" for _ in vowels["header"])
    row_strings = "|\n| ".join(" | ".join(x) for x in vowels["oral_rows"])

    vowel_chart = f"| {header_row_string} |\n"
    vowel_chart += f"| {alignment_row_string} |\n"
    vowel_chart += f"| {row_strings} |"
    if nasal_rows:
        vowel_chart = "#### Oral vowels\n\n" + vowel_chart
        header_row_string = " | ".join(vowels["header"])
        alignment_row_string = " | ".join(":----:" for _ in vowels["header"])
        row_strings = "|\n| ".join(" | ".join(x) for x in vowels["nasal_rows"])

        vowel_chart += f"\n\n#### Nasal vowels\n\n| {header_row_string} |\n"
        vowel_chart += f"| {alignment_row_string} |\n"
        vowel_chart += f"| {row_strings} |"

    if dictionary_mapping["diphthong"]:
        vowel_chart += "\n\n##### Diphthongs\n"
        for diphthong in dictionary_mapping["diphthong"]:
            vowel_chart += f"* {diphthong}\n"

    if dictionary_mapping["triphthong"]:
        vowel_chart += "\n\n##### Triphthongs\n"
        for triphthong in dictionary_mapping["triphthong"]:
            vowel_chart += f"* {triphthong}\n"

    if dictionary_mapping["other"]:
        vowel_chart += "\n\n#### Other\n"
        for other in dictionary_mapping["other"]:
            vowel_chart += f"* {other}\n"
    data = {
        "consonant_chart": consonant_chart,
        "vowel_chart": vowel_chart,
        "phone_examples": phone_examples,
        "num_words": dictionary.word_count,
        "num_graphemes": dictionary.grapheme_count,
        "num_phones": len(
            plotted
            | dictionary_mapping["other"]
            | dictionary_mapping["diphthong"]
            | dictionary_mapping["triphthong"]
        ),
    }
    for k in ["stress", "tones"]:
        if k in dictionary_mapping:
            data[k] = dictionary_mapping[k]
    return data
