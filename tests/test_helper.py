import yaml

from montreal_forced_aligner.data import CtmInterval
from montreal_forced_aligner.helper import align_phones, mfa_open


def test_align_phones(basic_corpus_dir, basic_dict_path, temp_dir, eval_mapping_path):
    with mfa_open(eval_mapping_path) as f:
        mapping = yaml.safe_load(f)
    reference_phoneset = set()
    for v in mapping.values():
        if isinstance(v, str):
            reference_phoneset.add(v)
        else:
            reference_phoneset.update(v)

    reference_sequence = [
        "HH",
        "IY0",
        "HH",
        "AE1",
        "D",
        "Y",
        "ER0",
        "G",
        "R",
        "IY1",
        "S",
        "IY0",
        "S",
        "UW1",
        "T",
        "IH0",
        "N",
        "D",
        "ER1",
        "T",
        "IY0",
        "W",
        "AA1",
        "SH",
        "W",
        "AO1",
        "T",
        "ER0",
        "AO1",
        "L",
        "sil",
        "Y",
        "IH1",
        "R",
    ]
    reference_sequence = [CtmInterval(i, i + 1, x) for i, x in enumerate(reference_sequence)]
    comparison_sequence = [
        "ç",
        "i",
        "h",
        "æ",
        "d",
        "j",
        "ɚ",
        "ɟ",
        "ɹ",
        "iː",
        "s",
        "i",
        "s",
        "ʉː",
        "t",
        "sil",
        "ɪ",
        "n",
        "d",
        "ɝ",
        "ɾ",
        "i",
        "w",
        "ɑː",
        "ʃ",
        "w",
        "ɑː",
        "ɾ",
        "ɚ",
        "ɑː",
        "ɫ",
        "sil",
        "j",
        "ɪ",
        "ɹ",
    ]
    comparison_sequence = [CtmInterval(i, i + 1, x) for i, x in enumerate(comparison_sequence)]
    score, phone_errors = align_phones(
        reference_sequence,
        comparison_sequence,
        silence_phone="sil",
        custom_mapping=mapping,
        debug=True,
    )

    assert score < 1
    assert phone_errors < 1
