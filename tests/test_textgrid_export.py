from pathlib import Path
from unittest.mock import MagicMock

from montreal_forced_aligner import textgrid


class FakeResult:
    def __init__(self, rows):
        self.rows = rows
        self.iterated = False

    def __iter__(self):
        if self.iterated:
            raise AssertionError("SQL result streams must only be consumed once")
        self.iterated = True
        return iter(self.rows)


def run_export(
    tmp_path,
    monkeypatch,
    phones,
    words,
    file_batch,
    utterances=None,
    include_original_text=False,
):
    captured = {}
    result_sets = [phones, words]
    if include_original_text:
        result_sets.append(utterances or [])
    session = MagicMock()
    session.execute.side_effect = lambda _query: FakeResult(result_sets.pop(0))

    def fake_construct_output_path(name, _relative, output_directory, _text_path, _format):
        return Path(output_directory) / f"{name}.TextGrid"

    def fake_export(data, output_path, duration, _frame_shift, _format):
        captured[Path(output_path).name] = {
            "duration": duration,
            "data": {
                speaker: {
                    tier: [
                        (interval.begin, interval.end, interval.label) for interval in intervals
                    ]
                    for tier, intervals in tiers.items()
                }
                for speaker, tiers in data.items()
            },
        }

    monkeypatch.setattr(textgrid, "construct_output_path", fake_construct_output_path)
    monkeypatch.setattr(textgrid, "export_textgrid", fake_export)
    paths = list(
        textgrid.construct_textgrid_output(
            session,
            file_batch,
            True,
            "'",
            tmp_path,
            0.01,
            "long_textgrid",
            include_original_text,
        )
    )
    return paths, captured


def test_export_groups_filtered_streams_by_file_id(tmp_path, monkeypatch):
    file_batch = {
        1: ("seg_a", Path("."), 1.0, Path("a.lab")),
        2: ("seg_b", Path("."), 1.0, Path("b.lab")),
        3: ("seg_c", Path("."), 1.0, Path("c.lab")),
    }
    phones = [
        (0.0, 0.4, "a", "speaker", 1),
        (0.0, 0.3, "c", "speaker", 3),
    ]
    words = [
        (0.0, 0.4, "aword", "speaker", 1),
        (0.0, 0.5, "word_only", "speaker", 2),
        (0.0, 0.3, "cword", "speaker", 3),
    ]

    paths, captured = run_export(tmp_path, monkeypatch, phones, words, file_batch)

    assert [path.name for path in paths] == [
        "seg_a.TextGrid",
        "seg_b.TextGrid",
        "seg_c.TextGrid",
    ]
    assert captured["seg_a.TextGrid"]["data"]["speaker"]["words"] == [(0.0, 0.4, "aword")]
    assert captured["seg_b.TextGrid"]["data"]["speaker"]["phones"] == []
    assert captured["seg_b.TextGrid"]["data"]["speaker"]["words"] == [(0.0, 0.5, "word_only")]
    assert captured["seg_c.TextGrid"]["data"]["speaker"]["phones"] == [(0.0, 0.3, "c")]


def test_export_creates_word_only_speaker_tier(tmp_path, monkeypatch):
    file_batch = {1: ("seg", Path("."), 1.0, Path("seg.lab"))}

    _, captured = run_export(
        tmp_path,
        monkeypatch,
        phones=[],
        words=[(0.0, 0.4, "word", "word_speaker", 1)],
        file_batch=file_batch,
    )

    assert captured["seg.TextGrid"]["data"]["word_speaker"] == {
        "words": [(0.0, 0.4, "word")],
        "phones": [],
    }


def test_export_preserves_clitics_and_original_text(tmp_path, monkeypatch):
    file_batch = {1: ("seg", Path("."), 1.0, Path("seg.lab"))}
    phones = [(0.0, 0.4, "phone", "speaker", 1)]
    words = [
        (0.0, 0.2, "don'", "speaker", 1),
        (0.21, 0.4, "t", "speaker", 1),
    ]
    utterances = [(0.0, 1.0, "don't", "speaker", 1)]

    _, captured = run_export(
        tmp_path,
        monkeypatch,
        phones,
        words,
        file_batch,
        utterances,
        include_original_text=True,
    )

    assert captured["seg.TextGrid"]["data"]["speaker"]["words"] == [(0.0, 0.4, "don't")]
    assert captured["seg.TextGrid"]["data"]["speaker"]["utterances"] == [(0.0, 1.0, "don't")]


def test_export_word_only_file_before_another_speaker(tmp_path, monkeypatch):
    """A missing phone group must not require that speaker in another file."""
    file_batch = {
        1: ("word_only", Path("."), 12.0, Path("word_only.lab")),
        2: ("with_phones", Path("."), 11.0, Path("with_phones.lab")),
    }
    phones = [(0.43, 0.93, "k", "second_speaker", 2)]
    words = [
        (9.93, 11.36, "afligirá", "first_speaker", 1),
        (0.43, 0.93, "quedóse", "second_speaker", 2),
    ]
    utterances = [
        (0.0, 12.0, "ni judá afligirá", "first_speaker", 1),
        (0.0, 11.0, "quedóse en donde estaba", "second_speaker", 2),
    ]

    paths, captured = run_export(
        tmp_path,
        monkeypatch,
        phones,
        words,
        file_batch,
        utterances,
        include_original_text=True,
    )

    assert [path.name for path in paths] == ["word_only.TextGrid", "with_phones.TextGrid"]
    first = captured["word_only.TextGrid"]["data"]
    second = captured["with_phones.TextGrid"]["data"]
    assert set(first) == {"first_speaker"}
    assert first["first_speaker"]["phones"] == []
    assert first["first_speaker"]["words"] == [(9.93, 11.36, "afligirá")]
    assert first["first_speaker"]["utterances"] == [(0.0, 12.0, "ni judá afligirá")]
    assert set(second) == {"second_speaker"}
    assert second["second_speaker"]["words"] == [(0.43, 0.93, "quedóse")]
    assert second["second_speaker"]["phones"] == [(0.43, 0.93, "k")]
    assert second["second_speaker"]["utterances"] == [
        (0.0, 11.0, "quedóse en donde estaba")
    ]
