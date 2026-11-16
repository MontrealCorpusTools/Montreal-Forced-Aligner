from praatio import textgrid as tgio
from praatio.utilities.constants import TextgridFormats

from montreal_forced_aligner.textgrid import Textgrid


def _save(tg, path):
    tg.save(str(path), format=TextgridFormats.LONG_TEXTGRID, includeBlankSpaces=True)
    return path.read_text()


def test_leading_blank_starts_at_the_grid(tmp_path):
    tg = Textgrid(minTimestamp=5.0, maxTimestamp=10.0)
    tg.addTier(tgio.IntervalTier("words", [(5.0, 10.0, "hello")], minT=5.0, maxT=10.0))
    text = _save(tg, tmp_path / "grid.TextGrid")
    assert "xmin = 0" not in text
    assert 'text = "hello"' in text


def test_leading_blank_still_fills_a_gap_at_zero(tmp_path):
    tg = Textgrid(minTimestamp=0.0, maxTimestamp=1.5)
    tg.addTier(tgio.IntervalTier("words", [(0.5, 1.0, "hi")], minT=0.0, maxT=1.5))
    text = _save(tg, tmp_path / "grid.TextGrid")
    assert "xmin = 0" in text
    assert 'text = "hi"' in text
