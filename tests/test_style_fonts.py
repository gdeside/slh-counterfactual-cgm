"""Figures must render without Arial (Linux, CI): fall back silently to a sans-serif font."""
import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from matplotlib import font_manager

from src.visualization.style import (
    FONT_CSS_FAMILY, FONT_SANS_SERIF, apply_lancet_style, apply_publication_style,
)


def _clear_font_cache():
    """findfont results are memoised (lru_cache); the attribute location depends on the matplotlib version."""
    for owner in (font_manager, font_manager.fontManager, type(font_manager.fontManager)):
        cached = getattr(owner, "_findfont_cached", None)
        if hasattr(cached, "cache_clear"):
            cached.cache_clear()


@pytest.fixture
def only_dejavu(monkeypatch):
    """Simulate a machine without Arial / Helvetica / Liberation: only the fonts bundled with matplotlib."""
    bundled = [f for f in font_manager.fontManager.ttflist if f.name == "DejaVu Sans"]
    monkeypatch.setattr(font_manager.fontManager, "ttflist", bundled)
    _clear_font_cache()
    yield
    _clear_font_cache()


def test_fallback_chain_ends_with_a_bundled_font():
    assert FONT_SANS_SERIF[0] == "Arial" and FONT_SANS_SERIF[-1] == "DejaVu Sans"
    assert "DejaVu Sans" in FONT_CSS_FAMILY and "sans-serif" in FONT_CSS_FAMILY


@pytest.mark.parametrize("apply", [apply_publication_style, apply_lancet_style])
def test_style_renders_without_arial_and_without_warnings(only_dejavu, caplog, apply):
    apply()
    assert matplotlib.rcParams["font.family"] == ["sans-serif"]
    with caplog.at_level(logging.WARNING, logger="matplotlib.font_manager"):
        fig, ax = plt.subplots()
        ax.set_title("Glucose (mg/dL)")
        fig.savefig(io.BytesIO(), format="pdf")
        fig.savefig(io.BytesIO(), format="svg")
        plt.close(fig)
    assert not [r for r in caplog.records if "not found" in r.getMessage()]
    resolved = font_manager.findfont(font_manager.FontProperties(family=matplotlib.rcParams["font.family"]))
    assert font_manager.FontProperties(fname=resolved).get_name() == "DejaVu Sans"
