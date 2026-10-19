"""save_figure_with_panels: one folder per figure, whole figure + one file per panel."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.visualization.style import save_figure_with_panels


def _figure():
    fig = plt.figure(figsize=(6, 3), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1)
    ax_a, ax_b = top.subplots(1, 2)
    ax_c = bottom.subplots()
    twin = ax_a.twinx()
    ax_a.plot([0, 1]); twin.plot([0, 10]); ax_b.plot([0, 1]); ax_c.plot([0, 1])
    legend = top.legend(handles=ax_b.lines, labels=["x"], loc="outside lower right")
    return fig, {"A": ax_a, "B": [ax_b, legend], "C": ax_c}


def test_writes_whole_figure_and_each_panel_in_its_own_folder(tmp_path):
    fig, panels = _figure()
    save_figure_with_panels(fig, tmp_path / "Fig1", "Fig1", panels, unit_suffix=False)
    names = {p.name for p in (tmp_path / "Fig1").iterdir()}
    expected = {"Fig1.pdf", "Fig1.svg"} | {f"Fig1_panel_{k}.{e}" for k in "ABC" for e in ("pdf", "svg")}
    assert names == expected


def test_panel_crops_are_smaller_than_the_figure(tmp_path):
    fig, panels = _figure()
    save_figure_with_panels(fig, tmp_path / "F", "F", panels, unit_suffix=False)
    whole = (tmp_path / "F" / "F.svg").read_text()
    for k in "ABC":
        assert len((tmp_path / "F" / f"F_panel_{k}.svg").read_text()) < len(whole)
