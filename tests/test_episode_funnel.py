"""Single-patient episode funnel: same classification and waterfall as manuscript Figure 2."""
import matplotlib.pyplot as plt
import pandas as pd

from scripts.episodes_detection import episode_funnel_single_patient as fun


def test_funnel_numbers_and_categories_on_the_synthetic_trace(synthetic_trace):
    table, stats, counts = fun.analyse_patient(synthetic_trace)
    # night spike (fails C3), daytime spike (SLH), 150-min excursion (fails C1)
    assert stats["n_raw"] == 3 and stats["n_after_c1"] == 2 and stats["n_slh"] == 1
    assert counts["SLH"] == 1 and counts["Night"] == 1 and counts["LLH"] == 1 and counts["Not isolated"] == 0
    assert set(table["category"]) == {"SLH", "Night", "LLH"}


def test_funnel_uses_the_figure_2_functions():
    from scripts.paper_figures import generate_figure_2_classification_summary as fig2
    assert fun.compute_funnel_stats is fig2.compute_funnel_stats
    assert fun.plot_simplified_waterfall is fig2.plot_simplified_waterfall
    assert fun.CATEGORY_ORDER == fig2.CATEGORY_ORDER


def test_figure_is_drawn_for_one_patient(synthetic_trace, tmp_path):
    _, stats, _ = fun.analyse_patient(synthetic_trace)
    fig = fun.make_figure(stats, "P001")
    assert "P001" in fig.axes[0].get_xlabel()
    fig.savefig(tmp_path / "f.png")
    plt.close(fig)
    assert (tmp_path / "f.png").stat().st_size > 0


def test_cli_defaults_to_the_patient_with_most_slh():
    args = fun.parse_arguments([])
    assert args.patient_id is None and args.out_dir == fun.DEFAULT_OUTPUT_DIR
    assert fun.parse_arguments(["--patient_id", "P7", "--out_dir", "x"]).patient_id == "P7"


def test_waterfall_without_panel_letter_in_figure_2_function(synthetic_trace):
    _, stats, _ = fun.analyse_patient(synthetic_trace)
    fig, ax = plt.subplots()
    fun.plot_simplified_waterfall(ax, stats, 3.0, panel_label=None)
    assert "A" not in [t.get_text() for t in ax.texts]
    plt.close(fig)
