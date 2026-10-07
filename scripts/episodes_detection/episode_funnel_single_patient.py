"""
scripts/episodes_detection/episode_funnel_single_patient.py
===========================================================
Episode classification funnel for ONE patient, with the classification of manuscript Figure 2.

The waterfall is the one of Figure 2 panel A (same order, labels and colours), drawn for a single patient:

    All hyperglycaemia episodes (> 180 mg/dL)
      → Duration [10–60 min]                 (C1; failures are LLH)
      → Normoglycaemia in the surrounding hour (C4 + C5 + C6; failures are "Not isolated")
      → Data completeness ≥ 75 %             (C2; episodes that cannot be judged are set aside)
      → Daytime (06:00–22:00) = SLH          (C3; failures are "Night")

and the episodes end up in the four categories SLH / Not isolated / Night / LLH (episodes failing C2 are
counted apart). Everything is imported from `scripts/paper_figures/generate_figure_2_classification_summary.py`
and `src/episodes/criteria.py`, so this figure cannot drift from Figure 2.

Usage
-----
    python3 -m scripts.episodes_detection.episode_funnel_single_patient                    # patient with the most SLH episodes
    python3 -m scripts.episodes_detection.episode_funnel_single_patient --patient_id ID
    python3 -m scripts.episodes_detection.episode_funnel_single_patient --out_dir DIR
    (or: make slh_funnel [PATIENT_ID=ID])

Output
------
    <CGM_DATA_ROOT>/Figures/Episodes/<patient_id>_episode_funnel.{pdf,svg}
    <CGM_DATA_ROOT>/Figures/Episodes/<patient_id>_episode_funnel.csv    (the numbers: waterfall steps and categories)
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.paper_figures.generate_figure_2_classification_summary import (
    CATEGORY_ORDER,
    assign_fine_category,
    compute_funnel_stats,
    plot_simplified_waterfall,
)
from src.config import FIG_DIR
from src.episodes.criteria import classify_episodes
from src.episodes.detection import detect_hyperglycemia_episodes
from src.preprocessing.load_data import load_patient_window, prepare_valid_data
from src.utils.patient_selection import resolve_patient_id
from src.visualization.style import apply_publication_style, save_figure

DEFAULT_OUTPUT_DIR = os.path.join(FIG_DIR, "Episodes")
FIG_WIDTH_IN, FIG_HEIGHT_IN = 7.24, 2.9


def analyse_patient(df: pd.DataFrame):
    """Detect and classify the episodes of one trace.

    Returns (episode table with the C1–C6 flags and the Figure 2 ``category``, Figure 2 funnel statistics,
    number of episodes per category). Episodes failing C2 get the category ``Excluded (incomplete data)``.
    """
    table = classify_episodes(df, detect_hyperglycemia_episodes(df))
    funnel_stats = compute_funnel_stats(table) if len(table) else None
    table = table.copy()
    table["category"] = "Excluded (incomplete data)"
    complete = table["C2"].astype(bool)
    if complete.any():
        table.loc[complete, "category"] = assign_fine_category(table[complete])["category"].to_numpy()
    counts = table["category"].value_counts()
    return table, funnel_stats, {c: int(counts.get(c, 0)) for c in CATEGORY_ORDER + ["Excluded (incomplete data)"]}


def make_figure(funnel_stats: dict, patient_id: str):
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), constrained_layout=True)
    plot_simplified_waterfall(
        ax, funnel_stats, FIG_HEIGHT_IN, panel_label=None,
        xlabel=f"Number of hyperglycemic episodes — patient {patient_id}")
    return fig


def print_summary(patient_id: str, funnel_stats: dict, counts: dict) -> None:
    print("\n" + "=" * 70)
    print(f"  Episode classification funnel — {patient_id}")
    print("=" * 70)
    print(f"  All episodes:                         {funnel_stats['n_raw']}")
    print(f"  Duration OK (C1):                     {funnel_stats['n_after_c1']}")
    print(f"  + isolated (C4+C5+C6):                {funnel_stats['n_after_c456']}")
    print(f"  + complete data (C2):                 {funnel_stats['n_after_c2']}")
    print(f"  + daytime (C3)  = SLH:                {funnel_stats['n_slh']}")
    print("  Categories: " + "  ".join(f"{c}: {n}" for c, n in counts.items() if n or c in CATEGORY_ORDER))
    print("=" * 70)


def generate_funnel_for_patient(patient_id: str, output_dir: str):
    """Load, classify, draw and save the funnel of one patient. Returns the figure statistics or None."""
    print(f"\nProcessing patient {patient_id}...")
    raw = load_patient_window(patient_id)
    if raw is None:
        print(f"  Failed to load data for {patient_id}")
        return None
    df = prepare_valid_data(raw)
    print(f"Loaded {len(df):,} valid data points")

    table, funnel_stats, counts = analyse_patient(df)
    if funnel_stats is None:
        print("No episode detected")
        return None
    print_summary(patient_id, funnel_stats, counts)

    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, f"{patient_id}_episode_funnel")
    fig = make_figure(funnel_stats, patient_id)
    save_figure(fig, prefix)
    plt.close(fig)
    pd.DataFrame(
        [{"item": "all_episodes", "n": funnel_stats["n_raw"]},
         {"item": "after_C1_duration", "n": funnel_stats["n_after_c1"]},
         {"item": "after_C4C5C6_isolation", "n": funnel_stats["n_after_c456"]},
         {"item": "after_C2_completeness", "n": funnel_stats["n_after_c2"]},
         {"item": "SLH_after_C3_daytime", "n": funnel_stats["n_slh"]}]
        + [{"item": f"category_{c}", "n": n} for c, n in counts.items()]
    ).to_csv(prefix + ".csv", index=False)
    return funnel_stats


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Episode classification funnel of one patient (classification of manuscript Figure 2).")
    parser.add_argument("--patient_id", default=None,
                        help="patient to show (default: the patient with the most SLH episodes)")
    parser.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR,
                        help="output folder (default: <CGM_DATA_ROOT>/Figures/Episodes)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_arguments(argv)
    patient_id = resolve_patient_id(args.patient_id)
    print("\n" + "=" * 70)
    print("  Episode Classification Funnel (Figure 2 classification)")
    print("=" * 70)
    print(f"  Patient: {patient_id}")
    print(f"  Output:  {args.out_dir}")
    if generate_funnel_for_patient(patient_id, args.out_dir) is None:
        print("\nFailed to generate the funnel")
        return 1
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
