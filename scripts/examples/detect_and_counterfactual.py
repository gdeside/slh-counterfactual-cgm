"""
scripts/examples/detect_and_counterfactual.py
=============================================

Illustration: run the two core steps — hyperglycaemia-episode detection / SLH classification and the
counterfactual signal — on ONE CGM file, without the full pipeline (no preprocessing, no cohort, no
``CGM_DATA_ROOT``).

Input file (CSV, TSV or Excel), one patient, at least two columns:
    time      timestamp of each reading
    glucose   glucose value (default unit mg/dL; use --unit mmol/L for mmol/L)
Column names are detected automatically (time: time/timestamp/datetime/date; glucose: glycemie/glucose/
glycemia/sgl/value) or given with --time_col / --glucose_col.

Usage
-----
    python3 -m scripts.examples.detect_and_counterfactual --demo
    python3 -m scripts.examples.detect_and_counterfactual --input my_patient.csv [--unit mmol/L]
                                                          [--offset 30] [--out_dir example_output]

Outputs (in --out_dir)
----------------------
    episodes.csv         every detected episode with the criteria C1–C6 and its category
                         (SLH / Not isolated / Night / LLH — same categories as manuscript Figure 2)
    counterfactual.csv   time, observed glucose G(t), counterfactual C(t)
    summary.txt          episode counts and the effect on time in range / median glucose
    counterfactual.png   observed vs counterfactual signal, SLH episodes shaded

``--demo`` uses a synthetic 3-day trace (arbitrary dates), so the example runs out of the box.

What is NOT done here (done by the full pipeline): quality flags, extraction of the 45-day analysis window
(>= 70 % coverage), cohort filtering. Give a clean, time-sorted trace of a 5-minute CGM.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.counterfactual.build_counterfactual import build_full_counterfactual_signal
from src.episodes.criteria import classify_episodes
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.episode_config import H_HYPER, LABEL_SLH
from src.metrics.metrics import compute_time_in_range_metrics
from src.preprocessing.timestep import detect_dominant_sampling_interval
from scripts.paper_figures.generate_figure_2_classification_summary import assign_fine_category, CATEGORY_ORDER
from src.utils.unit_conversion import UNIT_MMOLL, mmoll_to_mgdl, normalize_unit

TIME_NAMES = ("time", "timestamp", "datetime", "date", "date_time")
GLUCOSE_NAMES = ("glycemie", "glucose", "glycemia", "sgl", "value", "glucose_mgdl")
DEFAULT_OFFSET_MIN = 30


# =============================================================================
# INPUT
# =============================================================================
def _find_column(columns, candidates, explicit=None, what="column"):
    if explicit:
        if explicit not in columns:
            raise SystemExit(f"{what} '{explicit}' not found. Columns: {list(columns)}")
        return explicit
    lowered = {str(c).strip().lower(): c for c in columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    raise SystemExit(f"Could not find the {what}; pass it explicitly. Columns: {list(columns)}")


def load_trace(path, unit="mg/dL", time_col=None, glucose_col=None) -> pd.DataFrame:
    """Read one patient file -> DataFrame with columns ``time`` and ``glycemie`` (mg/dL), sorted."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path, sep=None, engine="python")           # CSV or TSV, delimiter auto-detected
    tcol = _find_column(raw.columns, TIME_NAMES, time_col, "time column")
    gcol = _find_column(raw.columns, GLUCOSE_NAMES, glucose_col, "glucose column")

    df = pd.DataFrame({"time": pd.to_datetime(raw[tcol], errors="coerce"),
                       "glycemie": pd.to_numeric(raw[gcol], errors="coerce")})
    df = df.dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if normalize_unit(unit) == UNIT_MMOLL:
        df["glycemie"] = mmoll_to_mgdl(df["glycemie"])
    if len(df) < 100:
        raise SystemExit(f"Only {len(df)} valid readings — need at least a few days of CGM data.")
    return df


def demo_trace() -> pd.DataFrame:
    """Synthetic 3-day trace (arbitrary dates): a night spike, a short daytime spike (SLH), a long excursion (LLH)."""
    idx = pd.date_range("2000-01-01", periods=3 * 288, freq="5min")
    g = np.full(len(idx), 120.0) + 4 * np.sin(np.arange(len(idx)) / 20)
    spike = [150, 200, 230, 220, 190, 150]
    for day, hour, values in [(0, 23, spike), (1, 12, spike), (1, 18, spike), (2, 14, [150] + [220] * 30 + [150])]:
        i = day * 288 + int(hour * 12)
        g[i:i + len(values)] = values
    return pd.DataFrame({"time": idx, "glycemie": g})


# =============================================================================
# CORE: detection -> classification -> counterfactual
# =============================================================================
EXCLUDED_LABEL = "Excluded (incomplete data)"


def categorise(table: pd.DataFrame) -> pd.DataFrame:
    """Add the Figure 2 ``category`` column: LLH (C1 fails) / Not isolated (C4–C6) / Night (C3) / SLH.
    Episodes failing C2 (data completeness) cannot be judged and are labelled ``Excluded (incomplete data)``."""
    out = table.copy()
    out["category"] = EXCLUDED_LABEL
    complete = out["C2"].astype(bool)
    if complete.any():
        out.loc[complete, "category"] = assign_fine_category(out[complete])["category"].to_numpy()
    return out


def run(df: pd.DataFrame, offset_min: float = DEFAULT_OFFSET_MIN):
    """Return (episode table, counterfactual DataFrame, timestep in minutes)."""
    timestep = detect_dominant_sampling_interval(df)                       # e.g. 5.0 min
    episodes = detect_hyperglycemia_episodes(df)                           # runs of glucose > 180 mg/dL
    table = categorise(classify_episodes(df, episodes))                    # criteria C1–C6 -> SLH / Not isolated / Night / LLH
    slh = (table.loc[table["classification"] == LABEL_SLH, ["onset_time", "offset_time"]]
           .to_dict("records"))
    counterfactual = build_full_counterfactual_signal(df, slh, offset_min, timestep)
    out = pd.DataFrame({"time": df["time"], "observed": df["glycemie"].to_numpy(),
                        "counterfactual": counterfactual})
    return table, out, timestep


def summarise(table: pd.DataFrame, cf: pd.DataFrame, offset_min: float) -> str:
    n_slh = int((table["classification"] == LABEL_SLH).sum()) if len(table) else 0
    obs_m = compute_time_in_range_metrics(cf["observed"])
    cf_m = compute_time_in_range_metrics(cf["counterfactual"])
    counts = table["category"].value_counts() if len(table) else pd.Series(dtype=int)
    by_category = "  ".join(f"{c}: {int(counts.get(c, 0))}" for c in CATEGORY_ORDER)
    if int(counts.get(EXCLUDED_LABEL, 0)):
        by_category += f"  {EXCLUDED_LABEL}: {int(counts[EXCLUDED_LABEL])}"
    lines = [f"Readings: {len(cf)}   Episodes (> {H_HYPER} mg/dL): {len(table)}   offset δ: {offset_min:g} min",
             f"Categories   {by_category}",
             f"TIR 70–180 %  observed {100 * obs_m['TIR70180']:.1f}   counterfactual {100 * cf_m['TIR70180']:.1f}   "
             f"Δ {100 * (cf_m['TIR70180'] - obs_m['TIR70180']):+.2f}",
             f"TAR > 180 %   observed {100 * obs_m['TAR180']:.1f}   counterfactual {100 * cf_m['TAR180']:.1f}   "
             f"Δ {100 * (cf_m['TAR180'] - obs_m['TAR180']):+.2f}",
             f"Median mg/dL  observed {cf['observed'].median():.1f}   counterfactual {cf['counterfactual'].median():.1f}"]
    return "\n".join(lines)


def plot(table: pd.DataFrame, cf: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(cf["time"], cf["counterfactual"], color="#E69F00", lw=2.4, label="Counterfactual C(t)")
    ax.plot(cf["time"], cf["observed"], color="#333333", lw=1, label="Observed G(t)")
    ax.axhline(H_HYPER, color="#888888", ls="--", lw=0.8)
    for k, ep in enumerate(table[table["classification"] == LABEL_SLH].itertuples()):
        ax.axvspan(ep.onset_time, ep.offset_time, color="#E69F00", alpha=0.25,
                   label="SLH episode" if k == 0 else None)
    ax.set_ylabel("Glucose (mg/dL)")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="CGM file of ONE patient (.csv / .tsv / .xlsx)")
    src.add_argument("--demo", action="store_true", help="use a synthetic 3-day trace")
    p.add_argument("--unit", default="mg/dL", help="glucose unit of the file: mg/dL (default) or mmol/L")
    p.add_argument("--time_col", help="name of the time column (auto-detected by default)")
    p.add_argument("--glucose_col", help="name of the glucose column (auto-detected by default)")
    p.add_argument("--offset", type=float, default=DEFAULT_OFFSET_MIN, help="zone-expansion offset δ in min (default 30)")
    p.add_argument("--out_dir", default="example_output", help="output folder (default: ./example_output)")
    args = p.parse_args(argv)

    df = demo_trace() if args.demo else load_trace(args.input, args.unit, args.time_col, args.glucose_col)
    table, cf, timestep = run(df, args.offset)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "episodes.csv", index=False)
    cf.to_csv(out / "counterfactual.csv", index=False)
    summary = summarise(table, cf, args.offset)
    (out / "summary.txt").write_text(summary + "\n")
    plot(table, cf, out / "counterfactual.png")

    print(f"Sampling interval: {timestep:g} min")
    print(summary)
    print(f"\nWritten to {out.resolve()}/  (episodes.csv, counterfactual.csv, summary.txt, counterfactual.png)")


if __name__ == "__main__":
    main()
