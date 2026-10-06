"""
scripts/patient_selection/patient_similarity_analysis.py
========================================================
Similarity between participants in standardised CGM-metric space, and its
relation with the number of SLH episodes (n_SLH).

Analysis
--------
  1. Load the filtered cohort (CGM metrics + n_SLH)
  2. Standardise the metrics (z-scores) and compute pairwise Euclidean distances
  3. Project the participants in 2-D / 3-D (PCA, t-SNE, and UMAP if installed),
     coloured by n_SLH
  4. List the pairs that are close in metric space (bottom `--distance_percentile`
     of all distances) but differ by at least `--n_slh_diff` SLH episodes:
     similar CGM profile, different SLH frequency

For a ranked selection of pairs for the article see `find_compelling_pairs.py`.

Inputs
------
  Data/Metrics/metrics_all_patients.csv
  Data/Episodes/episodes_summary.csv
  Data/Filtered/..._patient_status.csv      (skipped with --no_filter)

Outputs (RESULTS_PATH / Patient_Pair_Selection / Similarity_Analysis)
---------------------------------------------------------------------
  normalized_metrics.csv
  interesting_patient_pairs.csv
  pair_comparison_details.csv               (metrics of the top 10 pairs)
  fig1_distance_matrix.html                 distance heatmap sorted by n_SLH
  fig2_pca_2d.html, fig3_pca_3d.html        PCA projections
  fig4_tsne_2d.html                         t-SNE projection
  fig5_umap_2d.html                         UMAP projection (needs `umap-learn`)
  fig6_pairs_network.html                   top pairs on the PCA map

Usage
-----
    env/bin/python3 -m scripts.patient_selection.patient_similarity_analysis [--no_filter]
        [--distance_percentile 10] [--n_slh_diff 20]

Authors: Guillaume Deside
"""

import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from src.config import PATIENT_ID_COL
from src.visualization.style import FONT_CSS_FAMILY
from src.config.paths import RESULTS_PATH

from scripts.patient_selection.find_compelling_pairs import PROFILE_METRICS, load_cohort
from src.visualization.style import FONT_CSS_FAMILY

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = RESULTS_PATH / "Patient_Pair_Selection" / "Similarity_Analysis"

DISTANCE_PERCENTILE = 10     # "close" = bottom 10% of pairwise distances
N_SLH_DIFF_MIN = 20          # minimum n_SLH difference for an interesting pair
MIN_PARTICIPANTS = 10
N_PAIRS_NETWORK = 20
N_PAIRS_TABLE = 10
MAX_HEATMAP_PARTICIPANTS = 100

COLORSCALE = "Viridis"
RANDOM_STATE = 42


# =============================================================================
# DISTANCES
# =============================================================================
def standardise(cohort: pd.DataFrame, metrics: list) -> tuple[pd.DataFrame, np.ndarray]:
    """z-scored metrics (columns `<metric>_norm`, plus patient ID and n_SLH) and the Euclidean distance matrix."""
    z = StandardScaler().fit_transform(cohort[metrics].to_numpy())

    normalised = pd.DataFrame(z, columns=[f"{m}_norm" for m in metrics], index=cohort.index)
    normalised[PATIENT_ID_COL] = cohort[PATIENT_ID_COL].to_numpy()
    normalised["n_SLH"] = cohort["n_SLH"].to_numpy()

    distances = pairwise_distances(z, metric="euclidean")
    upper = distances[np.triu_indices_from(distances, k=1)]
    print(f"  {len(metrics)} metrics, {len(cohort)} participants; "
          f"distance mean {upper.mean():.2f}, min {upper.min():.2f}, max {upper.max():.2f}")
    return normalised, distances


def norm_matrix(normalised: pd.DataFrame) -> np.ndarray:
    return normalised[[c for c in normalised.columns if c.endswith("_norm")]].to_numpy()


# =============================================================================
# INTERESTING PAIRS
# =============================================================================
def find_interesting_pairs(normalised: pd.DataFrame, distances: np.ndarray,
                           distance_percentile: float, n_slh_diff_min: int) -> pd.DataFrame:
    """Pairs in the closest `distance_percentile`% of distances with an n_SLH difference >= `n_slh_diff_min`."""
    i, j = np.triu_indices(len(normalised), k=1)
    d = distances[i, j]
    cutoff = np.percentile(d, distance_percentile)

    ids, n_slh = normalised[PATIENT_ID_COL].to_numpy(), normalised["n_SLH"].to_numpy()
    diff = np.abs(n_slh[i] - n_slh[j])
    keep = (d <= cutoff) & (diff >= n_slh_diff_min)

    pairs = pd.DataFrame({
        "patient_1": ids[i[keep]], "patient_2": ids[j[keep]],
        "n_SLH_1": n_slh[i[keep]], "n_SLH_2": n_slh[j[keep]],
        "n_SLH_diff": diff[keep],
        "metric_distance": d[keep],
        "distance_percentile": [(d <= x).mean() * 100 for x in d[keep]],   # share of pairs at least as close
    })
    return pairs.sort_values("n_SLH_diff", ascending=False).reset_index(drop=True)


def pair_comparison_table(pairs: pd.DataFrame, cohort: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Raw metric values of both participants for the top pairs."""
    by_id = cohort.set_index(PATIENT_ID_COL)
    rows = []
    for _, r in pairs.head(N_PAIRS_TABLE).iterrows():
        row = {"pair": f"{r.patient_1} vs {r.patient_2}", "n_SLH_diff": r.n_SLH_diff,
               "distance": round(r.metric_distance, 3), "n_SLH_1": r.n_SLH_1, "n_SLH_2": r.n_SLH_2}
        for m in metrics:
            row[f"{m}_1"], row[f"{m}_2"] = round(by_id.loc[r.patient_1, m], 2), round(by_id.loc[r.patient_2, m], 2)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# EMBEDDINGS
# =============================================================================
def pca_embedding(X: np.ndarray, n_components: int = 3) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    print("  PCA variance explained: "
          + ", ".join(f"PC{k + 1} {v:.1%}" for k, v in enumerate(pca.explained_variance_ratio_))
          + f" (total {pca.explained_variance_ratio_.sum():.1%})")
    return coords


def tsne_embedding(X: np.ndarray) -> np.ndarray:
    return TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, len(X) - 1)).fit_transform(X)


def umap_embedding(X: np.ndarray) -> np.ndarray | None:
    try:
        from umap import UMAP
    except ImportError:
        print("  [WARN] UMAP not installed — skipping (pip install umap-learn).")
        return None
    return UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=min(15, len(X) - 1)).fit_transform(X)


# =============================================================================
# FIGURES (interactive HTML)
# =============================================================================
def _colorbar() -> dict:
    return dict(title="n_SLH", thickness=15, len=0.7)


def save_html(fig: go.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUTPUT_DIR / filename)
    print(f"  Saved: {filename}")


def plot_embedding(coords: np.ndarray, normalised: pd.DataFrame, title: str, axis_name: str, filename: str) -> None:
    """2-D or 3-D scatter of an embedding, coloured by n_SLH (hover: participant ID and n_SLH)."""
    three_d = coords.shape[1] >= 3
    marker = dict(size=6 if three_d else 8, color=normalised["n_SLH"], colorscale=COLORSCALE, showscale=True,
                  colorbar=_colorbar(), line=dict(width=0.5, color="white"))
    axes = {name: coords[:, k] for k, name in enumerate("xyz"[:coords.shape[1]])}
    trace_class = go.Scatter3d if three_d else go.Scatter

    fig = go.Figure(trace_class(
        mode="markers", marker=marker, text=normalised[PATIENT_ID_COL],
        customdata=normalised["n_SLH"], hovertemplate="<b>%{text}</b><br>n_SLH: %{customdata}<extra></extra>", **axes))

    labels = [f"{axis_name} {k + 1}" for k in range(coords.shape[1])]
    layout = dict(title=f"{title} (coloured by n_SLH)", font=dict(family=FONT_CSS_FAMILY, size=12))
    if three_d:
        layout.update(scene=dict(xaxis_title=labels[0], yaxis_title=labels[1], zaxis_title=labels[2],
                                 camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))), width=900, height=700)
    else:
        layout.update(xaxis_title=labels[0], yaxis_title=labels[1], width=800, height=600)
    fig.update_layout(**layout)
    save_html(fig, filename)


def plot_distance_matrix(distances: np.ndarray, normalised: pd.DataFrame, filename: str) -> None:
    """Distance heatmap with participants sorted by n_SLH (subsampled above MAX_HEATMAP_PARTICIPANTS)."""
    order = np.argsort(normalised["n_SLH"].to_numpy())
    if len(order) > MAX_HEATMAP_PARTICIPANTS:
        order = order[:: len(order) // MAX_HEATMAP_PARTICIPANTS]

    ids = normalised[PATIENT_ID_COL].to_numpy()[order]
    n_slh = normalised["n_SLH"].to_numpy()[order]
    z = distances[np.ix_(order, order)]
    hover = [[f"Row: {ids[a]} (n_SLH={n_slh[a]})<br>Col: {ids[b]} (n_SLH={n_slh[b]})<br>Distance: {z[a, b]:.3f}"
              for b in range(len(ids))] for a in range(len(ids))]

    fig = go.Figure(go.Heatmap(z=z, x=ids, y=ids, colorscale=COLORSCALE, hovertext=hover, hoverinfo="text",
                               colorbar=dict(title="Distance", thickness=15, len=0.7)))
    fig.update_layout(title="Distance matrix (sorted by n_SLH)", width=800, height=800, font=dict(family=FONT_CSS_FAMILY, size=11),
                      xaxis=dict(title="Participant", showticklabels=False), yaxis=dict(title="Participant", showticklabels=False))
    save_html(fig, filename)


def plot_pairs_network(pairs: pd.DataFrame, normalised: pd.DataFrame, filename: str) -> None:
    """Top pairs drawn as edges on the 2-D PCA map: close participants with different n_SLH."""
    coords = pca_embedding(norm_matrix(normalised), n_components=2)
    position = {pid: coords[k] for k, pid in enumerate(normalised[PATIENT_ID_COL])}
    n_slh = dict(zip(normalised[PATIENT_ID_COL], normalised["n_SLH"]))

    top = pairs.head(N_PAIRS_NETWORK)
    nodes = sorted(set(top["patient_1"]) | set(top["patient_2"]))
    edge_x, edge_y = [], []
    for _, r in top.iterrows():
        (x1, y1), (x2, y2) = position[r.patient_1], position[r.patient_2]
        edge_x += [x1, x2, None]
        edge_y += [y1, y2, None]

    fig = go.Figure([
        go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#888"), hoverinfo="none"),
        go.Scatter(x=[position[p][0] for p in nodes], y=[position[p][1] for p in nodes], mode="markers+text",
                   marker=dict(size=15, color=[n_slh[p] for p in nodes], colorscale=COLORSCALE, showscale=True,
                               colorbar=_colorbar(), line=dict(width=2, color="white")),
                   text=nodes, textposition="top center",
                   hovertext=[f"{p}<br>n_SLH: {n_slh[p]}" for p in nodes], hoverinfo="text"),
    ])
    fig.update_layout(title=f"Similarity network (top {len(top)} pairs)", showlegend=False, width=900, height=700,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), font=dict(family=FONT_CSS_FAMILY, size=11))
    save_html(fig, filename)


# =============================================================================
# MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--no_filter", action="store_true", help="Use all participants instead of the filtered cohort.")
    p.add_argument("--distance_percentile", type=float, default=DISTANCE_PERCENTILE,
                   help="A pair is 'close' if its distance is in the bottom X%% of all distances.")
    p.add_argument("--n_slh_diff", type=int, default=N_SLH_DIFF_MIN, help="Minimum n_SLH difference of a pair.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  Participant similarity in CGM-metric space and n_SLH")
    print("=" * 70)

    cohort = load_cohort(apply_filter=not args.no_filter)
    print(f"  Cohort: {len(cohort)} participants, n_SLH {cohort['n_SLH'].min():.0f}–{cohort['n_SLH'].max():.0f}")
    if len(cohort) < MIN_PARTICIPANTS:
        print(f"  Fewer than {MIN_PARTICIPANTS} participants: not enough for this analysis.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    normalised, distances = standardise(cohort, PROFILE_METRICS)
    normalised.to_csv(OUTPUT_DIR / "normalized_metrics.csv", index=False)

    pairs = find_interesting_pairs(normalised, distances, args.distance_percentile, args.n_slh_diff)
    print(f"\n  Interesting pairs (closest {args.distance_percentile:g}% of distances, "
          f"n_SLH difference ≥ {args.n_slh_diff}): {len(pairs)}")
    if len(pairs):
        pairs.to_csv(OUTPUT_DIR / "interesting_patient_pairs.csv", index=False)
        pair_comparison_table(pairs, cohort, PROFILE_METRICS).to_csv(OUTPUT_DIR / "pair_comparison_details.csv", index=False)
        print(pairs.head(5).round(3).to_string(index=False))

    print("\n  Figures")
    X = norm_matrix(normalised)
    plot_distance_matrix(distances, normalised, "fig1_distance_matrix.html")

    pca = pca_embedding(X, n_components=3)
    plot_embedding(pca[:, :2], normalised, "PCA projection", "Principal component", "fig2_pca_2d.html")
    plot_embedding(pca, normalised, "3D PCA projection", "PC", "fig3_pca_3d.html")
    plot_embedding(tsne_embedding(X), normalised, "t-SNE projection", "t-SNE component", "fig4_tsne_2d.html")

    umap = umap_embedding(X)
    if umap is not None:
        plot_embedding(umap, normalised, "UMAP projection", "UMAP component", "fig5_umap_2d.html")
    if len(pairs):
        plot_pairs_network(pairs, normalised, "fig6_pairs_network.html")

    print(f"\n  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
