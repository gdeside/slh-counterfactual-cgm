"""
scripts/statistics/slh_metrics_visualization_from_files.py
==========================================================
SLH Metrics Interactive Visualization (from existing files)
============================================================
Script: slh_metrics_visualization_from_files.py

Generates interactive Plotly figures showing relationships between
CGM metrics and SLH episode count (n_SLH) using pre-computed data.

Data sources:
  - Data/Metrics/metrics_all_patients.csv (CGM metrics)
  - Data/Episodes/{patient_id}_episodes.csv (SLH episode counts)

Multiple visualization types:
  - 3D scatter plots (3 metrics + n_SLH as color gradient)
  - 2D scatter plots with n_SLH gradient
  - Parallel coordinates
  - Bubble charts
  - Correlation heatmaps

Output: Interactive HTML files in Figures/SLH_Metrics_Interactive/

Authors: Guillaume Deside
Date: 2026-04-22

Usage
-----
    python3 -m scripts.statistics.slh_metrics_visualization_from_files [--help for options]
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import warnings
from pathlib import Path

# Add project root to path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# PROJECT IMPORTS
# =============================================================================
from src.config import (
    DATA_PATH,
    PATIENT_ID_COL,
    FILTER_STATUS_CSV,
)

from src.filtering.load_filtering import load_filtered_patient_list
from src.episodes.load_episodes import discover_episode_files

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
METRICS_CSV = DATA_PATH / "Metrics" / "metrics_all_patients.csv"
EPISODES_DIR = DATA_PATH / "Episodes"
FIGURES_DIR = DATA_PATH.parent / "Figures" / "SLH_Metrics_Interactive"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plotly color scale for n_SLH gradient
COLORSCALE = "Viridis"  # or "Plasma", "Inferno", "Turbo", "RdYlBu_r"

# Metric names to use
METRICS_OF_INTEREST = [
    "mean_glucose",
    "median_glucose", 
    "sd_glucose",
    "cv",
    "tir_70_180",
    "tar_180",
    "tar_250",
    "tbr_70",
]

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_patient_episodes(patient_id: str, episodes_dir: Path) -> pd.DataFrame | None:
    """
    Load episode data for a single patient.

    Parameters
    ----------
    patient_id : str
        Patient identifier
    episodes_dir : Path
        Directory containing episode CSV files

    Returns
    -------
    pd.DataFrame or None
        Episode data, or None if file not found
    """
    filepath = episodes_dir / f"{patient_id}_episodes.csv"

    if not filepath.exists():
        return None

    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"  [WARN] Error loading {patient_id}: {e}")
        return None


def count_slh_episodes(patient_id: str, episodes_dir: Path) -> int:
    """
    Count SLH episodes for a patient from pre-computed episode file.

    Parameters
    ----------
    patient_id : str
        Patient identifier
    episodes_dir : Path
        Directory containing episode CSV files

    Returns
    -------
    int
        Number of SLH episodes
    """
    episodes_df = load_patient_episodes(patient_id, episodes_dir)
    
    if episodes_df is None or len(episodes_df) == 0:
        return 0
    
    # Count episodes classified as SLH
    n_slh = (episodes_df["classification"] == "SLH").sum()
    
    return int(n_slh)


def collect_cohort_data(metrics_csv: Path, episodes_dir: Path, 
                        filter_csv: Path | None = None) -> pd.DataFrame:
    """
    Collect metrics and n_SLH for all patients from existing files.

    Parameters
    ----------
    metrics_csv : Path
        Path to metrics CSV file
    episodes_dir : Path
        Directory containing episode CSV files
    filter_csv : Path, optional
        Path to patient filter CSV

    Returns
    -------
    pd.DataFrame
        Combined metrics and n_SLH data
    """
    print("\n" + "=" * 70)
    print("  Loading data from existing files")
    print("=" * 70)
    
    # Load metrics
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_csv}")
    
    print(f"  Loading metrics from: {metrics_csv.name}")
    df_metrics = pd.read_csv(metrics_csv)
    
    # Filter to patients with valid windows
    df_metrics = df_metrics[df_metrics["has_window"]].copy()
    
    print(f"  Found {len(df_metrics)} patients with valid metrics")
    
    # Load filtered patient list if provided
    if filter_csv is not None and Path(filter_csv).exists():
        filtered_patients = load_filtered_patient_list(filter_csv)
        df_metrics = df_metrics[df_metrics[PATIENT_ID_COL].isin(filtered_patients)]
        print(f"  After filtering: {len(df_metrics)} patients")
    
    # Count SLH episodes for each patient
    print(f"\n  Counting SLH episodes from: {episodes_dir.name}/")
    
    n_slh_counts = []
    
    for pid in tqdm(df_metrics[PATIENT_ID_COL], desc="Processing patients", ncols=70):
        n_slh = count_slh_episodes(pid, episodes_dir)
        n_slh_counts.append(n_slh)
    
    df_metrics["n_SLH"] = n_slh_counts
    
    # Rename columns for compatibility with visualization functions
    column_mapping = {
        "mean_glucose": "Mean_glycemia",
        "median_glucose": "Median_glycemia",
        "sd_glucose": "SD",
        "cv": "CV",
        "tir_70_180": "TIR70180",
        "tar_180": "TAR180",
        "tar_250": "TAR250",
        "tbr_70": "TBR70",
    }
    
    df_metrics = df_metrics.rename(columns=column_mapping)
    
    # Select relevant columns
    viz_columns = [PATIENT_ID_COL, "n_SLH"] + list(column_mapping.values())
    df_viz = df_metrics[viz_columns].copy()
    
    print(f"\n  Collected data for {len(df_viz)} patients")
    print(f"  SLH range: {df_viz['n_SLH'].min():.0f} - {df_viz['n_SLH'].max():.0f}")
    print(f"  Patients with SLH > 0: {(df_viz['n_SLH'] > 0).sum()}")
    
    return df_viz


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_3d_scatter(df: pd.DataFrame, x_metric: str, y_metric: str, 
                    z_metric: str, filename: str):
    """3D scatter plot with n_SLH as color gradient."""
    
    fig = go.Figure(data=[go.Scatter3d(
        x=df[x_metric] * 100 if x_metric.startswith(('TIR', 'TAR', 'TBR', 'CV')) else df[x_metric],
        y=df[y_metric] * 100 if y_metric.startswith(('TIR', 'TAR', 'TBR', 'CV')) else df[y_metric],
        z=df[z_metric] * 100 if z_metric.startswith(('TIR', 'TAR', 'TBR', 'CV')) else df[z_metric],
        mode='markers',
        marker=dict(
            size=6,
            color=df['n_SLH'],
            colorscale=COLORSCALE,
            showscale=True,
            colorbar=dict(
                title="n_SLH",
                thickness=15,
                len=0.7
            ),
            line=dict(width=0.5, color='white')
        ),
        text=df[PATIENT_ID_COL],
        hovertemplate=(
            '<b>%{text}</b><br>' +
            f'{x_metric}: %{{x:.1f}}<br>' +
            f'{y_metric}: %{{y:.1f}}<br>' +
            f'{z_metric}: %{{z:.1f}}<br>' +
            'n_SLH: %{marker.color}<br>' +
            '<extra></extra>'
        )
    )])
    
    # Format axis labels
    x_label = x_metric.replace('_', ' ') + (' (%)' if x_metric.startswith(('TIR', 'TAR', 'TBR', 'CV')) else ' (mg/dL)')
    y_label = y_metric.replace('_', ' ') + (' (%)' if y_metric.startswith(('TIR', 'TAR', 'TBR', 'CV')) else ' (mg/dL)')
    z_label = z_metric.replace('_', ' ') + (' (%)' if z_metric.startswith(('TIR', 'TAR', 'TBR', 'CV')) else ' (mg/dL)')
    
    fig.update_layout(
        title=f'3D: {x_metric} vs {y_metric} vs {z_metric} (colored by n_SLH)',
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        font=dict(family="Arial", size=12),
        width=900,
        height=700
    )
    
    output_path = FIGURES_DIR / filename
    fig.write_html(output_path)
    print(f"  ✓ Saved: {filename}")


def plot_2d_scatter_grid(df: pd.DataFrame, metrics: list, filename: str):
    """2x2 grid of 2D scatter plots with n_SLH gradient."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'{m[0]} vs {m[1]}' for m in metrics],
        horizontal_spacing=0.12,
        vertical_spacing=0.12
    )
    
    for idx, (x_metric, y_metric) in enumerate(metrics):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        fig.add_trace(
            go.Scatter(
                x=df[x_metric] * 100 if x_metric.startswith(('TIR', 'TAR', 'TBR', 'CV')) else df[x_metric],
                y=df[y_metric] * 100 if y_metric.startswith(('TIR', 'TAR', 'TBR', 'CV')) else df[y_metric],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['n_SLH'],
                    colorscale=COLORSCALE,
                    showscale=(idx == 3),  # Only show colorbar on last plot
                    colorbar=dict(
                        title="n_SLH",
                        thickness=15,
                        len=0.8,
                        x=1.15
                    ) if idx == 3 else None,
                    line=dict(width=0.5, color='white')
                ),
                text=df[PATIENT_ID_COL],
                hovertemplate=(
                    '<b>%{text}</b><br>' +
                    f'{x_metric}: %{{x:.1f}}<br>' +
                    f'{y_metric}: %{{y:.1f}}<br>' +
                    'n_SLH: %{marker.color}<br>' +
                    '<extra></extra>'
                ),
                showlegend=False
            ),
            row=row, col=col
        )
        
        x_label = x_metric.replace('_', ' ') + (' (%)' if x_metric.startswith(('TIR', 'TAR', 'TBR', 'CV')) else '')
        y_label = y_metric.replace('_', ' ') + (' (%)' if y_metric.startswith(('TIR', 'TAR', 'TBR', 'CV')) else '')
        
        fig.update_xaxes(title_text=x_label, row=row, col=col)
        fig.update_yaxes(title_text=y_label, row=row, col=col)
    
    fig.update_layout(
        title_text='CGM Metrics Grid (colored by n_SLH)',
        font=dict(family="Arial", size=11),
        width=1000,
        height=800
    )
    
    output_path = FIGURES_DIR / filename
    fig.write_html(output_path)
    print(f"  ✓ Saved: {filename}")


def plot_bubble_chart(df: pd.DataFrame, x_metric: str, y_metric: str, filename: str):
    """2D scatter with n_SLH as both color and size."""
    
    # Normalize n_SLH for bubble size (min=5, max=30)
    n_slh_normalized = 5 + 25 * (df['n_SLH'] - df['n_SLH'].min()) / (df['n_SLH'].max() - df['n_SLH'].min() + 0.001)
    
    fig = go.Figure(data=[go.Scatter(
        x=df[x_metric] * 100 if x_metric.startswith(('TIR', 'TAR', 'TBR', 'CV')) else df[x_metric],
        y=df[y_metric] * 100 if y_metric.startswith(('TIR', 'TAR', 'TBR', 'CV')) else df[y_metric],
        mode='markers',
        marker=dict(
            size=n_slh_normalized,
            color=df['n_SLH'],
            colorscale=COLORSCALE,
            showscale=True,
            colorbar=dict(
                title="n_SLH",
                thickness=15,
                len=0.7
            ),
            line=dict(width=1, color='white'),
            opacity=0.7
        ),
        text=df[PATIENT_ID_COL],
        hovertemplate=(
            '<b>%{text}</b><br>' +
            f'{x_metric}: %{{x:.1f}}<br>' +
            f'{y_metric}: %{{y:.1f}}<br>' +
            'n_SLH: %{marker.color}<br>' +
            '<extra></extra>'
        )
    )])
    
    x_label = x_metric.replace('_', ' ') + (' (%)' if x_metric.startswith(('TIR', 'TAR', 'TBR', 'CV')) else ' (mg/dL)')
    y_label = y_metric.replace('_', ' ') + (' (%)' if y_metric.startswith(('TIR', 'TAR', 'TBR', 'CV')) else ' (mg/dL)')
    
    fig.update_layout(
        title=f'Bubble: {x_metric} vs {y_metric} (size & color = n_SLH)',
        xaxis_title=x_label,
        yaxis_title=y_label,
        font=dict(family="Arial", size=12),
        width=800,
        height=600
    )
    
    output_path = FIGURES_DIR / filename
    fig.write_html(output_path)
    print(f"  ✓ Saved: {filename}")


def plot_parallel_coordinates(df: pd.DataFrame, metrics: list, filename: str):
    """Parallel coordinates plot with n_SLH gradient."""
    
    # Prepare data - convert percentages to 0-100 scale
    df_plot = df.copy()
    for m in metrics:
        if m.startswith(('TIR', 'TAR', 'TBR', 'CV')):
            df_plot[m] = df_plot[m] * 100
    
    dimensions = []
    for m in metrics:
        label = m.replace('_', ' ') + (' (%)' if m.startswith(('TIR', 'TAR', 'TBR', 'CV')) else ' (mg/dL)')
        dimensions.append(dict(
            label=label,
            values=df_plot[m]
        ))
    
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=df['n_SLH'],
            colorscale=COLORSCALE,
            showscale=True,
            colorbar=dict(
                title="n_SLH",
                thickness=15,
                len=0.7
            )
        ),
        dimensions=dimensions
    ))
    
    fig.update_layout(
        title='Parallel Coordinates: CGM Metrics (colored by n_SLH)',
        font=dict(family="Arial", size=11),
        width=1200,
        height=600
    )
    
    output_path = FIGURES_DIR / filename
    fig.write_html(output_path)
    print(f"  ✓ Saved: {filename}")


def plot_correlation_heatmap(df: pd.DataFrame, metrics: list, filename: str):
    """Correlation heatmap including n_SLH."""
    
    # Convert percentages for correlation
    df_corr = df.copy()
    for m in metrics:
        if m.startswith(('TIR', 'TAR', 'TBR', 'CV')):
            df_corr[m] = df_corr[m] * 100
    
    corr_matrix = df_corr[metrics].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[m.replace('_', ' ') for m in corr_matrix.columns],
        y=[m.replace('_', ' ') for m in corr_matrix.index],
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(
            title="Correlation",
            thickness=15,
            len=0.7
        )
    ))
    
    fig.update_layout(
        title='Correlation Heatmap: Metrics with n_SLH',
        font=dict(family="Arial", size=11),
        width=700,
        height=700,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    output_path = FIGURES_DIR / filename
    fig.write_html(output_path)
    print(f"  ✓ Saved: {filename}")


def plot_n_slh_distribution(df: pd.DataFrame, filename: str):
    """Distribution of n_SLH across cohort."""
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=df['n_SLH'],
        nbinsx=30,
        marker=dict(
            color=df['n_SLH'],
            colorscale=COLORSCALE,
            line=dict(width=1, color='white')
        ),
        hovertemplate='n_SLH: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Distribution of n_SLH Across Cohort',
        xaxis_title='Number of SLH Episodes (n_SLH)',
        yaxis_title='Number of Patients',
        font=dict(family="Arial", size=12),
        width=800,
        height=500,
        showlegend=False
    )
    
    output_path = FIGURES_DIR / filename
    fig.write_html(output_path)
    print(f"  ✓ Saved: {filename}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  SLH Metrics Visualization with Plotly (from existing files)")
    print("=" * 70)
    print(f"  Metrics:  {METRICS_CSV}")
    print(f"  Episodes: {EPISODES_DIR}")
    print(f"  Output:   {FIGURES_DIR}")
    print("=" * 70)
    
    # Collect data from existing files
    df = collect_cohort_data(
        metrics_csv=METRICS_CSV,
        episodes_dir=EPISODES_DIR,
        filter_csv=FILTER_STATUS_CSV
    )
    
    if len(df) == 0:
        print("\n  [!] No data collected. Exiting.")
        return
    
    # Save combined data for reference
    combined_csv = FIGURES_DIR / "combined_metrics_slh.csv"
    df.to_csv(combined_csv, index=False)
    print(f"\n  Combined data saved to: {combined_csv.name}")
    
    print("\n" + "=" * 70)
    print("  Generating interactive figures...")
    print("=" * 70 + "\n")
    
    # 1. 3D scatter plots (various metric combinations)
    print("  [1] 3D Scatter Plots")
    plot_3d_scatter(df, 'TIR70180', 'TAR180', 'Mean_glycemia', 
                   'fig1_3d_TIR_TAR_Mean.html')
    plot_3d_scatter(df, 'TIR70180', 'CV', 'Median_glycemia', 
                   'fig2_3d_TIR_CV_Median.html')
    plot_3d_scatter(df, 'TAR180', 'TAR250', 'SD', 
                   'fig3_3d_TAR180_TAR250_SD.html')
    
    # 2. 2D scatter grid
    print("\n  [2] 2D Scatter Grid")
    plot_2d_scatter_grid(df, [
        ('TIR70180', 'Mean_glycemia'),
        ('TAR180', 'Median_glycemia'),
        ('CV', 'SD'),
        ('TIR70180', 'TAR180')
    ], 'fig4_2d_grid.html')
    
    # 3. Bubble charts
    print("\n  [3] Bubble Charts")
    plot_bubble_chart(df, 'TIR70180', 'Mean_glycemia', 
                     'fig5_bubble_TIR_Mean.html')
    plot_bubble_chart(df, 'TAR180', 'CV', 
                     'fig6_bubble_TAR_CV.html')
    
    # 4. Parallel coordinates
    print("\n  [4] Parallel Coordinates")
    plot_parallel_coordinates(df, ['n_SLH', 'TIR70180', 'TAR180', 
                                  'Mean_glycemia', 'CV'], 
                             'fig7_parallel_coords.html')
    
    # 5. Correlation heatmap
    print("\n  [5] Correlation Heatmap")
    plot_correlation_heatmap(df, ['n_SLH', 'TIR70180', 'TAR180', 'TAR250',
                                 'Mean_glycemia', 'Median_glycemia', 'CV', 'SD'], 
                            'fig8_correlation_heatmap.html')
    
    # 6. n_SLH distribution
    print("\n  [6] n_SLH Distribution")
    plot_n_slh_distribution(df, 'fig9_nslh_distribution.html')
    
    # 7. Additional metric combinations
    print("\n  [7] Additional Combinations")
    plot_3d_scatter(df, 'TIR70180', 'TAR250', 'CV', 
                   'fig10_3d_TIR_TAR250_CV.html')
    plot_bubble_chart(df, 'Median_glycemia', 'SD', 
                     'fig11_bubble_Median_SD.html')
    
    print("\n" + "=" * 70)
    print("  ✓ All figures generated successfully!")
    print(f"  Output directory: {FIGURES_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
