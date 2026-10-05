"""
scripts/regression_analysis_poisson.py
=======================================

Negative Binomial regression: CGM-derived metrics as predictors of
SLH episode count (Num_SLH).

All 135 patients have complete CGM metrics → no missing data imputation
needed for the main models.

Proportional metrics (TIR, TAR, TBR, CV) are rescaled to 0–100 (%)
so that Rate Ratios are expressed per 1 % change.

Sections
--------
1.  Data loading & preparation
2.  Overdispersion check
3.  Correlation matrix of CGM metrics
4.  Univariate NB regression
5.  Multivariate NB regression (full model)
6.  Multivariate NB regression (parsimonious)
7.  Presentation figures  (8 figures, PDF + SVG + PNG)
      Fig 1 — Outcome distribution
      Fig 2 — Correlation heatmap
      Fig 3 — Univariate forest plot
      Fig 4 — Multivariate full forest plot
      Fig 5 — Multivariate parsimonious forest plot
      Fig 6 — Observed vs Predicted
      Fig 7 — Residuals 4-panel diagnostic
      Fig 8 — Hanging rootogram

Usage
-----
    python3 -m scripts.statistics.regression_analysis_poisson
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.special import gammaln
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

# =============================================================================
# PATHS
# =============================================================================
DATA_PATH = Path(
    "/Users/guillaume/Library/CloudStorage/"
    "OneDrive-UCL/Shared - Zain - Philippe - Laura - Guillaume/"
    "Data/Final cohort SLH.csv"
)
OUTPUT_DIR = Path(
    "/Users/guillaume/Documents/Projects_Lysy/"
    "counterfactual-cgm-slh/Results/NB_Regression_CGM_Metrics"
)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# =============================================================================
# PRESENTATION STYLE
# =============================================================================
FS_TITLE  = 17
FS_LABEL  = 14
FS_TICK   = 12
FS_ANNOT  = 11
FS_LEGEND = 12

C_SIG = "#1B4F8A"   # dark blue  — significant
C_NS  = "#AAAAAA"   # grey       — non-significant
C_REF = "#CC3311"   # red        — reference line
C_NB  = "#009E73"   # green      — NB variance curve

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FS_LABEL,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# =============================================================================
# HELPERS
# =============================================================================

def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def save_fig(fig, stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(Path(stem).with_suffix(f".{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved → {Path(stem).name}.pdf / .svg / .png")


def forest_plot(df_res, title, subtitle, stem, n_obs):
    """Horizontal forest plot sorted by RR (protective → harmful)."""
    df_p = df_res.sort_values("RR").reset_index(drop=True)
    n    = len(df_p)

    fig_h = max(5.5, 0.65 * n + 2.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    fig.subplots_adjust(left=0.36, right=0.86, top=0.88, bottom=0.10)

    for i, row in df_p.iterrows():
        sig   = row["p"] < 0.05
        color = C_SIG if sig else C_NS
        ms    = 10 if sig else 7
        ax.errorbar(
            row["RR"], i,
            xerr=[[row["RR"] - row["RR_lo"]], [row["RR_hi"] - row["RR"]]],
            fmt="o", color=color, markersize=ms,
            capsize=4, capthick=1.8, elinewidth=1.8, zorder=4,
        )
        p_str = ("p < 0.001  ***" if row["p"] < 0.001
                 else f"p = {row['p']:.3f}  {sig_label(row['p'])}")
        ax.text(ax.get_xlim()[1], i, f"  {p_str}",
                va="center", ha="left", fontsize=FS_ANNOT, color=color)

    ax.axvline(1.0, color=C_REF, lw=1.8, ls="--", zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df_p["Variable"], fontsize=FS_TICK + 1)
    ax.set_xlabel("Rate Ratio  (95 % CI)", fontsize=FS_LABEL, labelpad=8)
    ax.set_title(f"{title}\n{subtitle}", fontsize=FS_TITLE,
                 fontweight="bold", pad=14)

    handles = [
        mpatches.Patch(color=C_SIG, label="p < 0.05"),
        mpatches.Patch(color=C_NS,  label="p ≥ 0.05"),
        plt.Line2D([0], [0], color=C_REF, lw=1.8, ls="--", label="RR = 1"),
    ]
    ax.legend(handles=handles, fontsize=FS_LEGEND, loc="lower right",
              framealpha=0.92, edgecolor="#CCCCCC")
    ax.text(0.01, 0.02, f"n = {n_obs}", transform=ax.transAxes,
            fontsize=FS_ANNOT, color="#666666", va="bottom")
    ax.xaxis.grid(True, ls=":", lw=0.5, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    x_lo, x_hi = ax.get_xlim()
    ax.set_xlim(x_lo, x_hi * 1.40)
    save_fig(fig, stem)


def nb_mle(y, X):
    return sm.NegativeBinomial(y.astype(float), X.astype(float)).fit(disp=False)


def extract_nb(res, idx):
    coef  = float(res.params[idx])
    pval  = float(res.pvalues[idx])
    ci_lo = float(res.conf_int()[idx, 0])
    ci_hi = float(res.conf_int()[idx, 1])
    return coef, pval, ci_lo, ci_hi, np.exp(coef), np.exp(ci_lo), np.exp(ci_hi)


def nb_pmf_avg(k_arr, mu_arr, alpha):
    r = 1.0 / alpha
    out = []
    for k in k_arr:
        lp = (gammaln(r + k) - gammaln(r) - gammaln(k + 1)
              + r * np.log(r / (r + mu_arr))
              + k * np.log(mu_arr / (r + mu_arr)))
        out.append(np.exp(lp).mean() * len(mu_arr))
    return np.array(out)


def nb_dev_resid(y, mu, alpha):
    sign = np.sign(y - mu)
    out  = np.zeros_like(y, dtype=float)
    for k in range(len(y)):
        yk, mk = float(y[k]), float(mu[k])
        t1 = yk * np.log(yk / mk) if yk > 0 else 0.0
        t2 = (yk + 1.0 / alpha) * np.log((1 + alpha * yk) / (1 + alpha * mk))
        out[k] = sign[k] * np.sqrt(max(0.0, 2.0 * (t1 - t2)))
    return out


# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n" + "=" * 72)
print("  NB REGRESSION — CGM METRICS → SLH COUNT")
print("=" * 72 + "\n")

df    = pd.read_csv(DATA_PATH, sep=";", decimal=",")
print(f"  {len(df)} patients loaded")

OUTCOME = "Num_SLH"
y_all   = df[OUTCOME].values.astype(float)
mu_y    = y_all.mean()
var_y   = y_all.var(ddof=1)

# Rescale proportions → percent (0–100)
for col in ["CV", "TIR70180", "TIR70140", "TBR70", "TBR54", "TAR180", "TAR250"]:
    if col in df.columns:
        df[f"{col}_pct"] = df[col] * 100

# =============================================================================
# 2. OVERDISPERSION CHECK
# =============================================================================
pois0   = sm.GLM(y_all, np.ones((len(y_all), 1)),
                 family=sm.families.Poisson()).fit()
lhs_ct  = (y_all - pois0.fittedvalues) ** 2 - y_all
aux_ct  = sm.OLS(lhs_ct, pois0.fittedvalues ** 2).fit()
alpha_ct = float(np.atleast_1d(aux_ct.params)[0])
p_ct     = float(np.atleast_1d(aux_ct.pvalues)[0])

print(f"\n  Var/Mean = {var_y/mu_y:.2f}  (Poisson expects 1.0)")
p_ct_str = "< 0.001" if p_ct < 0.001 else f"= {p_ct:.4f}"
print(f"  Cameron-Trivedi  alpha = {alpha_ct:.4f},  p {p_ct_str}")
print(f"  → NB appropriate: {'YES ✓' if p_ct < 0.05 else 'not confirmed'}\n")

# =============================================================================
# PREDICTOR SETS
# =============================================================================
# Only metrics significant at p < 0.001 in univariate screening
PRED_ALL = {
    "Mean_glycemia":    "Mean glucose (mg/dL)",
    "Median_glycemia":  "Median glucose (mg/dL)",
    "SD":               "SD glucose (mg/dL)",
    "CV_pct":           "CV (%)",
    "TIR70180_pct":     "TIR 70–180 (%)",
    "TIR70140_pct":     "TIR 70–140 (%)",
    "TAR180_pct":       "TAR >180 (%)",
    "TAR250_pct":       "TAR >250 (%)",
    "TBR70_pct":        "TBR <70 (%)",
}

# Parsimonious: one representative per construct to remove collinearity
# Mean glucose  (level) | CV (variability) | TIR 70–180 (in-range) | TBR <70 (hypoglycaemia)
PRED_PARSI = {
    "Mean_glycemia":    "Mean glucose (mg/dL)",
    "CV_pct":           "CV (%)",
    "TIR70180_pct":     "TIR 70–180 (%)",
    "TBR70_pct":        "TBR <70 (%)",
}

CORR_COLS = {
    OUTCOME:            "N SLH",
    "Mean_glycemia":    "Mean\nglucose",
    "Median_glycemia":  "Median\nglucose",
    "SD":               "SD",
    "CV_pct":           "CV (%)",
    "TIR70180_pct":     "TIR\n70–180",
    "TIR70140_pct":     "TIR\n70–140",
    "TAR180_pct":       "TAR\n>180",
    "TAR250_pct":       "TAR\n>250",
    "TBR70_pct":        "TBR\n<70",
}

# =============================================================================
# 3. CORRELATION MATRIX
# =============================================================================
avail    = {k: v for k, v in CORR_COLS.items() if k in df.columns}
df_corr  = df[list(avail.keys())].copy()
df_corr.columns = list(avail.values())
corr_mat = df_corr.corr(method="spearman")
print(f"  Correlation matrix: {df_corr.shape[0]} patients, "
      f"{df_corr.shape[1]} variables\n")

# =============================================================================
# 4. UNIVARIATE NB REGRESSION
# =============================================================================
print("─" * 72)
print("  UNIVARIATE NB REGRESSION — CGM METRICS")
print("─" * 72 + "\n")

uni_results = []
for col, label in PRED_ALL.items():
    sub = df[[col, OUTCOME]].dropna()
    y_  = sub[OUTCOME].values.astype(float)
    X_  = sm.add_constant(sub[col].values.reshape(-1, 1).astype(float))
    try:
        res  = nb_mle(y_, X_)
        null = nb_mle(y_, np.ones((len(y_), 1)))
        coef, pval, ci_lo, ci_hi, rr, rr_lo, rr_hi = extract_nb(res, 1)
        pr2 = 1 - res.llf / null.llf
        uni_results.append({
            "Variable": label, "col": col, "N": len(sub),
            "Coef": coef, "SE": float(res.bse[1]), "p": pval,
            "CI_lo": ci_lo, "CI_hi": ci_hi,
            "RR": rr, "RR_lo": rr_lo, "RR_hi": rr_hi, "Pseudo_R2": pr2,
        })
        print(f"  {label:35s}  RR={rr:6.3f} [{rr_lo:.3f}–{rr_hi:.3f}]  "
              f"p={pval:.4f} {sig_label(pval)}")
    except Exception as e:
        print(f"  {label:35s}  [failed: {e}]")

df_uni = pd.DataFrame(uni_results).sort_values("p").reset_index(drop=True)
df_uni.to_csv(OUTPUT_DIR / "univariate_NB_cgm_results.csv", index=False)
print(f"\n  Saved → univariate_NB_cgm_results.csv\n")

# =============================================================================
# 5. MULTIVARIATE NB — FULL MODEL
# =============================================================================
print("─" * 72)
print("  MULTIVARIATE NB — FULL MODEL  (9 metrics, p<0.001 univariate)")
print("─" * 72 + "\n")

X_full_cols = list(PRED_ALL.keys())
df_full     = df[X_full_cols + [OUTCOME]].dropna()
y_full      = df_full[OUTCOME].values.astype(float)
X_full      = sm.add_constant(df_full[X_full_cols].values.astype(float))
nb_full     = nb_mle(y_full, X_full)
null_full   = nb_mle(y_full, np.ones((len(y_full), 1)))
pr2_full    = 1 - nb_full.llf / null_full.llf
alpha_full  = float(nb_full.params[-1])

print(f"  n = {len(df_full)}  |  α = {alpha_full:.4f}  |  "
      f"pseudo-R² = {pr2_full:.4f}  |  AIC = {nb_full.aic:.1f}\n")

full_results = []
for i, col in enumerate(X_full_cols):
    coef, pval, ci_lo, ci_hi, rr, rr_lo, rr_hi = extract_nb(nb_full, i + 1)
    full_results.append({
        "Variable": PRED_ALL[col], "col": col,
        "Coef": coef, "SE": float(nb_full.bse[i + 1]), "p": pval,
        "CI_lo": ci_lo, "CI_hi": ci_hi,
        "RR": rr, "RR_lo": rr_lo, "RR_hi": rr_hi,
    })
    print(f"  {PRED_ALL[col]:35s}  RR={rr:.3f} [{rr_lo:.3f}–{rr_hi:.3f}]  "
          f"p={pval:.4f} {sig_label(pval)}")

df_full_mv = pd.DataFrame(full_results).sort_values("p").reset_index(drop=True)
df_full_mv.to_csv(OUTPUT_DIR / "multivariate_NB_full_results.csv", index=False)

# =============================================================================
# 6. MULTIVARIATE NB — PARSIMONIOUS MODEL
# =============================================================================
print("\n" + "─" * 72)
print("  MULTIVARIATE NB — PARSIMONIOUS MODEL")
print("─" * 72 + "\n")

X_par_cols = list(PRED_PARSI.keys())
df_par     = df[X_par_cols + [OUTCOME]].dropna()
y_par      = df_par[OUTCOME].values.astype(float)
X_par      = sm.add_constant(df_par[X_par_cols].values.astype(float))
nb_par     = nb_mle(y_par, X_par)
null_par   = nb_mle(y_par, np.ones((len(y_par), 1)))
pr2_par    = 1 - nb_par.llf / null_par.llf
alpha_par  = float(nb_par.params[-1])

print(f"  n = {len(df_par)}  |  α = {alpha_par:.4f}  |  "
      f"pseudo-R² = {pr2_par:.4f}  |  AIC = {nb_par.aic:.1f}\n")

par_results = []
for i, col in enumerate(X_par_cols):
    coef, pval, ci_lo, ci_hi, rr, rr_lo, rr_hi = extract_nb(nb_par, i + 1)
    par_results.append({
        "Variable": PRED_PARSI[col], "col": col,
        "Coef": coef, "SE": float(nb_par.bse[i + 1]), "p": pval,
        "CI_lo": ci_lo, "CI_hi": ci_hi,
        "RR": rr, "RR_lo": rr_lo, "RR_hi": rr_hi,
    })
    print(f"  {PRED_PARSI[col]:35s}  RR={rr:.3f} [{rr_lo:.3f}–{rr_hi:.3f}]  "
          f"p={pval:.4f} {sig_label(pval)}")

df_par_mv = pd.DataFrame(par_results).sort_values("p").reset_index(drop=True)
df_par_mv.to_csv(OUTPUT_DIR / "multivariate_NB_parsimonious_results.csv", index=False)

# Parsimonious model used for diagnostics
y_mv   = y_par
mu_hat = nb_par.fittedvalues

# =============================================================================
# 7. PRESENTATION FIGURES
# =============================================================================
print("\n" + "─" * 72)
print("  GENERATING PRESENTATION FIGURES")
print("─" * 72 + "\n")

pb = dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", alpha=0.92)

# ── Fig 1: Outcome distribution ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6.5))
fig.subplots_adjust(left=0.11, right=0.96, top=0.88, bottom=0.13)
ax.hist(y_all, bins=range(int(y_all.min()), int(y_all.max()) + 2),
        color=C_SIG, edgecolor="white", linewidth=0.6, alpha=0.85)
x_rng = np.arange(int(y_all.min()), int(y_all.max()) + 1)
ax.plot(x_rng + 0.5, stats.poisson.pmf(x_rng, mu_y) * len(y_all),
        color=C_REF, lw=2.2, ls="--", label=f"Poisson fit  (μ = {mu_y:.1f})")
ax.axvline(mu_y, color=C_SIG, lw=1.5, ls=":", alpha=0.7)
ax.text(0.97, 0.97,
        f"n = {len(y_all)}\nMean = {mu_y:.1f}\nVariance = {var_y:.1f}\n"
        f"Var / Mean = {var_y/mu_y:.1f}\nC-T  p < 0.001",
        transform=ax.transAxes, fontsize=FS_ANNOT, va="top", ha="right", bbox=pb)
ax.set_xlabel("Number of SLH episodes", fontsize=FS_LABEL)
ax.set_ylabel("Number of patients", fontsize=FS_LABEL)
ax.set_title("Distribution of SLH episode counts\n"
             "(Var/Mean = 6.0 → strong overdispersion → Negative Binomial)",
             fontsize=FS_TITLE, fontweight="bold", pad=12)
ax.legend(fontsize=FS_LEGEND, framealpha=0.9)
ax.yaxis.grid(True, ls=":", lw=0.5, color="#DDDDDD", zorder=0)
ax.set_axisbelow(True)
save_fig(fig, OUTPUT_DIR / "fig1_outcome_distribution")

# ── Fig 2: Correlation matrix ─────────────────────────────────────────────
n_v  = len(corr_mat)
fig, ax = plt.subplots(figsize=(max(10, n_v * 1.15), max(8, n_v * 1.0)))
fig.subplots_adjust(left=0.15, right=0.97, top=0.91, bottom=0.17)
mask_up = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
sns.heatmap(corr_mat, mask=mask_up, ax=ax,
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": FS_ANNOT},
            linewidths=0.5, linecolor="#EEEEEE", square=True,
            cbar_kws={"shrink": 0.70, "label": "Spearman ρ"})
ax.set_title(
    f"Spearman correlation — CGM metrics & SLH count  (n = {df_corr.shape[0]})",
    fontsize=FS_TITLE, fontweight="bold", pad=14)
plt.setp(ax.xaxis.get_majorticklabels(), fontsize=FS_TICK, rotation=0, ha="center")
plt.setp(ax.yaxis.get_majorticklabels(), fontsize=FS_TICK, rotation=0)
save_fig(fig, OUTPUT_DIR / "fig2_correlation_matrix")

# ── Fig 3: Univariate forest ──────────────────────────────────────────────
forest_plot(df_uni,
            title="Univariate NB Regression — CGM Metrics",
            subtitle="Rate Ratios (95 % CI)  |  proportional metrics: per 1 %",
            stem=OUTPUT_DIR / "fig3_univariate_forest",
            n_obs=135)

# ── Fig 4: Multivariate full forest ──────────────────────────────────────
forest_plot(df_full_mv,
            title="Multivariate NB — All Significant Metrics",
            subtitle=(f"9 metrics (p < 0.001 univariate)  |  n = {len(df_full)}  "
                      f"|  α = {alpha_full:.3f}  "
                      f"|  pseudo-R² = {pr2_full:.3f}  "
                      f"|  AIC = {nb_full.aic:.0f}"),
            stem=OUTPUT_DIR / "fig4_multivariate_full_forest",
            n_obs=len(df_full))

# ── Fig 5: Multivariate parsimonious forest ───────────────────────────────
forest_plot(df_par_mv,
            title="Multivariate NB — Parsimonious Model",
            subtitle=(f"Mean glucose + CV + TIR 70–180 + TBR <70  "
                      f"|  n = {len(df_par)}  "
                      f"|  α = {alpha_par:.3f}  "
                      f"|  pseudo-R² = {pr2_par:.3f}  "
                      f"|  AIC = {nb_par.aic:.0f}"),
            stem=OUTPUT_DIR / "fig5_multivariate_parsimonious_forest",
            n_obs=len(df_par))

# ── Fig 6: Observed vs Predicted ─────────────────────────────────────────
r_sp, p_sp = stats.spearmanr(y_mv, mu_hat)
fig, ax = plt.subplots(figsize=(8.5, 7.5))
fig.subplots_adjust(left=0.13, right=0.95, top=0.88, bottom=0.13)
ax.scatter(y_mv, mu_hat, color=C_SIG, alpha=0.65, s=60,
           edgecolors="white", linewidths=0.5, zorder=3)
lo = min(y_mv.min(), mu_hat.min()) - 1
hi = max(y_mv.max(), mu_hat.max()) + 1
ax.plot([lo, hi], [lo, hi], color=C_REF, lw=2.0, ls="--",
        label="Perfect prediction")
m, b = np.polyfit(y_mv, mu_hat, 1)
x_fit = np.linspace(lo, hi, 200)
ax.plot(x_fit, m * x_fit + b, color="#555555", lw=1.6, ls="-", alpha=0.75,
        label=f"Regression fit (slope = {m:.2f})")
p_str = "< 0.001" if p_sp < 0.001 else f"= {p_sp:.3f}"
ax.text(0.04, 0.96,
        f"Spearman ρ = {r_sp:.3f}\np {p_str}",
        transform=ax.transAxes, fontsize=FS_ANNOT + 1,
        va="top", ha="left", bbox=pb)
ax.set_xlabel("Observed SLH count", fontsize=FS_LABEL)
ax.set_ylabel("Predicted SLH count (μ̂)", fontsize=FS_LABEL)
ax.set_title("NB Parsimonious Model — Observed vs Predicted",
             fontsize=FS_TITLE, fontweight="bold", pad=12)
ax.legend(fontsize=FS_LEGEND, framealpha=0.9)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.grid(True, ls=":", lw=0.4, color="#DDDDDD", zorder=0)
ax.set_axisbelow(True)
save_fig(fig, OUTPUT_DIR / "fig6_observed_vs_predicted")

# ── Fig 7: Residuals 4-panel ──────────────────────────────────────────────
var_hat = mu_hat + alpha_par * mu_hat ** 2
res_p   = (y_mv - mu_hat) / np.sqrt(var_hat)
res_d   = nb_dev_resid(y_mv, mu_hat, alpha_par)
sq_abs  = np.sqrt(np.abs(res_p))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(left=0.09, right=0.97, top=0.91, bottom=0.09,
                    hspace=0.42, wspace=0.35)
fig.suptitle("NB Regression — Residual Diagnostics  (CGM metrics)",
             fontsize=FS_TITLE + 1, fontweight="bold", y=0.98)

# 7a Pearson vs fitted
ax = axes[0, 0]
ax.scatter(mu_hat, res_p, color=C_SIG, alpha=0.55, s=45,
           edgecolors="white", linewidths=0.3, zorder=3)
ax.axhline(0, color=C_REF, lw=1.5, ls="--")
ax.axhline( 2, color="#AAAAAA", lw=0.8, ls=":")
ax.axhline(-2, color="#AAAAAA", lw=0.8, ls=":")
sm_xy = lowess(res_p, mu_hat, frac=0.5)
ax.plot(sm_xy[:, 0], sm_xy[:, 1], color="#E69F00", lw=2.2, label="Lowess")
ax.set_xlabel("Fitted μ̂", fontsize=FS_LABEL)
ax.set_ylabel("Pearson residuals", fontsize=FS_LABEL)
ax.set_title("Pearson residuals vs Fitted\n(flat around 0 = good)",
             fontsize=FS_LABEL, fontweight="bold")
ax.legend(fontsize=FS_ANNOT)
ax.yaxis.grid(True, ls=":", lw=0.4, color="#DDDDDD"); ax.set_axisbelow(True)

# 7b Scale-location
ax = axes[0, 1]
ax.scatter(mu_hat, sq_abs, color=C_SIG, alpha=0.55, s=45,
           edgecolors="white", linewidths=0.3, zorder=3)
sm_xy2 = lowess(sq_abs, mu_hat, frac=0.5)
ax.plot(sm_xy2[:, 0], sm_xy2[:, 1], color="#E69F00", lw=2.2, label="Lowess")
ax.set_xlabel("Fitted μ̂", fontsize=FS_LABEL)
ax.set_ylabel("√|Pearson residuals|", fontsize=FS_LABEL)
ax.set_title("Scale-Location\n(flat = homogeneous dispersion)",
             fontsize=FS_LABEL, fontweight="bold")
ax.legend(fontsize=FS_ANNOT)
ax.yaxis.grid(True, ls=":", lw=0.4, color="#DDDDDD"); ax.set_axisbelow(True)

# 7c Q-Q deviance residuals
ax = axes[1, 0]
(osm, osr), (slope, intercept, _) = stats.probplot(res_d, dist="norm")
ax.scatter(osm, osr, color=C_SIG, alpha=0.60, s=45,
           edgecolors="white", linewidths=0.3, zorder=3)
x_qq = np.array([osm.min(), osm.max()])
ax.plot(x_qq, slope * x_qq + intercept, color=C_REF, lw=2.0, ls="--")
ax.set_xlabel("Theoretical quantiles", fontsize=FS_LABEL)
ax.set_ylabel("Deviance residuals", fontsize=FS_LABEL)
ax.set_title("Q-Q Deviance Residuals\n(close to line = good fit)",
             fontsize=FS_LABEL, fontweight="bold")
ax.yaxis.grid(True, ls=":", lw=0.4, color="#DDDDDD"); ax.set_axisbelow(True)

# 7d Variance vs Mean
ax = axes[1, 1]
n_bins  = 10
bins    = np.percentile(mu_hat, np.linspace(0, 100, n_bins + 1))
bin_ids = np.digitize(mu_hat, bins[1:-1])
bmu     = np.array([mu_hat[bin_ids == b].mean()
                    for b in range(n_bins) if (bin_ids == b).sum() > 3])
bvar    = np.array([y_mv[bin_ids == b].var(ddof=1)
                    for b in range(n_bins) if (bin_ids == b).sum() > 3])
x_ln = np.linspace(bmu.min() * 0.9, bmu.max() * 1.1, 100)
ax.plot(x_ln, x_ln, color=C_REF, lw=1.8, ls="--", label="Poisson: Var = μ")
ax.plot(x_ln, x_ln + alpha_par * x_ln ** 2, color=C_NB, lw=1.8, ls="-.",
        label=f"NB: Var = μ + {alpha_par:.3f}·μ²")
ax.scatter(bmu, bvar, color=C_SIG, s=75, zorder=5, label="Empirical (binned)")
ax.set_xlabel("Fitted mean (μ̂, binned)", fontsize=FS_LABEL)
ax.set_ylabel("Observed variance", fontsize=FS_LABEL)
ax.set_title("Empirical Variance vs Mean\n(NB curve should track data)",
             fontsize=FS_LABEL, fontweight="bold")
ax.legend(fontsize=FS_ANNOT)
ax.yaxis.grid(True, ls=":", lw=0.4, color="#DDDDDD"); ax.set_axisbelow(True)

fig.text(0.50, 0.005,
         f"NB α = {alpha_par:.4f}  |  pseudo-R² = {pr2_par:.3f}  "
         f"|  AIC = {nb_par.aic:.1f}  |  n = {len(df_par)}",
         ha="center", va="bottom", fontsize=FS_ANNOT + 0.5, bbox=pb)
save_fig(fig, OUTPUT_DIR / "fig7_residuals_diagnostic")

# ── Fig 8: Rootogram ──────────────────────────────────────────────────────
count_range = np.arange(int(y_mv.min()), int(y_mv.max()) + 1)
observed    = np.array([(y_mv == k).sum() for k in count_range], dtype=float)
expected_nb = nb_pmf_avg(count_range, mu_hat, alpha_par)
sqrt_exp    = np.sqrt(expected_nb)
hanging     = sqrt_exp - np.sqrt(observed)

fig, ax = plt.subplots(figsize=(14, 6.5))
fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.13)
bar_c = [C_REF if h > 0 else C_SIG for h in hanging]
ax.bar(count_range, hanging, bottom=sqrt_exp,
       width=0.85, color=bar_c, alpha=0.80, edgecolor="white", linewidth=0.4)
ax.plot(count_range, sqrt_exp, color="#333333", lw=1.8, ls="-",
        marker="o", markersize=5, label="√ Expected (NB)")
ax.axhline(0, color="#555555", lw=0.9, ls="-")
ax.set_xlabel("Number of SLH episodes", fontsize=FS_LABEL)
ax.set_ylabel("√ Frequency", fontsize=FS_LABEL)
ax.set_title("Hanging Rootogram — NB model  (CGM metrics)\n"
             "Bars above 0 = over-predicted  |  Below 0 = under-predicted",
             fontsize=FS_TITLE, fontweight="bold", pad=12)
ax.legend(handles=[
    mpatches.Patch(color=C_REF, alpha=0.80, label="Over-predicted"),
    mpatches.Patch(color=C_SIG, alpha=0.80, label="Under-predicted"),
    plt.Line2D([0], [0], color="#333333", lw=1.8, marker="o",
               markersize=5, label="√ Expected (NB)"),
], fontsize=FS_LEGEND, framealpha=0.9, loc="upper right")
ax.yaxis.grid(True, ls=":", lw=0.4, color="#DDDDDD", zorder=0)
ax.set_axisbelow(True)
save_fig(fig, OUTPUT_DIR / "fig8_rootogram")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 72)
print("  SUMMARY — NB REGRESSION  (CGM METRICS)")
print("=" * 72)
print(f"\n  Var/Mean = {var_y/mu_y:.2f}  →  overdispersion confirmed")
print(f"\n  Parsimonious model  (n = {len(df_par)}):")
print(f"    NB dispersion α  = {alpha_par:.4f}")
print(f"    Pseudo-R²        = {pr2_par:.4f}")
print(f"    AIC              = {nb_par.aic:.1f}")
print(f"\n  Full model  (n = {len(df_full)}):")
print(f"    NB dispersion α  = {alpha_full:.4f}")
print(f"    Pseudo-R²        = {pr2_full:.4f}")
print(f"    AIC              = {nb_full.aic:.1f}")

print("\n  Significant predictors — parsimonious model (p < 0.05):")
sig_p = df_par_mv[df_par_mv["p"] < 0.05]
if len(sig_p):
    for _, r in sig_p.iterrows():
        d = "↑" if r["RR"] > 1 else "↓"
        print(f"  {d} {r['Variable']:35s}  "
              f"RR={r['RR']:.3f} [{r['RR_lo']:.3f}–{r['RR_hi']:.3f}]  "
              f"p={r['p']:.4f}")
else:
    print("  None")

print(f"\n  All outputs in: {OUTPUT_DIR}")
print("=" * 72 + "\n")
