# Counterfactual CGM Signals for Short-Lasting Hyperglycaemia (SLH)

Code, analysis pipeline and figure scripts for the manuscript:

> *Short-Lived Hyperglycaemia Contributes to Continuous Glucose Monitoring Metrics Used for Therapeutic Decision-Making*
> Deside G, Syed Z, Symul L, Lysy P. *The Lancet Digital Health*.
>
> Archived on Zenodo: [10.5281/zenodo.23102518](https://doi.org/10.5281/zenodo.23102518)

---

## Table of contents

- [Overview](#overview)
- [Method in brief](#method-in-brief)
- [Manuscript figure and table map](#manuscript-figure-and-table-map)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start with the sample dataset](#quick-start-with-the-sample-dataset)
- [Data availability](#data-availability)
- [How to use the data](#how-to-use-the-data)
- [How to reproduce](#how-to-reproduce)
- [Glucose units](#glucose-units-mgdl-and-mmoll)
- [Makefile reference](#makefile-reference)
- [Tests](#tests)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)

---

## Overview

This repository implements a **counterfactual signal reconstruction** approach for quantifying the metabolic
weight of short-lasting hyperglycaemia (SLH) episodes in paediatric continuous glucose monitoring (CGM) data:
what would the glucose signal, and the consensus CGM metrics derived from it, look like if the SLH episodes had not occurred?

**What the code does**

1. Preprocesses raw CGM exports into 45-day windows (≥ 70 % coverage, 5-min sampling) and computes standard CGM metrics.
2. Filters the cohort (median-glucose criterion) and detects hyperglycaemic episodes (> 180 mg/dL).
3. Classifies each episode as **SLH** or **LLH** (long-lasting) using a six-criterion scheme (C1–C6).
4. Builds a counterfactual signal *C(t)* by linear interpolation across each SLH zone, expanded by an offset δ.
5. Quantifies Δ metrics (TIR, TAR, median glucose, CV, …) between the observed signal *G(t)* and *C(t)*,
   with a sensitivity analysis over δ.
6. Explores determinants of SLH burden (CGM metrics, phenotype, HbA1c) and a clock-lock rhythm index Λ.
7. Produces all manuscript figures and tables, and a HIPAA/GDPR-anonymised open dataset.

**Cohort:** 245 paediatric patients (IDs A001–A245), 45-day preprocessed windows, daytime-only SLH (22:00–06:00 excluded).

---

## Method in brief

| Step | Definition | Where |
|---|---|---|
| Hyperglycaemia | glucose > 180 mg/dL (context threshold 140 mg/dL) | `src/episodes/episode_config.py` |
| Episode merging | gaps ≤ 30 min merged; internal gaps ≤ 15 min | `episode_config.py` |
| SLH criteria | C1 duration 10–60 min · C2 ≥ 75 % data completeness in a 60-min context window · C3 daytime onset · C4/C5 no hyperglycaemia before/after · C6 crossing condition | `src/episodes/criteria.py` |
| Counterfactual | linear interpolation over the SLH zone expanded by δ (default **30 min**; tested 0–90 min) | `src/counterfactual/build_counterfactual.py` |
| Rhythm index Λ | permutation-sensitive clock-lock metric | `src/metrics/rhythm.py` |

All thresholds live in `src/config/` (`constants.py`, `episode_config`, `counterfactual_config`) — nothing is hard-coded in scripts.

---

## Manuscript figure and table map

| Manuscript item | Script | Notes |
|---|---|---|
| Fig. 1 — SLH detection schema | `scripts/fig_slh_schema.py` | |
| Fig. 2 — SLH/LLH classification summary | `scripts/paper_figures/generate_figure_2_classification_summary.py` | |
| Fig. 3 — Two patients, similar metrics, different SLH burden (high vs low SLH) | `scripts/paper_figures/generate_figure_3.py` | |
| Fig. 4 — Counterfactual construction and sensitivity | `scripts/paper_figures/generate_figure_4_counterfactual_construction.py` | needs `make sensitivity_analysis` |
| Fig. 5 — Distribution of ΔTIR / Δmedian | `scripts/paper_figures/generate_figure_5_counterfactual_distribution.py` | |
| Fig. 5 (alt.) — Determinants of SLH burden | `scripts/paper_figures/generate_figure_5_slh_determinants.py` | needs the clinical phenotype table |
| Table 1 — Cohort characteristics | `scripts/paper_figures/generate_table1.py` | needs `pysummaries` |
| Suppl. Fig. — Cohort flowchart | `scripts/paper_figures/generate_cohort_flowchart.py` | needs `rsvg-convert` (`brew install librsvg`) |
| Suppl. Fig. 1 — Data quality and cohort selection | `scripts/paper_figures/generate_supplementary_figure_1.py` | |
| Suppl. Fig. — Phenotype determinants | `scripts/paper_figures/generate_supplementary_figure_phenotype_determinants.py` | |

Run any script as a module from the repository root, e.g.
`python3 -m scripts.paper_figures.generate_figure_3`.

---

## Repository structure

```
counterfactual-cgm-slh/
├── src/                          # Reusable, tested library code
│   ├── config/                   #   thresholds (constants.py), paths (paths.py), figure settings
│   ├── preprocessing/            #   loading, quality flags, coverage, 45-day window extraction
│   ├── filtering/                #   cohort filtering
│   ├── episodes/                 #   episode detection, criteria C1–C6, classification
│   ├── counterfactual/           #   linear-interpolation counterfactual construction
│   ├── metrics/                  #   CGM metrics (metrics.py) and clock-lock index (rhythm.py)
│   ├── agp/                      #   ambulatory glucose profile
│   ├── visualization/            #   plotting helpers and colour palette
│   └── utils/                    #   time and unit-conversion utilities
│
├── scripts/                      # Executable analysis scripts (`python3 -m scripts.<pkg>.<name>`)
│   ├── preprocessing/            #   raw → windows → metrics → filtered cohort; open-data release; unit conversion
│   ├── episodes_detection/       #   episode detection + classification
│   ├── counterfactual/           #   counterfactual figures + sensitivity analysis over δ
│   ├── statistics/               #   descriptive stats, regressions, quartiles, clock-lock index (`src/metrics/rhythm.py`)
│   ├── patient_selection/        #   participant pairs / similarity map illustrating SLH beyond standard CGM metrics
│   ├── paper_figures/            #   manuscript figures and tables
│   ├── presentation_figures/     #   talk figures, synthetic-data friendly (`make presentation_figures`; not part of the paper)
│   └── supplementary_code/       #   exploratory / superseded analyses
│
├── notebooks/                    # 8 example notebooks (01_cgm_metrics … 08_phenotype_cgm_slh_association)
├── tests/                        # pytest unit tests
├── docs/UNITS.md                 # mg/dL ↔ mmol/L handling
├── Makefile                      # all execution commands — `make help`
├── requirements.txt
└── LICENSE                       # MIT
```

---

## Installation

Requires **Python ≥ 3.11** (developed and tested on 3.13).

```bash
git clone https://github.com/gdeside/counterfactual-cgm-slh.git
cd counterfactual-cgm-slh
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Point the code at your data.** Set the `CGM_DATA_ROOT` environment variable (or edit `BASE_PATH` in
[`src/config/paths.py`](src/config/paths.py)). Expected layout under `BASE_PATH`:

```
Data/Raw/  Data/Preprocessed/  Data/Filtered/  Data/Metrics/  Data/Episodes/
Data/Final cohort SLH.csv        # clinical phenotype table
Results/                         # all outputs are written here
```

---

## Quick start with the sample dataset

No access to patient data is needed to try the pipeline. `scripts/sample_data/` generates **4 synthetic patients**
(A001–A004, 50 days, 5-min sampling, sensor gaps, meal excursions producing both SLH and LLH episodes;
fixed seed). The same file is attached to each GitHub release as `sample_data.zip`.

```bash
# Option A — generate locally
make sample_data                 # → sample_data/ and sample_data.zip
make sample_data_run             # preprocessing → metrics → filtering → episodes → sensitivity (~2 min)

# Option B — from the release asset
unzip sample_data.zip
export CGM_DATA_ROOT=$PWD/sample_data
make pipeline
```

`CGM_DATA_ROOT` overrides `BASE_PATH` in `src/config/paths.py`; all outputs are written under it
(`Data/…`, `Results/…`, `Figures/…`). The sample data are **synthetic**: they demonstrate that the code runs and
what the outputs look like, and are not expected to reproduce any manuscript number. Steps that need the clinical
phenotype table (`Data/Final cohort SLH.csv`: Table 1, determinant figures) are not covered.

---

## Data availability

- **Raw CGM data** are patient-level clinical data and **cannot be shared** publicly.
- An **anonymised open dataset** (HIPAA Safe Harbor + GDPR: relative timestamps, Gaussian glucose noise, shuffled IDs)
  is generated by `make open_data` and deposited at **[10.5281/zenodo.23102518](https://doi.org/10.5281/zenodo.23102518)** under CC BY 4.0.
  Its `DATACARD.md` documents the de-identification steps.
- Because noise is added to glucose values, results computed from the open dataset will differ slightly from
  those reported in the manuscript.

---

## How to use the data

Three kinds of data can be used with this code. In every case, point the code at a working folder with
`CGM_DATA_ROOT` (default: `./data`); inputs are read from, and outputs written to, sub-folders of it.

| Data | Use it for | How |
|---|---|---|
| **Synthetic sample** (`make sample_data`) | trying the code, the notebooks | `export CGM_DATA_ROOT=$PWD/sample_data` then `make pipeline` |
| **Open anonymised dataset** (Zenodo) | re-running the analysis | convert it to the raw layout (below), then `make pipeline` |
| **Your own CGM exports** | applying the method to new data | put one file per patient in `Data/Raw/` (below) |

### Option 1 — Open dataset (Zenodo)

The release contains `cohort_open_data.csv` (all patients) and `per_patient/OD_xxx.csv`, with three columns:

| Column | Meaning |
|---|---|
| `patient_id` | anonymous code (`OD_001` …); the mapping to original IDs does not exist |
| `time_h` | relative time since monitoring start (hours, 5-min steps) |
| `glucose_mgdl` (or `glucose_mmoll`) | glucose with added Gaussian noise (mg/dL, or mmol/L) |

```bash
export CGM_DATA_ROOT=$PWD/od_run                       # any empty folder
make open_data_to_raw OPEN_DATA_CSV=/path/to/cohort_open_data.csv
make pipeline                                          # preprocessing → metrics → filtering → episodes → sensitivity
```

`open_data_to_raw` writes `Data/Raw/<patient_id>.xlsx` (columns `time`, `glycemie` in mg/dL) and maps the relative
time onto an arbitrary origin (2000-01-01 00:00): no real date is involved. Because noise was added to glucose, results
differ slightly from the manuscript. The phenotype table (`Data/Final cohort SLH.csv`) is **not** part of the open
dataset, so Table 1 and the determinant figures cannot be reproduced from it.

### Option 2 — Your own CGM data

Create `Data/Raw/<patient_id>.xlsx` under `CGM_DATA_ROOT`, one file per patient, with two columns:

| Column | Content |
|---|---|
| `time` | timestamp of each reading (date-time) |
| `glycemie` (also accepted: `glycemia`, `glucose`) | glucose in **mg/dL** (convert mmol/L × 18.0156 first) |

The pipeline expects 5-minute sampling and extracts a 45-day window with ≥ 70 % coverage per patient (patients without
one are listed in `Data/Preprocessed/patients_status.csv`). Then run `make pipeline`.

### Which patients are shown in the illustrations?

Scripts that illustrate one patient (episode detection, counterfactual construction, AGP, …) or two patients
(Figures 3–5, patient comparison) **do not hard-code participant IDs**. Participant IDs differ between datasets
(original study codes, `A001`… in the sample data, `OD_001`… in the open dataset), so the code picks patients itself
unless you tell it which ones to use.

**Default (automatic).** From the filtered cohort, the script takes the patient with the **most** SLH episodes
and, for two-patient figures, also the patient with the **fewest** (at least one). The chosen IDs are printed
(`[auto] patient with most SLH episodes: …`).

**Choosing your own patients**

| Figure type | How | Example |
|---|---|---|
| One patient | `--patient_id <ID>` | `python3 -m scripts.fig_slh_schema --patient_id A174` |
| Two patients (Figures 3, 4, 5, `patient_comparison`, two-patient presentation figures) | environment variables `CGM_PATIENT_HIGH` and `CGM_PATIENT_LOW` | see below |
| `lininterp_*` scripts | set `SELECTED_PID` at the top of the script (default `None` = automatic) | `SELECTED_PID = "A224"` |

`CGM_PATIENT_HIGH` is the patient with a **high** SLH burden (many SLH episodes) and `CGM_PATIENT_LOW` the one with a
**low** burden; they are the two patients contrasted in Figures 3–5. They are *environment variables*: the
`NAME=value` part placed before a command sets them for that command only, so they work with `make` and with `python3 -m`:

```bash
# Figures 3, 4 and 5 for two chosen patients
CGM_PATIENT_HIGH=A174 CGM_PATIENT_LOW=A224 make fig3 fig4 fig5

# the same, for every figure of the shell session
export CGM_PATIENT_HIGH=A174
export CGM_PATIENT_LOW=A224
make fig3 fig4 fig5 patient_comparison
```

`A174` and `A224` are the two participants shown in the manuscript (high vs low SLH burden); they exist only in the
original dataset. If an ID is not found in the data, the script prints a warning and falls back to the automatic choice,
so the same command also runs on the sample or open data (it then shows other patients). Selection logic:
[`src/utils/patient_selection.py`](src/utils/patient_selection.py).

### Presentation figures

`scripts/presentation_figures/` holds the 16:9 figures made for talks (not part of the manuscript). They use the same
episode classification as Figure 2 and choose their patients automatically (see
[Which patients are shown](#which-patients-are-shown-in-the-illustrations)). `make presentation_figures` runs all of
them; outputs go to `scripts/presentation_figures/output/` (git-ignored).

### Where the results go

| Folder under `CGM_DATA_ROOT` | Content |
|---|---|
| `Data/Preprocessed/`, `Data/Metrics/`, `Data/Filtered/`, `Data/Episodes/` | windows, CGM metrics, filtered cohort, classified episodes |
| `Results/` | tables and sensitivity-analysis outputs (`Results/Open_Data/` for the anonymised release) |
| `Figures/` | all figures (PDF + SVG) |

---

## How to reproduce

`make help` lists every target. Run the pipeline in this order.

```bash
# 1. Data pipeline
make time_step_analysis      # raw timestep distribution
make preprocessing           # 45-day windows
make coverage_analysis       # coverage distribution
make metrics_preprocessing   # CGM metrics (TIR, TAR, SD, CV, …)
make filtering               # median-glucose filter
make episode_detection       # episode detection + SLH/LLH classification

# 2. Statistics
make slh_stats               # SLH/LLH descriptive statistics
make quartile_analysis       # stratification by n_SLH
make scatter_analysis        # n_SLH vs Δ metrics

# 3. Counterfactual + sensitivity (δ = 0…90 min)
make lininterp_stepbystep
make lininterp_offset_comp
make lininterp_45days
make sensitivity_analysis    # all patients × all δ (slowest step)
make sensitivity_figures

# 4. Manuscript figures / tables
make paper                   # Fig. 2-5, Suppl. Fig. 1, cohort flowchart
make table1 fig5_determinants suppfig_phenotype   # need the clinical phenotype table
# individual figures: make fig2 fig3 fig4 fig5 … (see the figure map above)

# 5. Open-data release
make open_data               # σ = 2 mg/dL noise, shuffled IDs
make open_data_phase_shift   # + random monitoring-start shift
```

Example notebooks (`notebooks/01_…` to `08_…`: metrics, preprocessing + filtering, episode detection, counterfactual,
sensitivity analysis, median-profile difference (t_max), patient comparison, phenotype/CGM association with n_SLH) run on the sample data
(`08_…` needs the clinical file for real results and falls back to a flagged random demo phenotype without it):

```bash
make sample_data_run
CGM_DATA_ROOT=$PWD/sample_data jupyter lab notebooks/
```

---

## Glucose units (mg/dL and mmol/L)

The pipeline computes and stores glucose in **mg/dL**; mmol/L is a presentation layer, so switching units cannot change a result.

```bash
python3 -m scripts.preprocessing.convert_data_units            # mmol/L copies of data files
CGM_GLUCOSE_UNIT="mmol/L" python3 -m scripts.paper_figures.generate_figure_3
python3 -m scripts.preprocessing.generate_open_data --unit mmol/L
```

Figure filenames carry the unit (`..._mgdl.pdf` / `..._mmoll.pdf`). Details in [`docs/UNITS.md`](docs/UNITS.md).

---

## Makefile reference

`make help` prints the authoritative list (descriptions live next to each target in the `Makefile`).

| Group | Targets |
|---|---|
| Setup | `install`, `requirements-check` (reports missing imports, never overwrites `requirements.txt`) |
| Data pipeline | `time_step_analysis`, `preprocessing`, `coverage_analysis`, `metrics_preprocessing`, `filtering`, `filtered_metrics_distribution`, `episode_detection`, **`pipeline`** (core chain) |
| Statistics | `slh_stats`, `quartile_analysis`, `scatter_analysis`, `regression_analysis` |
| SLH figures | `slh_schema`, `slh_funnel`, `slh_overview`, `slh_zoom_panels`, `agp_recurrence`, `patient_comparison` |
| Counterfactual | `lininterp_stepbystep`, `lininterp_offset_comp`, `lininterp_45days` |
| Sensitivity | `sensitivity_analysis`, `lines_`/`boxplots_`/`cdf_`/`agp_`/`timeseries_sensitivity`, `sensitivity_figures` |
| Manuscript | `table1`, `fig2`, `fig3`, `fig4`, `fig5`, `fig5_determinants`, `suppfig1`, `suppfig_phenotype`, `cohort_flowchart`, **`paper`** |
| Open data / units | `open_data`, `open_data_phase_shift`, `convert_units` |
| Sample dataset | `sample_data`, `sample_data_run` |
| Utility | `figures`, `tests`, `clean` |

---

## Tests

```bash
make tests        # or: python3 -m pytest tests -v
```

Covers data loading, quality flags, coverage, metric computation, episode criteria and the clock-lock index (182 tests).

---

## Citation

If you use this code or the open dataset, please cite:

```
Deside G, Syed Z, Symul L, Lysy P. Short-Lived Hyperglycaemia Contributes to Continuous Glucose Monitoring Metrics Used for Therapeutic Decision-Making. The Lancet Digital Health. Code and data: Zenodo, doi:10.5281/zenodo.23102518
```


---

## Authors

Guillaume Deside, Zain Syed, Laura Symul, Philippe Lysy.
Contact: guillaume.deside28@gmail.com

---

## License

Code: **MIT** (see [`LICENSE`](LICENSE)).
Open anonymised dataset: **CC BY 4.0** (see `Results/Open_Data/DATACARD.md`).
