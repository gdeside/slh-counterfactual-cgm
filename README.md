# Counterfactual CGM Signals for Short-Lasting Hyperglycaemia (SLH)

[![Tests](https://github.com/gdeside/counterfactual-cgm-slh/actions/workflows/tests.yml/badge.svg)](https://github.com/gdeside/counterfactual-cgm-slh/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg)](#installation)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-green.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data%20license-CC%20BY%204.0-lightgrey.svg)](#data-availability)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23102518.svg)](https://doi.org/10.5281/zenodo.23102518)

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
- [Use the two core steps on your own file](#use-the-two-core-steps-on-your-own-file)
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
6. Explores determinants of SLH frequency (CGM metrics, phenotype, HbA1c).
7. Produces all manuscript figures and tables

**Cohort (manuscript):** 245 paediatric patients (IDs A001–A245), 45-day preprocessed windows, daytime-only SLH (22:00–06:00 excluded).


---

## Method in brief

| Step | Definition | Where |
|---|---|---|
| Hyperglycaemia | glucose > 180 mg/dL (context threshold 140 mg/dL) | `src/episodes/episode_config.py` |
| SLH criteria | C1 duration 10–60 min · C2 ≥ 75 % data completeness in a 60-min context window · C3 daytime onset · C4/C5 no hyperglycaemia before/after · C6 crossing condition | `src/episodes/criteria.py` |
| Counterfactual | linear interpolation over the SLH zone expanded by δ (default **30 min**; tested 0–90 min) | `src/counterfactual/build_counterfactual.py` |

All thresholds live in `src/config/` (`constants.py`, `episode_config`, `counterfactual_config`).

---

## Manuscript figure and table map

| Manuscript item | Script | Notes |
|---|---|---|
| Suppl. Fig. 3 — SLH detection schema | `scripts/paper_figures/generate_supplementary_figure_3_slh_detection_schema.py` | `make suppfig3` |
| Fig. 2 — SLH/LLH classification summary | `scripts/paper_figures/generate_figure_2_classification_summary.py` | |
| Fig. 3 — Two patients, similar metrics, different SLH frequency (high vs low SLH) | `scripts/paper_figures/generate_figure_3_patient_comparison.py` | |
| Fig. 4 — Counterfactual construction and sensitivity | `scripts/paper_figures/generate_figure_4_counterfactual_construction.py` | needs `make sensitivity_analysis` |
| Fig. 5 — Distribution of ΔTIR / Δmedian | `scripts/paper_figures/generate_figure_5_counterfactual_distribution.py` | |
| Fig. 6 — Determinants of SLH frequency | `scripts/paper_figures/generate_figure_6_slh_determinants.py` | needs the clinical phenotype table |
| Table 1 — Cohort characteristics | `scripts/paper_figures/generate_table1.py` | needs `pysummaries` |
| Suppl. Fig. — Cohort flowchart | `scripts/paper_figures/generate_cohort_flowchart.py` | needs `rsvg-convert` (`brew install librsvg`) |
| Suppl. Fig. 1 — Data quality and cohort selection | `scripts/paper_figures/generate_supplementary_figure_1_data_quality.py` | |
| Suppl. Fig. — Phenotype determinants | `scripts/paper_figures/generate_supplementary_figure_phenotype_determinants.py` | |

Run any script as a module from the repository root, e.g.
`python3 -m scripts.paper_figures.generate_figure_3_patient_comparison`.

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
│   ├── metrics/                  #   CGM metrics (metrics.py)
│   ├── agp/                      #   ambulatory glucose profile
│   ├── visualization/            #   plotting helpers and colour palette
│   └── utils/                    #   time and unit-conversion utilities
│
├── scripts/                      # Executable analysis scripts (`python3 -m scripts.<pkg>.<name>`)
│   ├── preprocessing/            #   raw → windows → metrics → filtered cohort; open-data release; unit conversion
│   ├── episodes_detection/       #   episode detection + classification
│   ├── counterfactual/           #   counterfactual figures + sensitivity analysis over δ
│   ├── statistics/               #   descriptive stats, quartiles, scatter plots
│   ├── patient_selection/        #   participant pairs / similarity map illustrating SLH beyond standard CGM metrics
│   ├── paper_figures/            #   manuscript figures and tables
│   ├── examples/                 #   detection + counterfactual on one standalone file (`make example`)
│   ├── presentation_figures/     #   talk figures, synthetic-data friendly (`make presentation_figures`; not part of the paper)
│   └── supplementary_code/       #   exploratory / superseded analyses
│
├── notebooks/                    # 9 example notebooks (01_cgm_metrics … 09_detect_and_counterfactual_on_your_file)
├── tests/                        # pytest unit tests
├── docs/UNITS.md                 # mg/dL ↔ mmol/L handling
├── Makefile                      # all execution commands — `make help`
├── requirements.txt
└── LICENSE                       # MIT
```

**Naming conventions.** Every script is run as `python3 -m scripts.<package>.<name>` (or through `make`) and starts with a header
giving its path, what it does, its **module command** (`Usage`) and its **output location** (`Output`).


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

**Fonts.** Figures use Arial when it is installed; otherwise matplotlib falls back silently to a metric-compatible font
(Helvetica, Liberation Sans, Nimbus Sans or DejaVu Sans, which ships with matplotlib) - nothing to install on Linux, though
`sudo apt install fonts-liberation` gives the closest match to Arial.

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

### A more realistic sample dataset (maintainers, needs the real data)

`scripts/sample_data/generate_sample_data_from_real.py` (`make sample_data_real`) builds 4 **fake patients** from the *shape* of real
CGM data, for a more realistic illustration than the simulated sample: the filtered real cohort is split into 4 groups by SLH frequency
(high → low); each fake patient interleaves randomly chosen, never-reused **24-h segments from ≥ 5 different real patients**, adds
patient-level scaling (±5 %) and Gaussian noise (σ = 2 mg/dL), replaces the time by an **arbitrary relative origin** and gets a
fresh ID (A001–A004); no mapping to real patients is stored. It must be run where the real data are
(`CGM_DATA_ROOT=<real root> make sample_data_real`) and its output must **not** be committed or released without the data-protection
officer's approval. Details in the script header.

---

## Use the two core steps on your own file

To run only **episode detection / SLH classification** and the **counterfactual** on a new CGM file — no pipeline, no
cohort, no `CGM_DATA_ROOT` — use [`scripts/examples/detect_and_counterfactual.py`](scripts/examples/detect_and_counterfactual.py):

```bash
make example                                    # synthetic demo trace, runs out of the box
make example INPUT=my_patient.csv               # your file (.csv / .tsv / .xlsx), glucose in mg/dL
make example INPUT=my_patient.csv UNIT=mmol/L   # glucose in mmol/L (converted to mg/dL)

# the same, with all options
python3 -m scripts.examples.detect_and_counterfactual --input my_patient.csv \
        [--unit mmol/L] [--time_col Time] [--glucose_col Glucose] [--offset 30] [--out_dir example_output]
```

The file holds **one patient** with a time column and a glucose column (names are auto-detected: `time`/`timestamp`/…,
`glycemie`/`glucose`/…). Outputs go to `example_output/` (git-ignored): `episodes.csv` (every episode with C1–C6 and its
SLH/LLH label), `counterfactual.csv` (observed G(t) and counterfactual C(t)), `summary.txt` (counts, ΔTIR, ΔTAR, Δmedian)
and `counterfactual.png`.

The notebook [`notebooks/09_detect_and_counterfactual_on_your_file.ipynb`](notebooks/09_detect_and_counterfactual_on_your_file.ipynb)
walks through the same steps (loading, episodes and criteria, counterfactual, effect on the metrics, sensitivity to δ).
The script is also a short template for calling the library from your own code:

```python
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import classify_episodes
from src.counterfactual.build_counterfactual import build_full_counterfactual_signal

episodes = detect_hyperglycemia_episodes(df)            # df: columns "time" (datetime) and "glycemie" (mg/dL)
table = classify_episodes(df, episodes)                 # C1–C6 -> SLH / LLH
slh = table.loc[table["classification"] == "SLH", ["onset_time", "offset_time"]].to_dict("records")
counterfactual = build_full_counterfactual_signal(df, slh, offset_min=30, timestep_min=5.0)
```

Not done here (done by the full pipeline): quality flags, extraction of the 45-day window (≥ 70 % coverage) and cohort
filtering — give a clean, time-sorted trace from a 5-minute CGM. Thresholds are in `src/episodes/episode_config.py`.

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
| `glycemie` (also accepted: `glycemia`, `glucose`) | glucose in **mg/dL** (convert mmol/L × 18.0182 first) |

The pipeline expects 5-minute sampling and extracts a 45-day window with ≥ 70 % coverage per patient (patients without
one are listed in `Data/Preprocessed/patients_status.csv`). Then run `make pipeline`. The patients processed are all the `*.xlsx` files of `Data/Raw/` (the file name is the patient ID, any format);
to process only some of them: `python3 -m scripts.preprocessing.cgm_preprocessing --patients ID1 ID2` (the summary tables then cover this
subset only).

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
| One patient | `--patient_id <ID>` | `python3 -m scripts.paper_figures.generate_supplementary_figure_3_slh_detection_schema --patient_id A174` |
| Two patients (Figures 3, 4, 5, `patient_comparison`, two-patient presentation figures) | environment variables `CGM_PATIENT_HIGH` and `CGM_PATIENT_LOW` | see below |
| `lininterp_*` scripts | set `SELECTED_PID` at the top of the script (default `None` = automatic) | `SELECTED_PID = "A224"` |

`CGM_PATIENT_HIGH` is the patient with a **high** SLH frequency (many SLH episodes) and `CGM_PATIENT_LOW` the one with a
**low** frequency; they are the two patients contrasted in Figures 3–5. They are *environment variables*: the
`NAME=value` part placed before a command sets them for that command only, so they work with `make` and with `python3 -m`:

```bash
# Figures 3, 4 and 5 for two chosen patients
CGM_PATIENT_HIGH=A174 CGM_PATIENT_LOW=A224 make fig3 fig4 fig5

# the same, for every figure of the shell session
export CGM_PATIENT_HIGH=A174
export CGM_PATIENT_LOW=A224
make fig3 fig4 fig5 patient_comparison
```

`A174` and `A224` are the two participants shown in the manuscript (high vs low SLH frequency); they exist only in the
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
make paper                   # Fig. 2-4, Suppl. Fig. 1 and 3, cohort flowchart (no clinical table needed)
make paper_clinical          # Table 1, Fig. 5-6, Suppl. phenotype figure (need Data/Final cohort SLH.csv)
# individual figures: make fig2 fig3 fig4 fig5 … (see the figure map above)

# 5. Open-data release
make open_data               # σ = 2 mg/dL noise, shuffled IDs
make open_data_phase_shift   # + random monitoring-start shift
```

Example notebooks (`notebooks/01_…` to `09_…`: metrics, preprocessing + filtering, episode detection, counterfactual,
sensitivity analysis, median-profile difference (t_max), patient comparison, phenotype/CGM association with n_SLH) run on the sample data
(`08_…` needs the clinical file for real results and falls back to a flagged random demo phenotype without it; `09_…` runs detection + counterfactual on a single file, with no pipeline and no `CGM_DATA_ROOT`):

```bash
make sample_data_run
CGM_DATA_ROOT=$PWD/sample_data jupyter lab notebooks/
```

---

## Glucose units (mg/dL and mmol/L)

The pipeline computes and stores glucose in **mg/dL**; mmol/L is a presentation layer, so switching units cannot change a result.

```bash
python3 -m scripts.preprocessing.convert_data_units            # mmol/L copies of data files
CGM_GLUCOSE_UNIT="mmol/L" python3 -m scripts.paper_figures.generate_figure_3_patient_comparison
python3 -m scripts.preprocessing.generate_open_data --unit mmol/L
```

Figure filenames carry the unit (`..._mgdl.pdf` / `..._mmoll.pdf`). Details in [`docs/UNITS.md`](docs/UNITS.md).

---

## Makefile reference

`make help` prints the authoritative list (descriptions live next to each target in the `Makefile`).

| Group | Targets |
|---|---|
| Setup | `install`, `requirements-check` (reports missing imports, never overwrites `requirements.txt`), `check_clinical_table` |
| Data pipeline | `time_step_analysis`, `preprocessing`, `coverage_analysis`, `metrics_preprocessing`, `filtering`, `filtered_metrics_distribution`, `episode_detection`, **`pipeline`** (core chain) |
| Statistics | `slh_stats`, `quartile_analysis`, `scatter_analysis` |
| Patient selection | `compelling_pairs`, `patient_similarity` |
| SLH illustrations | `slh_funnel`, `slh_overview`, `slh_zoom_panels`, `agp_recurrence`, `patient_comparison` |
| Counterfactual | `lininterp_stepbystep`, `lininterp_offset_comp`, `lininterp_45days` |
| Sensitivity | `sensitivity_analysis`, `lines_`/`boxplots_`/`cdf_`/`agp_`/`timeseries_sensitivity`, `sensitivity_figures` |
| Manuscript | `fig2`, `fig3`, `fig4`, `suppfig1`, `suppfig3`, `cohort_flowchart`, **`paper`**; with the clinical table*: `table1`, `fig5`, `fig6`, `suppfig_phenotype`, **`paper_clinical`** |
| Single-file example | `example` (`make example INPUT=file.csv`) |
| Talk figures | `presentation_figures` |
| Open data / units | `open_data`, `open_data_phase_shift`, `open_data_to_raw`, `convert_units` |
| Sample dataset | `sample_data`, `sample_data_run`, `sample_data_real` (maintainers, real data) |
| Utility | `figures`, `tests`, `clean` |

\* needs `Data/Final cohort SLH.csv` (not in the open dataset): these targets first run `check_clinical_table`, which stops with a
clear message when the file is missing.

---|---|
| Setup | `install`, `requirements-check` (reports missing imports, never overwrites `requirements.txt`) |
| Data pipeline | `time_step_analysis`, `preprocessing`, `coverage_analysis`, `metrics_preprocessing`, `filtering`, `filtered_metrics_distribution`, `episode_detection`, **`pipeline`** (core chain) |
| Statistics | `slh_stats`, `quartile_analysis`, `scatter_analysis` |
| SLH figures | `slh_schema`, `slh_funnel`, `slh_overview`, `slh_zoom_panels`, `agp_recurrence`, `patient_comparison` |
| Counterfactual | `lininterp_stepbystep`, `lininterp_offset_comp`, `lininterp_45days` |
| Sensitivity | `sensitivity_analysis`, `lines_`/`boxplots_`/`cdf_`/`agp_`/`timeseries_sensitivity`, `sensitivity_figures` |
| Manuscript | `table1`, `fig2`, `fig3`, `fig4`, `fig5`, `fig6`, `suppfig1`, `suppfig_phenotype`, `cohort_flowchart`, **`paper`** |
| Open data / units | `open_data`, `open_data_phase_shift`, `convert_units` |
| Sample dataset | `sample_data`, `sample_data_run` |
| Utility | `figures`, `tests`, `clean` |

---

## Tests

```bash
make tests        # or: python3 -m pytest tests -v
```

Runs in a few seconds, needs no patient data, and is executed on every push by GitHub Actions
(`.github/workflows/tests.yml`). Covered:

| Area | Test file |
|---|---|
| Data loading, quality flags, coverage | `test_load_data.py`, `test_quality_flags.py`, `test_coverage.py` |
| CGM metrics | `test_metrics.py` |
| Episode criteria C1–C6 | `test_criteria.py` |
| **End to end** on a synthetic trace: detection → SLH/LLH classification → counterfactual → Δ metrics | `test_pipeline_end_to_end.py` |
| mg/dL ↔ mmol/L conversion | `test_unit_conversion.py` |
| Automatic / explicit patient selection | `test_patient_selection.py` |
| Open-data anonymisation, open data → raw conversion, synthetic sample data | `test_open_data.py` |
| Patient discovery (no hard-coded cohort size, any ID format, `--patients`) and 45-day window extraction | `test_patient_discovery.py`, `test_window.py` |
| Single-file example (detection + counterfactual on a CSV / Excel file) | `test_example_script.py` |
| Figure fonts fall back cleanly when Arial is missing (Linux) | `test_style_fonts.py` |

---

## Citation

If you use this code or the open dataset, please cite:

```
Deside G, Syed Z, Symul L, Lysy P. Short-Lived Hyperglycaemia Contributes to Continuous Glucose Monitoring Metrics Used for Therapeutic Decision-Making. The Lancet Digital Health. Code and data: Zenodo, doi:10.5281/zenodo.23102518
```


---

## Authors

Guillaume Deside, Zain Syed, Laura Symul, Philippe Lysy.
Contact: guillaume.deside@uclouvain.be

---

## License

Code: **MIT** (see [`LICENSE`](LICENSE)).
Open anonymised dataset: **CC BY 4.0** (see `Results/Open_Data/DATACARD.md`).
