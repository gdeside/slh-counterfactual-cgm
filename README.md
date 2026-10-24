# Counterfactual CGM signals for short-lived hyperglycaemia (SLH): code

[![Tests](https://github.com/gdeside/slh-counterfactual-cgm/actions/workflows/tests.yml/badge.svg)](https://github.com/gdeside/slh-counterfactual-cgm/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg)](#installation)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-green.svg)](LICENSE)
[![Example data: BrisT1D-Open, CC BY 4.0](https://img.shields.io/badge/example%20data-BrisT1D--Open%20(CC%20BY%204.0)-lightgrey.svg)](#data-availability)

Code, analysis pipeline and figure scripts for the manuscript:

> *Short-Lived Hyperglycaemia Contributes to Continuous Glucose Monitoring Metrics Used for Therapeutic Decision-Making*
> Deside G, Syed Z, Symul L, Lysy P. *The Lancet Diabetes & Endocrinology* (submitted; preprint on medRxiv).
>
> Code archived on Zenodo (code only, **no patient data**): DOI to come with the next release.

---

## Table of contents

- [Overview](#overview)
- [Method in brief](#method-in-brief)
- [Three ways to use the code](#three-ways-to-use-the-code)
- [Manuscript figure and table map](#manuscript-figure-and-table-map)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Example with the BrisT1D-Open data](#example-with-the-brist1d-open-data)
- [Run detection and counterfactual directly on your files (no pipeline)](#run-detection-and-counterfactual-directly-on-your-files-no-pipeline)
- [Data availability](#data-availability)
- [Full pipeline on your own data](#full-pipeline-on-your-own-data)
- [How to reproduce](#how-to-reproduce)
- [Glucose units](#glucose-units-mgdl-and-mmoll)
- [Makefile reference](#makefile-reference)
- [Tests](#tests)
- [Citation](#citation)
- [Authors and contact](#authors-and-contact)
- [License](#license)

---

## Overview

This repository implements a **counterfactual signal reconstruction** approach for quantifying the metabolic
weight of short-lasting hyperglycaemia (SLH) episodes in paediatric continuous glucose monitoring (CGM) data:
what would the glucose signal, and the consensus CGM metrics derived from it, look like if the SLH episodes had not occurred?

**What the code does**

1. Preprocesses raw CGM exports into 45-day windows (≥ 70 % coverage, 5-min sampling) and computes standard CGM metrics.
2. Filters the cohort (median-glucose criterion) and detects hyperglycaemic episodes (> 180 mg/dL).
3. Classifies each episode with a six-criterion scheme (C1–C6) into four categories, as in Figure 2: **SLH** (short, isolated,
   complete data, daytime), **Not isolated**, **Night** (would be SLH but starts at night) and **LLH** (long-lasting).
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
| Criteria | C1 duration 10–60 min · C2 ≥ 75 % data completeness in a 60-min context window · C3 daytime onset · C4/C5 no hyperglycaemia before/after · C6 crossing condition | `src/episodes/criteria.py` |
| Episode categories (Figure 2) | episodes failing C2 are set aside (cannot be judged); then **C1 fails → LLH** · **C4/C5/C6 not all met → Not isolated** · **C3 fails → Night** (would be SLH, but onset at night) · **all met → SLH** | `scripts/paper_figures/generate_figure_2_classification_summary.py` |
| Counterfactual | linear interpolation over the SLH zone expanded by δ (default **30 min**; tested 0–90 min) | `src/counterfactual/build_counterfactual.py` |

All thresholds live in `src/config/` (`constants.py`, `episode_config`, `counterfactual_config`).

---

## Three ways to use the code

| | **A. Full pipeline** (the paper) | **B. One file** | **C. A folder of files** |
|---|---|---|---|
| What it does | raw files → 45-day windows → metrics → cohort filter → episodes → counterfactual + sensitivity → manuscript figures and tables | episode detection / classification, counterfactual and effect on the metrics | the same, for every file of a folder (a whole cohort) |
| Input | `Data/Raw/<patient_id>.xlsx` under `CGM_DATA_ROOT` | any CSV / TSV / Excel file of one patient | a folder with one such file per patient (file name = patient ID) |
| Preprocessing, quality flags, 45-day window, filtering | yes | **no** - give a clean trace | **no** - give clean traces |
| Command | `make pipeline`, then `make paper` | `make example INPUT=my_patient.csv` | `make example INPUT_DIR=my_folder` |
| Output | `Data/`, `Results/`, `Figures/` under `CGM_DATA_ROOT` | `example_output/` (episodes, counterfactual, summary, figure) | `example_output/`: per-patient folders + `cohort_summary.csv` + `cohort_episodes.csv` |
| Reproduces the manuscript | **yes**, with the study data (available on request, see [Data availability](#data-availability)) | no | no |
| Details | [Full pipeline on your own data](#full-pipeline-on-your-own-data), [How to reproduce](#how-to-reproduce) | [B and C below](#run-detection-and-counterfactual-directly-on-your-files-no-pipeline) | [B and C below](#run-detection-and-counterfactual-directly-on-your-files-no-pipeline) |

Which one? **A** to reproduce or re-run the study; **B** to look at one recording; **C** to analyse a whole set of recordings
quickly, without organising them in the `Data/Raw/…` layout or filtering the cohort. Counts obtained with B or C differ from
the pipeline's for the same patients, because the pipeline first extracts a 45-day window with ≥ 70 % coverage and applies the
cohort filter.

**A in short** - put the files in `<CGM_DATA_ROOT>/Data/Raw/`, then:

```bash
export CGM_DATA_ROOT=/path/to/your/data
make pipeline        # preprocessing → metrics → filtering → episode detection (+ sensitivity analysis)
make paper           # Figures 2–4, Suppl. Fig. 1 and 3
make paper_clinical  # Table 1, Figures 5–6, phenotype figure (need the clinical table)

# illustrations for a patient of your choice (default: the patient with the most SLH episodes)
make suppfig3 lininterp_45days PATIENT_ID=A224
CGM_PATIENT_HIGH=A174 CGM_PATIENT_LOW=A224 make fig3 fig4     # two patients: high vs low SLH frequency
```

No data at hand? Use the public [BrisT1D-Open example data](#example-with-the-brist1d-open-data): they are the only
data this repository points to, and they are not the study data.

---

## Manuscript figure and table map

| Manuscript item | Script | Notes |
|---|---|---|
| Table 1 - Cohort characteristics | `scripts/paper_figures/generate_table1.py` | `make table1`; needs `pysummaries` and the clinical table |
| Fig. 2 - Episode classification summary (SLH / Not isolated / Night / LLH) | `scripts/paper_figures/generate_figure_2_classification_summary.py` | `make fig2` |
| Fig. 3 - Two patients, similar metrics, different SLH frequency (high vs low SLH) | `scripts/paper_figures/generate_figure_3_patient_comparison.py` | `make fig3` |
| Fig. 4 - Counterfactual construction and sensitivity | `scripts/paper_figures/generate_figure_4_counterfactual_construction.py` | `make fig4`; needs `make sensitivity_analysis` |
| Fig. 5 - Distribution of ΔTIR / Δmedian / max ΔAGP, with the correlation matrix | `scripts/paper_figures/generate_figure_5_counterfactual_distribution.py` | `make fig5`; needs the clinical table |
| Fig. 6 - Determinants of SLH frequency | `scripts/paper_figures/generate_figure_6_slh_determinants.py` | `make fig6`; needs the clinical table |
| Suppl. Fig. 1 - Data quality and cohort selection | `scripts/paper_figures/generate_supplementary_figure_1_data_quality.py` | `make suppfig1` |
| Suppl. Fig. 3 - SLH detection schema | `scripts/paper_figures/generate_supplementary_figure_3_slh_detection_schema.py` | `make suppfig3` |
| Suppl. Fig. - Cohort flowchart | `scripts/paper_figures/generate_cohort_flowchart.py` | `make cohort_flowchart`; needs `rsvg-convert` (`brew install librsvg`) |
| Suppl. Fig. - Phenotype determinants | `scripts/paper_figures/generate_supplementary_figure_phenotype_determinants.py` | `make suppfig_phenotype`; needs the clinical table |

Every manuscript figure is saved in its **own folder** under `Figures/Manuscript_Figures/` (supplementary ones under
`Figures/Supplementary/`): the whole figure as `<name>.{pdf,svg}` and each panel on its own as `<name>_panel_A.{pdf,svg}`,
`…_panel_B…`, etc. (cropped from the same rendering, so panels are identical to what the combined figure shows). Figure 6
keeps its separately rendered panels in `Figure6_Determinants/`. The cohort flowchart and Table 1 are not matplotlib figures.

"Clinical table" = `Data/Final cohort SLH.csv` (study data, not in this repository; `make paper` skips these items, `make paper_clinical` runs them).

Run any script as a module from the repository root, e.g.
`python3 -m scripts.paper_figures.generate_figure_3_patient_comparison`.

---

## Repository structure

```
slh-counterfactual-cgm/
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
│   ├── preprocessing/            #   raw → windows → metrics → filtered cohort; unit conversion
│   ├── episodes_detection/       #   episode detection + classification; single-patient classification funnel (`make slh_funnel`)
│   ├── counterfactual/           #   counterfactual figures + sensitivity analysis over δ
│   ├── statistics/               #   descriptive stats, quartiles, scatter plots; exploratory scripts (below)
│   ├── patient_selection/        #   participant pairs / similarity map illustrating SLH beyond standard CGM metrics
│   ├── paper_figures/            #   manuscript figures and tables
│   ├── examples/                 #   detection + counterfactual on one file or a folder (`make example`)
│   ├── presentation_figures/     #   talk figures, synthetic-data friendly (`make presentation_figures`; not part of the paper)
│   └── *.py                      #   single-patient illustrations at the top level (overview, zoom panels, AGP recurrence, patient comparison)
│
├── example_data/BrisT1D/raw/     # where to put the BrisT1D-Open files (NOT in the repository; contents git-ignored)
├── notebooks/                    # 9 example notebooks (01_cgm_metrics … 09_detect_and_counterfactual_on_your_file)
├── tests/                        # pytest suite (238 tests; no patient data needed) - `make tests`
├── docs/UNITS.md                 # mg/dL ↔ mmol/L handling
├── .github/workflows/tests.yml   # CI: pytest + example smoke test on every push
├── Makefile                      # all execution commands - `make help`
├── requirements.txt              # pinned dependencies (Python ≥ 3.11)
├── TODO.md                       # pre-release checklist (maintainers; removed from the archived version)
├── CITATION.cff                  # how to cite (Zenodo DOI, code only)
└── LICENSE                       # MIT
```

`scripts/statistics/` also holds four exploratory scripts without a `make` target, run as modules when needed:
`slh_baseline_metrics_multivariate_exploration` (CGM metrics vs n_SLH), `slh_combined_cgm_phenotype_pca` (combined CGM + phenotype PCA),
`slh_metrics_visualization_from_files` (interactive Plotly figures) and `slh_patient_characteristics_exploration` (phenotype vs n_SLH;
its functions are reused by the supplementary phenotype figure). The last two need the clinical table or earlier pipeline outputs.

**Naming conventions.** Every script is run as `python3 -m scripts.<package>.<name>` (or through `make`) and starts with a header
giving its path, what it does, its **module command** (`Usage`) and its **output location** (`Output`).


---

## Installation

Requires **Python ≥ 3.11** (developed and tested on 3.13).

```bash
git clone https://github.com/gdeside/slh-counterfactual-cgm.git
cd slh-counterfactual-cgm
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

## Example with the BrisT1D-Open data

The example data of this repository are the open part of the **BrisT1D** dataset (young adults with type 1 diabetes, University of Bristol):

> Sam Gordon James, Miranda Armstrong, Aisling O'Kane, Harry Emerson, Zahraa Abdallah (2025): BrisT1D-Open Dataset. https://doi.org/10.5523/bris.33z5jc8fa6tob21ptrugzqog08 (CC BY 4.0)

They are **not distributed with this code** (the Zenodo archive contains code only): download them from the University of Bristol
Data Repository (link above), which also describes the files and their licence. Only the open part is needed.

**Where to put the files**

```
example_data/BrisT1D/
└── raw/          <- put the ORIGINAL BrisT1D-Open files here, as downloaded (git-ignored: the data stay on your machine)
```

> ⚠️ **The data are modified.** BrisT1D-Open is not in the format and units of this project.
> [`scripts/examples/convert_brist1d.py`](scripts/examples/convert_brist1d.py) (`make brist1d`) writes one file per participant to
> `example_data/BrisT1D/converted/Data/Raw/<participant>.xlsx` (git-ignored) with only two columns, `time` and `glycemie`:
> - **columns**: everything but the timestamp and the glucose (insulin, carbs, heart rate, steps, device ...) is dropped, as are rows without glucose;
> - **units**: mmol/L -> mg/dL (x 18.0182);
> - **time grid**: each reading is moved to the nearest 5-minute mark and readings landing on the same mark are averaged; nothing is interpolated
>   (15-minute FreeStyle Libre 2 recordings stay 15-minute).
>
> The converted files are *not* the original BrisT1D files, and any result obtained from them illustrates what the code does, not a finding about BrisT1D.

```bash
make brist1d            # convert
make brist1d_run        # convert + run the core pipeline (preprocessing -> metrics -> filtering -> episodes -> sensitivity)
CGM_DATA_ROOT=$PWD/example_data/BrisT1D/converted jupyter lab notebooks/    # notebooks 01-08 on these data
make example INPUT_DIR=example_data/BrisT1D/converted/Data/Raw              # no pipeline, a whole folder
```

---

## Run detection and counterfactual directly on your files (no pipeline)

[`scripts/examples/detect_and_counterfactual.py`](scripts/examples/detect_and_counterfactual.py) runs **episode detection and
classification**, the **counterfactual** and the **effect on the glucose metrics** (TIR, TAR, median, CV) straight from CGM
files - no preprocessing, no filtering, no `Data/Raw` layout, no `CGM_DATA_ROOT`.

**Expected files** (CSV, TSV or Excel), one patient each, with a time column and a glucose column; names are auto-detected
(`time`/`timestamp`/…, `glycemie`/`glucose`/…) or given with `--time_col` / `--glucose_col`. Glucose in mg/dL (`--unit mmol/L`
converts). Give clean, time-sorted traces from a 5-minute CGM.

### B. One file

```bash
make example                                    # synthetic demo trace, runs out of the box
make example INPUT=my_patient.csv               # your file, glucose in mg/dL
make example INPUT=my_patient.csv UNIT=mmol/L   # glucose in mmol/L

# all options
python3 -m scripts.examples.detect_and_counterfactual --input my_patient.csv \
        [--unit mmol/L] [--time_col Time] [--glucose_col Glucose] [--offset 30] [--out_dir example_output]
```

Outputs in `example_output/` (git-ignored): `episodes.csv` (every episode with C1–C6 and its category - SLH / Not isolated /
Night / LLH), `counterfactual.csv` (observed G(t) and counterfactual C(t)), `summary.txt` (counts, ΔTIR, ΔTAR, Δmedian) and
`counterfactual.png`.

### C. A folder of files (a whole cohort)

Put one file per patient in a folder - any folder, no required sub-structure. The **file name is the patient ID**
(`P001.csv` → patient `P001`), and CSV, TSV and Excel files can be mixed.

```bash
make example INPUT_DIR=my_folder                # every file of the folder
make example INPUT_DIR=my_folder UNIT=mmol/L

python3 -m scripts.examples.detect_and_counterfactual --input_dir my_folder/ \
        [--unit mmol/L] [--offset 30] [--out_dir example_output] [--no_plots]
```

Outputs in `example_output/`:

| File | Content |
|---|---|
| `<patient_id>/episodes.csv`, `counterfactual.csv`, `counterfactual.png` | the single-file outputs, for each patient (`--no_plots` skips the figures - faster for large cohorts) |
| `cohort_episodes.csv` | all episodes of all patients, `patient_id` first |
| `cohort_summary.csv` | one row per patient: readings, days, number of episodes per category, TIR 70–180 / TAR > 180 / median glucose observed vs counterfactual and the differences Δ = counterfactual − observed |
| `skipped_files.csv` | files that could not be processed and why (unreadable, columns not found, fewer than 100 readings) - one bad file never stops the others |

In both modes the zone-expansion offset δ is `--offset` (default 30 min; the manuscript tests 0–90 min). The thresholds are in
`src/episodes/episode_config.py`.

The notebook [`notebooks/09_detect_and_counterfactual_on_your_file.ipynb`](notebooks/09_detect_and_counterfactual_on_your_file.ipynb)
walks through the same steps for one file (loading, episodes and criteria, counterfactual, effect on the metrics, sensitivity
to δ). The script is also a short template for calling the library from your own code:

```python
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import classify_episodes
from src.counterfactual.build_counterfactual import build_full_counterfactual_signal

episodes = detect_hyperglycemia_episodes(df)            # df: columns "time" (datetime) and "glycemie" (mg/dL)
table = classify_episodes(df, episodes)                 # C1–C6; "classification" is SLH or not-SLH
slh = table.loc[table["classification"] == "SLH", ["onset_time", "offset_time"]].to_dict("records")
counterfactual = build_full_counterfactual_signal(df, slh, offset_min=30, timestep_min=5.0)
```

The `classification` column of `classify_episodes` is binary (SLH, or not). The four categories of Figure 2 (SLH / Not isolated /
Night / LLH) are obtained from the C1–C6 columns, as done by `categorise()` in the example script; episodes failing C2 are set
aside first.

**Not done here** (done by the full pipeline A): quality flags, extraction of the 45-day window (≥ 70 % coverage) and cohort
filtering. For that reason the episode counts differ from those of the pipeline for the same patients.

---

## Data availability

- **This repository and its Zenodo archive contain code only. No patient data are shared.**
- **Study data.** The CGM data used in the manuscript are patient-level clinical data. They are available on request from the
  corresponding author of the paper, **Philippe Lysy**.
- **Example data (the only data this repository points to).** The examples use the open part of the public **BrisT1D** dataset
  (Sam Gordon James, Miranda Armstrong, Aisling O'Kane, Harry Emerson, Zahraa Abdallah (2025): BrisT1D-Open Dataset, [10.5523/bris.33z5jc8fa6tob21ptrugzqog08](https://doi.org/10.5523/bris.33z5jc8fa6tob21ptrugzqog08),
  CC BY 4.0), which you download yourself. It is converted to the format of this project (see
  [Example](#example-with-the-brist1d-open-data)); numbers obtained from it are not manuscript results.

---

## Full pipeline on your own data

This section is about the **full pipeline (A)**; it needs data in the layout below, for example the study data (on request) or
your own recordings. Point the code at a working folder with `CGM_DATA_ROOT` (default: `./data`); inputs are read from, and outputs
written to, sub-folders of it. To run only detection and the counterfactual on files, without this layout, see
[B and C](#run-detection-and-counterfactual-directly-on-your-files-no-pipeline).

### Your CGM data

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
(original study codes, `A001`… in the synthetic sample data, file names in the BrisT1D example), so the code picks patients itself
unless you tell it which ones to use.

**Default (automatic).** From the filtered cohort, the script takes the patient with the **most** SLH episodes
and, for two-patient figures, also the patient with the **fewest** (at least one). The chosen IDs are printed
(`[auto] patient with most SLH episodes: …`).

**Choosing your own patients - with `make`**

One patient: add `PATIENT_ID=<id>` to the command (any number of targets can share it):

```bash
make suppfig3 PATIENT_ID=A224                                   # Supplementary Figure 3 (detection schema)
make lininterp_45days PATIENT_ID=A224                           # 45-day observed vs counterfactual
make lininterp_45days lininterp_stepbystep suppfig3 PATIENT_ID=A224    # several figures, same patient
make lininterp_45days                                           # no PATIENT_ID: patient with the most SLH episodes
```

`PATIENT_ID=` works with: `suppfig3`, `lininterp_45days`, `lininterp_stepbystep`, `lininterp_offset_comp`, `lines_sensitivity`,
`agp_sensitivity`, `timeseries_sensitivity`, `slh_funnel`, `slh_overview`, `slh_zoom_panels`.

Two patients (Figures 3–5, `patient_comparison`: high vs low SLH frequency): use the environment variables `CGM_PATIENT_HIGH`
and `CGM_PATIENT_LOW`, which are the patient with many SLH episodes and the one with few. The `NAME=value` part placed before a
command sets them for that command only, so they work with `make` and with `python3 -m`:

```bash
# Figures 3, 4 and 5 for two chosen patients
CGM_PATIENT_HIGH=A174 CGM_PATIENT_LOW=A224 make fig3 fig4 fig5

# the same, for every figure of the shell session
export CGM_PATIENT_HIGH=A174
export CGM_PATIENT_LOW=A224
make fig3 fig4 fig5 patient_comparison
```

**Without `make`** - every one of these scripts accepts `--patient_id <ID>` (and, where relevant, `--offset`, `--offsets`,
`--n_episodes`, `--out_dir`; run with `-h`):

```bash
python3 -m scripts.counterfactual.lininterp_45days --patient_id A224 --offset 45
python3 -m scripts.paper_figures.generate_supplementary_figure_3_slh_detection_schema --patient_id A224
```

`A174` and `A224` are the two participants shown in the manuscript (high vs low SLH frequency); they exist only in the
study data. If an ID is not found in the data, the script prints a warning and falls back to the automatic choice,
so the same command also runs on other data (it then shows other patients). Selection logic:
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
| `Results/` | tables and sensitivity-analysis outputs |
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
make episode_detection       # episode detection + classification (C1–C6)

# 2. Statistics
make slh_llh_cohort_descriptive_stats  # episode-category descriptive statistics
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
```

Example notebooks (`notebooks/01_…` to `09_…`: metrics, preprocessing + filtering, episode detection, counterfactual,
sensitivity analysis, median-profile difference (t_max), patient comparison, phenotype/CGM association with n_SLH) run on the converted BrisT1D-Open example data (`make brist1d_run`)
(`08_…` needs the clinical file for real results and falls back to a flagged random demo phenotype without it; `09_…` runs detection + counterfactual on a single file, with no pipeline and no `CGM_DATA_ROOT`):

```bash
make brist1d_run
CGM_DATA_ROOT=$PWD/example_data/BrisT1D/converted jupyter lab notebooks/
```

---

## Glucose units (mg/dL and mmol/L)

The pipeline computes and stores glucose in **mg/dL**; mmol/L is a presentation layer, so switching units cannot change a result.

```bash
python3 -m scripts.preprocessing.convert_data_units            # mmol/L copies of data files
CGM_GLUCOSE_UNIT="mmol/L" python3 -m scripts.paper_figures.generate_figure_3_patient_comparison
```

Figure filenames carry the unit (`..._mgdl.pdf` / `..._mmoll.pdf`). Details in [`docs/UNITS.md`](docs/UNITS.md).

---

## Makefile reference

`make help` prints the authoritative list (descriptions live next to each target in the `Makefile`).

| Group | Targets |
|---|---|
| Setup | `install`, `requirements-check` (reports missing imports, never overwrites `requirements.txt`), `check_clinical_table` |
| Data pipeline | `time_step_analysis`, `preprocessing`, `coverage_analysis`, `metrics_preprocessing`, `filtering`, `filtered_metrics_distribution`, `episode_detection`, **`pipeline`** (core chain) |
| Statistics | `slh_llh_cohort_descriptive_stats`, `quartile_analysis`, `scatter_analysis` |
| Patient selection | `compelling_pairs`, `patient_similarity` |
| SLH illustrations | `slh_funnel`, `slh_overview`, `slh_zoom_panels`, `agp_recurrence`, `patient_comparison` |
| Counterfactual | `lininterp_stepbystep`, `lininterp_offset_comp`, `lininterp_45days` |
| Sensitivity | `sensitivity_analysis`, `lines_sensitivity`, `boxplots_sensitivity`, `cdf_sensitivity`, `agp_sensitivity`, `timeseries_sensitivity`, `sensitivity_figures` |
| Manuscript | `fig2`, `fig3`, `fig4`, `suppfig1`, `suppfig3`, `cohort_flowchart`, **`paper`**; with the clinical table*: `table1`, `fig5`, `fig6`, `suppfig_phenotype`, **`paper_clinical`** |
| Single file / folder, no pipeline | `example` (`make example INPUT=file.csv` or `INPUT_DIR=folder`) |
| Talk figures | `presentation_figures` |
| Units | `convert_units` |
| Example data | `brist1d`, `brist1d_run` |
| Utility | `figures`, `tests`, `clean` |

\* needs `Data/Final cohort SLH.csv` (study data, not in this repository): these targets first run `check_clinical_table`, which stops with a
clear message when the file is missing.

---

## Tests

```bash
make tests        # or: python3 -m pytest tests -v
```

Runs in a few seconds, needs no patient data, and is executed on every push by GitHub Actions
(`.github/workflows/tests.yml`). 

---

## Citation

If you use this code, please cite the Zenodo archive (code only) and the manuscript:

```
Deside G, Syed Z, Symul L, Lysy P. Short-Lived Hyperglycaemia Contributes to Continuous Glucose Monitoring Metrics Used for Therapeutic Decision-Making. The Lancet Diabetes & Endocrinology (submitted; medRxiv). Code: Zenodo (DOI to come)
```

If you use the example data, also cite the dataset: Sam Gordon James, Miranda Armstrong, Aisling O'Kane, Harry Emerson, Zahraa Abdallah (2025): BrisT1D-Open Dataset. https://doi.org/10.5523/bris.33z5jc8fa6tob21ptrugzqog08 - and state that the data were modified by this project's converter.

---

## Authors and contact

Guillaume Deside, Zain Syed, Laura Symul, Philippe Lysy.

- **Code** (this repository, issues, questions): Guillaume Deside - guillaume.deside@uclouvain.be
- **Study data** (available on request): Philippe Lysy, corresponding author of the manuscript - philippe.lysy@uclouvain.be

---

## License

Code: **MIT** (see [`LICENSE`](LICENSE)).
The BrisT1D-Open example data are not part of this repository; they belong to their authors and are released under **CC BY 4.0**.
