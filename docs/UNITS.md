# Glucose units: mg/dL and mmol/L

The pipeline computes in **mg/dL**. That is the invariant everything else is
built on:

* every threshold in `src/config` (`THRESHOLD_HYPO`, `THRESHOLD_HYPER`,
  `H_HYPER`, `H_CONTEXT`, `MEDIAN_GLUCOSE_MIN/MAX`, `GLUCOSE_MIN/MAX`),
* every stored data file (window `.xlsx`, metrics, episodes, counterfactual
  signals, `cohort_sensitivity.csv`),
* every metric (`Mean_glycemia`, `SD`, `Median_night`, `delta_median`, …).

mmol/L is a **presentation and publication** layer on top of that. Episode
detection, filtering and the counterfactual construction never see it, so
switching units cannot change a single result.

Conversion factor (glucose molar mass 180.156 g/mol):

```
1 mmol/L = 18.0182 mg/dL          # src.utils.unit_conversion.MGDL_PER_MMOLL
```

---

## 1. Converting the data files

`scripts/preprocessing/convert_data_units.py` reads the produced data files and
writes mmol/L copies to a **separate tree**. The mg/dL originals stay untouched
and remain the pipeline's input.

```bash
# What would be converted, and which columns — writes nothing
python3 -m scripts.preprocessing.convert_data_units --dry-run

# Write mmol/L copies under <BASE_PATH>/Converted_mmoll/
python3 -m scripts.preprocessing.convert_data_units

# Selected products, chosen destination
python3 -m scripts.preprocessing.convert_data_units \
    --targets metrics signals --output-dir ~/Desktop/mmol

# Convert a mmol/L tree back to mg/dL
python3 -m scripts.preprocessing.convert_data_units --to mg/dL \
    --source-dir <BASE_PATH>/Converted_mmoll --output-dir /tmp/back
```

Targets: `windows`, `metrics`, `distribution`, `episodes`, `signals`,
`sensitivity`, `cohort`, `opendata` (all by default).

The output root gets a `CONVERSION_REPORT.md` listing, per target, exactly
which columns were converted.

### What is and is not converted

Column classification lives in
`src.utils.unit_conversion.is_glucose_valued_column` and is applied to numeric
columns only.

| Converted | Left alone |
| --- | --- |
| `glycemie`, `glucose_original`, `glucose_ctf_delta_*` | `CV`, `orig_cv`, … (unitless / %) |
| `Mean_glycemia`, `Median_glycemia`, `Minimum_/Maximum_glycemia` | `TIR*`, `TAR*`, `TBR*` (%) |
| `SD`, `orig_sd`, `ctf_sd`, `delta_sd` | `CGM_days_measured`, `n_points` |
| `Median_morning/noon/afternoon/evening/night` | `duration_min`, `gap_before_min`, `dominant_step_min` |
| `orig_mean`, `ctf_mean`, `delta_mean`, `*_median`, `daytime_*` variants | `coverage_pct`, timestamps, flags, `HbA1c` / `IDAA1c` / `GTAA1c` |
| `max_median_diff_abs`, `max_median_diff_signed` | episode files (no glucose column — copied verbatim) |

Differences (`delta_*`, `max_median_diff_*`) convert with the same factor as
concentrations, since the conversion is purely multiplicative.

### Notes

* **Never point `BASE_PATH` at a converted tree.** The pipeline's thresholds
  are mg/dL; feeding it mmol/L files would silently classify everything as
  hypoglycaemia.
* `--in-place` exists but overwrites the originals and asks for confirmation.
  The default (a mirrored copy) is the intended workflow.
* Values are rounded to 2 decimals by default (`--decimals`, `-1` disables);
  at that precision the round-trip mg/dL → mmol/L → mg/dL is exact for
  integer CGM readings.
* The `cohort` target round-trips a `;`-separated, decimal-comma file through
  pandas, so untouched integers may be re-serialised as `7,0` instead of `7`.
  Values are unchanged.
* For the open-data release, prefer regenerating rather than converting, so
  that `DATACARD.md` matches:

  ```bash
  python3 -m scripts.preprocessing.generate_open_data --unit mmol/L
  ```

  Noise and clipping are still applied in mg/dL; only the published column is
  converted (and renamed `glucose_mmoll`), so both releases describe the same
  perturbed signal.

---

## 2. Plotting in mmol/L

### Output filenames

`save_figure` appends the display unit to the filename of any figure that
shows glucose, so the two versions coexist instead of overwriting each other:

```
Figure4_Counterfactual_Construction_A174_vs_A224_mgdl.pdf
Figure4_Counterfactual_Construction_A174_vs_A224_mmoll.pdf
```

Producing both is just the same script run twice:

```bash
python3 -m scripts.paper_figures.generate_figure_4_counterfactual_construction
CGM_GLUCOSE_UNIT="mmol/L" \
    python3 -m scripts.paper_figures.generate_figure_4_counterfactual_construction
```

The tag is added only where it means something. `save_figure` inspects the
figure's text (`figure_shows_glucose_unit`) and tags it when the unit appears
on an axis label, colorbar, table header or threshold annotation — so
unit-free figures (coverage, funnels, sampling intervals) keep their plain
names and are not duplicated.

Overrides, per call or globally:

```python
save_figure(fig, path, unit_suffix=False)   # keep a fixed name
save_figure(fig, path, unit_suffix=True)    # tag regardless of content

import src.visualization.style as style
style.FIGURE_UNIT_SUFFIX = False            # restore plain names everywhere
```

### Switching the unit

Figures follow a process-wide **display unit**. Nothing else changes: the data
handed to matplotlib stays in mg/dL.

```bash
# Whole run in mmol/L
CGM_GLUCOSE_UNIT="mmol/L" python3 -m scripts.paper_figures.generate_figure_3
```

```python
from src.utils.unit_conversion import set_display_unit, display_unit, UNIT_MMOLL

set_display_unit("mmol/L")          # process-wide, before building figures

with display_unit(UNIT_MMOLL):      # or scoped — e.g. to emit both versions
    generate_figure_3()
```

Console logs, `DATACARD.md`, and the parameter banners printed by the pipeline
stay in mg/dL: they report canonical values, not presentation.

### The mechanism

Converting the plotted series would mean touching every `plot`,
`fill_between`, `axhline` and `axhspan` call, and any one missed would produce
a figure silently mixing units. So the data is left in mg/dL and only the
axis is relabelled: **tick positions stay mg/dL data coordinates, tick labels
are round numbers in the display unit.**

```python
from src.visualization.glucose_axis import apply_glucose_axis

-   ax.set_ylabel("Glucose (mg/dL)", fontsize=9, labelpad=4)
+   apply_glucose_axis(ax, fontsize=9, labelpad=4)
```

Call it *after* the axis limits are final — in mmol/L the ticks are derived
from them. In mg/dL the tick locator is left untouched, so existing figures
are reproduced exactly.

### The four cases

| Situation | Use |
| --- | --- |
| A y- or x-axis of glucose values | `apply_glucose_axis(ax)` / `apply_glucose_axis(ax, axis="x")` |
| A panel that already picks its own clinical ticks | `apply_glucose_axis(ax, mgdl_ticks=[70, 100, 140, 180], mgdl_ticklabels=[...])` — kept in mg/dL, recomputed in mmol/L |
| A colorbar scaled in mg/dL | `apply_glucose_colorbar(cbar, quantity="Mean glucose")` |
| A number rendered as **text** (annotation, legend, title, table cell) | `format_glucose(180)` → `"180 mg/dL"` / `"10.0 mmol/L"`; `glucose_label("Median glucose")` → `"Median glucose (mmol/L)"`; `glucose_cell(value)` for table cells |

One exception needs value conversion instead of axis relabelling: an axis that
mixes units. `plot_distribution_metrics`' panel B puts `SD` (a concentration)
next to `CV` (%) on one axis, so there the *values* are converted with
`to_display(values)` and the axis is left alone. `generate_table1.py` does the
same, for the same reason — a table has no axis.

### Where this is already wired in

* `src/visualization/` — the whole shared library: `agp_plot`,
  `slh_schema_plots`, `metrics_plots`, `filtering_plots`, `episode_plots`.
* `scripts/paper_figures/` — figures 3, 4, 5, supplementary 1, and Table 1.
* `scripts/counterfactual/` — the sensitivity visualisations and the
  `lininterp_*` construction figures.
* `scripts/generate_zoom_panels.py`, `scripts/generate_overview_figure.py`.

### Adapting anything else

Any remaining `"... (mg/dL)"` literal in a figure is a one-line change:

1. axis label → `apply_glucose_axis(ax, quantity="…", **kwargs)`;
2. a threshold or value inside a string → `format_glucose(value)`;
3. a label built from a constant → `glucose_label("Mean glucose")`;
4. a table cell → `glucose_cell(value)`.

Grep for candidates with:

```bash
grep -rn "mg/dL" src scripts --include="*.py"
```

Hits inside docstrings, comments and `print()` calls are intentional — they
describe the canonical unit.

---

## 3. API reference

`src/utils/unit_conversion.py` (also re-exported from `src.config`):

| Function | Purpose |
| --- | --- |
| `mgdl_to_mmoll(v, decimals=None)` / `mmoll_to_mgdl(v, …)` | Value conversion; scalars, arrays, Series, DataFrames |
| `normalize_unit("mmol/l")` | `"mmol/L"` — accepts any spelling |
| `convert_glucose_column(df, to_unit, column=…)` | One column of a DataFrame |
| `convert_dataframe_units(df, to_unit)` | Every glucose-valued column; returns `(df, converted_names)` |
| `convert_metric_table(df, metric_column="metric")` | Long-format table: converts glucose *rows* |
| `is_glucose_valued_column(name)` | The classification used above |
| `get_display_unit()` / `set_display_unit(u)` / `display_unit(u)` | Read / set / scope the display unit |
| `to_display(v)` / `from_display(v)` | mg/dL ↔ display unit (identity while displaying mg/dL) |
| `glucose_label(q)` / `glucose_unit_label()` / `format_glucose(v)` | Labels and formatted values |
| `glucose_ticks(vmin, vmax)` | Round ticks: positions in mg/dL, labels in the display unit |
| `unit_tag(unit=None)` | `"mgdl"` / `"mmoll"` — filename token for per-unit outputs |

`src/visualization/glucose_axis.py`:

| Function | Purpose |
| --- | --- |
| `apply_glucose_axis(ax, axis="y", …)` | Label a glucose axis; relabel its ticks in mmol/L |
| `apply_glucose_colorbar(cbar, …)` | Same for a colorbar scaled in mg/dL |
| `glucose_cell(value)` | Table-cell value in the display unit |

`src/visualization/style.py`:

| Function | Purpose |
| --- | --- |
| `save_figure(fig, path, unit_suffix=None)` | Save PDF + SVG, appending the display unit to glucose figures |
| `figure_shows_glucose_unit(fig)` | The test behind the `"auto"` suffix policy |
