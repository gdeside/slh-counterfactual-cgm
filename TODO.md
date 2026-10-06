# To-do before the v1.0.0 release (ordered)

Only open items are listed (finished work is summarised under **Done** at the end). Everything below gets done; the
numbering is the order to do it in. Steps 1–10 modify the repository, so they come **before** the release: publishing the
GitHub release triggers the Zenodo snapshot, which cannot be edited afterwards.

> ⚠️ State: the clean history lives on the local branch `release-clean` (single commit, plus the commits made on top).
> The remote still has the old history, the tag `v1.0.0` (commit `905a5e8`) and PR #53. **No GitHub release is published
> yet**, so Zenodo has not archived anything.

## A. Safety first (nothing public until these pass)

1. [ ] **Get the clean history onto GitHub — a force-push is NOT enough.** Audit result: the repository is currently
       **private**, 0 forks, 53 PRs. GitHub keeps every PR head as a hidden ref (`refs/pull/N/head`, plus branches such as
       `<branch>`), and a force-push does **not** remove them: the old notebooks, plots, IDs and real
       dates stay reachable by commit hash and become public when the repo is flipped to public. Choose one:
       **(a, recommended)** publish from a **new repository** (create it empty, push `release-clean` as `master`; keep the old
       one private or delete it after checking the backup bundle) — then update the repo URL in README / `CITATION.cff` /
       badges; **(b)** delete and recreate the repository under the same name (also drops the PR refs; loses issues/PRs);
       **(c)** force-push + ask GitHub Support to purge the PR refs (slow, not guaranteed).
       Full old history is saved outside the repo in `../counterfactual-cgm-slh_FULL_HISTORY_BACKUP.bundle`.
2. [ ] **Ethics / DPO check**: (a) releasing the open dataset and the figures is covered by the approval;
       (b) the study codes `A###` (code, README, figures) cannot be linked back to patients;
       (c) the **realistic sample dataset** built from real data (`make sample_data_real`: day-mixing across ≥ 5 patients, noise,
       relative time) can be published as an illustration dataset.

## B. Clean the repository content

3. [ ] **Remove non-code leftovers**: untracked `TLDE_submission_review.md`, `talk_15min_plan.md` (delete the local
       `analyzer.py` are git-ignored (don't re-add).
4. [ ] **Finish `scripts/supplementary_code/`**: it is now empty (its last file was deleted) — remove the folder, the empty
       `scripts/poster_figures/`, and the `supplementary_code` line in the README structure tree.
5. [ ] **Re-check the README figure/table map** against the final manuscript numbering (two "Fig. 5" scripts) and the
       script / Makefile references after steps 3–8.
6. [ ] **Tidy the loose scripts at the root of `scripts/`** (`cgm_agp_recurrence.py`,
       `generate_episode_funnel_single_patient.py`, `generate_overview_figure.py`, `generate_zoom_panels.py`,
       `patient_comparison.py`): move into a sub-package (e.g. `scripts/slh_illustrations/`) or `paper_figures/`; update the
       Makefile targets, the README map and cross-imports (`generate_zoom_panels` is imported by
       `scripts/presentation_figures/episode_classification_zoom_panels.py`).
8. [ ] **Review `scripts/statistics/` and `scripts/patient_selection/`**: keep only what feeds a figure, table or statement
       of the paper (`slh_baseline_metrics_multivariate_exploration.py`, `slh_combined_cgm_phenotype_pca.py`,
       `slh_metrics_visualization_from_files.py`, `slh_patient_characteristics_exploration.py` look exploratory); confirm
       the `compelling_pairs` / `patient_similarity` Makefile targets are needed (they select real patients).
9. [ ] **Final consistency checks**: `git grep "/Users/"` empty; `git grep -n "presentation/"` only intended hits; every
       Makefile target points to an existing module; `make help` clean; `make tests` green; no unused imports.

9c. [ ] **README fixes**: contact e-mail differs between README (`guillaume.deside@uclouvain.be`) and `CITATION.cff`
        (`guillaume.deside28@gmail.com`) — keep one; the structure tree still lists `scripts/supplementary_code/` (folder
        no longer exists) and omits `.github/`; mention that `scripts/statistics/` has 4 exploratory scripts with no Makefile target
        (or delete them, step 8).
9d. [ ] **Lint pass**: pyflakes reports 157 unused imports and one duplicate definition
        (`scripts/preprocessing/cgm_filtering.py:28`, `FIG_FILTERING_DIR`); remove them (no undefined names, every script
        imports cleanly).
9e. [ ] **Cosmetic leftovers**: comments mentioning "OneDrive" (`src/config/paths.py` lines 51–56,
        `scripts/statistics/slh_patient_characteristics_exploration.py:29`) — reword to "the clinical table in `CGM_DATA_ROOT`".
        prefer them out of the public history). Authorship uses your GitHub noreply address — no personal e-mail leaks.

9g. [ ] **Optional — switch to the realistic sample data** (only after DPO approval, item 2c): run
        `CGM_DATA_ROOT=<real root> make sample_data_real`, check the 4 fake patients (`make pipeline` on them: SLH burden should
        decrease from A001 to A004 — it did in the test run: 38 / 27 / 15 / 10 SLH), then replace `sample_data` (Makefile
        `sample_data` target or a `--from-real` option), re-execute notebooks 01–09 on it, regenerate `sample_data.zip`,
        and update the README (sample-data description, "synthetic" wording, notebook banners). Never commit the real inputs.

## C. Validate

10. [ ] **Regenerate every manuscript figure/table from the open dataset** (values differ slightly because of the noise;
        stated in the README) and run `make sample_data_run` on the cleaned tree. (Figure 5 / Table 1 / determinants need the
        clinical table, which is not in the open dataset.)

## D. Release

11. [ ] **GitHub repository settings** (before going public): description, topics, enable the Zenodo ↔ GitHub integration for this
        repository (Zenodo → Account → GitHub), keep Actions enabled. **Final commit on `master`**, run `make sample_data` to produce `sample_data.zip`, and **(re)create the tag**
        `v1.0.0` on that commit (`git tag -f -a v1.0.0 -m "v1.0.0" && git push -f origin v1.0.0`) — only while no release
        exists. Check that the first **GitHub Actions** run (`tests.yml`) is green; the README "Tests" badge only works after it.
12. [ ] **Zenodo record**:
        - archive the open dataset (`make open_data`, output in `Results/Open_Data/`) and put its DOI in the README,
          `CITATION.cff` and the manuscript's data-sharing statement;
        - check whether `10.5281/zenodo.23102518` is the code record, the dataset record, or a concept DOI, and update
          README / `CITATION.cff` accordingly;
        - **licence**: the record is CC BY 4.0, but the code is MIT (`LICENSE`, `CITATION.cff`). If the record contains the
          code, add **MIT as a second licence** next to CC BY 4.0 (Zenodo accepts several); if it contains only the data,
          keep CC BY 4.0. Keep the README sentence "Code: MIT · open dataset: CC BY 4.0".
13. [ ] **Publish the GitHub release** `v1.0.0` with `sample_data.zip` attached
        (`gh release create v1.0.0 sample_data.zip --title "v1.0.0" --notes "First public release"`) → triggers the Zenodo
        snapshot. Check it appears on Zenodo and that the version DOI matches `CITATION.cff`.
14. [ ] Link `sample_data.zip` from the README Quick start.

## E. After release (optional, can be a v1.0.1)

15. [ ] Add ORCIDs and the missing affiliations (Syed, Lysy) to `CITATION.cff` / README.
16. [ ] Add a `docs/` page describing each output file's columns, and a short note on the config layout
        (`src/config/` is the single source of thresholds).
17. [ ] Add the journal year / volume / pages to the citation once published.

## Switching to the clean history (step 1, option (c) only — destructive; options (a)/(b) are preferable)

```bash
git checkout release-clean
git branch -M master                       # the old local master is kept in the bundle
git push --force origin master             # replaces the remote history
git push origin :refs/tags/v1.0.0          # delete the old remote tag (re-created in step 11)
git push origin --delete cleanup-for-publication
```
This does NOT delete `refs/pull/*`; ask GitHub Support to purge them, or prefer a new repository (step 1 (a)).
Steps 3–10 can be done as normal commits on `release-clean` before or after this switch.

## Done

- **Fonts on Linux**: `font.family` is now a sans-serif fallback chain (Arial → Helvetica → Liberation Sans → Nimbus Sans → Arimo → DejaVu Sans) in `src/visualization/style.py`, the talk figure and the plotly scripts; `tests/test_style_fonts.py` simulates a machine without Arial.
- **Dependencies**: unused `statsmodels` removed from `requirements.txt`; the optional-UMAP warning in `patient_similarity_analysis.py` now says `pip install umap-learn`.
- **Naming and headers**: both "Figure 5" scripts disambiguated (`generate_figure_5_counterfactual_distribution`, determinants → `generate_figure_6_slh_determinants`, `make fig6`); `generate_figure_3` → `generate_figure_3_patient_comparison`; `generate_supplementary_figure_1` → `..._1_data_quality`; all 66 scripts now start with their path and carry a module command and an output location; obsolete folder names (`Data_Preprocessed_45d`, …) fixed; naming conventions documented in the README.
- `scripts/fig_slh_schema.py` → `scripts/paper_figures/generate_supplementary_figure_3_slh_detection_schema.py` (`make suppfig3`, output in `Figures/Supplementary/`, part of `make paper`).
- **Single-file example**: `scripts/examples/detect_and_counterfactual.py` (`make example`, README section, tests; 220 tests pass) and notebook `09_detect_and_counterfactual_on_your_file.ipynb` (executed on a synthetic trace, "Illustration only" banner).
- **History and safety**: orphan branch `release-clean` created (largest blob 1.4 MB, no old notebooks / GIF / presentation
  outputs); full old history backed up as a git bundle.
- **Notebooks**: the 8 notebooks re-executed on synthetic sample data (fake patients A001–A004, time relative to an
  arbitrary t = 0 = 2000-01-01), each with an "Illustration only" banner; no real dates or paths remain.
- **Paths**: `src/config/paths.py` uses `CGM_DATA_ROOT` (default `./data`); no hard-coded `/Users/…` left.
- **Patient IDs**: no hard-coded `A159/A174/A224` defaults; `src/utils/patient_selection.py` (automatic most/fewest-SLH
  choice, `--patient_id`, `CGM_PATIENT_HIGH/LOW`), documented in the README.
- **Presentation figures**: `scripts/presentation_figures/` published (19 scripts, code only), headers/paths fixed, same
  episode categories as Figure 2, `make presentation_figures` runs on the sample data, outputs git-ignored; duplicates kept
  locally in the git-ignored `presentation/_dropped/`.
- **Makefile**: `clock_lock` removed; targets for every figure/table, `pipeline`, `open_data_to_raw`, `presentation_figures`.
- **Tests and CI**: 216 tests (incl. end-to-end synthetic trace, unit conversion, patient selection, open data);
  `.github/workflows/tests.yml`.
- **README**: cleaned for Zenodo (title, DOI, citation, authors), data-usage section, patient-selection section, badges
  (tests, Python, licences, DOI); `CITATION.cff` added.
- **Other**: stale `src/config.py` removed; `make install` works; `requirements.txt` tested on a clean venv; Python ≥ 3.11
  documented (tested on 3.13); sample dataset tested end to end; open-data → raw converter; no csv/xlsx/parquet/pkl tracked.
