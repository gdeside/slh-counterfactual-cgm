# To-do before the v1.0.0 release (ordered)

Everything below gets done; the numbering is the order to do it in. Steps 1–22 modify the repository, so they
come **before** the release: publishing the GitHub release triggers the Zenodo snapshot, which cannot be edited afterwards.

> ⚠️ Status checked on `master` (PR #53 merged): the tag `v1.0.0` **already exists on the remote** (commit `905a5e8`),
> but **no GitHub release is published yet**, so Zenodo has not archived anything. The tag must be moved to the final
> commit in step 23 (`git tag -f` + force-push of the tag) — only safe as long as no release exists.

## A. Safety first (nothing public until these pass)

1. [x] **Purge patient-derived outputs from the history** — done locally: orphan branch `release-clean` (single commit,
       largest blob 1.4 MB, no old notebooks / GIF / presentation outputs). Full old history saved outside the repo in
       `../counterfactual-cgm-slh_FULL_HISTORY_BACKUP.bundle`. **Still to do:** replace the remote history (force-push
       `release-clean` as `master`, delete remote tag `v1.0.0`, close/ignore PR #53) — see "Switching the remote" below.
       Anything already pushed (old `master`, PR #53 branch `cleanup-for-publication`) still exposes the old history
       until it is overwritten/deleted; if the repo was ever public or forked, treat those blobs as leaked.
2. [x] **Notebook outputs replaced by synthetic ones** — the 8 notebooks were re-executed on the synthetic sample data
       (fake patients A001–A004, time relative to an arbitrary t = 0 = 2000-01-01), each with an "Illustration only"
       banner; no real dates, local paths or patient data remain in them. `make tests`: 182 passed.
3. [x] **Real-patient presentation outputs** (old `scripts/presentation_figures/output/`, GIF, PDFs) removed from the clean branch history; see B2 for the rest.
4. [ ] **Ethics / DPO check** that releasing the open dataset (and the figures) is covered by the approval.

## B. Clean the repository content

5. [x] **Remove hard-coded local paths** — `src/config/paths.py` now uses `CGM_DATA_ROOT` (default `./data`),
       `regression_analysis_poisson.py` deleted, notebooks re-executed. `git grep "/Users/"` is empty (re-run at the end).
6. [ ] **Remove non-code artefacts** — `TLDE_submission_review.md`, `talk_15min_plan.md` (untracked leftovers: delete the
       local files); `env/`, `presentation/`, `analyzer.py` are git-ignored (don't re-add). Also delete local junk:
7. [ ] **Decide the fate of `scripts/supplementary_code/`** (1 file left: `local_slh_analysis_cohort.py`) and delete the
       empty `scripts/poster_figures/`: keep only what supports a figure, table or statement of the paper, say so in the README.
8. [x] **Makefile** — `clock_lock` target removed; every target points to an existing module (re-check after steps 7, 9–13).
9. [ ] **Re-check the README figure/table map** against the final manuscript numbering (two "Fig. 5" scripts), and the
       script/Makefile references after steps 6–13.

### B2. Presentation material (`presentation/` and `scripts/presentation_figures/`)

Principle: nothing from `presentation/` is published (git-ignored, contains patient figures, LaTeX builds, PDFs).
Only reusable *code* is migrated to `scripts/presentation_figures/`, cleaned, and tested on the sample data.

10. [x] **Decide: publish `scripts/presentation_figures/` or not.** It is currently untracked in the clean branch. If yes, — **DONE:** published (code only).
        do steps 11–15; if no, keep it local (add to `.gitignore`) and drop it from the README structure tree (line ~109).
11. [x] **Migrate or drop the remaining scripts still in `presentation/`** (10): `agp_counterfactual_comparison.py`, — **DONE:** migrated to `scripts/presentation_figures/`: `agp_counterfactual_recurrence`, `agp_single_patient`, `agp_recurrence_two_patients`, `counterfactual_seven_day_overview`, `nine_days_two_patients`; duplicates moved to the local, git-ignored `presentation/_dropped/` (`agp_recurrence`, `agp_two_patients_comparison`, `boxplots_tir_cv_median` ⊂ `visualize_boxplots_sensitivity.py`, `cgm_meal_spikes_animation`, `cgm_meal_spikes_insulin` ⊂ `cgm_meal_spikes.py`).
        `agp_presentation.py`, `agp_recurrence.py`, `agp_recurrence_two_patients.py`, `agp_two_patients_comparison.py`,
        `boxplots_tir_cv_median.py`, `cgm_7day_counterfactual.py`, `cgm_meal_spikes_animation.py`,
        `cgm_meal_spikes_insulin.py`, `cgm_nine_days_comparison.py` — many duplicate scripts already in
        `scripts/` or `scripts/presentation_figures/` (e.g. `cgm_agp_recurrence.py`, `patient_comparison.py`,
        `cgm_nine_days.py`, `cgm_meal_spikes.py`): delete duplicates, migrate the rest as done for
        `slh_distributions_by_category`, `offset_sensitivity_single_patient`, `offset_counterfactual_single_episode`,
        `episode_classification_zoom_panels`.
12. [x] **Fix headers and paths of the older scripts already in `scripts/presentation_figures/`**: `cgm_nine_days.py`, — **DONE:** headers, `sys.path` depth, default output folder (`scripts/presentation_figures/output/<script>/`) and `__init__.py` fixed in all scripts.
        `cgm_overview.py`, `counterfactual_steps.py` still show `presentation/...` in their docstring (file name and
        `--out_dir presentation/output`); check the `sys.path.insert` depth (must be `"..", ".."`) and the default output
        folder in every script; add `scripts/presentation_figures/__init__.py` (the other script packages have one).
13. [x] **Align presentation scripts with the paper's classification** (as done for `slh_distributions_by_category`): — **DONE:** `LABEL_SLH` from `classify_episodes` equals the Figure 2 SLH set; the only mislabelled output was the "LLH" series of `agp_recurrence_two_patients` (it included non-isolated / night episodes) — now the four Figure 2 categories. `agp_counterfactual_recurrence` keeps SLH/LLH like paper Figure 3 (shared `src` panel).
        every script that classifies episodes must use the Figure 2 rules (C1 → LLH, C4/C5/C6 → Not isolated, C3 → Night,
        else SLH; C2 dropped) — check `slh_count_vs_counterfactual_deltas.py`, `slh_twin_comparison.py`,
        `twin_two_weeks.py`, `metric_permutation_invariance.py`, `counterfactual_steps.py`.
14. [x] **Delete the old figures in `presentation/output/`** of every script migrated or dropped in step 11 — **DONE:** `presentation/output/` and the LaTeX decks are gone locally; only `presentation/_dropped/` (git-ignored) remains.
        (already gone: `zoom_panels_individual/`, `sensitivity_analysis/`, `episode_counterfactual_comparison/`,
        `slh_distributions/`; remaining folders: `agp`, `agp_counterfactual_comparison`, `agp_recurrence`,
        `agp_recurrence_two_patients`, `agp_two_patients_comparison`, `boxplots_tir_cv_median`, `cgm_7day_counterfactual`,
        `cgm_meal_spikes`, `cgm_nine_days_comparison`), plus the LaTeX decks (`Poster/`, `local_slh/`, `onset_duration/`:
        `.pdf`, `.aux`, `.log`…) — local files only, but they contain real-patient figures: keep them out of any archive/zip.
15. [x] **Test every kept presentation script on the sample data** (`CGM_DATA_ROOT=…/sample_data`, `--patient_id A001`); — **DONE:** `make presentation_figures` runs all 19 scripts on the sample data without error; `output/` is git-ignored and was deleted.
        delete generated `scripts/presentation_figures/output/` before the release (outputs from real data must never be
        committed; add `scripts/presentation_figures/output/` to `.gitignore`). Optionally add a `make presentation_figures` target.

### B3. Patient IDs and scripts layout

16. [x] **Remove hard-coded real patient IDs from defaults** — DONE: new `src/utils/patient_selection.py` (automatic most/fewest-SLH selection, `--patient_id`, `CGM_PATIENT_HIGH/LOW`), applied to all scripts, README note added, tested on the sample data (Fig. 5 needs the clinical table). **Remaining:** the DPO check on `A###` codes. Original task: (`A159`, `A174`, `A224` in `lininterp_*`,
        `visualize_*_sensitivity`, `fig_slh_schema.py`, `generate_*`, `patient_comparison.py`, `scatter_analysis.py`, and the
        paper Figures 3–5). On the open dataset (IDs shuffled to `OD_xxx`) and on the sample data these patients do not
        exist, so the scripts fail with defaults. Options: automatic selection (most / fewest SLH, as in notebooks 04, 07),
        or `--patient_id` required + README note that Figures 3–5 use specific patients that cannot be located in the open
        dataset. Also confirm with the DPO that `A###` study codes cannot be linked back to patients.
17. [ ] **Tidy the loose scripts at the root of `scripts/`** (`cgm_agp_recurrence.py`, `fig_slh_schema.py`,
        `generate_episode_funnel_single_patient.py`, `generate_overview_figure.py`, `generate_zoom_panels.py`,
        `patient_comparison.py`): move into a sub-package (e.g. `scripts/slh_illustrations/`) or into `paper_figures/`,
        update the Makefile targets, the README map and any cross-import (`generate_zoom_panels` is imported by
        `episode_classification_zoom_panels.py`).
18. [ ] **Naming consistency**: the paper scripts use `generate_*`, the others do not; the two "Fig. 5" scripts need clear
        names; check every script has a header with its module command and its output location.
19. [ ] **Review `scripts/statistics/` and `scripts/patient_selection/`**: keep only what feeds a figure/table/statement
        of the paper (e.g. `slh_baseline_metrics_multivariate_exploration.py`, `slh_combined_cgm_phenotype_pca.py`,
        `slh_metrics_visualization_from_files.py`, `slh_patient_characteristics_exploration.py` look exploratory); the
        `compelling_pairs` / `patient_similarity` Makefile targets select real patients — confirm they are needed.
20. [ ] **Final consistency checks**: `git grep "/Users/"` empty; `git grep -n "presentation/"` only intended hits;
        no module in the Makefile missing (script in the repo); `make help` clean; `make tests` green; no unused imports
        (`make lint` if available).

## C. Validate

21. [ ] **Regenerate every manuscript figure/table from the open dataset** (values differ slightly because of the noise;
        stated in the README) and run `make tests` + `make sample_data_run` on the cleaned tree.
22. [x] **Add one end-to-end test** (synthetic trace: detection → counterfactual → Δ metrics) and a **GitHub Actions
        workflow** running `make tests`. (Do this before the release so CI is in the archived snapshot.)

## D. Release
        **DONE:** `tests/test_pipeline_end_to_end.py` (+ unit conversion, patient selection, open data), `.github/workflows/tests.yml`; 216 tests pass locally. **Remaining:** check the first GitHub Actions run is green after pushing.
23. [ ] **Final commit on `master`**, then `make sample_data` to produce `sample_data.zip`, and **move the tag**:
        `git tag -f -a v1.0.0 -m "v1.0.0" && git push -f origin v1.0.0`
        (or delete and recreate it). Only while no release exists.
24. [ ] **Archive the open dataset** (`make open_data`, output in `Results/Open_Data/`) on Zenodo and put its DOI in the
        README, `CITATION.cff` and the manuscript's data-sharing statement. Check whether `10.5281/zenodo.23102518`
        is the code record, the dataset record, or a concept DOI, and update the README/CITATION accordingly.
25. [ ] **Publish the GitHub release** `v1.0.0` with `sample_data.zip` attached
        (`gh release create v1.0.0 sample_data.zip --title "v1.0.0" --notes "First public release"`) → triggers the Zenodo
        snapshot. Check it appears on Zenodo and the version DOI matches `CITATION.cff`.
26. [ ] Link `sample_data.zip` from the README Quick start and add the release/DOI badge.

## E. After release (optional, can be a v1.0.1)

27. [ ] Add ORCIDs and the missing affiliations (Syed, Lysy) to `CITATION.cff` / README.
28. [ ] *(badges done: tests, Python, licences, DOI)* Add a `docs/` page describing each output file's columns, and a short note on the
        config layout (`src/config/` is the single source of thresholds).
29. [ ] Add the journal year/volume/pages to the citation once published.

## Done

- [x] Removed stale `src/config.py`; `make install` works; `requirements.txt` tested on a clean venv (182 tests pass).
- [x] `Makefile` rewritten (targets for every figure/table, `pipeline`, `make help`, non-destructive `requirements-check`).
- [x] Python version pinned (≥ 3.11, tested on 3.13); sample dataset (`make sample_data`) tested end-to-end.
- [x] README cleaned for Zenodo (title, DOI, citation, authors); `CITATION.cff` added; PR #53 merged.
- [x] No csv/xlsx/parquet/pkl tracked in the current tree.
- [x] `requirements-optional.txt` references removed.

## Switching the remote to the clean history (needs your OK — destructive)

```bash
git checkout release-clean
git branch -M master                       # local old master is kept in the bundle
git push --force origin master             # replaces the remote history
git push origin :refs/tags/v1.0.0          # delete the old remote tag
git push origin --delete cleanup-for-publication
```
Then ask GitHub Support to purge cached views of PR #53 / dangling commits if the repo was public. Do the remaining
steps (6–22) as normal commits on top, then re-create the tag in step 23.
