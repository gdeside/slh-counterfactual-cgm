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
       `update-cohort-plots-…`), and a force-push does **not** remove them: the old notebooks, plots, IDs and real
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
       files); local junk `.DS_Store`, `.pytest_cache`, editor / tool folders. `env/`, `presentation/`,
       `analyzer.py` are git-ignored (don't re-add).
5. [ ] **Confirm the last manuscript numbers** (only you know them): the cohort flowchart and the phenotype-determinants figure are
       still "Suppl. Fig. —" without a number (rename the scripts / Makefile targets once known, e.g. `suppfig2`), there is no script for
       Fig. 1, and `generate_cohort_flowchart.py` may be dropped altogether (see the discussion: hand-drawn SVG, no computation).
6. [ ] **Tidy the loose scripts at the root of `scripts/`** (`cgm_agp_recurrence.py`,
       `generate_episode_funnel_single_patient.py`, `generate_overview_figure.py`, `generate_zoom_panels.py`,
       `patient_comparison.py`): move into a sub-package (e.g. `scripts/slh_illustrations/`) or `paper_figures/`; update the
       Makefile targets, the README map and cross-imports (`generate_zoom_panels` is imported by
       `scripts/presentation_figures/episode_classification_zoom_panels.py`).
8. [ ] **Review `scripts/statistics/` and `scripts/patient_selection/`** (4 statistics scripts have no Makefile target): keep only what feeds a figure, table or statement
       of the paper (`slh_baseline_metrics_multivariate_exploration.py`, `slh_combined_cgm_phenotype_pca.py`,
       `slh_metrics_visualization_from_files.py`, `slh_patient_characteristics_exploration.py` look exploratory); confirm
       the `compelling_pairs` / `patient_similarity` Makefile targets are needed (they select real patients).
9. [ ] **Final consistency checks** (and delete `TODO.md` from the archived version): `git grep "/Users/"` empty; `git grep -n "presentation/"` only intended hits; every
       Makefile target points to an existing module; `make help` clean; `make tests` green; no unused imports.

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

