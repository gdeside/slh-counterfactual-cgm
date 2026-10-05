# To-do before the v1.0.0 release (ordered)

Everything below gets done; the numbering is the order to do it in. Steps 1–10 modify the repository, so they
come **before** the release: publishing the GitHub release triggers the Zenodo snapshot, which cannot be edited afterwards.

> ⚠️ Status checked on `master` (PR #53 merged): the tag `v1.0.0` **already exists on the remote** (commit `905a5e8`),
> but **no GitHub release is published yet**, so Zenodo has not archived anything. The tag must be moved to the final
> commit in step 12 (`git tag -f` + force-push of the tag) — only safe as long as no release exists.

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
3. [x] **Presentation figures with real-patient outputs** (`scripts/presentation_figures/`) removed from the clean branch.
4. [ ] **Ethics / DPO check** that releasing the open dataset (and the figures) is covered by the approval.

## B. Clean the repository content

5. [ ] **Remove hard-coded local paths** (still present in): `src/config/paths.py` (default `BASE_PATH` + commented line 24),
       `scripts/supplementary_code/regression_analysis_poisson.py`, `notebooks/01_cgm_metrics.ipynb`.
       Make `CGM_DATA_ROOT` required or default to `./data`-style relative path with a clear error message.
6. [x] **Non-code artefacts removed** from the clean branch (`TLDE_submission_review.md`, `talk_15min_plan.md`,
       `scripts/presentation_figures/`); `env/`, `presentation/`, `analyzer.py` are git-ignored — don't re-add.
7. [ ] **Decide the fate of `scripts/supplementary_code/`** (2 files left: `local_slh_analysis_cohort.py`,
       `regression_analysis_poisson.py`) and the empty `scripts/poster_figures/`: keep only what supports a figure,
       table or statement of the paper, delete the rest, and say so in the README.
8. [ ] **Fix the `Makefile`**: `make clock_lock` (line 96-97) calls the deleted `scripts.statistics.clock_lock_analysis`
       — drop or redirect the target (also remove it from the `.PHONY` list, line 22), and remove targets pointing at
       anything deleted in steps 6–7.
9. [ ] **Re-check the README figure/table map** against the final manuscript numbering (two "Fig. 5" scripts), and the
       script/Makefile references after steps 6–8.

## C. Validate

10. [ ] **Regenerate every manuscript figure/table from the open dataset** (values differ slightly because of the noise;
        stated in the README) and run `make tests` + `make sample_data_run` on the cleaned tree.
11. [ ] **Add one end-to-end test** (synthetic trace: detection → counterfactual → Δ metrics) and a **GitHub Actions
        workflow** running `make tests`. (Do this before the release so CI is in the archived snapshot.)

## D. Release

12. [ ] **Final commit on `master`**, then `make sample_data` to produce `sample_data.zip`, and **move the tag**:
        `git tag -f -a v1.0.0 -m "v1.0.0" && git push -f origin v1.0.0`
        (or delete and recreate it). Only while no release exists.
13. [ ] **Archive the open dataset** (`make open_data`, output in `Results/Open_Data/`) on Zenodo and put its DOI in the
        README, `CITATION.cff` and the manuscript's data-sharing statement. Check whether `10.5281/zenodo.23102518`
        is the code record, the dataset record, or a concept DOI, and update the README/CITATION accordingly.
14. [ ] **Publish the GitHub release** `v1.0.0` with `sample_data.zip` attached
        (`gh release create v1.0.0 sample_data.zip --title "v1.0.0" --notes "First public release"`) → triggers the Zenodo
        snapshot. Check it appears on Zenodo and the version DOI matches `CITATION.cff`.
15. [ ] Link `sample_data.zip` from the README Quick start and add the release/DOI badge.

## E. After release (optional, can be a v1.0.1)

16. [ ] Add ORCIDs and the missing affiliations (Syed, Lysy) to `CITATION.cff` / README.
17. [ ] Add badges (licence, Python, tests), a `docs/` page describing each output file's columns, and a short note on the
        config layout (`src/config/` is the single source of thresholds).
18. [ ] Add the journal year/volume/pages to the citation once published.

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
steps (5, 7–11) as normal commits on top, then re-create the tag in step 12.
