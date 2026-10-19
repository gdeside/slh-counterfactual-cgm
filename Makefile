# =============================================================================
# Counterfactual CGM-SLH — run `make help` for the list of targets.
#
# * Every scripts/ entry point is run as a module from the repository root.
# * Data root: set CGM_DATA_ROOT (see src/config/paths.py), e.g.
#       CGM_DATA_ROOT=$PWD/sample_data make pipeline
# * Help text = the `## comment` written after each target.
# =============================================================================

PYTHON ?= python3
RUN     = $(PYTHON) -m

.DEFAULT_GOAL := help

# Clinical phenotype table (not in the repository; available on request, see README)
CLINICAL_TABLE = $(or $(CGM_DATA_ROOT),data)/Data/Final cohort SLH.csv

.PHONY: help install check_clinical_table paper_clinical requirements-check \
	time_step_analysis preprocessing coverage_analysis metrics_preprocessing filtering \
	filtered_metrics_distribution episode_detection pipeline \
	slh_llh_cohort_descriptive_stats compelling_pairs patient_similarity slh_funnel slh_overview slh_zoom_panels agp_recurrence patient_comparison \
	lininterp_stepbystep lininterp_offset_comp lininterp_45days \
	sensitivity_analysis lines_sensitivity boxplots_sensitivity cdf_sensitivity agp_sensitivity \
	timeseries_sensitivity sensitivity_figures \
	quartile_analysis scatter_analysis \
	table1 fig2 fig3 fig4 fig5 fig6 suppfig1 suppfig3 suppfig_onset_time suppfig_phenotype suppfig_s10 cohort_flowchart paper \
	example shanghai_convert shanghai_example presentation_figures convert_units sample_data sample_data_real sample_data_run \
	figures tests clean


# =============================================================================
# HELP
# =============================================================================

help:  ## List all targets
	@echo ""
	@echo "Counterfactual CGM-SLH — make targets"
	@echo "======================================"
	@awk 'BEGIN {FS = ":.*## "} \
	     /^# ---- / {sub(/^# ---- /, ""); sub(/ -*$$/, ""); printf "\n  %s\n", $$0} \
	     /^[a-zA-Z0-9_]+:.*## / {printf "    %-31s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Data root: CGM_DATA_ROOT=/path/to/data make <target>  (default: ./data)"
	@echo ""


# ---- SETUP ------------------------------------------------------------------

check_clinical_table:  ## Check that the clinical phenotype table exists (needed by table1, fig5, fig6, ...)
	@test -f "$(CLINICAL_TABLE)" || { echo "Clinical table not found: $(CLINICAL_TABLE)"; \
	  echo "  -> this target needs 'Data/Final cohort SLH.csv' under CGM_DATA_ROOT (not in the repository)."; exit 1; }


install:  ## Install dependencies (requirements.txt)
	$(PYTHON) -m pip install -r requirements.txt

requirements-check:  ## List imports missing from requirements.txt (never overwrites it)
	$(PYTHON) -m pip install -q pipreqs
	$(PYTHON) -m pipreqs.pipreqs . --force --encoding=utf-8 --savepath /tmp/requirements.detected.txt \
		--ignore env,venv,build,dist,presentation,notebooks
	@echo "Detected imports written to /tmp/requirements.detected.txt — compare with requirements.txt by hand."


# ---- DATA PIPELINE (run in order) -------------------------------------------

time_step_analysis:  ## Raw timestep distribution
	$(RUN) scripts.preprocessing.cgm_timestep_analysis_raw

preprocessing:  ## 45-day window extraction
	$(RUN) scripts.preprocessing.cgm_preprocessing

coverage_analysis:  ## Data-coverage analysis
	$(RUN) scripts.preprocessing.cgm_coverage_analysis

metrics_preprocessing:  ## CGM metric computation (TIR, TAR, SD, CV, ...)
	$(RUN) scripts.preprocessing.cgm_metrics

filtering:  ## Median-glucose cohort filter
	$(RUN) scripts.preprocessing.cgm_filtering

filtered_metrics_distribution:  ## Metric distributions of the filtered cohort
	$(RUN) scripts.preprocessing.cgm_filtered_metrics_distribution

episode_detection:  ## Episode detection + SLH/LLH classification
	$(RUN) scripts.episodes_detection.episode_detection

pipeline: preprocessing metrics_preprocessing filtering episode_detection sensitivity_analysis  ## Core pipeline: windows -> metrics -> filter -> episodes -> counterfactual sensitivity


# ---- STATISTICS -------------------------------------------------------------

slh_llh_cohort_descriptive_stats:  ## SLH/LLH cohort descriptive statistics
	$(RUN) scripts.statistics.slh_llh_cohort_descriptive_stats

compelling_pairs:  ## Participant pairs illustrating SLH beyond standard metrics
	$(RUN) scripts.patient_selection.find_compelling_pairs

patient_similarity:  ## Participant similarity in CGM-metric space vs n_SLH (interactive HTML)
	$(RUN) scripts.patient_selection.patient_similarity_analysis

quartile_analysis:  ## Quartile stratification by n_SLH
	$(RUN) scripts.statistics.quartile_analysis

scatter_analysis:  ## n_SLH vs delta-metric scatter plots
	$(RUN) scripts.statistics.scatter_analysis


# ---- SLH EPISODE FIGURES ----------------------------------------------------

slh_funnel:  ## Episode classification funnel of one patient (Figure 2 categories)
	$(RUN) scripts.episodes_detection.episode_funnel_single_patient $(if $(PATIENT_ID),--patient_id $(PATIENT_ID))

slh_overview:  ## 45-day episode overview
	$(RUN) scripts.generate_overview_figure $(if $(PATIENT_ID),--patient_id $(PATIENT_ID))

slh_zoom_panels:  ## Zoom panels per classification criterion
	$(RUN) scripts.generate_zoom_panels $(if $(PATIENT_ID),--patient_id $(PATIENT_ID))

agp_recurrence:  ## AGP + recurrence figures
	$(RUN) scripts.cgm_agp_recurrence

patient_comparison:  ## Side-by-side patient comparison
	$(RUN) scripts.patient_comparison


# ---- COUNTERFACTUAL CONSTRUCTION --------------------------------------------

lininterp_stepbystep:  ## Step-by-step construction figure
	$(RUN) scripts.counterfactual.lininterp_stepbystep $(if $(PATIENT_ID),--patient_id $(PATIENT_ID))

lininterp_offset_comp:  ## Offset comparison (delta = 0...90 min)
	$(RUN) scripts.counterfactual.lininterp_offset_comp $(if $(PATIENT_ID),--patient_id $(PATIENT_ID))

lininterp_45days:  ## 45-day G(t) vs C(t) overview
	$(RUN) scripts.counterfactual.lininterp_45days $(if $(PATIENT_ID),--patient_id $(PATIENT_ID))


# ---- SENSITIVITY ANALYSIS ---------------------------------------------------

sensitivity_analysis:  ## Core computation (all patients x all delta) — slowest step
	$(RUN) scripts.counterfactual.sensitivity_analysis

lines_sensitivity:  ## Delta metric vs delta line plots
	$(RUN) scripts.counterfactual.visualize_sensitivity_lines $(if $(PATIENT_ID),--patient_id $(PATIENT_ID))

boxplots_sensitivity:  ## Cohort delta-metric boxplots per delta
	$(RUN) scripts.counterfactual.visualize_boxplots_sensitivity

cdf_sensitivity:  ## CDF of delta metric per delta
	$(RUN) scripts.counterfactual.visualize_cdf_sensitivity

agp_sensitivity:  ## AGP sensitivity overlay
	$(RUN) scripts.counterfactual.visualize_agp_sensitivity $(if $(PATIENT_ID),--patient_id $(PATIENT_ID))

timeseries_sensitivity:  ## Original vs counterfactual time series
	$(RUN) scripts.counterfactual.visualize_timeseries_sensitivity $(if $(PATIENT_ID),--patient_id $(PATIENT_ID))

sensitivity_figures: lines_sensitivity boxplots_sensitivity cdf_sensitivity agp_sensitivity timeseries_sensitivity  ## All sensitivity figures


# ---- MANUSCRIPT FIGURES & TABLES --------------------------------------------

table1: check_clinical_table  ## Table 1 — cohort characteristics (needs clinical phenotype table)
	$(RUN) scripts.paper_figures.generate_table1

fig2:  ## Figure 2 — episode classification summary (SLH / Not isolated / Night / LLH)
	$(RUN) scripts.paper_figures.generate_figure_2_classification_summary

fig3:  ## Figure 3 — two patients, similar metrics, different SLH frequency
	$(RUN) scripts.paper_figures.generate_figure_3_patient_comparison

fig4:  ## Figure 4 — counterfactual construction + sensitivity
	$(RUN) scripts.paper_figures.generate_figure_4_counterfactual_construction

fig5: check_clinical_table  ## Figure 5 — distribution of delta-TIR / delta-median / max delta-AGP + correlation matrix
	$(RUN) scripts.paper_figures.generate_figure_5_counterfactual_distribution

fig6: check_clinical_table  ## Figure 6 — SLH determinants (needs clinical phenotype table)
	$(RUN) scripts.paper_figures.generate_figure_6_slh_determinants

suppfig1:  ## Suppl. Fig. 1 — data quality and cohort selection
	$(RUN) scripts.paper_figures.generate_supplementary_figure_1_data_quality

suppfig3:  ## Suppl. Fig. 3 — SLH detection schema (one patient, automatic choice unless PATIENT_ID given)
	$(RUN) scripts.paper_figures.generate_supplementary_figure_3_slh_detection_schema $(if $(PATIENT_ID),--patient_id $(PATIENT_ID))

suppfig_onset_time:  ## Suppl. figure — SLH/LLH onset time of day for every patient (supports Fig. 2D)
	$(RUN) scripts.paper_figures.generate_supplementary_figure_onset_time_per_patient

suppfig_phenotype: check_clinical_table  ## Suppl. figure — phenotype determinants (needs clinical phenotype table)
	$(RUN) scripts.paper_figures.generate_supplementary_figure_phenotype_determinants

suppfig_s10: check_clinical_table  ## Suppl. Figure S10 — phenotype overview: MI, GTAA1C vs duration, n_SLH by category (needs clinical phenotype table)
	$(RUN) scripts.paper_figures.generate_supplementary_figure_s10_phenotype_overview

cohort_flowchart:  ## Suppl. cohort flowchart (PDF needs rsvg-convert)
	$(RUN) scripts.paper_figures.generate_cohort_flowchart

paper: fig2 fig3 fig4 suppfig1 suppfig3 suppfig_onset_time cohort_flowchart  ## Manuscript figures that need no clinical table

paper_clinical: table1 fig5 fig6 suppfig_phenotype suppfig_s10  ## Table 1 and figures that need the clinical phenotype table


# ---- EXAMPLE DATA & UNITS ---------------------------------------------------

example:  ## Detection + counterfactual without the pipeline: demo, INPUT=file, or INPUT_DIR=folder [UNIT=mmol/L]
	$(RUN) scripts.examples.detect_and_counterfactual $(if $(INPUT_DIR),--input_dir $(INPUT_DIR),$(if $(INPUT),--input $(INPUT),--demo)) $(if $(UNIT),--unit $(UNIT))

presentation_figures:  ## Talk figures (scripts/presentation_figures/, output ignored by git; patients chosen automatically)
	$(RUN) scripts.presentation_figures.agp_counterfactual_recurrence
	$(RUN) scripts.presentation_figures.agp_recurrence_two_patients
	$(RUN) scripts.presentation_figures.agp_single_patient
	$(RUN) scripts.presentation_figures.cgm_meal_spikes
	$(RUN) scripts.presentation_figures.cgm_nine_days
	$(RUN) scripts.presentation_figures.cgm_overview
	$(RUN) scripts.presentation_figures.cgm_slh_repeat_gif
	$(RUN) scripts.presentation_figures.counterfactual_seven_day_overview
	$(RUN) scripts.presentation_figures.counterfactual_steps
	$(RUN) scripts.presentation_figures.episode_classification_zoom_panels
	$(RUN) scripts.presentation_figures.insulin_types
	$(RUN) scripts.presentation_figures.metric_permutation_invariance
	$(RUN) scripts.presentation_figures.nine_days_two_patients
	$(RUN) scripts.presentation_figures.offset_counterfactual_single_episode
	$(RUN) scripts.presentation_figures.offset_sensitivity_single_patient
	$(RUN) scripts.presentation_figures.slh_count_vs_counterfactual_deltas
	$(RUN) scripts.presentation_figures.slh_distributions_by_category
	$(RUN) scripts.presentation_figures.slh_twin_comparison
	$(RUN) scripts.presentation_figures.twin_two_weeks

shanghai_convert:  ## Put the ShanghaiT1DM files (example_data/ShanghaiT1DM/raw/) into the project format -> converted/ (MODIFIED data)
	$(RUN) scripts.examples.convert_shanghai_t1dm

shanghai_example: shanghai_convert  ## Detection + counterfactual on the converted ShanghaiT1DM example data -> example_output/
	$(RUN) scripts.examples.detect_and_counterfactual --input_dir example_data/ShanghaiT1DM/converted

convert_units:  ## Write mmol/L copies of the data files (mg/dL originals untouched)
	$(RUN) scripts.preprocessing.convert_data_units


# ---- SAMPLE DATASET (synthetic, for reviewers) ------------------------------

sample_data:  ## Generate 4 synthetic patients + sample_data.zip
	$(RUN) scripts.sample_data.generate_sample_data --out sample_data --zip

sample_data_real:  ## Maintainers only: 4 fake patients re-assembled from REAL data (needs real CGM_DATA_ROOT; DPO approval before publishing)
	$(RUN) scripts.sample_data.generate_sample_data_from_real --out sample_data_real --zip

sample_data_run: sample_data  ## Run the core pipeline on sample_data/ (~2 min)
	CGM_DATA_ROOT=$(CURDIR)/sample_data $(MAKE) pipeline


# ---- AGGREGATE --------------------------------------------------------------

figures: time_step_analysis coverage_analysis lininterp_offset_comp sensitivity_figures  ## Key exploratory figures


# ---- TESTS & UTILITIES ------------------------------------------------------

tests:  ## Run all unit tests
	$(PYTHON) -m pytest tests -v

clean:  ## Remove __pycache__, .pyc, pytest artefacts
	find . -type d -name __pycache__ -not -path "./env/*" -not -path "./venv/*" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -not -path "./env/*" -not -path "./venv/*" -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -not -path "./env/*" -not -path "./venv/*" -delete
