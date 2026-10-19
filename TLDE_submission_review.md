Legend: 🔴 blocking for submission · 🟠 should fix before submission · 🟡 style / copy-edit

---

## 1. Journal requirements — hard limits

| Requirement (TLDE) | Manuscript | Status |

| Statement that ≥1 author (named) **accessed and verified the underlying data** | **Absent** | 🔴 |
| Observational studies must be reported per **STROBE** and **submitted with their protocol** | No STROBE checklist / no protocol referenced | 🔴 (submission package) |
| Observational studies encouraged to be registered in a **WHO-compliant registry** | No registration statement | 🟠 |
| Ethics committee **and** approval number stated in the Article | Present — but see §3 (stray underscore, and no informed-consent / waiver statement) | 🟠 |

**Also part of the submission package (not in the .docx):** covering letter; signed author statement form; ICMJE conflict-of-interest forms for every author; figures as separate ≥300 dpi / ≥107 mm files (TIF/JPG for images, editable vector for line art); supplementary material as **one PDF with a table of contents and numbered pages**; supplementary text 10 pt Times New Roman, headings 10 pt bold, main heading 12 pt bold, tables 8 pt.

---

## 2. Journal house style — formatting

| Rule | Issue | Status |
|---|---|---|
| **Midline decimal point** (`23·4`, not `23.4`) throughout | Tables 1–3 use `·` correctly, but **the entire main text and abstract use `.`** (`13.68`, `1.53%`, `140.0`, `ρ = −0.49`). Inconsistent within the same document. | 🟠 |
| References cited as **superscript numbers**, Vancouver | Text uses bracketed `[1]`, `[5-7]`, `[8-10]` | 🟠 |
| Consecutive references joined with an **en rule**, two refs separated by comma with no space | `[5-7]`, `[8-10]` use hyphens; `[10, 14]` and `[2, 3]` have spaces | 🟡 |
| One citation style only | **Two author–date citations survive in Methods**: `(American Diabetes Association Professional Practice 2023)` and `(Doupis and Horton 2022)` — the latter is already reference 14 | 🟠 |
| ≤6 authors → list all six; ≥7 → first three + *et al* | Every multi-author reference is `First author, et al.` (refs 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 16) | 🟠 |
| **p values to two significant figures** unless p<0·0001 | All reported as `p<0.001` / `p < 0.001` | 🟠 |
| **Numbers one to ten spelled out** unless a unit of measurement | `2 consecutive CGM readings`, `12 consecutive CGM readings` (Methods, SLH criterion 1) | 🟡 |
| **Numbers must be given whenever a % is shown** | Results §Cohort characteristics gives bare percentages: `15% of patients presented at least one comorbidity`, `7%`, `67%`, `33%`, `54% / 23% / 22%`, `4%`, `63% / 37%` | 🟠 |

Degrees missing; addresses are one-line affiliations only; **no telephone**; affiliation 2 (`UCLouvain, Pediatrics, Brussels, Belgium`) is terse and does not name Cliniques universitaires Saint-Luc, which is the site named in the Methods | 🟠 |
| Consistent en dashes for ranges | `TIR 70-180` (Methods) vs `70–180` elsewhere | 🟡 |
| Latin terms | `in-silico` (hyphenated, roman) used in headings and text — journal style is *in silico* | 🟡 |


**Reporting sex/gender/race (SAGER + TLDE policy):**
- Table 1 row is labelled **"Gender"** with categories Male/Female, while the Methods say the variable extracted was **"sex"** — pick one, and state how it was ascertained (self-report, medical record). 🟠
- Methods list sex as a categorical predictor tested against SLH count, but **no sex-related result is reported anywhere**. 🟠
- Race/ethnicity not reported — the guidelines ask you to say why it was not collected/analysed if that is the case. 🟡
- Discussion should explicitly address **representativeness of the study population** — currently only partly covered by limitation 1. 🟡

---

## 3. Internal inconsistencies — numbers


🟠 **3. Abstract IQR does not match Table 2 / Results.**
Abstract: daytime TIR "+1·53% (IQR 1·11–**2·14**)" vs Results and Table 2 (δ=30): "1·53 [1·11, **2·16**]".

🟠 **6. Results text vs Table 1 on severe hypoglycaemia.**
Results: "After diagnosis, DKA and severe hypoglycemia **each occurred in 4%**" — Table 1 gives 3% for severe hypoglycaemia.

🟠 **7. Coefficient of variation reported on two different scales.**
Results: "median coefficient of variation was **0·43** [IQR 0·37–0·48]"; Table 1: "Coefficient of variation (**%**) **43** [37–48]"; Table 2 reports CV changes in percentage points.

🟠 **8. Non-SLH episode count for patient A174 vs Figure S3 — RESOLVED, but the manuscript must say which denominator it uses.**
Results: A174 had "substantially more non-SLH episodes (**83** vs 58), resulting in a total … count of 142". Figure S3 caption: "long-lasting hyperglycemia (**LLH, n=57**)".


🟠 **9. Two unresolved `XX` placeholders in the Results.**
"improvements ranging from **+3.XX%** to **+4.XX%**" (Results §In-silico intervention).

🟠 **11. Figure/panel cross-reference `Fig. SX`** in Results §"Short-lasting hyperglycemic episodes are highly prevalent" is an unresolved placeholder.

🟡 **12. Time-of-day values are reported in two incompatible formats.**
Results: "13.45 for A174 and 12.30 for A224" (decimal hours? 12:30?) vs "modes at 9:00, 13:00, 17:00, 20:00" (clock) vs Table 3 "14·08 [10·75, 18·17]" (decimal hours). `12.30` in particular reads as a clock time.

🟡 **13. BMI z-score IQR written ambiguously in the text.**
Results: "[IQR: −0·20–1·42]" — Table 1 correctly uses "−0·20 **to** 1·42" because of the negative bound; the text should match.

---

## 4. Internal inconsistencies — logic, definitions and missing results

🟠 **14. Exclusion criterion does not match its stated rationale.**
Methods: "Patients with **severe hyperglycemia or hypoglycemia** within the 45-day analysis window were excluded … Specifically, patients with **median glucose outside 110–170 mg/dL** were excluded." A median-glucose window excludes patients with an abnormal *average*, not patients with severe excursions. Related tension: the retained cohort has a **median TBR <70 mg/dL of 6% (IQR 3–11%)**, well above the consensus <4% target, yet the Discussion states the cohort "excluded individuals with marked hypo- or hyperglycemia". Either soften the Discussion claim or make the criterion explicit.

🟠 **15. Predictor set differs between Methods and Results.**
Methods §Association list only **three** baseline CGM metrics (median glucose, CV, TIR 70–180). Results report **"All seven baseline CGM metrics"** (TIR, SD, TAR>180, CV, mean glucose, median glucose, TBR<70). Align the Methods list.

🟠 **16. Discussion reports associations that never appear in the Results.**
"**HbA1c, diabetes duration, and BMI z-score** were also inversely associated with SLH frequency" — no ρ, MI or p is reported for HbA1c or BMI z-score anywhere. Similarly, the Abstract claims "**younger participants**" have more SLH, but no age–SLH statistic is reported in the Results. (A co-author comment already flags this: *"I'd propose to add a short paragraph … on the association between age & diabetes duration and # of SLH."*)

🟠 **17. Correlation coefficients are unlabelled and inconsistent with the stated method.**
Results §Contribution reports "**r** = 0·98" and "**r** = 0·42", while the Methods specify Spearman ρ. State which coefficient, and add p values / CIs.

🟠 **18. "Robust to parameter choices" is overstated for the glucose metrics.**
Abstract and Discussion both say sensitivity analyses showed robustness. Table 2 shows daytime median glucose moving from **0·00 (δ=0)** to **−5·00 (δ=60)** and mean glucose from −0·29 to −2·50 — the Results text itself concedes these "increased with δ". Restrict the robustness claim to TIR/TAR (which do plateau at δ=30) and say so explicitly in the abstract.

🟠 **19. Three Table 2 findings are never discussed.**
- **TBR <70 mg/dL increases** after SLH removal at δ=45 and δ=60 (up to +0·07 daytime) — i.e. the counterfactual creates hypoglycaemia.
- **CV increases** after removal at δ≥30 (up to +0·89 daytime) — counter-intuitive and worth a sentence.
- **SD changes sign** across δ (−0·14 → +0·37 daytime).
These matter because they bound the validity of the interpolation.

🟠 **20. No confidence intervals anywhere.** All effects are medians with IQR. TLDE expects effect estimates with CIs; at minimum, add bootstrap CIs for the cohort-level TIR and median-glucose changes.

🟠 **21. Study period of the CGM data is never given.** Methods state only "Patient enrollment and the analysis phase extended from March 2025 to May 2026", which describes the analysis, not the calendar range of the CGM recordings. Also, "patient **enrollment**" is odd wording for a retrospective chart/data review.

🟠 **22. `statsmodels 0.14.0 for regression analyses`** is listed in Software, but **no regression analysis is reported** in the manuscript.

🟡 **23. "LLH" is used before it is defined.** Methods §Definition: "All remaining peaks were classified as **LLH**" — the expansion (*long-lasting hyperglycemia*) appears only in the Figure 3 caption. Also, "**non-isolated SLH**" is a confusing label for episodes that are, by definition, *not* SLH.

🟡 **24. Threshold operator drifts.** Methods: readings "smaller than 140 mg/dL" (**<140**). Discussion: "the requirement for glucose values **≤140 mg/dL**". Abstract: "return to <140 mg/dL". Pick one.

🟡 **25. Abstract regroups the three SLH criteria differently from the Methods** (it folds data completeness into the duration criterion, giving the impression of three criteria in a different order). Mirror the Methods wording.

🟡 **26. Eligibility vs inclusion criteria are used interchangeably** (Methods §Study design; Results "met all eligibility criteria"; Figure 1 caption "all eligibility and inclusion criteria"). A co-author comment raises exactly this. Fix once, globally.

🟡 **27. Insulin-regimen terminology.** "basal-**bolus** therapy" (Results) vs "basal-**prandial**" (Methods, Table 1).

🟡 **28. Abbreviation capitalisation drifts:** `GTAA1c` (Methods, Discussion) vs `GTAA1C` (Abstract, Results, Table 1); `IDAA1c` (Methods) vs `IDAA1C` (Results, Table 1). Also `HbA1c` vs `hemoglobin A1C` vs `A1c` in the eligibility paragraph.

🟡 **29. Terminology for the phenomenon itself drifts:** "short-**lived** hyperglycemia" (title, abstract, definition) vs the Results heading "Short-**lasting** hyperglycemic episodes are highly prevalent"; and "SLH **episodes**" vs "SLH **peaks**" used interchangeably.

🟡 **30. Anchor-point notation is confusing.** The right anchor is written `Lb` (an "L" for the *right* anchor); the offset is `Eℓ`. Suggest `Aleft/Aright` or `L`/`R`. The same paragraph also switches from past to present tense ("the signal **is** smoothed").

🟡 **31. Table 1 footnote defines IDAA1C but not GTAA1C**, and does not define TIR/TAR/TBR, IQR, or z-score.

🟡 **32. Table 1 lists "Time in range 70–140 mg/dL"**, which appears nowhere else in the manuscript (not in the Methods metric list, not in Table 2).

🟡 **33. Results subheading punctuation is inconsistent** — some end with a full stop ("SLH prevalence is variable across patients."), some do not. "Software and packages." is a bold run-in paragraph rather than a Heading 2 like the other Methods subsections.

🟡 **34. Ethics approval number has a stray leading underscore:** `_2022/02FEV/043`. Also, there is **no statement about informed consent or its waiver** — required by the guidelines.

---

## 5. Figures, tables and supplement

🟠 **35. Panels are cited out of order** (a co-author has already flagged this: *"on doit citer les figures (et panels) dans l'ordre"*).
- Figure 2: cited as **2A → 2C → 2E → 2D**.
- Figure 4: cited as **4A,B → 4C-F → 4E,F → 4C,D**.
Either reorder the text or reorder the panels.

🟠 **36. Panels that are never cited in the text:** Figure 2**B**, Figure 2**F**; Figure 3 panels **A–F** (only "Fig. 3" and "Fig. 3G" are cited); Figure 5 has all panels cited ✅.

🔴 **37. No supplementary figure is cited in the main text.** Figures S1–S10 exist but the only supplementary cross-reference is the placeholder "(Fig. SX)". Every supplementary item must be called out.

🔴 **38. Supplementary captions incomplete:** **Figure S5 has no caption at all**, **Figure S10 has no caption at all**, and **Figures S7, S8, S9 have `[caption]` placeholders**.

🟡 **39. Figure 2 caption has a stray space before the full stop:** "Classification of hyperglycemic episodes ."

🟡 **40. Figure 3 caption panel description mismatch:** "(C–D) Recurrence patterns by episode classification" then "Heatmaps display the number of **SLH** episodes per time-of-day bin" — say whether all classes or SLH only are shown.

🟡 **41. Nighttime window is stated three ways:** Methods "after 6 AM and before 10 PM" (→ night 22:00–06:00), Figure S2/S3 captions "nocturnal periods (22:00–06:00)". Consistent, but use one notation (24 h clock) throughout, including "6 AM – 10 PM" in the Methods.

🟡 **42. Display-item count.** Five figures plus three tables in the main paper. The guidelines ask you to keep in the main paper only what is essential, with the rest in the appendix — consider moving Table 3 (2 rows) into Table 2 or the supplement.

---

## 6. Citation appropriateness

🟠 **43. Reference 4** (Gaster B, Hirsch IB. *The Effects of Improved Glycemic Control on Complications in **Type 2** Diabetes*. Arch Intern Med **1998**) is cited in the Introduction to support long-term micro/macrovascular complications **in type 1 diabetes**. DCCT/EDIC would be the appropriate citation.

🟡 **44. Reference 11** (Glaser et al., ISPAD DKA/HHS guidelines) is cited for "prolonged periods of sustained hyperglycemia" — a weak match; a glycaemic-variability or AGP-pattern reference would fit better.

🟡 **45. Reference 14** (Doupis & Horton) is cited both as a numbered reference `[7,10,14]` and as an author–date parenthetical in the Methods (see §2).

---

## 7. Open co-author comments in the file

The .docx still carries **43 unresolved comments** (Laura Symul, Philippe Lysy, Guillaume Deside, "Microsoft Office User"). Several encode substantive open decisions, not just wording:

- Define "recurrent SLH episodes" formally, or drop the word.
- "This statement needs some data support." (on a Results claim)
- "Nice! … add a short paragraph … on the association between age & diabetes duration and # of SLH." → see item 16.
- "on doit citer les figures (et panels) dans l'ordre … je proposerais plutôt de changer l'ordre des figures." → see item 35.
- "This needs some minor editing to account for categorical variables." (Methods §Association)
- "I wonder if it would make sense to continue expanding on the delta to include 75 and 90 minutes to see if the delta-median reaches a plateau." → directly relevant to item 18.
- Two suggestions to move interpretive sentences from Results to Discussion.
- Declaration of interests: "Nothing to declare" (LS) / "Philippe: what do you usually do?" — still open.
- Data sharing: "You can make the GitHub repo public upon submission and link it to a Zenodo repo that has a doi."
- Philippe Lysy: "Je pense que ceci est trop complexe pour le lecteur « moyen »."

**All comments must be resolved and removed before submission.** (No tracked insertions/deletions remain — only comments.)

---

## 8. Suggested order of work

1. Fix the two hard contradictions first: **SLH count (item 1)** and **GTAA1C direction (item 2)** — regenerate the abstract numbers from the current analysis output.
2. Rewrite the abstract to **250 words in five paragraphs including Funding**, then cut the main text by ~1100 words to reach 3500 (the Discussion, at 1170 words over five paragraphs, and the Methods §Counterfactual maths are the obvious candidates — the formal construction can move to the appendix).
3. Add the missing required sections: **Research in context**, **Role of the funding source**, **Contributors**, **Declaration of interests**, **Data sharing**, **data-access-and-verification statement**, informed-consent statement.
4. Fill all placeholders: `XXX` word counts, `+3.XX%/+4.XX%`, `[]` code URL, `Fig. SX`, supplementary captions S5/S7–S10.
5. Fix Table 1 arithmetic (item 4) and the percentage/scale mismatches (items 5–7).
6. Reconcile Methods↔Results↔Discussion on the predictor set and the reported associations (items 15–17), and add CIs (item 20).
7. Reorder panels/text (items 35–37), then run a single house-style pass: midline decimal points, superscript Vancouver references, six-author reference format, p-value precision, n-with-every-%, spacing.
8. Resolve and delete all 43 comments; prepare STROBE checklist, protocol, forms, and the single-PDF supplement.

---

# Appendix — Figure/panel reordering plan (items 35–37)

Laura's comment: *"on doit citer les figures (et panels) dans l'ordre … J'aime bien l'ordre du texte, donc je proposerais plutôt de changer l'ordre des figures."*

Below, each figure is treated separately, because the right answer is **not** the same for all three.

## Figure 4 — change the figure (recommended)

**Current layout** (`scripts/paper_figures/generate_figure_4_counterfactual_construction.py`, lines 589–636):

| Row | Panels | Content |
|---|---|---|
| 0 (full width) | **A** | Episode-level counterfactual construction across δ |
| 1 (full width) | **B** | 7-day observed vs counterfactual trace |
| 2 | **C** \| **D** | Median daily (AGP) profile, A174 \| A224 |
| 3 | **E** \| **F** | ΔTIR vs δ \| Δmedian glucose vs δ |

**Citation order in the Results:** A,B → C–F → E,F → E,F → E,F → **C,D**. ✗

**Why the figure is the thing to change here — not the text.** The order "aggregate summary metrics first, local median-profile effect second" is used in *two other places already*:
- **Methods**: §"Contribution of SLH episodes to summary metrics" presents the aggregate metrics first, then Δm(t)/tmax.
- **Figure 5** (`generate_figure_5_counterfactual_distribution.py`): A = ΔTIR, B = Δmedian glucose, **then** C = max profile difference, D = time of max.

Figure 4 is the **only** display item that inverts it. Swapping its two bottom rows makes Methods, Figure 4, Figure 5 and both Results paragraphs agree, and requires **no text reordering at all** — only the panel letters in the two paragraphs and the caption.

**Proposed layout — swap rows 2 and 3:**

| Row | Panels | Content | Was |
|---|---|---|---|
| 0 | **A** | Counterfactual construction across δ | A |
| 1 | **B** | 7-day observed vs counterfactual | B |
| 2 | **C** \| **D** | ΔTIR vs δ \| Δmedian glucose vs δ | E \| F |
| 3 | **E** \| **F** | Median daily profile A174 \| A224 | C \| D |

New citation order: A,B → C–F → C,D → E,F ✓

**Code change:** move the `ax_c`/`ax_d` AGP block (lines 607–620) and the `ax_e`/`ax_f` sensitivity block (lines 622–635) past each other, i.e. plot the sensitivity metrics into `gs[2, 0]`/`gs[2, 1]` labelled C/D and the AGP comparison into `gs[3, 0]`/`gs[3, 1]` labelled E/F. Also swap the height ratios, since the short row must follow the content:

```python
height_ratios=[1.2, 1.2, 1.0, 1.2]   # was [1.2, 1.2, 1.2, 1.0]
```

Keep each `show_legend=` flag attached to its own plot call (AGP legend on the right-hand panel, both sensitivity panels keep theirs).

**Counter-argument to note:** the *current* order has a nice visual "zoom out" logic — one episode (A) → one week (B) → the day profile (C,D) → the δ summary (E,F). If you prefer to keep that, the equivalent fix is to **swap the two Results paragraphs** instead (median-profile paragraph before the aggregate-metrics paragraph), which also yields A,B → C–F → C,D → E,F. But then Figure 4 and Figure 5 tell their stories in opposite orders, so you would want to reorder Figure 5 to match as well. Changing Figure 4 is the smaller, more consistent move.

## Figure 2 — RESOLVED by deleting the two redundant panels (implemented)

Supervisor's objection: the top row is redundant, and rows 3–4 show the same information. Both are correct, and confirmed in the code:

- **Panel A vs panel B.** `plot_simplified_waterfall` already prints the exclusion delta above every bar (`f"−{delta:,}"`, bold, colour-coded by destination category, with the Night call-out). `plot_funnel_exclusions` then plotted *those exact five numbers* as bars. Panel B was a bar chart of annotations already on panel A.
- **Panel E vs panel F.** `plot_category_composition_vertical` built one `counts` frame, one `patient_order`, and drew it twice — `props = counts.div(totals)` on top, `counts` below, same colours, same x-axis. E was F normalised.

**The two redundant panels (B and F) are exactly the two the Results text never cites.** Deleting them leaves four panels that fall into the text's citation order by themselves:

| Text cites | Was | Now |
|---|---|---|
| waterfall, 1st | 2A | **2A** |
| SLH count per patient, 2nd | 2C | **2B** |
| per-patient proportions, 3rd | 2E | **2C** |
| onset by time of day, 4th | 2D | **2D** |

Citation order becomes A → B → C → D, sequential, with **no sentence moved**. The only manuscript edit is two find-replaces in the citation letters — `(Fig. 2C)` → `(Fig. 2B)` and `(Fig. 2E)` → `(Fig. 2C)`; 2A and 2D are unchanged.

**Implemented layout** (`generate_figure_2_classification_summary.py`), 10.5 × 12.3 in (2.6 in shorter than before):

| Row | Panels | Content |
|---|---|---|
| 1 | **A** (2.0) \| **B** (1.15) | Classification waterfall \| SLH episodes per patient |
| 2 (full width) | **C** | Per-patient composition (proportions) + total episodes overlaid on a secondary axis |
| 3 (full width) | **D** | Episode onset by time of day |

The one thing panel F carried that proportions destroy — each patient's *total* episode count — is preserved as a grey line on a secondary axis in panel C. It also doubles as a legend for the x ordering, which the reader previously had to take on trust from the axis label.

**Replacement caption:**

> **Figure 2: Classification of hyperglycemic episodes.** (A) Classification waterfall: episodes surviving each successive criterion (duration, context, data completeness, daytime onset), with the number excluded at each step annotated and coloured by the category those episodes join. (B) Distribution of the number of daytime SLH episodes per patient. (C) Per-patient composition of hyperglycemic episodes by category, as a proportion of each patient's total; patients are ordered by total episode count (ascending) and that total is overlaid (grey line, right axis). (D) Episode onset by time of day for each category; shaded bands mark the night window (22:00–06:00).

**Optional follow-up:** panel B still uses one bin per integer count (61 bins for 135 patients), which reads as a comb. Bins of width 5 would suit the sample size — a one-line change to `plot_slh_count_distribution`.

## Figure 3 — not an ordering problem; the panels are simply never discussed

The text cites `(Fig. 3)` and then `(Fig. 3G)`. Panels **A–F are never cited at all**, so there is nothing to reorder — the first citation is the whole figure, which is legitimate.

The real problem is that six panels (AGP, recurrence heatmaps, 7-day traces) are displayed and never described in the Results. **Recommended fix:** add one sentence after "…patient A224 with only 19 SLH episodes", walking the panels in order — the ambulatory glucose profiles (A, B), the recurrence patterns by classification (C, D) and the 7-day traces (E, F) — before the existing sentence that cites the metrics table (G).

Promoting the summary table from G to A would also fix the citation order, but a metrics table as the opening panel is unusual and would relabel every other panel; the added sentence is better science writing anyway.

## Repo hygiene — which script makes which figure? (relevant to code availability)

While mapping panels I could not unambiguously match scripts to paper figures, which matters because the code will be published:

- `generate_figure_3_patient_comparison.py` produces **A** construction, **B/C** 7-day, **D/E** AGP, **F/G** δ-sensitivity — i.e. a *counterfactual* figure that matches neither the paper's Figure 3 caption (AGP / heatmaps / traces / table) nor Figure 4. It looks like a superseded draft of Figure 4.
- `generate_figure4.py` (`OUTPUT_NAME = "Figure4_TIR_comprehensive_analysis"`, panels A/B/C, D, E/F) is a **third** file claiming Figure 4.
- Two files claim Figure 5: `generate_figure_5_counterfactual_distribution.py` (panels A–E) matches the paper's Figure 5 caption exactly; `generate_figure_5_slh_determinants.py` (panels A–G) does not and is presumably supplementary.
- The paper's **Figure 3 has no generator script** — its panels come from `scripts/cgm_agp_recurrence.py` plus trace plots and a table, assembled by hand (`Figures.pptx`). Any relabelling of Figure 3 is therefore a PowerPoint edit, not a code edit.
- Possible caption/figure mismatch: Figure 5's caption describes panel **E** as "Spearman correlation **and mutual information**", but `ax_e` in `generate_figure_5_counterfactual_distribution.py` is a correlation matrix only; the MI panel lives in `generate_figure_5_slh_determinants.py`. Worth checking whether the published panel E is a composite.

Before submission, rename the scripts to match the final figure numbers, delete or move the superseded ones to an `archive/` folder, and add a short README mapping each paper figure to the script (or PPT slide) that produces it.


---

# Appendix — Figure 4 caption (Lancet house style)

Written against the figure **as the code currently produces it** (panels C/D = ambulatory glucose profiles, E/F = sensitivity across δ). The row swap proposed above has not been applied; if it is, exchange the "(C, D)" and "(E, F)" sentences and relabel — nothing else changes.

## Recommended caption

> **Figure 4: *In silico* removal of short-lived hyperglycemia episodes in two representative patients**
>
> Counterfactual glucose trajectories were constructed for two patients with similar median glucose but contrasting short-lived hyperglycemia (SLH) frequency over the 45-day analysis window (patient A174, 59 daytime SLH episodes; patient A224, 19). (A) Construction of the counterfactual signal for a single SLH episode in patient A224, shown with 3 h of context on either side; the counterfactual is a linear interpolation between the lowest glucose readings within a contextual window of width δ before episode onset and after episode offset, drawn here for each δ tested (0, 15, 30, 45, and 60 min). (B) Observed glucose G(t) and counterfactual glucose C(t) over the first seven days of the analysis window in patient A174, at the main-analysis settings (δ=30 min; 60 min centered rolling-mean smoothing). (C, D) Ambulatory glucose profile before and after SLH removal (δ=30 min) in patient A174 (C) and patient A224 (D); lines show the median and shaded bands the IQR of glucose at each time of day across the 45-day window, and dashed horizontal lines mark the hyperglycemia (180 mg/dL) and hypoglycemia (70 mg/dL) thresholds. (E, F) Change in daytime (06:00–22:00) time in range 70–180 mg/dL (E) and daytime median glucose (F) after SLH removal in each patient, across all five contextual window widths; the shaded band marks δ=30 min, the width used in the main analysis. Glucose concentrations are reported in mg/dL rather than SI units (mmol/L). IQR=interquartile range. SLH=short-lived hyperglycemia.


## Factual errors corrected from the current caption

| Current caption | Actual figure |
|---|---|
| "(B) Observed and counterfactual glucose traces for **a representative day**" | `DAYS_PER_ROW = 7` — panel B is a **seven-day** trace |
| Panel A patient unstated (reads as A174 by proximity) | `pid0 = PATIENT_IDS[1]` — the episode in panel A comes from **A224** |
| Panel B patient unstated | `PATIENT_IDS[0]` — **A174** only |
| "(C, D) **Median** daily glucose profiles" | Panels also draw the **IQR band** (q25–q75) and the 180/70 mg/dL threshold lines |
| δ value for panels B/C/D unstated | All three use `SELECTED_OFFSET_MIN = 30` |
| Smoothing unstated | `KERNEL_WIDTH_MIN = 60`, centered rolling mean |

## House-style points applied

- *In silico* — italic, no hyphen (the manuscript has "In-silico" in the caption, the Results heading, and the Results text; fix all three).
- Abbreviations expanded in the legend so it stands alone: SLH and IQR defined; CGM, TIR and AGP written out rather than abbreviated.
- `δ=30 min`, `n=135` — no spaces around the equals sign.
- Numbers one to ten in words unless a unit of measurement: "two patients", "seven days" — but "3 h", "59 … episodes", "180 mg/dL".
- Serial comma before the final "and": "0, 15, 30, 45, and 60 min".
- Non-SI glucose units declared, matching the Table 1–3 footnotes.
- Ranges use en rules: 70–180, 06:00–22:00.
- Spelling is US ("centered", "hyperglycemia") to match the rest of the manuscript. *Lancet* house style is British English — convert the whole document at once rather than this caption alone (see §2).


---

# Appendix — Figure 5 caption (Lancet house style)

Written against `generate_figure_5_counterfactual_distribution.py`, whose panels A–E match the manuscript's Figure 5.

## Recommended caption

> **Figure 5: Cohort-wide effect of *in silico* removal of short-lived hyperglycemia episodes, and correlates of SLH frequency**
>
> Counterfactual removal of daytime short-lived hyperglycemia (SLH) episodes across the whole cohort (δ=30 min). (A, B) Distribution across patients (n=135) of the change in time in range 70–180 mg/dL (A) and in median glucose (B) after SLH removal, computed over the full 24 h (solid bars) and over the daytime period only (06:00–22:00; hatched bars). (C) Distribution of the largest absolute difference between each patient's observed and counterfactual median daily glucose profile; the dashed vertical line marks the cohort median. (D) Distribution of the time of day at which that largest difference occurred, in hourly bins. In (A)–(D), coloured vertical lines mark the two representative patients shown in figures 3 and 4 (A174 and A224). (E) Lower triangle of the Spearman correlation matrix between baseline CGM metrics, clinical phenotype variables, glycemic biomarkers, and the number of SLH episodes (last row and column, dashed line); dotted lines separate the three variable domains and asterisks denote p<0·05. (F) Mutual information between each of the same baseline variables and the number of SLH episodes, ranked in descending order; bars are coloured by domain as in (E). Panels (E) and (F) are restricted to the 132 patients with complete phenotype and biomarker data. Glucose concentrations are reported in mg/dL rather than SI units (mmol/L). CGM=continuous glucose monitoring. SLH=short-lived hyperglycemia.

## ✅ Panel F (mutual information) added

`generate_figure_5_counterfactual_distribution.py` now imports `compute_mutual_information` from the determinants module and draws it as panel F beside the correlation matrix in row 3 (`width_ratios=[2.35, 1.0]`, so the matrix keeps most of the width). Bars are sorted by MI rather than kept in the matrix's variable order — the ranking is the point of the panel — with the domain carried by bar colour, matching panel E's axis labels. The redundant domain legend from the determinants version is dropped, since panel E labels the same three domains in the same colours immediately to the left.

## 🔴 The mutual-information values in the Results do not match the pipeline

Regenerating the panel produced numbers that differ substantially from those quoted in the Results:

| Variable | Results text | Pipeline (n=132) |
|---|---|---|
| TIR 70–180 | 0·349 | 0·351 |
| Glucose SD | 0·237 | 0·243 |
| Glucose CV | 0·134 | 0·130 |
| GTAA1C | 0·163 | 0·145 |
| TAR >180 | 0·144 | **0·100** |
| Mean glucose | 0·092 | **0·121** |
| Median glucose | 0·042 | **0·105** |
| TBR <70 | 0·011 | **0·000** |
| IDAA1C | 0·164 | **0·050** |

TIR, SD and CV agree to rounding, but TAR, mean glucose, median glucose, TBR and above all **IDAA1C** do not. The full ranking is now TIR (0·351), Glucose SD (0·243), **diabetes duration (0·180)**, HbA1c (0·148), GTAA1C (0·145), CV (0·130), mean (0·121), median (0·105), TAR (0·100), BMI z-score (0·100), IDAA1C (0·050), TIDD (0·041), age (0·020), TBR (0·000).

This breaks a specific sentence in the Results: *"IDAA1C and GTAA1C ranked third and fourth by mutual information (MI = 0·164 and 0·163), ahead of TAR (MI = 0·144)."* On the current pipeline IDAA1C ranks **eleventh**, and third and fourth are diabetes duration and HbA1c. `mutual_info_regression` is seeded (`random_state=42`), so this is a data difference, not estimator noise — most likely the paragraph was written from an earlier cohort or filtering step. **Regenerate that whole paragraph from the current output before submission.**

## 🟠 Panels E and F use 132 patients, not 135

`build_dataset(apply_filter=True)` prints "Merged, complete-case dataset: 132 patients": three of the 135 lack complete phenotype or biomarker data and are dropped from both the correlation matrix and the mutual-information analysis. The current caption says "n = 135 patients" for the whole figure. The recommended caption above states the two denominators separately; the Results should do the same wherever ρ and MI values are quoted.

## Other corrections from the current caption

| Current caption | Actual figure |
|---|---|
| "(A) Change in daytime **and standard** time in range" / "(B) …daytime **and overall** median glucose" | Correct in substance, but "standard" and "overall" name the same thing (full 24 h) in adjacent sentences — use one term. The panels overlay the two distributions as solid vs hatched bars, which the caption should say. |
| Patient marker lines not mentioned | (A)–(D) all draw coloured vertical lines for A174 and A224 |
| Panel C median line not mentioned | A dashed vertical line marks the cohort median |
| Panel D binning not stated | Hourly bins |
| Panel E structure not described | Lower triangle only; dotted lines = domain boundaries; dashed line = the SLH-count row/column; `*` = p<0·05 |
| "frequency" | replaced with "frequency", per the same change made in Figure 4 |

House-style points are the same as for the Figure 4 caption: *in silico* italic and unhyphenated, midline decimal in p<0·05, `n=135`/`δ=30 min` without spaces, en rules for ranges, serial comma, numbers one to ten in words unless a unit, non-SI glucose units declared, and abbreviations expanded so the legend stands alone.
