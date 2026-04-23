# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app demonstrating a single explainable Logistic Regression model
that performs **baseline Diabetic Retinopathy (DR) staging** — classifying eyes
into ETDRS 35 (mild / moderate NPDR) vs ETDRS 43-47 (moderate-to-severe NPDR)
— from 5 features:

- `MA_turnover` (dynamic, longitudinal microaneurysm turnover over 6 months)
- `RT_OutR_Temp` (static, OCT retinal thickness — outer ring temporal)
- `RT_OutR_Inf` (static, OCT retinal thickness — outer ring inferior)
- `[VD/PD]_InR_SCP_SS6` (static, OCTA vessel/perfusion density ratio)
- `DiabetesYears` (systemic, duration of diabetes)

Predictions are explained with SHAP waterfall and decision plots. Accompanies
the ARVO 2026 submission by Mendes L et al. (AIBILI / University of Coimbra).

**This is a research demonstration, not a clinical tool.**

## Relationship to the Parent Project

This app is the deployment surface of the `stagingAndProgressionDR` Julia
research repo (at `/home/lgmendes/gitProjects/stagingAndProgressionDR`), where
the model was trained. Artifacts here (`logistic_model_*.pkl`, `scaler_*.pkl`,
`boundaries_*.pkl`) were produced by
`scripts/export_ARVO2026_6mo_StatDynSys.py` in that repo and copied in.

When the model changes upstream (new features, new scenario, re-trained
coefficients), re-export the `.pkl` files from the parent repo and replace
them here.

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

- **`app.py`** — Privacy-safe single-model version deployed to Streamlit Cloud.
  SHAP uses a synthetic zero-vector background in scaled space (equivalent to
  the training mean) instead of patient data.
- **`MODEL_CONFIGS`** dict in `app.py` — designed to host multiple scenarios.
  Currently holds only `arvo_6mo`. Adding a model means adding a config entry
  and the corresponding `.pkl` files; the code already supports multiple
  entries.
- **Model artifacts** (committed as `.pkl` files):
  - `logistic_model_ARVO2026_6mo_StatDynSys.pkl`
  - `scaler_ARVO2026_6mo_StatDynSys.pkl`
  - `boundaries_ARVO2026_6mo_StatDynSys.pkl`

### Prediction Pipeline

1. User enters 5 raw feature values
2. Values validated against `boundaries.pkl` (min/max from original, pre-SMOTE data)
3. Raw values scaled via `scaler.transform()` (StandardScaler, fit on SMOTE-balanced data)
4. Scaled values passed to `model.predict()` → binary class (0=ETDRS 35, 1=ETDRS 43-47)
5. `model.decision_function()` provides the raw log-odds score shown under the result card
6. SHAP `LinearExplainer` computes feature contributions on the scaled input
7. Waterfall and decision plots rendered at 200 DPI via matplotlib

### Key Design Decisions

- **No `predict_proba`** — the UI shows only the binary class and the raw
  log-odds decision score. No calibrated probabilities are claimed.
- **SHAP background** — `np.zeros((1, n_features))` represents the training
  mean in scaled space; produces the correct base value without shipping
  patient data.
- **Widget input ranges** — clamped to training `[min, max]` (no extrapolation).
- **Integer inputs** — `RT_OutR_Temp`, `RT_OutR_Inf`, `DiabetesYears` use
  integer number_inputs (step=1). `MA_turnover` and `[VD/PD]_InR_SCP_SS6` are
  floats with step = SD/10.
- **SHAP waterfall shows raw values** — the `data` field in
  `shap.Explanation` receives the user's unscaled input so the plot shows
  clinical values, not z-scores.
- **Display vs raw names** — `FEATURE_LABELS` maps raw column names
  (`MA_turnover`) to human-readable labels (`MA Turnover (6mo)`). The scaler
  and model still operate on the raw names via positional order; display
  names are only used in the UI.
- **Single model, extensible layout** — `MODEL_CONFIGS` is already a dict
  even though only `arvo_6mo` is populated. To add the 12-month variant or
  a London progression model, add a new config key and the related `.pkl`
  files, then extend `main()` (or iterate over `MODEL_CONFIGS`) to render
  more sections.

## Adding a New Scenario

To add a 12-month variant or a second model:

1. Export artifacts from the parent Julia repo
   (`stagingAndProgressionDR/scripts/export_*.py`)
2. Copy the `.pkl` files here
3. Add a new entry to `MODEL_CONFIGS` in `app.py` — mimic the `arvo_6mo`
   entry
4. Either (a) switch between scenarios via a selectbox, or (b) stack both
   in the UI (keratoconus-predictor style).

## Privacy

Patient data must **never** be committed. Git-ignored:

- `localData/` — any local patient data
- `model-evaluation/` — if you ever add an evaluation folder with CSVs
- `X_train_*.pkl` — training feature matrices (NOT needed by the privacy-safe app)
- `training_metrics.md` — generated reports

The `app.py` here is designed to run without any patient data. Only the
**model**, **scaler**, and **boundary statistics** (aggregated summary
statistics, not raw data) are shipped.

## Deployment

Intended for Streamlit Community Cloud. Pushing to `main` triggers
automatic redeployment.

### Streamlit Cloud Requirements

`requirements.txt` must include **all** transitive dependencies used by the
app, including ones that appear indirectly (e.g. `pandas`, `ipython`,
`sparklines`). Streamlit Cloud installs only what is listed — missing
transitive dependencies can cause the deployment to hang silently in
"preparing" without an error. See the keratoconus-predictor sibling repo's
`lessons_learned.md` for the original incident that informed this list.

### Git Authentication (GitHub)

GitHub no longer supports password authentication over HTTPS. Use:
```bash
gh auth login  # GitHub.com → HTTPS → web browser → one-time code
```

## Domain Glossary

| Abbrev. | Meaning |
|---|---|
| DR | Diabetic Retinopathy |
| NPDR | Non-Proliferative DR |
| ETDRS | Early Treatment Diabetic Retinopathy Study severity scale |
| OCT | Optical Coherence Tomography |
| OCTA | OCT Angiography |
| RT | Retinal Thickness |
| MA | Microaneurysm |
| SCP / DCP | Superficial / Deep Capillary Plexus |
| VD / PD | Vessel Density / Perfusion Density |
| FAZ | Foveal Avascular Zone |
| SS 6×6 | 6×6 mm OCTA acquisition |
| FFS | Forward Feature Selection |
| SMOTE | Synthetic Minority Over-sampling Technique |
