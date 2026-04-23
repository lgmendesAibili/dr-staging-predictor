# DR Baseline Staging Predictor

Interactive demonstration of an explainable logistic regression model that
classifies Diabetic Retinopathy (DR) baseline severity from 5 clinical features.

> Mendes L et al. *"Baseline Staging of Diabetic Retinopathy (ETDRS 35 vs 43-47)
> from Static, Dynamic, and Systemic Features."* ARVO 2026 submission
> (AIBILI / University of Coimbra).

## Binary Task

| Class | ETDRS grade | Clinical label |
|---|---|---|
| `0` | 35 | Mild / moderate NPDR |
| `1` | 43 – 47 | Moderate-to-severe NPDR |

NPDR = Non-Proliferative Diabetic Retinopathy. The target is **baseline
severity** — the ETDRS grade at enrolment — not a progression prediction.

## Features

- **5 clinical feature inputs** with real-time validation against training
  data boundaries
- **Binary classification** — ETDRS 35 vs ETDRS 43-47
- **SHAP waterfall plot** — per-feature contribution to the prediction
- **SHAP decision plot** — cumulative path from base value to final output
- **Privacy-safe** — no patient data required at runtime; SHAP uses a
  synthetic background derived from the fitted StandardScaler

## The 5 Features

| Input label | Raw column | Type | Source |
|---|---|---|---|
| MA Turnover (6mo) | `MA_turnover` | Dynamic (slope) | Longitudinal microaneurysm turnover over 6 months |
| RT Outer Ring Temporal (µm) | `RT_OutR_Temp` | Static | OCT retinal thickness, ETDRS outer ring, temporal quadrant |
| RT Outer Ring Inferior (µm) | `RT_OutR_Inf` | Static | OCT retinal thickness, ETDRS outer ring, inferior quadrant |
| VD/PD Inner Ring SCP | `[VD/PD]_InR_SCP_SS6` | Static | OCTA vessel-density / perfusion-density ratio (inner ring, superficial plexus, 6×6 mm scan) |
| Diabetes Duration (years) | `DiabetesYears` | Systemic | Years since diabetes diagnosis |

## Live Demo

Deployed on Streamlit Community Cloud from
[`lgmendesAibili/dr-staging-predictor`](https://github.com/lgmendesAibili/dr-staging-predictor).
Push to `main` to redeploy.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model

- **Algorithm:** Logistic Regression (scikit-learn)
- **Training:** SMOTE-balanced (k=5, ratio=1.0) with StandardScaler preprocessing
- **Data:** 161 eyes from the CHART clinical dataset (85 at ETDRS 43-47, 76 at ETDRS 35)
- **Features:** 5 FFS-selected features (static + dynamic + systemic)
- **Performance (50-fold repeated grouped CV):** AUC = 0.84 ± 0.06, Balanced Accuracy = 0.76 ± 0.06
- **Performance (training set after SMOTE):** AUC = 0.84, Balanced Accuracy = 0.76

Full training-set and held-out metrics are in `training_metrics_*.md` and
`evaluation_*.md`.

### Model Coefficients

| Feature | Coefficient |
|---|---|
| MA_turnover | +1.462 |
| RT_OutR_Temp | +0.529 |
| RT_OutR_Inf | +0.544 |
| [VD/PD]_InR_SCP_SS6 | −0.403 |
| DiabetesYears | −0.358 |
| **Intercept** | +0.177 |

## Project Structure

```
├── app.py                                            # Streamlit application (privacy-safe, no patient data)
├── requirements.txt                                  # Python dependencies
├── logistic_model_ARVO2026_6mo_StatDynSys.pkl        # Trained logistic regression
├── scaler_ARVO2026_6mo_StatDynSys.pkl                # StandardScaler fitted on SMOTE-balanced data
├── boundaries_ARVO2026_6mo_StatDynSys.pkl            # Per-feature min/max/mean/std/median (original data)
├── training_metrics_*.md                             # Training-set performance report
├── evaluation_*_all.md                               # Evaluation on 161 samples (incl. imputed)
├── evaluation_*_nonmissing.md                        # Evaluation on 124 non-missing samples
├── .streamlit/
│   └── config.toml                                   # Streamlit theme + viewer toolbar mode
└── lessons_learned.md                                # Deployment / infra incidents and fixes
```

## Dependencies

- streamlit
- numpy, pandas
- scikit-learn, joblib
- shap
- matplotlib

## Disclaimer

**Not for clinical use.** This application is a research demonstration
intended solely to illustrate the algorithm described in the accompanying
submission. It is not a certified or validated medical device and must not
be used for clinical decision-making.

## License

Research code — part of ongoing DR staging/progression research at AIBILI
(Association for Innovation and Biomedical Research on Light and Image) in
collaboration with the University of Coimbra.
