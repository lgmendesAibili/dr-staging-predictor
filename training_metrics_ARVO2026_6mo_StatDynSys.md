# Training Set Performance Report — ARVO 2026 6mo StatDynSys

*Generated: 2026-04-10 18:08:38*

**Task:** ETDRS 35 (stable) vs. ETDRS 43-47 (progression) baseline staging
**Features:** MA_turnover, RT_OutR_Temp, RT_OutR_Inf, [VD/PD]_InR_SCP_SS6, DiabetesYears
**Time window:** 6-month slopes

## Dataset Summary

| | Samples | ETDRS 43-47 (Pos) | ETDRS 35 (Neg) |
|---|---|---|---|
| Original | 161 | 85 | 76 |
| After SMOTE | 170 | 85 | 85 |

## Model Coefficients

| Feature | Coefficient |
|---|---|
| MA_turnover | +1.4616 |
| RT_OutR_Temp | +0.5292 |
| RT_OutR_Inf | +0.5439 |
| [VD/PD]_InR_SCP_SS6 | -0.4028 |
| DiabetesYears | -0.3576 |
| **Intercept** | +0.1767 |

## Metrics (on training data)

| Metric | Value |
|---|---|
| Accuracy | 0.7588 |
| Balanced Accuracy | 0.7588 |
| Precision | 0.7750 |
| Recall (Sensitivity) | 0.7294 |
| Specificity | 0.7882 |
| F1 Score | 0.7515 |
| AUC-ROC | 0.8364 |
| Log Loss | 0.4960 |

## Confusion Matrix

| | Predicted Neg (35) | Predicted Pos (43-47) |
|---|---|---|
| **Actual Neg (35)** | 67 | 18 |
| **Actual Pos (43-47)** | 23 | 62 |

> **Note:** These metrics are on the training set (after SMOTE) and are
> intended as a quick sanity check, not as an estimate of generalization.
> Cross-validation performance (50-fold): AUC = 0.84 +/- 0.06, bAcc = 0.76 +/- 0.06.
