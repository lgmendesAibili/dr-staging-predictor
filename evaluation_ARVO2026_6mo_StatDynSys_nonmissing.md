# Evaluation Report — ARVO 2026 6mo StatDynSys — Non-Missing Only

*Generated: 2026-04-10 17:57:19*

**Task:** ETDRS 35 (stable) vs. ETDRS 43-47 (progression) baseline staging
**Features:** MA_turnover, RT_OutR_Temp, RT_OutR_Inf, [VD/PD]_InR_SCP_SS6, DiabetesYears
**Time window:** 6-month slopes

## Dataset Summary

| Samples | ETDRS 43-47 (Pos) | ETDRS 35 (Neg) |
|---|---|---|
| 124 | 66 | 58 |

## Metrics

| Metric | Value |
|---|---|
| Accuracy | 0.7742 |
| Balanced Accuracy | 0.7795 |
| Precision | 0.8519 |
| Recall (Sensitivity) | 0.6970 |
| Specificity | 0.8621 |
| F1 Score | 0.7667 |
| AUC-ROC | 0.8584 |
| Log Loss | 0.4739 |

## Confusion Matrix

| | Predicted Neg (35) | Predicted Pos (43-47) |
|---|---|---|
| **Actual Neg (35)** | 50 | 8 |
| **Actual Pos (43-47)** | 20 | 46 |

> **Note:** These metrics are evaluated on the original (unbalanced) data
> using a pre-trained model. The model was trained on SMOTE-balanced data.
> Cross-validation performance (50-fold): AUC = 0.84 +/- 0.06, bAcc = 0.76 +/- 0.06.
