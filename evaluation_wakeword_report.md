# Quantitative Wake-Word Evaluation Report (`bot clip that`)

- **Calibrated Decision Threshold:** `0.750`
- **Total Evaluated Clips:** 85 (41 positive, 44 negative)
- **Streaming Mean Chunk Latency (100 ms input):** `1.248 ms` (P95: `2.089 ms`, RTF: `0.0125x`)
- **Continuous Negative Stream False Alarm Rate (Group Negatives):** `0` triggers over `196.0s` (`0.00 FP/hr`)
- **E5 Open-Set Held-Out False Alarm Rate (Unseen Speech/Music/Noise):** `0` triggers over `3.21 hours` (**`0.00 FP/hr`**, window FAR `0.000%`)

## 1. Baseline & Experimental Conditions (PDF Section 12–13)

| Experiment | Condition | Accuracy | Precision | Recall | F1 | FRR | FAR | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Clean Speech | 100.0% | 100.0% | 100.0% | 1.000 | 0.0% | 0.0% | 1.000 |
| E2 (Noise) | SNR_20dB | 98.8% | 100.0% | 97.6% | 0.988 | 2.4% | 0.0% | 1.000 |
| E2 (Noise) | SNR_10dB | 94.1% | 100.0% | 87.8% | 0.935 | 12.2% | 0.0% | 0.996 |
| E2 (Noise) | SNR_5dB | 78.8% | 100.0% | 56.1% | 0.719 | 43.9% | 0.0% | 0.952 |
| E3 (Acoustic) | Quiet_0.35x | 100.0% | 100.0% | 100.0% | 1.000 | 0.0% | 0.0% | 1.000 |
| E3 (Acoustic) | Loud_1.40x | 100.0% | 100.0% | 100.0% | 1.000 | 0.0% | 0.0% | 1.000 |
| E3 (Acoustic) | Slow_0.88x | 97.7% | 100.0% | 95.1% | 0.975 | 4.9% | 0.0% | 0.999 |
| E3 (Acoustic) | Fast_1.12x | 100.0% | 100.0% | 100.0% | 1.000 | 0.0% | 0.0% | 1.000 |

### Baseline Confusion Matrix

| | Predicted Positive | Predicted Negative |
|---|---:|---:|
| **Actual Positive (`bot clip that`)** | 41 (TP) | 0 (FN) |
| **Actual Negative (Speech / Noise)** | 0 (FP) | 44 (TN) |

## 3. Threshold Sensitivity Sweep (Baseline)

| Threshold | Precision | Recall | F1 | FRR | FAR |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 100.0% | 100.0% | 1.000 | 0.0% | 0.0% |
| 0.40 | 100.0% | 100.0% | 1.000 | 0.0% | 0.0% |
| 0.50 | 100.0% | 100.0% | 1.000 | 0.0% | 0.0% |
| 0.60 | 100.0% | 100.0% | 1.000 | 0.0% | 0.0% |
| 0.70 | 100.0% | 100.0% | 1.000 | 0.0% | 0.0% |
| 0.80 | 100.0% | 100.0% | 1.000 | 0.0% | 0.0% |

## 4. Error Analysis (Baseline Misclassifications)

Zero misclassifications on the baseline clean evaluation set.
