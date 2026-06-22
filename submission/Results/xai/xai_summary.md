# Output-Level XAI Summary

This report explains the final validation outputs using only submission-local files.
It is not a SHAP report and does not require fitted model objects.

## Dataset Context

- Validation games: `20,000`.
- Validation positive rate: `0.4964`.
- Final profile: `boosting`.

## Key Model Metrics

| Task | Primary metric | Value |
|---|---:|---:|
| White win before | ROC-AUC | 0.578805 |
| White win after 3 | ROC-AUC | 0.578667 |
| White win after 10 | ROC-AUC | 0.622593 |
| Elo after 10 | Avg MAE | 29.309 |
| Elo expected-score baseline | ROC-AUC | 0.578497 |

## Calibration And Lift

- Top 10% predicted after-10 White-win rate: `0.7170`.
- Bottom 10% predicted after-10 White-win rate: `0.2825`.
- Top-minus-bottom 10% lift: `0.4345`.
- Top 20% predicted after-10 White-win rate: `0.6620`.
- Bottom 20% predicted after-10 White-win rate: `0.3463`.

The after-10 model is useful for ranking games: the highest-probability bucket has a much higher actual White-win rate than the lowest bucket.

## Calibration Bins

|   bin |   count |   mean_predicted_prob |   actual_white_win_rate |   calibration_gap_pred_minus_actual |
|------:|--------:|----------------------:|------------------------:|------------------------------------:|
|     1 |    2000 |              0.275829 |                  0.2825 |                        -0.00667127  |
|     2 |    2000 |              0.388055 |                  0.41   |                        -0.0219447   |
|     3 |    2000 |              0.429363 |                  0.4475 |                        -0.0181369   |
|     4 |    2000 |              0.458072 |                  0.459  |                        -0.000928231 |
|     5 |    2000 |              0.481453 |                  0.4805 |                         0.000952993 |
|     6 |    2000 |              0.503539 |                  0.4915 |                         0.0120391   |
|     7 |    2000 |              0.526503 |                  0.5195 |                         0.00700257  |
|     8 |    2000 |              0.554357 |                  0.5495 |                         0.00485662  |
|     9 |    2000 |              0.597572 |                  0.607  |                        -0.00942751  |
|    10 |    2000 |              0.720125 |                  0.717  |                         0.00312485  |

## Elo Error By Rating Band

| elo_band   |   count |   white_mae |   black_mae |   avg_mae |   white_rmse |   black_rmse |   avg_error_p50 |   avg_error_p90 |   avg_error_p99 |
|:-----------|--------:|------------:|------------:|----------:|-------------:|-------------:|----------------:|----------------:|----------------:|
| <1000      |     975 |     70.4958 |     74.5081 |   72.5019 |     171.988  |     176.043  |        17.0249  |        163.307  |         746.437 |
| 1000-1399  |    4212 |     37.8157 |     37.9444 |   37.88   |      94.4503 |      95.3764 |        10.9932  |         71.0607 |         459.434 |
| 1400-1799  |    7572 |     23.6266 |     23.443  |   23.5348 |      52.5991 |      51.2247 |        10.6686  |         49.7101 |         230.282 |
| 1800-2199  |    6302 |     21.6199 |     22.0901 |   21.855  |      70.3074 |      72.3274 |         6.15815 |         34.4552 |         367.378 |
| 2200-2599  |     900 |     39.8465 |     36.3671 |   38.1068 |     120.276  |     116.278  |        10.5744  |         61.5476 |         705.351 |
| 2600+      |      39 |    148.86   |    143.656  |  146.258  |     285.556  |     259.088  |        79.1101  |        204.293  |        1114.07  |

## Interpretation

- Pre-game prediction is dominated by Elo difference and stays close to the Elo expected-score baseline.
- After-10 prediction gains signal from observed board structure and clock/time-pressure features.
- Elo prediction is very strong because causal player-history features are highly predictive for repeat players.
- For completely unseen players, the Elo model should be expected to degrade toward weaker history-free behavior.
