# Output-Level XAI Summary

This report explains the final validation outputs using only submission-local files.
It is not a SHAP report and does not require fitted model objects.

## Scope and Limitations

This is output-level XAI, not SHAP, LIME, or exact feature-attribution XAI. It analyzes the behavior of saved validation predictions rather than introspecting fitted model internals.

The report can analyze probability calibration, lift/ranking behavior, Elo error segments, representative success/failure examples, and validation-level behavior. It cannot provide exact per-feature attribution because the submitted package does not require fitted model objects, SHAP/LIME dependencies, Stockfish, raw PGN files, or model refitting.

## Dataset Context

- Validation games: `20,000`.
- Validation positive rate: `0.4964`.
- Final profile: `report_best`.

## Key Model Metrics

| Task | Primary metric | Value |
|---|---:|---:|
| White win before | ROC-AUC | 0.579185 |
| White win after 3 | ROC-AUC | 0.579614 |
| White win after 10 | ROC-AUC | 0.621742 |
| Elo after 10 | Avg MAE | 28.827 |
| Elo expected-score baseline | ROC-AUC | 0.578497 |

## Calibration And Lift

- Top 10% predicted after-10 White-win rate: `0.7120`.
- Bottom 10% predicted after-10 White-win rate: `0.2860`.
- Top-minus-bottom 10% lift: `0.4260`.
- Top 20% predicted after-10 White-win rate: `0.6620`.
- Bottom 20% predicted after-10 White-win rate: `0.3565`.

The after-10 model is useful for ranking games: the highest-probability bucket has a much higher actual White-win rate than the lowest bucket.

## Calibration Bins

|   bin |   count |   mean_predicted_prob |   actual_white_win_rate |   calibration_gap_pred_minus_actual |
|------:|--------:|----------------------:|------------------------:|------------------------------------:|
|     1 |    2000 |              0.287474 |                  0.286  |                         0.00147382  |
|     2 |    2000 |              0.39154  |                  0.427  |                        -0.0354601   |
|     3 |    2000 |              0.429938 |                  0.4195 |                         0.0104384   |
|     4 |    2000 |              0.459353 |                  0.4655 |                        -0.00614721  |
|     5 |    2000 |              0.483709 |                  0.4675 |                         0.0162088   |
|     6 |    2000 |              0.504184 |                  0.5115 |                        -0.00731634  |
|     7 |    2000 |              0.524669 |                  0.525  |                        -0.000330523 |
|     8 |    2000 |              0.550929 |                  0.538  |                         0.0129288   |
|     9 |    2000 |              0.593089 |                  0.612  |                        -0.0189112   |
|    10 |    2000 |              0.70932  |                  0.712  |                        -0.0026801   |

## Elo Error By Rating Band

| elo_band   |   count |   white_mae |   black_mae |   avg_mae |   white_rmse |   black_rmse |   avg_error_p50 |   avg_error_p90 |   avg_error_p99 |
|:-----------|--------:|------------:|------------:|----------:|-------------:|-------------:|----------------:|----------------:|----------------:|
| <1000      |     975 |     72.5387 |     76.373  |   74.4559 |     180.126  |     184.293  |         14.35   |         186.018 |         736.596 |
| 1000-1399  |    4212 |     37.2404 |     37.6491 |   37.4448 |      96.7412 |      98.2316 |         10.0075 |          66.232 |         471.354 |
| 1400-1799  |    7572 |     22.3533 |     22.3501 |   22.3517 |      51.3068 |      51.2644 |          8.52   |          51.36  |         229.033 |
| 1800-2199  |    6302 |     22.4842 |     22.9466 |   22.7154 |      71.7547 |      73.3748 |          6.8275 |          35.43  |         393.067 |
| 2200-2599  |     900 |     35.7854 |     32.1502 |   33.9678 |     119.293  |     113.94   |          8.9225 |          45.416 |         703.205 |
| 2600+      |      39 |     93.1546 |     73.6769 |   83.4158 |     263.408  |     233.168  |         19.755  |          75.242 |        1096.89  |

## Representative Prediction Examples

The script also exports the full examples to `prediction_examples.json`. The table below summarizes the same representative validation rows.

| note | game_index | result | p_white_win_before | p_white_win_after_10 | white_elo_abs_error | black_elo_abs_error |
|---|---:|---|---:|---:|---:|---:|
| High-confidence correct after-10 prediction | 206351 | 1-0 | 0.9035 | 0.9552 | 11.85 | 9.69 |
| High-confidence miss | 178196 | 0-1 | 0.8799 | 0.9218 | 17.06 | 19.86 |
| Largest after-10 probability increase relative to before-game | 206812 | 1-0 | 0.4913 | 0.9249 | 144.86 | 201.03 |
| Largest after-10 probability decrease relative to before-game | 177759 | 0-1 | 0.5276 | 0.0567 | 32.73 | 216.31 |
| Lowest Elo error example | 202474 | 1-0 | 0.4818 | 0.4521 | 0.00 | 0.05 |
| Highest Elo error example | 186670 | 1-0 | 0.6288 | 0.6602 | 1260.54 | 1100.86 |

These examples show that confidence can be useful but is not perfect. Future mistakes, tactical swings, or time-pressure collapses after move 10 are not visible to the model. Elo prediction can be extremely accurate for repeat or history-rich players, but it can fail badly for sparse-history players or unusual rating-band cases.

## Interpretation

- Pre-game prediction is dominated by Elo difference and stays close to the Elo expected-score baseline.
- After-10 prediction gains signal from observed board structure and clock/time-pressure features.
- Elo prediction is very strong because causal player-history features are highly predictive for repeat players.
- The Elo model should be interpreted as a same-stream rating reconstruction model rather than a pure cold-start Elo estimator. Its strongest signal comes from causal player history and repeat-player structure. This is leakage-safe because the history is computed only from earlier games, but the headline MAE is most reliable when the validation/test stream has similar player-overlap patterns.
