# Detailed Experiment Report

## 1. Executive Summary

This report documents the experiments performed for the Lichess Blitz prediction assessment. It explains what was tested, how each experiment was run, the feature sets and model parameters used, the observed results, and the conclusions that led to the final selected pipeline.

Final selected submission profile:

| Task | Final selected model | Why selected | Main validation metric |
|---|---|---|---:|
| T1 White win before | LogisticRegression C=1.0 | Stable, beats tree models for pre-game data | ROC-AUC 0.5788 |
| T2 White win after 3 | XGBoost conservative, no Stockfish | Best portable no-engine after-3 model | ROC-AUC 0.5787 |
| T3 White win after 10 | XGBoost balanced, no Stockfish, clock | Best portable no-engine after-10 model | ROC-AUC 0.6226 |
| T4 Elo after 10 | LightGBM balanced, no Stockfish, causal history | Best portable no-engine Elo model | Avg MAE 29.31 |

The strongest exploratory after-10 classifier used Stockfish and reached ROC-AUC around `0.6483`, but Stockfish was not selected for the final portable submission because it requires an external engine/cache. The selected pipeline therefore optimizes the tradeoff between performance, reproducibility, and package simplicity.

## 2. Shared Experiment Protocol

All serious experiments followed the same high-level validation rule:

- Use Lichess Blitz games from the selected monthly stream.
- Preserve chronological order.
- Train on the first 80% of eligible games.
- Validate on the last 20% of eligible games.
- Do not use validation rows for fitting models or preprocessing.
- Do not use post-game leakage features.
- For Elo regression, do not use current `WhiteElo`, `BlackElo`, `elo_diff`, or `mean_elo` as inputs.

Main metrics:

- Classification: ROC-AUC, log loss, Brier score, accuracy.
- Elo regression: White MAE, Black MAE, average MAE, RMSE, R2.

## 3. Experiment Sources

The experiment report is based on these artifacts:

| Source | Purpose |
|---|---|
| `experiment/run_experiments.py` | Main multi-phase experiment runner |
| `experiment/models.py` | Model builders and default model parameterization |
| `experiment/config.py` | Experiment paths, seed, selected month, Stockfish settings |
| `experiment/outputs/experiment_results.csv` | Multi-phase 100k experiment results |
| `experiment/outputs/comparison_report.md` | Vietnamese detailed experiment narrative |
| `artifacts/experiments/boosting_no_stockfish_100k/experiment_results.csv` | Optional LightGBM/XGBoost no-Stockfish comparison |
| `artifacts/experiments/ab_testing_100k/ab_testing_results.csv` | A/B Stockfish vs no-Stockfish metrics |
| `artifacts/experiments/report_best_models_100k/experiment_results.csv` | Heavy/Stockfish best-model summary |
| `outputs_solution_improvements_100k_final/metrics.json` | Final verified 100k production run |

## 4. Experiment Runner Design

The exploratory runner is `experiment/run_experiments.py`. It executes multiple phases:

1. Baseline recap.
2. Enhanced feature engineering.
3. Tree-based models.
4. Stockfish evaluation.
5. Deep learning experiments were scaffolded in code but not selected in final reported artifacts.
6. Ensemble and stacking.
7. Final selected validation.

The runner loads cached datasets through `experiment/data_loader.py`, enhances features through `experiment/features.py`, and builds models through `experiment/models.py`.

Important fixed settings from `experiment/config.py`:

| Setting | Value |
|---|---:|
| Random seed | 42 |
| Month | 2023-11 |
| Time control | Blitz |
| Train ratio | 0.8 |
| Hashing features | 2^15 |
| Stockfish depth | 10 |
| Stockfish cache file | `experiment/stockfish_cache.json` |

## 5. Feature Sets Tested

The experiments tested several feature families.

| Feature group | Description | Leakage status |
|---|---|---|
| `base_before` | Elo and time-control features before game | Allowed for White-win classification |
| `base_after3+text` | Before features + after-3 board/move features + hashed text | Uses only first 6 plies |
| `base_after10+text` | Before features + after-10 board/move features + hashed text | Uses only first 20 plies |
| `base_elo+text` | After-10 board/move/text features, no current Elo | Allowed for Elo regression |
| `before+history` | Pre-game features + causal player history | History computed before current-game update |
| `after3+enhanced` | After-3 numeric board features + engineered chess proxies | No future plies |
| `after10+enhanced` | After-10 numeric board features + engineered chess proxies | No future plies |
| `after10+enhanced_clock` | After-10 enhanced features + clock features up to ply 20 | No future clock values |
| `elo+enhanced` | Elo-safe after-10 features + enhanced board features | Current Elo excluded |
| `elo_enhanced_history` | Elo-safe after-10 features + causal history | Final Elo feature set |
| `+SF` | Adds Stockfish centipawn/mate features | Exploratory only, not final |

## 6. Model Families And Parameters

### 6.1 Baseline Linear Models

Baseline classification uses:

```text
LogisticRegression(C=<selected>, solver="liblinear", max_iter=5000, random_state=42)
```

Baseline Elo regression uses:

```text
Ridge(alpha=10.0)
```

Text features use:

```text
HashingVectorizer(n_features=2^15, alternate_sign=False)
```

Numeric features use median imputation and standard scaling.

### 6.2 General Tree Models In `experiment/`

Tree experiments used sklearn/LightGBM/XGBoost wrappers:

- LightGBM: `LGBMClassifier` / `LGBMRegressor`
- XGBoost: `XGBClassifier(eval_metric="logloss")` / `XGBRegressor`
- HistGradientBoosting
- RandomForest
- GradientBoosting

The generic tree runner used default model parameters plus `random_state=42`, with numeric imputation/scaling before the estimator.

### 6.3 No-Stockfish Boosting Candidate Parameters

The final no-Stockfish boosting profile tested two parameter families.

Conservative LightGBM:

```json
{
  "n_estimators": 200,
  "learning_rate": 0.03,
  "max_depth": 4,
  "num_leaves": 15,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "reg_lambda": 5.0,
  "random_state": 42,
  "n_jobs": 1
}
```

Balanced LightGBM:

```json
{
  "n_estimators": 400,
  "learning_rate": 0.05,
  "max_depth": 6,
  "num_leaves": 31,
  "subsample": 0.9,
  "colsample_bytree": 0.9,
  "reg_lambda": 2.0,
  "random_state": 42,
  "n_jobs": 1
}
```

Conservative XGBoost:

```json
{
  "n_estimators": 200,
  "learning_rate": 0.03,
  "max_depth": 3,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "reg_lambda": 5.0,
  "tree_method": "hist",
  "random_state": 42,
  "n_jobs": 1
}
```

Balanced XGBoost:

```json
{
  "n_estimators": 400,
  "learning_rate": 0.05,
  "max_depth": 4,
  "subsample": 0.9,
  "colsample_bytree": 0.9,
  "reg_lambda": 2.0,
  "tree_method": "hist",
  "random_state": 42,
  "n_jobs": 1
}
```

## 7. Phase 1 - Baseline Recap

### Goal

Establish simple, reproducible reference models for all four tasks.

### How It Was Run

`experiment/run_experiments.py` trains:

- T1: LogisticRegression C=1.0 on `base_before`.
- T2: LogisticRegression C=0.25 + hashed text on `base_after3+text`.
- T3: LogisticRegression C=0.25 + hashed text on `base_after10+text`.
- T4: Ridge alpha=10.0 + hashed text on `base_elo+text`.

### Results

| ID | Task | Model | Features | ROC-AUC | Log loss | Brier | Accuracy | Avg MAE |
|---|---|---|---|---:|---:|---:|---:|---:|
| B1 | T1 | LogReg(C=1.0) | base_before | 0.5788 | 0.6788 | 0.2433 | 0.5526 | n/a |
| B2 | T2 | LogReg(C=0.25)+Hash | base_after3+text | 0.5741 | 0.6805 | 0.2440 | 0.5493 | n/a |
| B3 | T3 | LogReg(C=0.25)+Hash | base_after10+text | 0.6149 | 0.6675 | 0.2380 | 0.5765 | n/a |
| B4 | T4 | Ridge(alpha=10)+Hash | base_elo+text | n/a | n/a | n/a | n/a | 90.51 |

### Conclusion

The baseline already shows that:

- Elo difference is a meaningful before-game signal.
- After-10 board/move information improves classification over after-3.
- Elo regression is much easier than White-win classification once causal history and early-game behavior are available.

## 8. Phase 2 - Enhanced Feature Engineering

### Goal

Test whether additional causal history and handcrafted board features improve simple models.

### What Was Added

- `before+history` for T1.
- `after3+enhanced` board features for T2.
- `after10+enhanced` board features for T3.
- `elo+enhanced+text` for T4.

### Results

| ID | Task | Model | Features | ROC-AUC | Log loss | Brier | Accuracy | Avg MAE |
|---|---|---|---|---:|---:|---:|---:|---:|
| F1 | T1 | LogReg(C=1.0)+Hist | before+history | 0.5792 | 0.6787 | 0.2432 | 0.5522 | n/a |
| F2 | T2 | LogReg(C=0.5) | after3+enhanced | 0.5797 | 0.6787 | 0.2431 | 0.5552 | n/a |
| F3 | T2 | LogReg(C=0.25)+Hash | after3+enhanced+text | 0.5741 | 0.6805 | 0.2440 | 0.5493 | n/a |
| F4 | T3 | LogReg(C=0.25)+Hash | after10+enhanced+text | 0.6149 | 0.6675 | 0.2380 | 0.5765 | n/a |
| F5 | T4 | Ridge(alpha=10)+Hash | elo+enhanced+text | n/a | n/a | n/a | n/a | 90.51 |

### Conclusion

- T1 history improves AUC only slightly: `0.5788 -> 0.5792`.
- T2 enhanced numeric features improve over hashed move text: `0.5741 -> 0.5797`.
- T3 and T4 did not improve under the linear/hash setup because the baseline text + board representation already captured most of what Ridge/LogReg could use.
- This suggested that nonlinear models might be needed to exploit the enhanced board features.

## 9. Phase 3 - Tree-Based Models

### Goal

Evaluate whether nonlinear models can exploit enhanced chess features better than logistic/Ridge models.

### Models Tested

- LightGBM
- XGBoost
- HistGradientBoosting
- RandomForest
- GradientBoosting

### T1 Before-Game Results

| Model | Features | ROC-AUC | Log loss | Brier | Accuracy |
|---|---|---:|---:|---:|---:|
| LogReg + history | before+history | 0.5792 | 0.6787 | 0.2432 | 0.5522 |
| LightGBM | before+history | 0.5718 | 0.6819 | 0.2446 | 0.5442 |
| XGBoost | before+history | 0.5508 | 0.6987 | 0.2516 | 0.5318 |
| HistGB | before+history | 0.5765 | 0.6800 | 0.2438 | 0.5474 |
| RandomForest | before+history | 0.5453 | 0.7053 | 0.2549 | 0.5299 |
| GradientBoosting | before+history | 0.5773 | 0.6796 | 0.2436 | 0.5461 |

Conclusion: tree models do not beat logistic regression before the game. The relationship is mostly monotonic in Elo difference, so linear logistic regression is sufficient and more stable.

### T2 After-3 Results

| Model | Features | ROC-AUC | Log loss | Brier | Accuracy |
|---|---|---:|---:|---:|---:|
| LogReg enhanced | after3+enhanced | 0.5797 | 0.6787 | 0.2431 | 0.5552 |
| LightGBM | after3+enhanced | 0.5750 | 0.6798 | 0.2438 | 0.5502 |
| XGBoost | after3+enhanced | 0.5618 | 0.6889 | 0.2477 | 0.5379 |
| HistGB | after3+enhanced | 0.5776 | 0.6790 | 0.2433 | 0.5497 |
| RandomForest | after3+enhanced | 0.5462 | 0.7068 | 0.2556 | 0.5342 |
| GradientBoosting | after3+enhanced | 0.5796 | 0.6786 | 0.2432 | 0.5526 |

Conclusion: after 3 moves, enhanced numeric features help, but the signal is still weak. GradientBoosting and LogisticRegression are similar in AUC.

### T3 After-10 Results

| Model | Features | ROC-AUC | Log loss | Brier | Accuracy |
|---|---|---:|---:|---:|---:|
| LogReg baseline | base_after10+text | 0.6149 | 0.6675 | 0.2380 | 0.5765 |
| LightGBM | after10+enhanced | 0.6201 | 0.6655 | 0.2370 | 0.5803 |
| XGBoost | after10+enhanced | 0.6070 | 0.6759 | 0.2414 | 0.5731 |
| HistGB | after10+enhanced | 0.6203 | 0.6652 | 0.2369 | 0.5791 |
| RandomForest | after10+enhanced | 0.6030 | 0.6737 | 0.2406 | 0.5677 |
| GradientBoosting | after10+enhanced | 0.6215 | 0.6651 | 0.2368 | 0.5810 |
| LightGBM tuned | after10+enhanced | 0.6227 | 0.6639 | 0.2364 | 0.5807 |

Conclusion: after 10 moves, nonlinear models begin to help. Tuned LightGBM and GradientBoosting improve over the logistic baseline.

### T4 Elo Results

| Model | Features | White MAE | Black MAE | Avg MAE | Avg RMSE | Avg R2 |
|---|---|---:|---:|---:|---:|---:|
| Ridge baseline | base_elo+text | 90.10 | 90.91 | 90.51 | 132.19 | 0.8711 |
| LightGBM | elo+enhanced | 29.05 | 28.98 | 29.01 | 80.58 | 0.9521 |
| XGBoost | elo+enhanced | 31.12 | 31.82 | 31.47 | 82.26 | 0.9501 |
| HistGB | elo+enhanced | 29.06 | 29.17 | 29.11 | 80.58 | 0.9521 |
| RandomForest | elo+enhanced | 26.44 | 26.54 | 26.49 | 80.68 | 0.9520 |
| GradientBoosting | elo+enhanced | 37.32 | 37.76 | 37.54 | 86.55 | 0.9448 |

Conclusion: nonlinear models dramatically improve Elo prediction. This is mainly because causal history features and rating-band patterns interact nonlinearly.

## 10. Phase 4 - Stockfish Exploratory Features

### Goal

Measure how much a chess engine can improve performance when engine evaluation is available.

### How It Was Run

Stockfish features were cached and mapped to board positions:

- `sf3_cp`
- `sf3_mate`
- `sf10_cp`
- `sf10_mate`
- `sf10_cp_diff`

Stockfish depth in the experiment config was `10`.

### Results From Multi-Phase Experiment

| ID | Task | Model | Features | ROC-AUC / Avg MAE |
|---|---|---|---|---:|
| S1 | T2 | GradientBoosting + Stockfish | after3+enhanced+SF | ROC-AUC 0.5832 |
| S2 | T3 | GradientBoosting + Stockfish | after10+enhanced+SF | ROC-AUC 0.6480 |
| S3 | T4 | RandomForest + Stockfish | elo+enhanced+SF | Avg MAE 26.42 |

### Conclusion

- Stockfish is mildly useful after 3 moves.
- Stockfish is highly useful after 10 moves.
- Stockfish barely improves Elo prediction because Elo is already explained by history.
- Stockfish is not selected for final submission because it creates an external dependency and cache/reproducibility risk.

## 11. Phase 5 - Ensemble And Stacking

### Goal

Test whether combining strong tree models improves after-10 classification and Elo regression.

### Results

| ID | Task | Model | Features | Metric |
|---|---|---|---|---:|
| E1 | T3 | Voting(LGB+XGB+RF) | after10+enhanced | ROC-AUC 0.6183 |
| E2 | T3 | Stacking(LGB+XGB+RF) | after10+enhanced | ROC-AUC 0.6209 |
| E3 | T4 | Voting Regressor | elo+enhanced | Avg MAE 27.19 |

### Conclusion

- Ensembles did not beat the best tuned after-10 classifier.
- Voting regression was strong but still did not justify additional complexity over the final LightGBM profile.
- Stacking increases training complexity and makes leakage control more delicate, so it was not selected.

## 12. No-Stockfish Boosting Experiment

### Goal

Find the best no-Stockfish model profile that improves metrics while staying lightweight enough for a compact submission.

### How It Was Run

The experiment compared:

- Existing production models.
- Conservative LightGBM/XGBoost.
- Balanced LightGBM/XGBoost.

This was run on the same 100k `2023-11` dataset and chronological validation split.

### T1 Before-Game

| Config | Algorithm | Features | ROC-AUC | Log loss | Brier | Accuracy |
|---|---|---|---:|---:|---:|---:|
| production_logreg_C1.0 | LogisticRegression | production | 0.5788 | 0.6788 | 0.2433 | 0.5526 |
| lightgbm_conservative_before_history | LightGBM | before_history | 0.5778 | 0.6794 | 0.2435 | 0.5487 |
| xgboost_conservative_before_history | XGBoost | before_history | 0.5774 | 0.6794 | 0.2435 | 0.5466 |
| lightgbm_balanced_before_history | LightGBM | before_history | 0.5690 | 0.6839 | 0.2455 | 0.5458 |
| xgboost_balanced_before_history | XGBoost | before_history | 0.5703 | 0.6819 | 0.2446 | 0.5438 |

Conclusion: keep LogisticRegression for T1.

### T2 After-3

| Config | Algorithm | Features | ROC-AUC | Log loss | Brier | Accuracy |
|---|---|---|---:|---:|---:|---:|
| production_logreg_identity_C0.25 | LogisticRegression | production | 0.5667 | 0.6837 | 0.2455 | 0.5428 |
| lightgbm_conservative_after3_enhanced | LightGBM | after3_enhanced | 0.5779 | 0.6794 | 0.2435 | 0.5498 |
| xgboost_conservative_after3_enhanced | XGBoost | after3_enhanced | 0.5787 | 0.6793 | 0.2434 | 0.5504 |
| lightgbm_balanced_after3_enhanced | LightGBM | after3_enhanced | 0.5669 | 0.6827 | 0.2451 | 0.5433 |
| xgboost_balanced_after3_enhanced | XGBoost | after3_enhanced | 0.5714 | 0.6811 | 0.2443 | 0.5461 |

Conclusion: select conservative XGBoost for T2.

### T3 After-10

| Config | Algorithm | Features | ROC-AUC | Log loss | Brier | Accuracy |
|---|---|---|---:|---:|---:|---:|
| production_logreg_identity_clock_C0.25 | LogisticRegression | production | 0.6107 | 0.6698 | 0.2391 | 0.5718 |
| lightgbm_conservative_after10_enhanced_clock | LightGBM | after10_enhanced_clock | 0.6208 | 0.6654 | 0.2369 | 0.5796 |
| lightgbm_balanced_after10_enhanced_clock | LightGBM | after10_enhanced_clock | 0.6211 | 0.6652 | 0.2369 | 0.5808 |
| xgboost_conservative_after10_enhanced_clock | XGBoost | after10_enhanced_clock | 0.6202 | 0.6660 | 0.2371 | 0.5796 |
| xgboost_balanced_after10_enhanced_clock | XGBoost | after10_enhanced_clock | 0.6219 | 0.6642 | 0.2364 | 0.5803 |

Conclusion: select balanced XGBoost for T3.

### T4 Elo

| Config | Algorithm | Features | White MAE | Black MAE | Avg MAE | White R2 | Black R2 |
|---|---|---|---:|---:|---:|---:|---:|
| production_ridge_history_identity | Ridge | production | 91.05 | 91.97 | 91.51 | 0.8709 | 0.8687 |
| lightgbm_conservative_elo_enhanced_history | LightGBM | elo_enhanced_history | 33.26 | 33.16 | 33.21 | 0.9446 | 0.9435 |
| lightgbm_balanced_elo_enhanced_history | LightGBM | elo_enhanced_history | 29.24 | 29.38 | 29.31 | 0.9503 | 0.9499 |
| xgboost_conservative_elo_enhanced_history | XGBoost | elo_enhanced_history | 36.99 | 37.85 | 37.42 | 0.9397 | 0.9378 |
| xgboost_balanced_elo_enhanced_history | XGBoost | elo_enhanced_history | 30.82 | 31.00 | 30.91 | 0.9489 | 0.9483 |

Conclusion: select balanced LightGBM for T4.

## 13. A/B Stockfish vs No-Stockfish Experiment

### Goal

Quantify whether Stockfish is worth adding to the final solution.

### T2 After-3 A/B Results

| Model | No Stockfish AUC | With Stockfish AUC | Delta AUC | No-SF fit time | SF fit time |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 0.5817 | 0.5818 | +0.0002 | 8.13s | 19.86s |
| GradientBoosting | 0.5811 | 0.5827 | +0.0015 | 62.42s | 62.77s |
| HistGradientBoosting | 0.5807 | 0.5826 | +0.0019 | 1.28s | 1.19s |
| RandomForest | 0.5723 | 0.5731 | +0.0009 | 2.48s | 2.44s |
| LightGBM | 0.5818 | 0.5831 | +0.0013 | 1.40s | 1.30s |
| XGBoost | 0.5819 | 0.5831 | +0.0013 | 1.31s | 1.47s |

Conclusion: Stockfish has minimal impact after 3 moves.

### T3 After-10 A/B Results

| Model | No Stockfish AUC | With Stockfish AUC | Delta AUC | No-SF fit time | SF fit time |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 0.6188 | 0.6447 | +0.0258 | 14.84s | 41.32s |
| GradientBoosting | 0.6219 | 0.6486 | +0.0267 | 77.33s | 84.23s |
| HistGradientBoosting | 0.6225 | 0.6495 | +0.0270 | 1.83s | 2.23s |
| RandomForest | 0.6130 | 0.6404 | +0.0274 | 3.65s | 3.68s |
| LightGBM | 0.6220 | 0.6485 | +0.0264 | 2.09s | 2.23s |
| XGBoost | 0.6205 | 0.6478 | +0.0273 | 1.55s | 1.59s |

Conclusion: Stockfish strongly improves after-10 classification. This is expected because engine evaluation captures tactical/positional advantage at ply 20.

### T4 Elo A/B Results

| Model | No Stockfish Avg MAE | With Stockfish Avg MAE | Delta MAE | No-SF fit time | SF fit time |
|---|---:|---:|---:|---:|---:|
| Ridge | 89.39 | 89.38 | -0.01 | 0.14s | 0.08s |
| GradientBoosting | 37.54 | 37.44 | -0.10 | 139.90s | 152.44s |
| HistGradientBoosting | 30.58 | 30.31 | -0.27 | 10.00s | 9.66s |
| RandomForest | 38.85 | 38.85 | +0.00 | 33.05s | 37.06s |
| LightGBM | 33.55 | 33.34 | -0.21 | 5.25s | 5.38s |
| XGBoost | 38.40 | 38.96 | +0.55 | 4.13s | 3.87s |

Conclusion: Stockfish does not materially help Elo regression. Causal player history is the primary signal.

## 14. Heavy / Stockfish Best-Model Reference

The strongest exploratory models from `artifacts/experiments/report_best_models_100k/experiment_results.csv` were:

| Task | Best heavy model | Metric |
|---|---|---:|
| T1 Before | LogReg + history | ROC-AUC 0.5792 |
| T2 After-3 | GradientBoosting + Stockfish | ROC-AUC 0.5827 |
| T3 After-10 | HistGradientBoosting + Stockfish | ROC-AUC 0.6483 |
| T4 Elo | RandomForest + Stockfish | Avg MAE 28.02 |
| T4 Elo | RandomForest no Stockfish | Avg MAE 28.05 |

The heavy reference shows the ceiling available with engine and high-memory model choices. The final submission intentionally avoids this route.

## 15. Final Verified Run

The final production-style run was:

```bash
python solution.py --target-games 100000 --output-dir outputs_solution_improvements_100k_final --model-profile boosting
```

Final metrics:

| Task | Metric |
|---|---:|
| White win before ROC-AUC | 0.578805 |
| White win after 3 ROC-AUC | 0.578667 |
| White win after 10 ROC-AUC | 0.622593 |
| Elo White MAE | 29.241 |
| Elo Black MAE | 29.376 |
| Runtime | 645.82 seconds |
| Parsed games | 213,463 |
| Header-eligible games | 104,005 |
| Eligible games | 100,000 |

This run also tested additional time-pressure features and stream retry/resume support. Bayesian smoothing for history was implemented and tested, but it was disabled because it worsened Elo MAE.

## 16. Decision Matrix

| Candidate | Performance | Portability | Leakage safety | Final decision |
|---|---|---|---|---|
| Strict lightweight sklearn | Medium | Excellent | Excellent | Keep as fallback |
| No-Stockfish boosting | High | Good | Good | Use as final reported profile |
| Stockfish-heavy | Highest for after-10 | Weak | Manageable but more complex | Research appendix only |
| Deep learning | Not selected / not needed | Weak | Higher complexity | Exclude |
| Ensembles / stacking | Mixed | Medium | More complex | Exclude |

## 17. Conclusions By Task

### T1 Before-Game White-Win Prediction

Pre-game outcome prediction is dominated by Elo. Logistic regression is sufficient and more robust than tree models. Additional history gives only tiny gains, so the final profile uses the stable production logistic model.

### T2 After-3 White-Win Prediction

After 3 moves, chess games are still early. Enhanced board features help modestly, and conservative XGBoost is the best no-Stockfish option. Stockfish adds little at this horizon.

### T3 After-10 White-Win Prediction

After 10 moves, board and clock features become meaningfully predictive. XGBoost with enhanced features and clock is the best portable no-Stockfish model. Stockfish would improve AUC further, but it is not selected for the final compact pipeline.

### T4 Elo Prediction

Elo regression benefits massively from causal player history and nonlinear boosting. LightGBM balanced gives a strong MAE around 29 Elo. This is valid for the chronological same-month stream, but the result should be interpreted carefully for unseen-player generalization.

## 18. Final Recommendation

Use the no-Stockfish boosting profile as the final reported solution:

```bash
pip install -r requirements.txt
pip install -r requirements-experiments.txt
python Solution.py --target-games 100000 --output-dir outputs_full --model-profile boosting
```

Keep the strict lightweight profile documented as a fallback:

```bash
pip install -r requirements.txt
python Solution.py --target-games 100000 --output-dir outputs_lightweight
```

