# Boosting No-Stockfish Experiment

This experiment compares optional LightGBM/XGBoost candidates against the current production solution path.

## Scope

- Stockfish: not used.
- Deep learning: not used.
- LightGBM/XGBoost are optional experiment dependencies, not part of `requirements.txt`.
- Current Elo and Elo-derived columns remain excluded from Elo regression features.

## Best Configs

- `white_win_before`: `production_logreg_C1.0`
- `white_win_after_3`: `xgboost_conservative_after3_enhanced`
- `white_win_after_10`: `xgboost_balanced_after10_enhanced_clock`
- `elo_after_10`: `lightgbm_balanced_elo_enhanced_history`

## Output Files

- `outputs_boosting_experiments_100k/experiment_results.csv`
- `outputs_boosting_experiments_100k/best_config.json`
- `outputs_boosting_experiments_100k/metrics.json`
