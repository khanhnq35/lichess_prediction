# Boosting No-Stockfish Experiment

This experiment compares optional LightGBM/XGBoost candidates against the current production solution path.

## Scope

- Stockfish: not used.
- Deep learning: not used.
- LightGBM/XGBoost are optional experiment dependencies, not part of `requirements.txt`.
- Current Elo and Elo-derived columns remain excluded from Elo regression features.

## Best Configs

- `white_win_before`: `lightgbm_conservative_before_history`
- `white_win_after_3`: `production_logreg_identity_C0.25`
- `white_win_after_10`: `production_logreg_identity_clock_C0.25`
- `elo_after_10`: `lightgbm_balanced_elo_enhanced_history`

## Output Files

- `outputs_improved_10k/experiment_results.csv`
- `outputs_improved_10k/best_config.json`
- `outputs_improved_10k/metrics.json`
