# Solution.py Improvement Run - 100k

Run date: 2026-06-22

Command:

```bash
python solution.py --target-games 100000 --output-dir outputs_solution_improvements_100k_final --model-profile boosting
```

## Implemented Changes

The following leakage-safe improvements were added to `solution.py` before the final run:

- Extra clock/time-pressure features:
  - `white_time_panic`
  - `black_time_panic`
  - `white_time_used_ratio`
  - `black_time_used_ratio`
  - `time_ratio_diff`
- Stream retry/resume support for interrupted Lichess downloads.
- Bayesian smoothing was implemented behind a constant, tested, and then disabled by default because it worsened the 100k Elo regression result.

Not ported from `solution_improved.py`:

- Stockfish features: excluded because the current target is lightweight/no-heavy-engine.
- In-sample text stacking meta-features: avoided because they are harder to keep leakage-safe without out-of-fold training.
- Isotonic calibration wrapper: not adopted because prior available 100k evidence did not show a clear production improvement.

## Run Summary

- Runtime: 645.82 seconds.
- Selected month: 2023-11.
- Time control: Blitz.
- Parsed games: 213,463.
- Header-eligible games: 104,005.
- Final eligible games: 100,000.
- Train games: 80,000.
- Validation games: 20,000.
- Train positive rate: 0.493950.
- Validation positive rate: 0.496400.

## Classification Metrics

| Model | ROC-AUC | Log loss | Brier | Accuracy |
|---|---:|---:|---:|---:|
| White win before | 0.578805 | 0.678818 | 0.243280 | 0.552550 |
| White win after 3 | 0.578667 | 0.679298 | 0.243440 | 0.550400 |
| White win after 10 | 0.622593 | 0.663965 | 0.236364 | 0.579900 |
| Elo expected-score baseline | 0.578497 | 0.680803 | 0.243974 | n/a |
| Majority baseline | n/a | n/a | n/a | 0.503600 |

## Elo Regression Metrics

| Model | White MAE | White RMSE | White R2 | Black MAE | Black RMSE | Black R2 |
|---|---:|---:|---:|---:|---:|---:|
| Elo after 10 | 29.241 | 82.072 | 0.950259 | 29.376 | 82.509 | 0.949865 |
| Mean baseline | 300.224 | 368.031 | -0.000222 | 300.586 | 368.529 | -0.000195 |

## Comparison

Compared with previous production full 100k:

- Before-game ROC-AUC: unchanged at 0.5788.
- After-3 ROC-AUC: improved from 0.5667 to 0.5787.
- After-10 ROC-AUC: improved from 0.6107 to 0.6226.
- Elo MAE improved from 91.05/91.97 to 29.24/29.38.

Compared with previous no-Stockfish boosting 100k:

- Before-game ROC-AUC: unchanged at 0.5788.
- After-3 ROC-AUC: unchanged at 0.5787.
- After-10 ROC-AUC: improved from 0.6219 to 0.6226.
- Elo MAE stayed at the same level: 29.24/29.38.

Interpretation:

- The added time-pressure features gave a small positive effect for after-10 classification.
- Bayesian history smoothing was not good for the current LightGBM Elo profile, so it is disabled by default.
- The best current no-Stockfish `solution.py --model-profile boosting` profile is still LightGBM/XGBoost based, with clock/time-pressure features for after-10 classification and causal history for Elo regression.

## Files

- `outputs_solution_improvements_100k_final/metrics.json`: 4.5 KB.
- `outputs_solution_improvements_100k_final/validation_predictions.csv`: 2.9 MB.

No raw `.pgn`, `.zst`, or `.pgn.zst` files were created by this run.

## Superseded Smoothing Trial

An intermediate 100k trial was run at `outputs_solution_improvements_100k` with Bayesian history smoothing enabled.

- After-10 ROC-AUC: 0.622593.
- Elo White MAE: 35.690.
- Elo Black MAE: 34.884.

This was worse than the unsmoothed Elo profile, so the final `solution.py` keeps the smoothing implementation available but disabled by default.
