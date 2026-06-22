# Quick Overview Report

This is the shortest navigation report for the Lichess Quant Research submission. Use it first, then open the linked documents for details.

## 1. Task

Build a reproducible pipeline from the Lichess open database to predict:

1. White-win probability before the game.
2. White-win probability after 3 full moves.
3. White-win probability after 10 full moves.
4. Both players' Elo after 10 full moves.

Full task interpretation: [01_Problem_definition.md](01_Problem_definition.md)

## 2. Final Submission Choice

The final reported profile is **portable report-best, no Stockfish**:

| Task | Final model | Main signal |
|---|---|---|
| Before-game White win | Logistic Regression + causal history | Pre-game Elo/time control + prior same-stream history |
| After-3 White win | LogisticRegression(C=0.5) | Early enhanced board + first-6-ply clock features |
| After-10 White win | sklearn HistGradientBoostingClassifier | Enhanced board + clock/time pressure |
| Elo after 10 | sklearn RandomForestRegressor | Causal player history + enhanced board |

Stockfish-based experiments achieved the highest after-10 classification AUC, but they are kept as research references because they require an external engine or cached engine evaluations. The final submission profile is chosen to remain compact and reproducible from the submitted code.

Pipeline details: [04_Proposal_pipeline.md](04_Proposal_pipeline.md)

## 3. Data And Validation

| Item | Value |
|---|---:|
| Month | 2023-11 |
| Time-control | Blitz |
| Parsed games | 213,463 |
| Final eligible games | 100,000 |
| Train / validation | 80,000 / 20,000 |
| Split | Chronological |

No validation rows are used for training or preprocessing fitting. No raw `.pgn`, `.zst`, or `.pgn.zst` files are included.

## 4. Final Results vs Baselines

| Task | Final model | Baseline | Interpretation |
|---|---:|---:|---|
| White win before | ROC-AUC 0.5792 | Elo expected-score ROC-AUC 0.5785 | Pre-game prediction is mostly Elo-driven; causal history gives only a tiny lift. |
| White win after 3 | ROC-AUC 0.5796 | Elo expected-score ROC-AUC 0.5785 | After 3 full moves adds only a small lift over Elo; this horizon should not be overclaimed. |
| White win after 10 | ROC-AUC 0.6217 | Elo expected-score ROC-AUC 0.5785 | Board state and clock/time-pressure features add meaningful signal beyond Elo. |
| Elo after 10 | White/Black MAE 28.72 / 28.93 | Mean Elo baseline MAE 300.22 / 300.59 | Causal same-stream player history makes Elo reconstruction much stronger than a global mean baseline. |

For classification, the main baseline is the Elo expected-score formula:

```text
p_white = 1 / (1 + 10 ** (-(WhiteElo - BlackElo) / 400))
```

This is a strong pre-game benchmark because Elo difference is known before the game. The after-10 model is the most useful outcome model because it improves from the Elo baseline ROC-AUC `0.5785` to `0.6217`.

The low Elo MAE is **not interpreted as a cold-start rating estimator**. It is best understood as same-stream rating reconstruction: the model uses causal player-history features computed only from earlier eligible games before the current game is processed. This is leakage-safe because no current Elo, rating diff, result, future games, or validation fitting is used as an input, but the headline MAE is most reliable when the validation stream has similar repeat-player/history patterns.

Full result analysis: [05_results_report.md](05_results_report.md)

## 5. Experiment Takeaways

| Finding | Decision |
|---|---|
| Before-game outcome is mostly Elo-driven | Use Logistic Regression with causal history |
| After-3 has limited signal | Use LogisticRegression(C=0.5) from `experiment/outputs` F2; do not overclaim vs Elo baseline |
| After-10 benefits from board + clock features | Use sklearn HistGradientBoosting |
| Elo prediction benefits strongly from causal history | Use sklearn RandomForest |
| Stockfish improves after-10 most | Keep as research appendix, not final pipeline |

Full experiment report: [03_Experiment.md](03_Experiment.md)

## 6. XAI And Diagnostics

Output-level XAI is generated from local validation outputs only. It covers:

- calibration bins,
- lift/ranking behavior,
- Elo error by rating band,
- representative success/failure examples.

Generated XAI summary: [../Results/xai/xai_summary.md](../Results/xai/xai_summary.md)

Regenerate XAI:

```bash
cd submission
python Results/xai/generate_xai_report.py
```

## 7. Key Caveat

White-win prediction is realistic but not engine-strength. Elo prediction is very strong because causal same-stream player history is highly informative; it should be interpreted as **same-stream rating reconstruction**, not pure cold-start Elo estimation.

Limitations and future work: [06_Limitation_and_future_work.md](06_Limitation_and_future_work.md)

## 8. Reading Order

1. This file.
2. [01_Problem_definition.md](01_Problem_definition.md)
3. [04_Proposal_pipeline.md](04_Proposal_pipeline.md)
4. [03_Experiment.md](03_Experiment.md)
5. [05_results_report.md](05_results_report.md)
6. [06_Limitation_and_future_work.md](06_Limitation_and_future_work.md)
7. [07_AI_workflow.md](07_AI_workflow.md)
