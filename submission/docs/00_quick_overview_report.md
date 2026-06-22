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

The final reported profile is **no-Stockfish boosting**:

| Task | Final model | Main signal |
|---|---|---|
| Before-game White win | Logistic Regression | Pre-game Elo/time control |
| After-3 White win | Conservative XGBoost | Early board features |
| After-10 White win | Balanced XGBoost | Board + clock/time pressure |
| Elo after 10 | Balanced LightGBM | Causal player history + board |

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

## 4. Final Results

| Task | Metric |
|---|---:|
| White win before | ROC-AUC 0.5788 |
| White win after 3 | ROC-AUC 0.5787 |
| White win after 10 | ROC-AUC 0.6226 |
| Elo after 10 | White/Black MAE 29.24 / 29.38 |

Full result analysis: [05_results_report.md](05_results_report.md)

## 5. Experiment Takeaways

| Finding | Decision |
|---|---|
| Before-game outcome is mostly Elo-driven | Keep Logistic Regression |
| After-3 has limited signal | Use conservative XGBoost, no Stockfish |
| After-10 benefits from board + clock features | Use balanced XGBoost |
| Elo prediction benefits strongly from causal history | Use balanced LightGBM |
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

