# Limitations And Future Work

## Limitations

## 1. White-Win Prediction Is Intrinsically Noisy

The best after-10 White-win ROC-AUC is `0.6226`. This is useful as a probabilistic signal, but it is not engine-strength prediction.

Reasons:

- Lichess matches players with similar Elo, so many games are close to 50/50.
- Draws are treated as non-White-wins.
- Many decisive blunders happen after move 10.
- The model does not use Stockfish or neural chess evaluation.
- Human factors such as tilt, mouse slips, and late time trouble are mostly unobserved.

## 2. Elo Prediction Benefits From Repeat Players

The Elo model performs very well because causal same-month player history is highly predictive for repeat players.

This is leakage-safe because history features are computed before updating with the current game. However, performance will be weaker for completely unseen players or a different platform/month with less repeat-player history.

## 3. Single-Month Validation

The main full-scale run uses a chronological split inside one selected month. This matches the assessment requirement but does not fully test cross-month stability.

## 4. Optional Boosting Dependencies

The best model profile uses LightGBM and XGBoost. These are still lightweight compared with deep learning or Stockfish, but they may require additional binary wheels on some systems.

The strict fallback profile remains available through `requirements.txt`.

## 5. No Engine Evaluation In Final Pipeline

Stockfish features improve after-10 prediction in experiments, but they are excluded from the final pipeline to keep the solution portable and compact.

## Future Work

1. Add cross-month validation to test robustness across time.
2. Add opening/ECO features without external engines.
3. Add out-of-fold text stacking if more complex training is allowed.
4. Add calibration methods and reliability diagrams for probability quality.
5. Report separate metrics for repeat-player and unseen-player subsets.
6. Add SHAP or permutation importance for boosting models if dependency constraints allow.
7. Explore Stockfish as an optional appendix pipeline, not as the core submission.
8. Train models on multiple months while preserving chronological evaluation.

