# Limitations and Future Work

## Limitations

### 1. White-Win Prediction Is Intrinsically Noisy

The best final portable after-10 White-win model achieves a ROC-AUC of `0.6217`. This is a useful probabilistic signal, but it should not be interpreted as engine-strength game outcome prediction.

This limitation is expected for several reasons. First, Lichess pairings often match players with relatively similar ratings, so many games are close to a 50/50 outcome before the game starts. Second, draws are treated as non-White-wins in the binary target, which makes the classification task stricter than predicting expected score. Third, many decisive events in blitz games occur after move 10, including tactical blunders, time-pressure mistakes, mouse slips, and resignations from later positions. These factors are not fully observable from early-game features.

The final pipeline also intentionally avoids using Stockfish or neural chess evaluation. As a result, the model relies on metadata, early move prefixes, board-state features, clock features, and player-history features rather than deep chess position evaluation. This improves portability and keeps the submission lightweight, but it limits the maximum achievable predictive accuracy for early White-win probability.

### 2. Elo Prediction Benefits Strongly From Repeat Players

The Elo prediction models perform well because causal same-month player-history features are highly informative for repeat players. For example, a player's previous observed rating, recent average rating, and number of prior games provide strong signals about their current Elo.

This design is leakage-safe because all player-history features are computed chronologically and are updated only after the current game has been processed. Therefore, the model does not use future games or validation-game labels when constructing training features.

However, this also means that performance may be weaker for completely unseen players, players with very few previous games in the selected month, or deployment settings where historical player information is unavailable. In those cases, the model must rely more heavily on early-game behavior, time usage, and board-state features, which are less direct proxies for player strength.

### 3. Single-Month Chronological Validation

The main full-scale experiment uses a chronological split within one selected Lichess month. This matches the assessment setting and prevents validation games from being used during training. It also reflects a realistic scenario where earlier games are used to predict later games.

However, this setup does not fully test cross-month robustness. Player pools, rating distributions, openings, time-control preferences, and platform behavior can shift over time. Therefore, the reported validation results should be interpreted as within-month generalization rather than a complete guarantee of stability across different months.

### 4. Dependence on PGN Parsing Quality

The pipeline depends on correctly parsing Lichess PGN headers, moves, clock comments, and legal board states. Invalid or incomplete games are skipped during eligibility filtering. Although this keeps the training data clean, it may introduce a small selection bias toward games with well-formed metadata and parseable move sequences.

The cutoff-based features also depend on the interpretation that 3 moves and 10 moves refer to 3 and 10 full moves, equivalent to 6 and 20 plies respectively. This convention is documented in the pipeline, but different interpretations of “move” could slightly change the eligible subset and extracted features.

### 5. Optional Boosting Dependencies

The final `report_best` profile uses sklearn tree/boosting models, so it does not require LightGBM or XGBoost. Earlier LightGBM/XGBoost experiments remain useful comparison profiles and produced competitive no-Stockfish results.

LightGBM/XGBoost may require additional binary wheels on some systems. To keep the final solution reproducible, they are not mandatory for the submitted pipeline. A stricter lightweight fallback profile is also provided through `requirements.txt`.

### 6. No Engine Evaluation in the Final Core Pipeline

Stockfish-based features can improve after-3 and after-10 White-win prediction because engine evaluations provide direct information about position quality. However, the final core pipeline excludes external engine analysis to keep the solution portable, compact, and reproducible within the file-size and time constraints.

This means the final model is not designed to evaluate chess positions at engine strength. Instead, it learns statistical patterns from early moves, board structure, clock usage, time control, ratings, and causal player history.

### 7. Limited Modeling of Human and Contextual Factors

The dataset contains useful metadata and move information, but many human factors are not directly observed. These include player fatigue, tilt, distraction, mouse slips, preparation, opening familiarity, device quality, and whether a player is playing casually or seriously.

Such factors can strongly influence blitz outcomes, especially in later phases of the game. Since they are not explicitly available in the PGN data, the model can only approximate them indirectly through rating, clock usage, move patterns, and recent player history.

## Future Work

1. **Cross-month validation**
   Evaluate the pipeline on later months after training on earlier months to measure robustness against temporal distribution shift.

2. **Unseen-player evaluation**
   Report separate metrics for repeat players and completely unseen players. This would clarify how much of the Elo-prediction performance comes from historical player information versus early-game behavior.

3. **Expected-score modeling**
   Add a secondary target where White win is `1.0`, draw is `0.5`, and Black win is `0.0`. This may better reflect chess outcome prediction than treating draws as non-White-wins.

4. **Probability calibration**
   Add calibration methods such as Platt scaling or isotonic regression, then report reliability diagrams and Brier score. This would make the predicted win probabilities more interpretable.

5. **Opening-prefix features without leakage**
   Add opening features derived only from the observed cutoff moves, rather than using PGN `Opening` or `ECO` tags directly. This would preserve leakage safety while improving early-game representation.

6. **Richer player-history features**
   Extend causal history features with recent win rate, recent color-specific performance, recent time-control-specific performance, and rating volatility. These features should still be computed strictly before the current game.

7. **Model explainability**
   Add permutation importance or SHAP analysis for the boosting models if dependency constraints allow. This would make it easier to understand whether predictions rely mostly on Elo, early board state, clock usage, or player history.

8. **Optional Stockfish appendix pipeline**
   Explore Stockfish evaluation as an optional experimental appendix rather than part of the core submission. This would help quantify the performance gap between lightweight statistical features and engine-informed features.

9. **Multiple-month training**
   Train on multiple months while preserving chronological evaluation. This could improve generalization, especially for rare openings, uncommon time controls, and players with sparse same-month history.

10. **Sequence-based models**
    Explore lightweight sequence models over the first 20 plies, such as move-token embeddings or compact recurrent models. These should only be considered after the tabular pipeline is stable, because they add complexity and may not outperform boosted trees under the 24-hour constraint.
