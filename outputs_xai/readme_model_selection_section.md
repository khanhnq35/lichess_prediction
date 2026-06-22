# Model Selection and Experimentation Phases

## 1. Experimentation Phases

Our quantitative research followed a structured, seven-phase development process:

1.  **Phase 1: Logistic / Ridge Baselines**: Established basic benchmark models using only Elo ratings and basic time controls.
2.  **Phase 2: Causal History Features**: Incorporated rolling statistics summarizing a player's performance in the monthly dataset prior to the current game (e.g., prior win rate, average opponent Elo, prior game count).
3.  **Phase 3: Player Identity Features**: Hashed player usernames via `HashingVectorizer` to capture individual rating adjustments and player-specific tendencies.
4.  **Phase 4: Clock Features**: Extracted clock times and move durations up to plies 6 and 20.
5.  **Phase 5: Tree and Boosting Models**: Tested Random Forest and Gradient Boosting models to evaluate non-linear relationships.
6.  **Phase 6: Stockfish Optional Features**: Integrated Stockfish engine evaluations at move 3 (ply 6) and move 10 (ply 20).
7.  **Phase 7: Repeat vs. Unseen Player Audit**: Evaluated how model performance degrades on players not previously observed in the training data.

---

## 2. Why the Highest Validation Score is Not Always Chosen

In a production quantitative research system, we prioritize **generalization, auditability, and safety** over pure validation metrics. The highest validation score in offline testing often hides overfitting or dependency on fragile features that fail in production.

### The RandomForest Elo MAE (~26.4) vs. Ridge Elo Model (~91.5) Dilemma
*   **The Mirage**: A Random Forest model incorporating username identities achieved a remarkable validation MAE of `~26.4`.
*   **The Audit Findings**: The repeat-player audit revealed a massive performance divergence:
    *   **Both players seen before**: MAE was `7.10` Elo points.
    *   **One player seen before**: MAE was `19.05` Elo points.
    *   **Both players unseen before**: MAE jumped to `95.05` Elo points (a 13-fold increase in error!).
*   **The Risk**: The Random Forest overfitted to specific username hash combinations. In a live setting where new players frequently enter the queue, this model's performance degrades rapidly.
*   **The Defensible Choice**: We selected a **Ridge Elo Regression** model. It has a validation MAE of `91.05` for White and `91.97` for Black. Ridge regression uses stable, regularized linear weights. On unseen players, it degrades gracefully, reverting to rating-based averages and time controls, making it significantly more robust and defensible for an interview.

---

## 3. Why After-10 Predictions Benefit More Than After-3 Predictions

*   **After 3 Moves (6 Plies)**: The game is still in the opening book. Very little has happened: castling is rare, material is equal, and clock differences are negligible (usually under 2–3 seconds). Consequently, the model's predictions after 3 moves are almost identical to the pre-game predictions, dominated by the Elo difference.
*   **After 10 Moves (20 Plies)**: The opening phase is complete. Critical signals have emerged:
    *   **Clock consumption**: Significant clock imbalances exist (players are thinking).
    *   **Material**: Early tactical mistakes or gambits may have changed the piece balance.
    *   **King Safety**: Players have castled or lost the ability to castle.
    This provides rich, actionable data, allowing the after-10 model to achieve a significant ROC-AUC lift (`+0.0322`) over the Elo baseline.
