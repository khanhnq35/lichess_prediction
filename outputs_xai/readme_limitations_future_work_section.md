# Limitations and Future Work

A key requirement in a Quantitative Research interview is showing **intellectual honesty** about your model's limitations and outlining a clear roadmap for improvements.

## 1. Limitations of the Current Approach

*   **Binary Simplification of Game Outcome**: The model treats chess outcomes as binary: White win (`1`) vs. non-White win (`0`). This groups draws (which represent ~4% of games in this Blitz dataset) with Black wins, failing to capture the distinct dynamics of drawing strategies.
*   **Data Window Constraints**: The model is fit on a single month of Lichess data. This short window results in partial player history, making it difficult to establish robust long-term rating averages or trend indicators.
*   **Over-reliance on Repeat-Player Statistics**: History and player identity features provide significant predictive power but only benefit repeat players. Unseen players suffer from high error rates as the model defaults to global means.
*   **Non-Interpretable Identity Hashes**: Player usernames and move text are mapped via `HashingVectorizer`. While this provides lightweight scalability and handles high cardinality, individual hash dimensions cannot be easily mapped back to specific strategic patterns or player names.
*   **No Live Engine Evaluations**: Stockfish features are excluded from the default production pipeline due to resource constraints. This limits the model's ability to assess exact tactical advantages in the middle game.
*   **Hidden Test Set Variance**: The hidden test set may have different player overlaps than the validation set. If the overlap is lower, models relying heavily on identity features will experience performance degradation.

---

## 2. Roadmap for Future Work

*   **Multi-Month Historical Databases**: Expanding the historical window to 3–6 months would compile richer player history profiles, reducing the proportion of unseen players and refining rating estimations.
*   **Explicit Draw Modeling**: Transitioning from binary classification to ordinal regression or multi-class classification (White Win / Draw / Black Win) would better capture chess dynamics.
*   **Richer Board Representation**: Moving beyond simple piece counts and basic mobility metrics to spatial board representations (e.g., convolutional layers or attention maps over squares).
*   **Unseen-Player Validation Folds**: Designing a specific cross-validation fold where the validation set contains *only* players unseen in the training fold, forcing the model to generalize based on game play rather than identities.
*   **Engine Integration at Scale**: If runtime constraints are relaxed, integrating Stockfish evaluations at move 3 (ply 6) and move 10 (ply 20) during training would inject an objective tactical feature into the classification tasks.
*   **Probability Calibration**: Applying Platt scaling or Isotonic Regression on the classifier outputs to ensure predicted probabilities match actual empirical win rates under all conditions.
*   **Deep Sequence Models**: Replacing the bag-of-words text hashing with sequence models (like LSTMs or lightweight Transformers) to model the exact chronological order of moves.
