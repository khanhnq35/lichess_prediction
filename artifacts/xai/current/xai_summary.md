# Model Explainability (XAI) Summary

This document explains the feature groups that drive predictions for each of the four quantitative tasks in the Lichess Blitz prediction pipeline.

## 1. Feature Importance Summary by Task

### Before-Game Prediction (`white_win_before`)
*   **Most Important Features**: Elo rating features.
    *   `elo_diff` (Coefficient: `+0.4495`): The rating difference between White and Black is the single most dominant factor.
    *   `black_elo` (Coefficient: `-0.0753`): Higher opponent rating reduces White's win probability.
    *   `white_elo` (Coefficient: `+0.0670`): Higher White rating increases White's win probability.
*   **Log-Odds Interpretation**: A positive coefficient indicates that higher values shift the log-odds of a White victory upward. The Elo rating difference acts as the primary causal prior. Time control configurations (`increment_seconds` coefficient: `+0.0152`) have a tiny but positive effect under longer blitz increments.

### After 3 Full Moves (`white_win_after_3`)
*   **Most Important Features**: Elo rating remains dominant (`elo_diff`: `+0.4489`), but early board state features begin to exert influence.
    *   `m3_king_move_count` (Coefficient: `-0.0777`): Moving the king early has a strong negative impact on White's chances because it permanently forfeits castling rights.
    *   `m3_black_can_castle_queenside` (Coefficient: `-0.0558`) and `m3_black_can_castle_kingside` (Coefficient: `-0.0431`): Black retaining castling rights reduces White's win probability.
    *   `m3_white_rooks` (Coefficient: `+0.0560`): White maintaining rook activity has a positive influence.
    *   `m3_white_center_occupancy` (Coefficient: `+0.0245`): Occupying center squares early (d4, e4, d5, e5) favors White.

### After 10 Full Moves (`white_win_after_10`)
*   **Most Important Features**: At move 10, the board state and clock consumption features become highly significant.
    *   **Clock Features**:
        *   `clk10_black_min_clock` (Coefficient: `-0.4194`): A higher remaining minimum clock for Black reduces White's win probability (Black is faster).
        *   `clk10_white_min_clock` (Coefficient: `+0.3760`): A higher remaining minimum clock for White increases White's win probability (White is faster).
        *   `clk10_clock_diff_last` (Coefficient: `-0.2371`): Points to a negative impact if White is behind on clock time relative to Black.
        *   `clk10_clock_used_diff` (Coefficient: `-0.1020`): A positive difference (White using more time than Black) decreases White's probability.
    *   **Board State Features**:
        *   `m10_material_diff` (Coefficient: `+0.1598`): A material advantage for White (White material - Black material) is the most dominant board-state feature.
        *   `m10_castle_count_white` (Coefficient: `+0.1457`): White having successfully castled has a strong positive effect.
        *   `m10_castle_count_black` (Coefficient: `-0.1098`): Black having castled has a strong negative effect.
        *   `m10_legal_moves_count` (Coefficient: `+0.0981`): Higher mobility (number of legal moves) correlates with control and increases probability.

### Elo Prediction (`white_elo_after_10` and `black_elo_after_10`)
*   **Most Important Features**: The model behaves like a causal historical prior, anchoring heavily on player history rather than temporary board states.
    *   **White Elo Prediction**:
        *   `white_prior_avg_elo_seen` (Coefficient: `+163.76`): The average rating of the opponents White played in previous games is the strongest predictor of White rating.
        *   `black_prior_avg_elo_seen` (Coefficient: `+122.92`): Opponent prior averages also correlate due to Lichess matchmaking pairing players of similar ratings.
        *   `white_prior_recent_score_rate_10` (Coefficient: `+9.30`): Recent performance over the last 10 games is a positive corrector.
    *   **Black Elo Prediction**:
        *   `black_prior_avg_elo_seen` (Coefficient: `+163.66`): Strongest anchor for Black rating.
        *   `white_prior_avg_elo_seen` (Coefficient: `+123.00`): Matchmaking-derived correlation.
        *   `black_prior_recent_score_rate_10` (Coefficient: `+9.24`): Recent corrector.
    *   **Board State Contribution**: Board characteristics at move 10 have extremely small coefficients (e.g., `m10_legal_moves_count` is `+6.41`), which indicates the model is highly robust to short-term board variance and relies on solid, pre-game history for rating estimation.

---

## 2. Leakage Audit

### Are there any suspicious features suggesting leakage?
No. The model passes the leakage check with a perfect audit score:
1.  **Strict Elo Exclusion**: Current game Elo values (`white_elo`, `black_elo`, `elo_diff`, `mean_elo`) are strictly excluded from the Elo regression model's input features. The Elo model only uses time controls, player history features computed *before* the current game, and moves/clock features up to ply 20.
2.  **No Game Outcome Features**: Post-game columns like `WhiteRatingDiff`, `BlackRatingDiff`, `Result`, `Termination`, or final move lengths are entirely excluded from all models.
3.  **Hashed Player Pair Hashing**: Player identities are hashed via `HashingVectorizer` to capture player-specific rating effects without introducing future data or circular labels.
4.  **Causal History Generation**: Player history stats (prior games, prior win rates, average opponent rating) are updated chronologically. A player's history feature for game $N$ is computed using only games $1$ to $N-1$, preventing future-information leak.
