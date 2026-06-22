# Data Leakage Prevention and Chronological Validation

Data leakage is the most common reason why quantitative models fail when transitioning from research to production. Our pipeline implements strict, system-level boundaries to guarantee that no future or forbidden information is exposed during training.

## 1. Chronological Train/Validation Split
We use a **strict chronological split** rather than a random shuffle.
*   **Method**: Games are ordered by their sequence in the monthly streaming file. The first `80%` of games are assigned to the training set, and the remaining `20%` are reserved for validation.
*   **Why?**: Random splits leak information because players play multiple games throughout the month. Shuffle-splitting would allow games from day 30 to appear in training while games from day 1 appear in validation, leaking player form and rating trends.

## 2. Train-Only Fitting
All stateful preprocessors (imputers, scalers, vectorizers) and model parameters are **fit exclusively on the training set**.
*   Validation rows are passed through `Pipeline.predict()` or `Pipeline.predict_proba()`, applying only the transformations learned from the training data. This prevents validation-set stats (like median imputation values) from leaking into training.

## 3. Strict Feature Availability Boundaries
Features are partitioned strictly based on when they would be observed in a live-game environment:

*   **Before-Game Model**: Only uses features available *prior* to the first move: player Elos, time control, increment, and causal history.
*   **After-3 Moves Model**: Only uses features extracted up to ply 6. Moves after ply 6, clock times after ply 6, and eventual game length are completely invisible.
*   **After-10 Moves Model**: Only uses features extracted up to ply 20. Moves after ply 20, clock times after ply 20, and the final outcome are invisible.

## 4. Forbidden Post-Game Features
We enforce a strict blocklist of columns that are generated at the end of the game:
*   Game outcome (`result`, `white_win`)
*   Lichess rating changes (`WhiteRatingDiff`, `BlackRatingDiff`)
*   Game termination reason (`Termination`)
*   Total moves played / game duration

## 5. Elo Regression Leakage Safeguards
To evaluate our ability to estimate player ratings *during* a game, the Elo regression model must predict `white_elo` and `black_elo` without using them as features:
*   **Current Ratings Excluded**: Current game `white_elo`, `black_elo`, `elo_diff`, and `mean_elo` are strictly excluded from the regression features.
*   **Causal History Only**: History features (like `white_prior_avg_elo_seen`) are computed chronologically. For game $N$, a player's history features are computed using only games $1, \dots, N-1$. The outcome and ratings of game $N$ are updated in the player's history record *after* predictions for game $N$ are made.
*   **Leakage Guard**: An automated runtime check in `solution.py` audits the input columns of the Elo regression model and raises a `ValueError` if any forbidden rating or outcome column is detected.
