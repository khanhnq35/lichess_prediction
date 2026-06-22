# Research

## Candidate Approaches

Several lightweight and heavier approaches were considered.

## 1. Elo Baselines

The simplest White-win baseline uses the standard Elo expected-score formula:

```text
p_white = 1 / (1 + 10 ** (-(WhiteElo - BlackElo) / 400))
```

This is useful as a baseline, but it is not identical to the binary target because Elo expected score treats draws as half a point while this task treats draws as non-White-wins.

For Elo regression, the mean baseline predicts the training mean `WhiteElo` and training mean `BlackElo`.

## 2. Linear Models

Logistic regression and Ridge regression are strong default choices:

- They are reproducible.
- They are easy to inspect.
- They work well with sparse hashed text features.
- They keep the solution small and dependency-light.

The strict lightweight profile uses these models.

## 3. Move Text And Board Features

Move text can encode opening choices and early tactical patterns. This project uses SAN plus UCI-style move tokens where available and encodes text with `HashingVectorizer`, avoiding a learned vocabulary file.

Board features are extracted after the legal prediction point:

- Material by side.
- Piece counts.
- Legal move count.
- Check flag.
- Side to move.
- Castling rights.
- Center occupancy and attacks.
- Move-behavior counts.

Enhanced board features were also tested:

- Piece-square table score.
- Pawn structure.
- Mobility.
- Development.
- King-safety proxies.

## 4. Clock And Time Pressure Features

Lichess PGN comments often contain clock data such as `[%clk H:MM:SS]` or `[%clk M:SS]`. Clock features are valid if they are computed only from moves already observed at the prediction point.

Clock features tested include:

- Last observed clock for each side.
- Total approximate time used.
- Average time per move.
- Clock difference.
- Missing clock counts.
- Time-pressure flags such as less than 15 seconds remaining.

These features helped the after-10 White-win classifier more than after-3 or Elo regression.

## 5. Causal Player History

Player history is highly informative, especially for Elo prediction. The key rule is causality:

1. Compute current-game history features from earlier eligible games only.
2. Make the prediction.
3. Update the player's history using the current game.

This avoids future-game leakage while allowing the model to use realistic information available before a later game in the same stream.

History features include:

- Prior game counts.
- Prior score rate.
- Side-specific win rates.
- Prior average opponent Elo.
- Prior average observed player Elo.
- Recent score rates over the last 10 and 30 games.

## 6. Player Identity

Player usernames are known before the game starts. Hashed player identity features were tested using text such as:

```text
white=<white_name> black=<black_name>
```

This can improve repeat-player prediction, especially Elo regression, but it may generalize less well to completely unseen players. The final no-Stockfish boosting profile does not rely on identity text for the selected best metrics.

## 7. Gradient Boosting Without Stockfish

LightGBM and XGBoost were tested as optional no-Stockfish models. They improve after-3, after-10, and Elo prediction substantially compared with the strict lightweight profile.

These dependencies are placed in `requirements-experiments.txt` rather than `requirements.txt`, so the strict assessment pipeline remains lightweight.

## 8. Stockfish And Heavy Models

Stockfish and heavier experimental models were investigated as references. They can improve after-10 outcome prediction because engine evaluation is highly informative after 20 plies.

They were not selected for the final submission pipeline because they require an external engine/cache and make reproducibility more fragile.

## Selected Direction

The selected submission strategy is dual:

- Keep a lightweight fallback profile that uses only the required Python packages.
- Report the best no-Stockfish boosting profile as the final high-performance configuration.

This gives both reproducibility and a stronger empirical result without relying on Stockfish or deep learning.

