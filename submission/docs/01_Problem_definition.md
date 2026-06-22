# Problem Definition

## Assessment Goal

The goal of this assessment is to build a reproducible Python pipeline using the public Lichess open database to train, validate, and generate predictions for four chess-related machine learning tasks:

1. Estimate the probability that White wins before the game starts.
2. Estimate the probability that White wins after 3 full moves.
3. Estimate the probability that White wins after 10 full moves.
4. Predict the Elo ratings of both players after observing the first 10 full moves.

The solution includes the full workflow from data downloading to final prediction outputs:

* download or stream the source data,
* decompress the `.zst` archive,
* parse PGN games,
* clean and filter eligible games,
* extract time-aware features,
* split data chronologically into training and validation sets,
* train models,
* evaluate metrics,
* write prediction and metrics artifacts.

The project is designed to be reproducible, leakage-safe, lightweight, and suitable for a quantitative research review.

---

## Data Source

The data source is the public Lichess open database:

```text
https://database.lichess.org/
```

This solution uses the standard rated monthly PGN archives:

```text
https://database.lichess.org/standard/lichess_db_standard_rated_{YYYY-MM}.pgn.zst
```

The compressed `.zst` archive is streamed and decompressed during processing. The decompressed PGN file is not written to disk, which keeps the solution lightweight and avoids storing large raw data files.

---

## Reproducible Month and Time Control

The assessment asks for a random month and one selected time-control. To make the result reproducible, this project defines a fixed candidate month list and uses a fixed random seed unless a month is explicitly provided by the user.

* Candidate months: `2023-01` through `2023-12`
* Random seed: `42`
* Optional override: `--selected-month YYYY-MM`
* Selected time-control: `Blitz`

For the final full-scale run reported in this submission:

* Selected month: `2023-11`
* Time-control: `Blitz`
* Target eligible games: `100,000`
* Train/validation split: first `80%` for training, last `20%` for validation

The pipeline stops after collecting the first `100,000` eligible games in chronological file order.

---

## Eligibility Definition

A game is considered eligible if all conditions below are satisfied:

* It comes from a Lichess standard rated monthly PGN archive.
* It belongs to the selected time-control category, `Blitz`.
* The PGN can be parsed successfully by `python-chess`.
* `WhiteElo` and `BlackElo` are present and valid integers.
* `Result` is one of:

  * `1-0`
  * `0-1`
  * `1/2-1/2`
* The mainline contains at least 20 plies, equivalent to 10 full moves.

The 20-ply requirement ensures that all four prediction tasks are evaluated on the same set of games. This keeps the comparison between before-game, after-3, after-10, and Elo prediction consistent.

---

## Ambiguous Terms Clarified

The original assessment statement leaves several implementation details open. This project resolves them explicitly as follows.

### Random Month

“Random month” is interpreted as reproducible random selection from a fixed candidate list using seed `42`. The user may override the selected month with `--selected-month`.

### Time-Control

“One time-control” is interpreted as `Blitz`. Games are filtered using Lichess PGN metadata, primarily the event/time-control information available in the PGN headers.

### 3 Moves and 10 Moves

Chess PGN moves can be interpreted either as plies or full moves. This project uses the standard chess interpretation:

* `3 moves` means 3 full moves, equivalent to 6 plies.
* `10 moves` means 10 full moves, equivalent to 20 plies.

Therefore:

* The after-3 model may only use information available up to ply 6.
* The after-10 model may only use information available up to ply 20.
* The Elo prediction model may only use information available up to ply 20, plus valid pre-game or causal historical information.

### White Win Rate

“Win rate of White” is modeled as a binary probability that White wins the game:

```text
white_win = 1 if Result == "1-0"
white_win = 0 if Result == "0-1" or Result == "1/2-1/2"
```

Draws are treated as non-White-wins. This is a modeling choice that turns the task into binary classification. The limitation is documented because Elo expected score treats draws differently from this binary target.

### Elo Prediction Target

The Elo prediction targets are the current-game PGN header values:

```text
WhiteElo
BlackElo
```

Although these Elo values are known in the PGN metadata, they are treated as prediction targets for the Elo regression task. The Elo model must not use the current-game `WhiteElo`, `BlackElo`, `elo_diff`, or `mean_elo` as input features.

The intended interpretation is:

> Given the first 10 full moves and any valid pre-game or causal historical information, estimate the Elo ratings of both players without directly using their current-game Elo headers as features.

---

## Prediction Times and Allowed Information

Each task has a different information boundary.

| Task                          | Prediction time | Allowed information                                                                                                                  |
| ----------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| White win before game         | Before move 1   | Pre-game metadata, time-control information, valid causal player history                                                             |
| White win after 3 full moves  | After ply 6     | Before-game information plus moves, board state, and clock information up to ply 6                                                   |
| White win after 10 full moves | After ply 20    | Before-game information plus moves, board state, and clock information up to ply 20                                                  |
| Elo after 10 full moves       | After ply 20    | Moves, board state, clock information up to ply 20, player identity, and causal player history; current-game Elo values are excluded |

No task may use information that would not be available at its prediction time.

---

## Validation Protocol

The data is split chronologically:

```text
First 80% eligible games  -> training set
Last 20% eligible games   -> validation set
```

A chronological split is used instead of a random split to better simulate a quantitative research setting, where models are trained on past observations and evaluated on later observations.

All preprocessing steps that require fitting, such as scaling or model fitting, are fit only on the training set. Validation rows are used only for evaluation.

---

## Leakage Constraints

The following fields are never used as input features:

* `Result`
* `Termination`
* `WhiteRatingDiff`
* `BlackRatingDiff`
* total game length
* total number of moves
* moves after the task-specific prediction point
* clock values after the task-specific prediction point
* future games
* validation rows for model or preprocessing fitting

For Elo regression, the model additionally excludes current-game rating-derived fields:

* `WhiteElo`
* `BlackElo`
* `elo_diff`
* `mean_elo`

Causal player-history features are allowed only if they are computed before updating the history with the current game. In other words, for game `N`, player-history features may use games `1` through `N-1`, but not game `N` itself or any future game.

---

## Baseline Definitions

The project reports model performance against simple baselines.

### Classification Baselines

For White-win prediction:

* Majority-class baseline: always predicts the majority validation class.
* Elo expected-score baseline: estimates White’s expected score from the pre-game Elo difference:

```text
E_white = 1 / (1 + 10 ** (-(WhiteElo - BlackElo) / 400))
```

The Elo expected-score baseline is not identical to the binary White-win target because the Elo formula treats draws as half a point, while this project treats draws as non-White-wins. However, it is still a strong and natural benchmark because it uses only pre-game rating information.

### Regression Baseline

For Elo prediction:

* Mean baseline: predicts the training-set mean Elo for both players.

A useful Elo model should substantially reduce MAE compared with this naive mean predictor.

---

## Output Artifacts

The pipeline writes the following core outputs:

```text
metrics.json
validation_predictions.csv
```

Depending on the run mode, it may also produce additional audit or explanation artifacts, such as:

```text
model_explainability.json
prediction_explanation_examples.json
calibration_bins_after10.csv
bootstrap_ci_after10.json
repeat_unseen_elo_diagnostics.csv
```

These additional artifacts are used for analysis and documentation, not as required raw data dependencies.

---

## Summary

This problem is framed as a time-aware, leakage-safe prediction task over Lichess Blitz games. The key design choices are:

* use a reproducible month selection process,
* collect the first `100,000` eligible Blitz games,
* define 3 moves as 6 plies and 10 moves as 20 plies,
* treat White-win prediction as binary classification,
* treat Elo prediction as regression without using current-game Elo as an input,
* use chronological validation,
* enforce strict feature availability at each prediction time.

These assumptions make the originally open-ended assessment precise, reproducible, and suitable for quantitative research evaluation.
