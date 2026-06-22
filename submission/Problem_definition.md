# Problem Definition

## Assessment Goal

The task is to build a reproducible Python pipeline using the Lichess open database to train and evaluate models for four prediction tasks:

1. Probability that White wins before the game starts.
2. Probability that White wins after 3 full moves.
3. Probability that White wins after 10 full moves.
4. Elo prediction for both players after 10 full moves.

The pipeline must include data downloading, streaming decompression, PGN parsing, cleaning, feature extraction, training, validation, metrics, and prediction outputs.

## Data Source

The data source is the public Lichess open database:

```text
https://database.lichess.org/
```

This solution uses standard rated monthly PGN archives:

```text
https://database.lichess.org/standard/lichess_db_standard_rated_{YYYY-MM}.pgn.zst
```

The compressed `.zst` file is streamed and decompressed in memory. The decompressed PGN is not written to disk.

## Reproducible Month And Time Control

The default candidate months are `2023-01` through `2023-12`. A fixed random seed (`42`) selects the month reproducibly unless the user provides `--selected-month`.

For the final full-scale run reported in this submission:

- Selected month: `2023-11`
- Time control: `Blitz`
- Target eligible games: `100,000`
- Train/validation split: first `80%` train, last `20%` validation

## Eligibility Definition

A game is eligible if all conditions below are satisfied:

- It comes from the Lichess standard rated monthly PGN archive.
- The event header contains `Blitz`.
- The PGN can be parsed by `python-chess`.
- `WhiteElo` and `BlackElo` are present and valid integers.
- `Result` is one of `1-0`, `0-1`, or `1/2-1/2`.
- The mainline contains at least 20 plies, so all tasks can be evaluated on the same game set.

The pipeline stops after collecting the first `100,000` eligible games in chronological file order.

## Ambiguous Terms Clarified

The original statement leaves several details open. This project fixes them explicitly:

- "Random month" means reproducible random selection from a fixed candidate list using seed `42`.
- "One time-control" is interpreted as Blitz, detected with `"Blitz" in Event`.
- "3 moves" means 3 full chess moves, or 6 plies.
- "10 moves" means 10 full chess moves, or 20 plies.
- "Win rate of White" is modeled as binary probability:
  - `white_win = 1` if `Result == "1-0"`
  - `white_win = 0` for both Black wins and draws
- Elo prediction targets are the current-game `WhiteElo` and `BlackElo`.

## Leakage Constraints

The following features are never used:

- `Result` as an input feature.
- `Termination`.
- `WhiteRatingDiff` or `BlackRatingDiff`.
- Total game length.
- Moves after the prediction point.
- Future games.
- Validation rows for model or preprocessing fitting.

For Elo regression, the model also excludes current-game `WhiteElo`, `BlackElo`, `elo_diff`, and `mean_elo`. Causal player-history features are allowed only because they are computed before updating with the current game.

