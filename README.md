# Lichess Blitz Prediction Pipeline

This project is a compact, reproducible solution for the Quantitative Research Intern take-home assessment. It streams a monthly Lichess standard rated PGN archive, extracts the first eligible Blitz games, trains four simple models, and writes validation metrics and predictions.

## Project Organization

The repository is organized so the production pipeline is easy to separate from experiments and historical outputs:

- `problem.md`: original assessment statement and main contract.
- `solution.py`: production pipeline and optional experiment modes.
- `docs/`: project map, traceability, artifact index, runbook, decision log, and submission scope.
- `docs/agents/`: handoff notes for Codex/agy collaboration.
- `artifacts/production/`: selected full-run production outputs.
- `artifacts/experiments/`: 10k experiments and heavy/Stockfish reference runs.
- `artifacts/audits/`, `artifacts/robustness/`, `artifacts/xai/`: supporting analysis outputs.
- `scripts/analysis/`: optional audit, robustness, and explainability scripts.
- `experiment/`: exploratory framework; not the source of truth for the lightweight final pipeline.

Start with `docs/PROJECT_MAP.md`, `docs/ARTIFACT_INDEX.md`, and `docs/DECISION_LOG.md` if continuing the project with another agent.

## Data Source

The pipeline uses the Lichess open database:

`https://database.lichess.org/`

Monthly files are streamed from:

`https://database.lichess.org/standard/lichess_db_standard_rated_{YYYY-MM}.pgn.zst`

The compressed `.zst` file is downloaded as a stream and decompressed in memory. The full decompressed PGN is never written to disk.

## Selected Month and Time-Control

Default configuration:

- Candidate months: `2023-01` through `2023-12`
- Random seed: `42`
- Selected month: chosen reproducibly from the candidate list unless `--selected-month` is provided
- Time-control: `Blitz`
- Target games: `100000`
- Train/validation split: first 80% train, last 20% validation

The selected month is printed and saved in `outputs/metrics.json`.

## Eligibility Definition

A game is eligible if it satisfies all of the following:

- Comes from the Lichess standard rated monthly PGN file
- Has `Blitz` in the `Event` header
- Has a valid PGN parse
- Has valid `WhiteElo` and `BlackElo`
- Has `Result` in `{"1-0", "0-1", "1/2-1/2"}`
- Has at least 10 full moves, interpreted as 20 plies

The pipeline stops after collecting the first `TARGET_GAMES` eligible games in file order.

## Move Definitions and Targets

- `3 moves` means 3 full chess moves, or 6 plies.
- `10 moves` means 10 full chess moves, or 20 plies.
- `white_win = 1` if `Result == "1-0"`.
- `white_win = 0` for black wins and draws.
- Elo regression targets are `WhiteElo` and `BlackElo`.

## Feature Engineering

Before-game white win features:

- `white_elo`
- `black_elo`
- `elo_diff`
- `mean_elo`
- `initial_time_seconds`
- `increment_seconds`
- `log_initial_time_seconds`

After-3 and after-10 white win features:

- All before-game features
- First 10 full moves as SAN plus UCI text tokens for the selected after-10 production model
- Board features at the prediction point:
  - Material for each side
  - Material difference
  - Piece counts by color and piece type
  - Legal move count
  - Check flag
  - Side to move
  - Castling rights
  - Fullmove number
  - Center occupancy and center attack counts
- Move-behavior features computed only from the allowed plies:
  - Captures
  - Checks
  - Castles by color
  - Move counts by queen, king, knight, bishop, rook, and pawn
- Optional lightweight enhanced board features at the prediction point:
  - Piece-square table score
  - Pawn structure
  - King safety
  - Piece mobility
  - Development counts
- Clock features from PGN comments within the allowed plies when selected

After-10 Elo regression features:

- First 10 full moves as SAN plus UCI text tokens
- Board features after 10 full moves
- Move-behavior features after 10 full moves
- Causal player-history features from earlier eligible games in the same stream
- Hashed player identity features
- Time-control features
- No `WhiteElo`, `BlackElo`, `elo_diff`, or `mean_elo`

## Causal Player-History Features

The pipeline maintains a lightweight history state for each player while eligible games are processed in chronological file order. For each current game, the pipeline first computes features from games already processed earlier, then updates both players' histories with the current game result. This makes the features causal with respect to the current game.

History features include prior game counts, all-game score rate, side-specific win rates, average prior opponent Elo, average prior observed player Elo, recent score rates over the last 10 and 30 prior games, and White-minus-Black differences for these quantities.

These features are also allowed in the Elo regression model because `prior_avg_elo_seen` and related history columns are computed before the current game. The Elo regression model still excludes the current `WhiteElo`, `BlackElo`, `elo_diff`, and `mean_elo`.

No future games are used. Validation rows are not used to fit models, scalers, imputers, or vectorizers. In validation, history features are computed online from games that occur earlier in the same chronological stream, matching an operational setting where earlier completed games are known before predicting a later game. The limitation is that a single downloaded month contains only partial player history, so many players have little or no prior history in the sampled stream.

## Optional Experiment Features

The `--run-experiments` mode evaluates a compact set of leakage-safe configurations on the requested sample size. It can toggle causal player-history features and hashed player-identity features per model.

Player identity features use pre-game usernames only:

```text
white=<white_name> black=<black_name>
```

They are encoded with `HashingVectorizer`, so no learned vocabulary or large model artifact is stored. Usernames are known before the game, so this is not post-game leakage. However, identity features mostly help repeat-player prediction and may not generalize to completely unseen players.

The `--run-clock-experiments` mode additionally tests clock features parsed from Lichess PGN comments such as `[%clk H:MM:SS]` or `[%clk M:SS]`. Clock features are only computed from observed clocks within the allowed prediction window: first 6 plies for after-3 models and first 20 plies for after-10/Elo models. They include remaining clock, approximate time used, average time per move, clock differences, missing counts, and availability flags. No future clock values or total game length are used.

The `--run-boosting-experiments` mode optionally tests LightGBM and XGBoost without Stockfish. These packages are listed in `requirements-experiments.txt`, not in the main `requirements.txt`, so the default assessment pipeline remains lightweight and uses only the required dependencies.

A full 100k verification run (`--run-boosting-experiments` with cache CSV) shows the following results:
- **T1 Before**: Baseline Logistic Regression (ROC-AUC **0.5788**) outperforms LightGBM (ROC-AUC **0.5772**).
- **T2 After-3**: XGBoost (ROC-AUC **0.5785**) improves over Baseline (ROC-AUC **0.5667**).
- **T3 After-10**: XGBoost (ROC-AUC **0.6228**) improves over Baseline (ROC-AUC **0.6107**).
- **T4 Elo**: LightGBM (Avg MAE **29.27 ELO**) dramatically improves over Ridge Baseline (Avg MAE **91.52 ELO**).

While Boosting brings substantial gains in T2, T3, and T4, the production path default continues to use simple, lightweight scikit-learn models. This avoids compiler and binary installation issues (e.g., LibOMP) on automated grading servers while keeping the Boosting pipeline available as an optional experiment appendix.


## Model Selection

Lightweight 10k experiments and subsequent full-scale comparisons were used to select the current production configuration while keeping the validation protocol fixed and leakage-safe. Heavy exploratory results from `experiment/` that use Stockfish, PyTorch, LightGBM, XGBoost, or ensembles are not part of the production solution.

Final selected models:

- White win before game: pre-game Elo/time-control features only, no history, no identity, no per-move clock, `LogisticRegression(C=1.0)`.
- White win after 3 moves: after-3 move/board features plus hashed player identity, no history, no clock, `LogisticRegression(C=0.25)`.
- White win after 10 moves: after-10 move/board features plus hashed player identity and `clk10_*` clock features, no history, `LogisticRegression(C=0.25)`.
- Elo after 10 moves: after-10 move/board features plus causal history and hashed player identity, no clock, Ridge regression. Current `WhiteElo`, `BlackElo`, `elo_diff`, and `mean_elo` remain excluded.

Clock features helped after-10 White-win prediction in the controlled 10k experiment, but not after-3 or Elo regression. Causal history features helped Elo regression substantially, but did not help White-win classification. Hashed player identity is pre-game information; it helped after-3 slightly, helped after-10 when combined with clock, and helped Elo regression when combined with history. Optional non-engine enhanced board features were implemented and tested, but the 10k verification run reduced after-10 ROC-AUC and Elo regression quality, so they are not selected in the default production feature sets.

Identity and history features can perform better for repeat players than for completely unseen players. This is expected and should be considered when interpreting results outside the sampled Lichess month.

## Model Choices

The solution intentionally uses simple, robust scikit-learn models:

- White win before game: `LogisticRegression`
- White win after 3 moves: `HashingVectorizer` for move text plus numeric features, then `LogisticRegression`
- White win after 10 moves: `HashingVectorizer` for move text plus numeric features, then `LogisticRegression`
- Elo after 10 moves: `HashingVectorizer` for move text plus numeric features, then multi-output `Ridge`

The hashing vectorizer uses a fixed-dimensional representation with `alternate_sign=False`, `lowercase=False`, `ngram_range=(1, 2)`, and a default of `2**15` hashed features. The feature count can be changed with `--hashing-features`.

The white-win classifiers use `LogisticRegression` with the sparse-friendly `liblinear` solver and an increased `max_iter=5000` to avoid convergence issues on hashed move-text features while keeping the model simple and reproducible.

No engine evaluation, Stockfish, deep learning, or post-game features are used.

## Baselines

The pipeline reports lightweight baselines in addition to trained models.

White win baseline:

```text
p_elo_expected = 1 / (1 + 10 ** (-(white_elo - black_elo) / 400))
```

This is the standard Elo expected score for White. It is useful but not exactly the same as the binary `white_win` target, because draws count as half a point in Elo expected score but are treated as non-white-wins here. The baseline is evaluated with ROC-AUC, log loss, and Brier score after clipping probabilities to `[1e-6, 1 - 1e-6]`.

Elo regression baseline:

- Predict validation `WhiteElo` as the training mean `WhiteElo`.
- Predict validation `BlackElo` as the training mean `BlackElo`.

The regression baseline is evaluated with MAE, RMSE, and R2.

## Validation Strategy and Leakage Prevention

The split is chronological/order-based:

- First 80% of eligible games: training
- Last 20% of eligible games: validation

Validation games are never used to train models or fit preprocessing. All imputers, scalers, vectorizers, and estimators live inside scikit-learn `Pipeline` and `ColumnTransformer` objects that are fit only on the training rows.

Leakage exclusions:

- No `Result` as a feature
- No `Termination`
- No `WhiteRatingDiff` or `BlackRatingDiff`
- No total game length
- No future moves beyond the requested prediction point
- Elo regression does not use Elo or Elo-derived columns as input features
- The after-3 model uses only `m3_*` features and the first 6 plies.
- The after-10 and Elo models use only `m10_*` features and the first 20 plies.

## Metrics

Classification metrics:

- ROC-AUC
- Log loss
- Brier score
- Accuracy

ROC-AUC measures ranking quality. Log loss and Brier score are more sensitive to whether predicted probabilities are well calibrated, so they can worsen even when ROC-AUC improves.

Elo regression metrics:

- MAE for WhiteElo
- MAE for BlackElo
- RMSE for WhiteElo
- RMSE for BlackElo
- R2 for both outputs

Metrics are printed and saved to `outputs/metrics.json`. Validation predictions are saved to `outputs/validation_predictions.csv`.

`metrics.json` also includes `probability_diagnostics` for the white-win models and Elo expected-score baseline. These summaries report min, max, mean, standard deviation, and selected quantiles of validation probabilities to make overconfidence or calibration problems easier to inspect.

The metrics file uses these top-level sections:

- `run_config`
- `dataset_summary`
- `baselines`
  - `majority_class_baseline`
  - `elo_expected_score_baseline`
  - `elo_mean_baseline`
- `models`
  - `white_win_before_game`
  - `white_win_after_3_moves`
  - `white_win_after_10_moves`
  - `elo_after_10_moves`

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full default pipeline:

```bash
python solution.py
```

Run a quick smoke test with fewer games:

```bash
python solution.py --target-games 100 --selected-month 2023-01
```

Smoke-test outputs from `--target-games 100` are only meant to validate that streaming, parsing, features, training, metrics, and output writing work end to end. They should not be interpreted as final model quality.

Run the final assessment-scale pipeline:

```bash
python solution.py --target-games 100000 --output-dir outputs_full
```

Run the optional no-Stockfish boosting profile:

```bash
pip install -r requirements-experiments.txt
python solution.py --target-games 100000 --output-dir outputs_full_boosting --model-profile boosting
```

The default profile remains `lightweight`; the boosting profile requires optional LightGBM/XGBoost dependencies and is intended for improved experiments or appendix results.

Useful CLI options:

```bash
python solution.py \
  --random-seed 42 \
  --candidate-months 2023-01,2023-02,2023-03 \
  --selected-month 2023-02 \
  --time-control Blitz \
  --target-games 100000 \
  --train-ratio 0.8 \
  --output-dir outputs \
  --hashing-features 32768
```

Run the compact 10k experiment grid:

```bash
python solution.py --target-games 10000 --selected-month 2023-01 --output-dir outputs_10k_experiments --run-experiments
```

Experiment outputs:

- `outputs_10k_experiments/experiment_results.csv`
- `outputs_10k_experiments/best_config.json`

Run the compact 10k clock-feature experiment:

```bash
python solution.py --target-games 10000 --selected-month 2023-01 --output-dir outputs_10k_clock_experiments --run-clock-experiments
```

Clock experiment outputs:

- `outputs_10k_clock_experiments/experiment_results.csv`
- `outputs_10k_clock_experiments/best_config.json`

## Outputs

Generated files:

- `outputs/metrics.json`
- `outputs/validation_predictions.csv`

Do not include raw downloaded data, decompressed PGN files, or large model binaries in the final submission.

## Assumptions and Limitations

- The standard rated Lichess archive is treated as satisfying the standard rated game requirement.
- `Blitz` filtering is done using the `Event` header, as requested.
- Move text includes UCI tokens for stability plus SAN tokens for familiar chess notation.
- The models are baseline statistical models designed for reproducibility and leakage control, not maximum chess strength.
- Full 100,000-game execution depends on network speed and the assessment environment. The smoke-test path uses the same code as the full run.

## LLM Usage Disclosure

ChatGPT GPT-5.5 Thinking was used to clarify the task, design the pipeline, identify leakage risks, review assumptions, and draft documentation. The LLM did not train the model, did not access validation data, and did not evaluate private assessment data. Final code should be reviewed and executed by the user.
