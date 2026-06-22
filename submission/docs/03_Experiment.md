# Experiment Report

## 1. Executive Summary

This document summarizes the experiments conducted for the Lichess Blitz prediction assessment. The goal was to compare several modeling approaches and select a final pipeline that balances predictive performance, reproducibility, leakage safety, and portability.

The assessment contains four prediction tasks:

1. Predict the probability that White wins before the game starts.
2. Predict the probability that White wins after 3 full moves.
3. Predict the probability that White wins after 10 full moves.
4. Predict both players' Elo ratings after 10 full moves.

The final selected profile is a **portable no-Stockfish `report_best` profile**. It uses stronger sklearn tabular models than the strict lightweight baseline, but it does not require Stockfish, LightGBM, or XGBoost. This provides a good trade-off between empirical performance, reproducibility, leakage safety, and submission portability.

Final selected submission profile:

| Task                        | Final selected model | Feature profile                                  | Main validation metric |
| --------------------------- | -------------------- | ------------------------------------------------ | ---------------------: |
| T1 White win before game    | Logistic Regression + causal history | Pre-game + causal history features |       ROC-AUC `0.5792` |
| T2 White win after 3 moves  | LogisticRegression(C=0.5) | After-3 enhanced + clock features, no Stockfish |       ROC-AUC `0.5796` |
| T3 White win after 10 moves | sklearn HistGradientBoostingClassifier | After-10 enhanced + clock features, no Stockfish |       ROC-AUC `0.6217` |
| T4 Elo after 10 moves       | sklearn RandomForestRegressor | Elo-safe enhanced + causal history features |        Avg MAE `28.83` |

The strongest exploratory after-10 classifier used Stockfish and reached ROC-AUC around `0.6483`. However, Stockfish was not selected for the final default submission because it requires an external engine or cached engine evaluations, which makes the pipeline less portable. The older LightGBM/XGBoost no-Stockfish profile remains useful as a comparison profile, but the final submitted `Solution.py` defaults to `report_best` for a dependency-light sklearn implementation.

The final selected solution prioritizes:

* strong validation performance,
* no required chess-engine dependency,
* chronological validation,
* leakage-safe feature construction,
* compact submission size,
* and clear reproducibility.

---

## 2. Shared Experiment Protocol

All major experiments used the same validation protocol.

### Dataset

* Data source: Lichess standard rated monthly PGN archive.
* Selected month: `2023-11`.
* Time-control: `Blitz`.
* Target eligible games: `100,000`.
* Eligible games were collected in chronological file order.

For the final full-scale run:

| Quantity                 |      Value |
| ------------------------ | ---------: |
| Parsed games             |  `213,463` |
| Header-eligible games    |  `104,005` |
| Final eligible games     |  `100,000` |
| Training rows            |   `80,000` |
| Validation rows          |   `20,000` |
| Train positive rate      | `0.493950` |
| Validation positive rate | `0.496400` |

### Split

The dataset was split chronologically:

```text
First 80% eligible games  -> training set
Last 20% eligible games   -> validation set
```

A chronological split was used instead of a random split to better approximate a real predictive setting. The model is trained on earlier games and evaluated on later games.

### Leakage Controls

The following leakage rules were enforced across experiments:

* Do not use validation rows for fitting models or preprocessing.
* Do not use the final `Result` as an input feature.
* Do not use `Termination`.
* Do not use `WhiteRatingDiff` or `BlackRatingDiff`.
* Do not use total game length.
* Do not use moves after the task-specific prediction point.
* Do not use clock values after the task-specific prediction point.
* Do not use future games.
* For Elo regression, do not use current-game `WhiteElo`, `BlackElo`, `elo_diff`, or `mean_elo` as input features.
* Causal player history must be computed before updating the player-history store with the current game.

### Metrics

Classification metrics:

* ROC-AUC,
* log loss,
* Brier score,
* accuracy.

Elo regression metrics:

* White MAE,
* Black MAE,
* average MAE,
* RMSE,
* R².

ROC-AUC was used as the primary classification metric because the White-win tasks are probabilistic and ranking-oriented. Log loss and Brier score were used to evaluate probability quality. Accuracy was reported for reference but was not the main selection metric.

For Elo regression, MAE was used as the primary metric because it is directly interpretable in Elo points.

---

## 3. Feature Groups Tested

Several feature families were tested.

| Feature group            | Description                                                                | Leakage status                                  |
| ------------------------ | -------------------------------------------------------------------------- | ----------------------------------------------- |
| Pre-game features        | Elo-derived features and time-control metadata                             | Allowed for White-win classification            |
| Move text features       | SAN/UCI-style early move tokens encoded with hashing                       | Only uses moves up to the prediction point      |
| Basic board features     | Material, piece counts, legal moves, check flags, castling rights          | Extracted only at the legal prediction point    |
| Enhanced board features  | Mobility, development, king-safety proxies, center control, pawn structure | Extracted only at the legal prediction point    |
| Clock features           | Clock remaining, time used, time-pressure proxies                          | Uses only clock data up to the prediction point |
| Causal player history    | Prior games, prior score rate, recent performance, prior observed Elo      | Computed before current-game update             |
| Player identity features | Hashed usernames or identity tokens                                        | Known before game, but repeat-player-sensitive  |
| Stockfish features       | Engine evaluation after 3 or 10 moves                                      | Exploratory only, not final default             |

### Feature Boundaries by Task

| Task            | Prediction point               | Allowed information                                         |
| --------------- | ------------------------------ | ----------------------------------------------------------- |
| T1 Before-game  | Before move 1                  | Pre-game information only                                   |
| T2 After-3      | After 3 full moves / 6 plies   | Information up to ply 6                                     |
| T3 After-10     | After 10 full moves / 20 plies | Information up to ply 20                                    |
| T4 Elo after 10 | After 10 full moves / 20 plies | Information up to ply 20, excluding current-game Elo inputs |

---

## 4. Model Families Tested

The experiments compared several model families.

### Linear Models

Linear models were used as the strict lightweight baseline.

Classification:

```text
LogisticRegression(
    solver="liblinear",
    max_iter=5000,
    random_state=42
)
```

Elo regression:

```text
Ridge(alpha=10.0)
```

These models are simple, reproducible, and easy to audit. They work well with sparse hashed text features and are suitable for a compact fallback profile.

### Tree-Based and Boosting Models

The experiments also tested nonlinear tabular models:

* LightGBM,
* XGBoost,
* HistGradientBoosting,
* RandomForest,
* GradientBoosting.

The motivation for testing tree-based models was that chess features can interact nonlinearly. For example, material balance, castling rights, clock pressure, and player history may jointly affect outcome probability.

### Stockfish-Based Models

Stockfish was tested as an exploratory feature generator. It evaluates a board position and returns engine-based positional information. These features are especially useful after 10 moves because the board state already contains meaningful tactical and strategic information.

However, Stockfish was not selected for the final default profile because it requires an external engine or cached evaluations.

---

## 5. Phase 1 — Baseline Models

### Goal

The first experiment established simple baseline models for all four tasks.

### Setup

| Task                     | Model                                           | Feature set                |
| ------------------------ | ----------------------------------------------- | -------------------------- |
| T1 Before-game White win | Logistic Regression `C=1.0`                     | Pre-game features          |
| T2 After-3 White win     | Logistic Regression `C=0.25` + hashed move text | After-3 features           |
| T3 After-10 White win    | Logistic Regression `C=0.25` + hashed move text | After-10 features          |
| T4 Elo after 10          | Ridge `alpha=10.0` + hashed move text           | Elo-safe after-10 features |

### Results

| ID | Task | Model                   | Features     |  ROC-AUC | Log loss |    Brier | Accuracy | Avg MAE |
| -- | ---- | ----------------------- | ------------ | -------: | -------: | -------: | -------: | ------: |
| B1 | T1   | LogReg `C=1.0`          | Pre-game     | `0.5788` | `0.6788` | `0.2433` | `0.5526` |     n/a |
| B2 | T2   | LogReg `C=0.25` + Hash  | After-3      | `0.5741` | `0.6805` | `0.2440` | `0.5493` |     n/a |
| B3 | T3   | LogReg `C=0.25` + Hash  | After-10     | `0.6149` | `0.6675` | `0.2380` | `0.5765` |     n/a |
| B4 | T4   | Ridge `alpha=10` + Hash | Elo after-10 |      n/a |      n/a |      n/a |      n/a | `90.51` |

### Conclusion

The baseline results showed that:

* Pre-game Elo is already a meaningful White-win signal.
* After-10 information is more predictive than after-3 information.
* Simple linear models provide a strong reproducible baseline.
* Elo prediction can be modeled much better than a naive mean baseline when early-game and history features are available.

---

## 6. Phase 2 — Enhanced Feature Engineering

### Goal

The second experiment tested whether handcrafted chess features and causal history improve the baseline models.

### Added Features

The enhanced feature set included:

* causal player history,
* material balance,
* mobility,
* development proxies,
* king-safety proxies,
* center control,
* pawn-structure proxies,
* castling information,
* move-behavior counts,
* clock and time-pressure features where applicable.

### Results

| ID | Task | Model                    | Features                 |  ROC-AUC | Log loss |    Brier | Accuracy | Avg MAE |
| -- | ---- | ------------------------ | ------------------------ | -------: | -------: | -------: | -------: | ------: |
| F1 | T1   | LogReg `C=1.0` + History | Before + history         | `0.5792` | `0.6787` | `0.2432` | `0.5522` |     n/a |
| F2 | T2   | LogReg `C=0.5`           | After-3 enhanced         | `0.5797` | `0.6787` | `0.2431` | `0.5552` |     n/a |
| F3 | T2   | LogReg `C=0.25` + Hash   | After-3 enhanced + text  | `0.5741` | `0.6805` | `0.2440` | `0.5493` |     n/a |
| F4 | T3   | LogReg `C=0.25` + Hash   | After-10 enhanced + text | `0.6149` | `0.6675` | `0.2380` | `0.5765` |     n/a |
| F5 | T4   | Ridge `alpha=10` + Hash  | Elo enhanced + text      |      n/a |      n/a |      n/a |      n/a | `90.51` |

### Conclusion

Enhanced features helped some tasks but not all of them under linear models.

Key observations:

* T1 improved only slightly from `0.5788` to `0.5792`.
* T2 improved from `0.5741` to `0.5797` with enhanced numeric features.
* T3 did not improve much under the linear setup.
* T4 remained around `90.51` MAE with Ridge.

This suggested that nonlinear models might be needed to better exploit the enhanced feature set.

---

## 7. Phase 3 — Tree-Based Models

### Goal

The third experiment tested whether nonlinear models could use enhanced chess features better than linear models.

### Models Tested

* LightGBM,
* XGBoost,
* HistGradientBoosting,
* RandomForest,
* GradientBoosting.

---

### 7.1 T1 Before-Game Results

| Model            | Features         |  ROC-AUC | Log loss |    Brier | Accuracy |
| ---------------- | ---------------- | -------: | -------: | -------: | -------: |
| LogReg + history | Before + history | `0.5792` | `0.6787` | `0.2432` | `0.5522` |
| LightGBM         | Before + history | `0.5718` | `0.6819` | `0.2446` | `0.5442` |
| XGBoost          | Before + history | `0.5508` | `0.6987` | `0.2516` | `0.5318` |
| HistGB           | Before + history | `0.5765` | `0.6800` | `0.2438` | `0.5474` |
| RandomForest     | Before + history | `0.5453` | `0.7053` | `0.2549` | `0.5299` |
| GradientBoosting | Before + history | `0.5773` | `0.6796` | `0.2436` | `0.5461` |

Conclusion: Tree models did not beat logistic regression before the game. The pre-game signal is mostly driven by Elo difference, where logistic regression is stable and sufficient.

---

### 7.2 T2 After-3 Results

| Model            | Features         |  ROC-AUC | Log loss |    Brier | Accuracy |
| ---------------- | ---------------- | -------: | -------: | -------: | -------: |
| LogReg enhanced  | After-3 enhanced | `0.5797` | `0.6787` | `0.2431` | `0.5552` |
| LightGBM         | After-3 enhanced | `0.5750` | `0.6798` | `0.2438` | `0.5502` |
| XGBoost          | After-3 enhanced | `0.5618` | `0.6889` | `0.2477` | `0.5379` |
| HistGB           | After-3 enhanced | `0.5776` | `0.6790` | `0.2433` | `0.5497` |
| RandomForest     | After-3 enhanced | `0.5462` | `0.7068` | `0.2556` | `0.5342` |
| GradientBoosting | After-3 enhanced | `0.5796` | `0.6786` | `0.2432` | `0.5526` |

Conclusion: After 3 moves, the signal is still limited. Enhanced features help modestly, but the task remains difficult.

---

### 7.3 T3 After-10 Results

| Model            | Features          |  ROC-AUC | Log loss |    Brier | Accuracy |
| ---------------- | ----------------- | -------: | -------: | -------: | -------: |
| LogReg baseline  | After-10 + text   | `0.6149` | `0.6675` | `0.2380` | `0.5765` |
| LightGBM         | After-10 enhanced | `0.6201` | `0.6655` | `0.2370` | `0.5803` |
| XGBoost          | After-10 enhanced | `0.6070` | `0.6759` | `0.2414` | `0.5731` |
| HistGB           | After-10 enhanced | `0.6203` | `0.6652` | `0.2369` | `0.5791` |
| RandomForest     | After-10 enhanced | `0.6030` | `0.6737` | `0.2406` | `0.5677` |
| GradientBoosting | After-10 enhanced | `0.6215` | `0.6651` | `0.2368` | `0.5810` |
| LightGBM tuned   | After-10 enhanced | `0.6227` | `0.6639` | `0.2364` | `0.5807` |

Conclusion: After 10 moves, nonlinear models begin to outperform logistic regression. Tuned LightGBM and GradientBoosting improve over the linear baseline.

---

### 7.4 T4 Elo Results

| Model            | Features          | White MAE | Black MAE | Avg MAE | Avg RMSE |   Avg R² |
| ---------------- | ----------------- | --------: | --------: | ------: | -------: | -------: |
| Ridge baseline   | Elo-safe features |   `90.10` |   `90.91` | `90.51` | `132.19` | `0.8711` |
| LightGBM         | Elo enhanced      |   `29.05` |   `28.98` | `29.01` |  `80.58` | `0.9521` |
| XGBoost          | Elo enhanced      |   `31.12` |   `31.82` | `31.47` |  `82.26` | `0.9501` |
| HistGB           | Elo enhanced      |   `29.06` |   `29.17` | `29.11` |  `80.58` | `0.9521` |
| RandomForest     | Elo enhanced      |   `26.44` |   `26.54` | `26.49` |  `80.68` | `0.9520` |
| GradientBoosting | Elo enhanced      |   `37.32` |   `37.76` | `37.54` |  `86.55` | `0.9448` |

Conclusion: Nonlinear models dramatically improve Elo prediction. The main signal comes from causal player history and repeat-player patterns. This result is valid under the chronological validation protocol, but should be interpreted with repeat-player and hidden-distribution caveats.

---

## 8. Phase 4 — Stockfish Exploratory Features

### Goal

The fourth experiment measured how much a chess engine can improve performance when engine evaluation is available.

Stockfish features were treated as **exploratory** because Stockfish requires an external chess engine or cached engine evaluations.

### Stockfish Features Tested

| Feature type                  | Description                                        |
| ----------------------------- | -------------------------------------------------- |
| After-3 Stockfish evaluation  | Engine evaluation of the board after 3 full moves  |
| After-10 Stockfish evaluation | Engine evaluation of the board after 10 full moves |
| Centipawn score               | Numeric positional advantage score                 |
| Mate score                    | Engine-detected mate-related score                 |

### Results

| ID | Task | Model                        | Features                      |           Metric |
| -- | ---- | ---------------------------- | ----------------------------- | ---------------: |
| S1 | T2   | GradientBoosting + Stockfish | After-3 enhanced + Stockfish  | ROC-AUC `0.5832` |
| S2 | T3   | GradientBoosting + Stockfish | After-10 enhanced + Stockfish | ROC-AUC `0.6480` |
| S3 | T4   | RandomForest + Stockfish     | Elo enhanced + Stockfish      |  Avg MAE `26.42` |

### Conclusion

Stockfish helped most for after-10 outcome prediction.

Key observations:

* After 3 moves, Stockfish adds only a small gain.
* After 10 moves, Stockfish adds a large gain because the position contains meaningful tactical and strategic information.
* For Elo regression, Stockfish adds almost no value because causal history is already the dominant signal.
* Stockfish was not selected for the final default submission because it requires an external engine or precomputed cache.

---

## 9. Phase 5 — Ensemble and Stacking

### Goal

The fifth experiment tested whether combining strong tree models improves performance.

### Results

| ID | Task | Model             | Features          |           Metric |
| -- | ---- | ----------------- | ----------------- | ---------------: |
| E1 | T3   | Voting ensemble   | After-10 enhanced | ROC-AUC `0.6183` |
| E2 | T3   | Stacking ensemble | After-10 enhanced | ROC-AUC `0.6209` |
| E3 | T4   | Voting regressor  | Elo enhanced      |  Avg MAE `27.19` |

### Conclusion

Ensembles did not justify their added complexity.

Key observations:

* Voting and stacking did not beat the best tuned after-10 classifier.
* Voting regression was strong but did not justify the additional complexity over a single final regressor.
* Stacking makes leakage control and documentation more complicated.
* Therefore, ensembles were excluded from the final selected pipeline.

---

## 10. Phase 6 — No-Stockfish Boosting Experiment

### Goal

The sixth experiment searched for the best no-Stockfish model profile. This was the most important experiment for final model selection because it avoids the external-engine dependency while still improving over the strict lightweight baseline.

The experiment compared:

* current lightweight production models,
* conservative LightGBM/XGBoost,
* balanced LightGBM/XGBoost.

All models were evaluated on the same 100k `2023-11` dataset with the same chronological split.

---

### 10.1 T1 Before-Game

| Config                                 | Algorithm          | Features               |  ROC-AUC | Log loss |    Brier | Accuracy |
| -------------------------------------- | ------------------ | ---------------------- | -------: | -------: | -------: | -------: |
| `production_logreg_C1.0`               | LogisticRegression | Production before-game | `0.5788` | `0.6788` | `0.2433` | `0.5526` |
| `lightgbm_conservative_before_history` | LightGBM           | Before + history       | `0.5778` | `0.6794` | `0.2435` | `0.5487` |
| `xgboost_conservative_before_history`  | XGBoost            | Before + history       | `0.5774` | `0.6794` | `0.2435` | `0.5466` |
| `lightgbm_balanced_before_history`     | LightGBM           | Before + history       | `0.5690` | `0.6839` | `0.2455` | `0.5458` |
| `xgboost_balanced_before_history`      | XGBoost            | Before + history       | `0.5703` | `0.6819` | `0.2446` | `0.5438` |

Conclusion: Logistic Regression remains the best T1 model. Pre-game prediction is mostly driven by Elo difference, and boosting does not improve generalization.

---

### 10.2 T2 After-3

| Config                                  | Algorithm          | Features           |  ROC-AUC | Log loss |    Brier | Accuracy |
| --------------------------------------- | ------------------ | ------------------ | -------: | -------: | -------: | -------: |
| `production_logreg_identity_C0.25`      | LogisticRegression | Production after-3 | `0.5667` | `0.6837` | `0.2455` | `0.5428` |
| `lightgbm_conservative_after3_enhanced` | LightGBM           | After-3 enhanced   | `0.5779` | `0.6794` | `0.2435` | `0.5498` |
| `xgboost_conservative_after3_enhanced`  | XGBoost            | After-3 enhanced   | `0.5787` | `0.6793` | `0.2434` | `0.5504` |
| `lightgbm_balanced_after3_enhanced`     | LightGBM           | After-3 enhanced   | `0.5669` | `0.6827` | `0.2451` | `0.5433` |
| `xgboost_balanced_after3_enhanced`      | XGBoost            | After-3 enhanced   | `0.5714` | `0.6811` | `0.2443` | `0.5461` |

Conclusion: Within the LightGBM/XGBoost no-Stockfish experiment, Conservative XGBoost was the strongest T2 candidate. It remains useful as a comparison profile, but it was not chosen as the final submitted default because the final `report_best` profile prioritizes sklearn-only portability.

---

### 10.3 T3 After-10

| Config                                         | Algorithm          | Features                  |  ROC-AUC | Log loss |    Brier | Accuracy |
| ---------------------------------------------- | ------------------ | ------------------------- | -------: | -------: | -------: | -------: |
| `production_logreg_identity_clock_C0.25`       | LogisticRegression | Production after-10       | `0.6107` | `0.6698` | `0.2391` | `0.5718` |
| `lightgbm_conservative_after10_enhanced_clock` | LightGBM           | After-10 enhanced + clock | `0.6208` | `0.6654` | `0.2369` | `0.5796` |
| `lightgbm_balanced_after10_enhanced_clock`     | LightGBM           | After-10 enhanced + clock | `0.6211` | `0.6652` | `0.2369` | `0.5808` |
| `xgboost_conservative_after10_enhanced_clock`  | XGBoost            | After-10 enhanced + clock | `0.6202` | `0.6660` | `0.2371` | `0.5796` |
| `xgboost_balanced_after10_enhanced_clock`      | XGBoost            | After-10 enhanced + clock | `0.6219` | `0.6642` | `0.2364` | `0.5803` |

Conclusion: Within the LightGBM/XGBoost no-Stockfish experiment, Balanced XGBoost gave the best after-10 ROC-AUC. The final `report_best` profile uses sklearn HistGradientBoosting instead, trading a very small metric difference for fewer optional dependencies.

---

### 10.4 T4 Elo

| Config                                       | Algorithm | Features               | White MAE | Black MAE | Avg MAE | White R² | Black R² |
| -------------------------------------------- | --------- | ---------------------- | --------: | --------: | ------: | -------: | -------: |
| `production_ridge_history_identity`          | Ridge     | Production Elo         |   `91.05` |   `91.97` | `91.51` | `0.8709` | `0.8687` |
| `lightgbm_conservative_elo_enhanced_history` | LightGBM  | Elo enhanced + history |   `33.26` |   `33.16` | `33.21` | `0.9446` | `0.9435` |
| `lightgbm_balanced_elo_enhanced_history`     | LightGBM  | Elo enhanced + history |   `29.24` |   `29.38` | `29.31` | `0.9503` | `0.9499` |
| `xgboost_conservative_elo_enhanced_history`  | XGBoost   | Elo enhanced + history |   `36.99` |   `37.85` | `37.42` | `0.9397` | `0.9378` |
| `xgboost_balanced_elo_enhanced_history`      | XGBoost   | Elo enhanced + history |   `30.82` |   `31.00` | `30.91` | `0.9489` | `0.9483` |

Conclusion: Balanced LightGBM dramatically improved over Ridge in this experiment. Later report-best experiments showed RandomForest was the stronger portable Elo regressor, so the final `report_best` profile uses RandomForest instead.

---

## 11. Phase 7 — Stockfish vs No-Stockfish A/B Test

### Goal

The seventh experiment quantified whether Stockfish is worth adding to the final solution.

### 11.1 T2 After-3 A/B Results

| Model                | No Stockfish AUC | With Stockfish AUC | Delta AUC |
| -------------------- | ---------------: | -----------------: | --------: |
| LogisticRegression   |         `0.5817` |           `0.5818` | `+0.0002` |
| GradientBoosting     |         `0.5811` |           `0.5827` | `+0.0015` |
| HistGradientBoosting |         `0.5807` |           `0.5826` | `+0.0019` |
| RandomForest         |         `0.5723` |           `0.5731` | `+0.0009` |
| LightGBM             |         `0.5818` |           `0.5831` | `+0.0013` |
| XGBoost              |         `0.5819` |           `0.5831` | `+0.0013` |

Conclusion: Stockfish has minimal impact after 3 moves. The improvement is too small to justify making Stockfish required.

### 11.2 T3 After-10 A/B Results

| Model                | No Stockfish AUC | With Stockfish AUC | Delta AUC |
| -------------------- | ---------------: | -----------------: | --------: |
| LogisticRegression   |         `0.6188` |           `0.6447` | `+0.0258` |
| GradientBoosting     |         `0.6219` |           `0.6486` | `+0.0267` |
| HistGradientBoosting |         `0.6225` |           `0.6495` | `+0.0270` |
| RandomForest         |         `0.6130` |           `0.6404` | `+0.0274` |
| LightGBM             |         `0.6220` |           `0.6485` | `+0.0264` |
| XGBoost              |         `0.6205` |           `0.6478` | `+0.0273` |

Conclusion: Stockfish strongly improves after-10 classification. This is expected because Stockfish directly evaluates tactical and positional quality after 20 plies. However, due to dependency and reproducibility risk, Stockfish was kept as an exploratory option rather than part of the default submission profile.

### 11.3 T4 Elo A/B Results

| Model                | No Stockfish Avg MAE | With Stockfish Avg MAE | Delta MAE |
| -------------------- | -------------------: | ---------------------: | --------: |
| Ridge                |              `89.39` |                `89.38` |   `-0.01` |
| GradientBoosting     |              `37.54` |                `37.44` |   `-0.10` |
| HistGradientBoosting |              `30.58` |                `30.31` |   `-0.27` |
| RandomForest         |              `38.85` |                `38.85` |   `+0.00` |
| LightGBM             |              `33.55` |                `33.34` |   `-0.21` |
| XGBoost              |              `38.40` |                `38.96` |   `+0.55` |

Conclusion: Stockfish does not materially help Elo regression. Elo prediction is better explained by causal player history than by engine evaluation of a single early-game position.

---

## 12. Heavy / Stockfish Best-Model Reference

The strongest exploratory models were:

| Task        | Best heavy model                 |           Metric |
| ----------- | -------------------------------- | ---------------: |
| T1 Before   | LogReg + history                 | ROC-AUC `0.5792` |
| T2 After-3  | GradientBoosting + Stockfish     | ROC-AUC `0.5827` |
| T3 After-10 | HistGradientBoosting + Stockfish | ROC-AUC `0.6483` |
| T4 Elo      | RandomForest + Stockfish         |  Avg MAE `28.02` |
| T4 Elo      | RandomForest no Stockfish        |  Avg MAE `28.05` |

This phase shows the upper-bound performance available with Stockfish and heavier model choices. The final submission intentionally avoids requiring Stockfish because the no-Stockfish `report_best` profile gives strong results while remaining more portable.

---

## 13. Final Verified Run

The final production-style run was:

```bash
python solution.py --target-games 100000 --output-dir outputs_full --model-profile report_best
```

### Dataset Summary

| Item                     |      Value |
| ------------------------ | ---------: |
| Runtime                  |  `769.13s` |
| Month                    |  `2023-11` |
| Time-control             |    `Blitz` |
| Parsed games             |  `213,463` |
| Header-eligible games    |  `104,005` |
| Final eligible games     |  `100,000` |
| Train rows               |   `80,000` |
| Validation rows          |   `20,000` |
| Train positive rate      | `0.493950` |
| Validation positive rate | `0.496400` |

### Classification Metrics

| Task                  |    ROC-AUC |   Log loss |      Brier |   Accuracy |
| --------------------- | ---------: | ---------: | ---------: | ---------: |
| Before-game           | `0.579185` | `0.678708` | `0.243225` | `0.552200` |
| After-3               | `0.579614` | `0.678700` | `0.243145` | `0.555150` |
| After-10              | `0.621742` | `0.665223` | `0.236834` | `0.583950` |
| Elo expected baseline | `0.578497` | `0.680803` | `0.243974` |        n/a |
| Majority baseline     |        n/a |        n/a |        n/a | `0.503600` |

### Elo Regression Metrics

| Model         | White MAE | Black MAE |          White R² |          Black R² |
| ------------- | --------: | --------: | ----------------: | ----------------: |
| Elo after-10  |  `28.719` |  `28.935` |        `0.948696` |        `0.947937` |
| Mean baseline | `300.224` | `300.586` | approximately `0` | approximately `0` |

### Interpretation

The final run shows strong performance:

* T1 is close to the Elo expected-score baseline, which is expected because before-game prediction is dominated by rating information.
* T2 improves over the earlier lightweight after-3 model while staying no-Stockfish.
* T3 meaningfully beats the Elo expected-score baseline.
* T4 substantially improves over the mean Elo baseline.

After-10 improvement over Elo baseline:

```text
ROC-AUC improvement = 0.621742 - 0.578497 = 0.043245
Log-loss improvement = 0.680803 - 0.665223 = 0.015580
Brier improvement = 0.243974 - 0.236834 = 0.007140
```

Elo improvement over mean baseline:

```text
White MAE: 300.224 -> 28.719
Black MAE: 300.586 -> 28.935
```

This validates the final no-Stockfish `report_best` profile as a strong and portable solution. Compared with the earlier LightGBM/XGBoost no-Stockfish profile, it improves T1, T2, and Elo regression, while T3 remains very close. The final choice is therefore a trade-off: stronger T2/Elo and fewer optional dependencies, with Stockfish-heavy results retained as research upper bounds.

---

## 14. Robustness and Calibration Summary

Additional checks were performed to evaluate model stability.

### Multi-Month Robustness

A 10k version of the pipeline was tested on multiple months:

| Month     | Before ROC-AUC | After-3 ROC-AUC | After-10 ROC-AUC | Elo Avg MAE |
| --------- | -------------: | --------------: | ---------------: | ----------: |
| `2023-03` |       `0.5845` |        `0.5773` |         `0.6373` |    `153.41` |
| `2023-07` |       `0.5947` |        `0.5840` |         `0.6406` |    `142.62` |
| `2023-11` |       `0.5786` |        `0.5719` |         `0.6126` |    `142.13` |

The direction of performance is stable across months. After-10 prediction is consistently stronger than after-3 and before-game prediction.

### Calibration and Lift

For the after-10 model:

| Metric                              |    Value |
| ----------------------------------- | -------: |
| Top-decile actual White win rate    | `70.15%` |
| Bottom-decile actual White win rate | `29.05%` |
| Top-decile lift                     | `41.10%` |
| Top-quintile lift                   | `28.32%` |

The model meaningfully separates high-probability and low-probability White-win games.

### Bootstrap Confidence Intervals

Bootstrap confidence intervals confirmed that after-10 improvements over the Elo expected-score baseline are statistically robust:

| Metric                               |     Mean |             95% CI |
| ------------------------------------ | -------: | -----------------: |
| ROC-AUC                              | `0.6109` | `[0.6040, 0.6185]` |
| AUC improvement vs Elo expected      | `0.0325` | `[0.0258, 0.0400]` |
| Brier improvement vs Elo expected    | `0.0049` | `[0.0037, 0.0061]` |
| Log-loss improvement vs Elo expected | `0.0111` | `[0.0084, 0.0136]` |

---

## 15. Repeat vs Unseen Player Diagnostics for Elo

Elo regression performance is strongly affected by player overlap. This is expected because causal history and identity-related features are powerful when players appear repeatedly in the same monthly stream.

A repeat/unseen diagnostic was used to compare conservative Ridge and high-score RandomForest behavior:

| Model        | Group                      |  Avg MAE |
| ------------ | -------------------------- | -------: |
| Ridge        | both players seen before   |  `58.52` |
| Ridge        | one player seen before     |  `97.24` |
| Ridge        | both players unseen before | `154.72` |
| RandomForest | both players seen before   |   `7.10` |
| RandomForest | one player seen before     |  `19.05` |
| RandomForest | both players unseen before |  `95.05` |

The tree-based Elo models are not necessarily leakage models. The leakage audit passed under the chronological setup. However, the strongest performance comes from causal same-month history and identity-like signals. Therefore, the Elo result should be reported with this caveat:

> The Elo model is leakage-safe under the chronological audit, but its strongest performance comes from causal same-month player history and identity-like information. The headline MAE is most reliable when validation data contains player overlap similar to the observed monthly stream.

---

## 16. Decision Matrix

| Candidate profile          | Performance          | Portability | Leakage safety              | Complexity  | Final decision          |
| -------------------------- | -------------------- | ----------- | --------------------------- | ----------- | ----------------------- |
| Strict lightweight sklearn | Medium               | Excellent   | Excellent                   | Low         | Keep as fallback        |
| No-Stockfish `report_best` | High                 | Excellent   | Good                        | Medium      | Final reported profile  |
| LightGBM/XGBoost no-Stockfish | High              | Good        | Good                        | Medium      | Comparison profile      |
| Stockfish-heavy            | Highest for after-10 | Weak        | Manageable but more complex | High        | Research reference only |
| Deep learning              | Not selected         | Weak        | More complex                | High        | Exclude                 |
| Ensembles / stacking       | Mixed                | Medium      | More delicate               | Medium/High | Exclude                 |

---

## 17. Final Conclusions by Task

### T1 Before-Game White-Win Prediction

Pre-game outcome prediction is dominated by Elo. Logistic Regression is sufficient and more robust than tree models. Additional history gives only very small gains, so the final profile uses the stable logistic model.

### T2 After-3 White-Win Prediction

After 3 moves, the game is still very early. In `experiment/outputs/experiment_results.csv`, the best no-Stockfish T2 row is `LogReg(C=0.5)` with after-3 enhanced numeric features, reaching ROC-AUC `0.5797`; the best overall T2 row is `GradientBoosting+SF` with ROC-AUC `0.5832`. The final profile selects the no-Stockfish `LogisticRegression(C=0.5)` variant because it improves T2 without requiring a Stockfish engine or cache.

### T3 After-10 White-Win Prediction

After 10 moves, board state and clock behavior become meaningfully predictive. sklearn `HistGradientBoostingClassifier` is selected for the final portable profile. Stockfish improves the metric further, but it is not selected because it requires an external engine or cache.

### T4 Elo Prediction

Elo prediction benefits heavily from causal player history and nonlinear tree models. sklearn `RandomForestRegressor` gives the best final portable result, with average MAE around `28.83` Elo. This result is valid under the chronological validation protocol, but it should be interpreted with repeat-player sensitivity in mind.

---

## 18. Final choose of model profile

Use the no-Stockfish `report_best` profile as the final reported solution:

```bash
pip install -r requirements.txt
python solution.py --target-games 100000 --output-dir outputs_full --model-profile report_best
```

Keep the strict lightweight profile as a fallback:

```bash
pip install -r requirements.txt
python solution.py --target-games 100000 --output-dir outputs_lightweight --model-profile lightweight
```

The final selected profile is recommended because it:

* keeps after-10 White-win prediction meaningfully above the Elo expected-score baseline,
* improves Elo regression substantially over Ridge and the mean baseline,
* does not require Stockfish,
* does not require LightGBM/XGBoost for the final submitted profile,
* keeps Stockfish-heavy models as research references only,
* and provides a good balance between empirical performance and reproducible execution.
