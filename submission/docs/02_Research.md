# Research

## 1. Research Goal

This document summarizes feasible modeling approaches for the Lichess Blitz prediction assessment. The goal is not only to maximize validation metrics, but also to select approaches that are:

* reproducible,
* leakage-safe,
* suitable for chronological validation,
* lightweight enough for a coding assessment,
* explainable enough for a Quantitative Research review,
* and robust to hidden-test distribution shifts.

The assessment requires a pipeline that predicts:

1. White-win probability before the game starts.
2. White-win probability after 3 full moves.
3. White-win probability after 10 full moves.
4. Elo ratings of both players after 10 full moves.

The problem combines elements of chess modeling, time-aware feature engineering, rating estimation, and tabular machine learning.

---

## 2. Data and Tooling Background

### 2.1 Lichess Open Database

The Lichess open database provides monthly PGN archives of rated games. Each standard rated monthly file contains games for one month only, and the archive format is `.pgn.zst`. This makes it suitable for large-scale reproducible experiments while still requiring careful streaming/decompression to avoid storing large raw files. [R1]

This project uses the standard rated monthly archive:

```text
https://database.lichess.org/standard/lichess_db_standard_rated_{YYYY-MM}.pgn.zst
```

The pipeline streams and decompresses the file, parses PGN games, filters eligible Blitz games, and stops after collecting the first 100,000 eligible games.

### 2.2 PGN Parsing

The project uses `python-chess` for PGN parsing. `python-chess` provides PGN reading utilities and can parse game headers, mainline moves, comments, and board states. This is important because the pipeline needs to reconstruct board positions after 3 and 10 full moves. [R2]

### 2.3 Text Hashing

Move sequences and player identities can be represented as sparse text features. The project uses `HashingVectorizer` instead of a learned vocabulary. This avoids storing a large vocabulary artifact and keeps the solution lightweight. The trade-off is that hashed dimensions are not directly human-interpretable. [R3]

---

## 3. Baseline Approaches

## 3.1 Elo Expected-Score Baseline

The simplest baseline for White-win prediction uses the standard Elo expected-score formula:

```text
p_white = 1 / (1 + 10 ** (-(WhiteElo - BlackElo) / 400))
```

This uses only pre-game rating information and therefore provides a strong natural benchmark for the before-game model.

However, this baseline is not exactly identical to the binary target used in this project. Elo expected score treats a draw as half a point, while the project defines:

```text
white_win = 1 if Result == "1-0"
white_win = 0 if Result == "0-1" or Result == "1/2-1/2"
```

Despite that mismatch, the Elo expected-score baseline is still useful because it captures the strongest simple pre-game signal: relative player strength.

### Why this baseline matters

For a Quantitative Research setting, a model should not only beat majority accuracy. It should also improve over a robust benchmark that is already known before the event. Therefore, the after-10 model is useful only if it adds signal beyond Elo expected score.

### Pros

* Very simple.
* No training required.
* Fully reproducible.
* Hard to overfit.
* Strong pre-game benchmark.

### Cons

* Ignores move quality.
* Ignores clock/time pressure.
* Ignores opening/position information.
* Does not match the binary White-win target perfectly because draws are treated differently.

---

## 3.2 Mean Elo Baseline

For Elo regression, the simplest baseline predicts the training-set mean rating:

```text
pred_white_elo = mean(train_white_elo)
pred_black_elo = mean(train_black_elo)
```

This baseline is intentionally naive. A useful Elo model should reduce MAE substantially compared with this mean predictor.

### Pros

* Simple and stable.
* No leakage risk.
* Useful as a lower bound.

### Cons

* Ignores all player-specific and game-specific information.
* Cannot model rating variation across players.

---

## 4. Linear Models

## 4.1 Logistic Regression for White-Win Prediction

Logistic regression is a strong default model for probability estimation. It is especially useful in this assessment because:

* it is lightweight,
* it trains quickly on 100,000 games,
* it supports sparse text features,
* it is easy to regularize,
* coefficients can be inspected for model-level explanation,
* and it is less likely to overfit noisy player-history features than deep trees.

This makes logistic regression a strong baseline for all three White-win classification tasks.

### Suitable features

* Pre-game Elo and time-control features.
* Move text features encoded with hashing.
* Board-state features after the legal prediction point.
* Clock features up to the prediction point.
* Optional player identity text, if treated carefully.

### Pros

* Fast and reproducible.
* Good for sparse features.
* Easy to inspect and audit.
* Simple dependency profile through scikit-learn.

### Cons

* Limited ability to model nonlinear interactions.
* Hashed text dimensions are not interpretable.
* May underfit complex board-state patterns.

---

## 4.2 Ridge Regression for Elo Prediction

Ridge regression is a strong conservative default for Elo prediction. It is linear, regularized, and more stable than high-capacity tree models when using repeat-player history and identity features.

Ridge regression is especially suitable when the goal is not merely to minimize validation MAE, but to produce a defensible model under possible hidden-test distribution shifts.

### Suitable features

* Causal player-history features.
* Hashed player identity features.
* Move/board features up to ply 20.
* Clock features up to ply 20 if they empirically help.
* Time-control features.

### Pros

* Simple and regularized.
* Easy to audit.
* Less likely than Random Forest to memorize repeat players.
* Coefficients can be inspected.
* Good safe default for a Quantitative Research assessment.

### Cons

* May underfit nonlinear relationships.
* May perform worse than tree ensembles on repeated-player validation data.
* Still benefits from same-month player history and identity, so repeat-player caveats remain necessary.

---

## 5. Move Text and Board Features

## 5.1 Move Text Features

Move sequences encode opening choices and early tactical patterns. For example, early move text may contain signals such as:

* common openings,
* early queen moves,
* captures,
* checks,
* castling,
* development patterns,
* and tactical deviations.

The project represents move text using SAN and UCI-style tokens where available. These are encoded using `HashingVectorizer`, avoiding the need to store a learned vocabulary. [R3]

### Pros

* Lightweight representation of opening and tactical sequences.
* Works well with sparse linear models.
* No vocabulary artifact required.
* Reproducible with fixed hashing settings.

### Cons

* Hashed dimensions are not directly interpretable.
* Collisions can occur.
* Text-only features do not fully understand the board position.

---

## 5.2 Basic Board Features

Board features are extracted only after the legal prediction point. For example:

* after 3 full moves: use board state at ply 6,
* after 10 full moves: use board state at ply 20.

Basic board features include:

* material difference,
* piece counts by side,
* legal move count,
* check flag,
* side to move,
* castling rights,
* castling completed flags,
* center occupancy,
* center attacks,
* capture counts,
* check counts,
* move-type counts.

These features provide interpretable chess-state information without requiring an external engine.

### Pros

* Lightweight and fully reproducible.
* No external binary required.
* Interpretable compared with hashed text.
* Useful for after-10 prediction because the position contains more information after 20 plies.

### Cons

* Hand-crafted features only approximate position quality.
* Cannot search tactics or evaluate long-term compensation.
* May miss strategic factors that a chess engine captures.

---

## 5.3 Enhanced Board Features

Enhanced board features were considered to capture richer chess concepts:

* piece-square table score,
* pawn structure,
* mobility,
* development,
* king-safety proxies,
* rook activity,
* bishop pair,
* queen activity,
* open-file proxies,
* attack/defense counts.

These features are still lightweight because they do not require a chess engine. However, experiments showed that enhanced board features did not always improve the selected production configuration. Therefore, they are kept as optional/experimental features rather than automatically selected for the default pipeline.

### Pros

* More expressive than basic board counts.
* Still reproducible without Stockfish.
* Can improve tree-based models.

### Cons

* More feature engineering complexity.
* May introduce noisy proxies.
* Did not consistently improve the selected lightweight production model.

---

## 6. Clock and Time-Pressure Features

Lichess PGN comments often include clock annotations such as:

```text
[%clk H:MM:SS]
[%clk M:SS]
```

Clock features are valid only if they are computed from clock comments available at or before the prediction point. For after-3 prediction, this means only clock data up to ply 6. For after-10 prediction, this means only clock data up to ply 20.

Clock-time modeling is supported by chess rating research. RatingNet, a CNN-LSTM model for rating estimation, uses move sequences and clock-time data from over one million Lichess games and reports an MAE of 182 rating points. This suggests that clock behavior contains information about player strength and decision quality. [R4]

### Clock features considered

* last observed White clock,
* last observed Black clock,
* total approximate time used by White,
* total approximate time used by Black,
* average time per move,
* clock difference,
* time-used difference,
* minimum observed clock,
* missing clock counts,
* time-pressure flags.

### Experimental finding

Clock features helped the after-10 White-win classifier more than the after-3 model or Elo regression. This is intuitive: after 10 full moves, time usage and board position jointly reveal more information about the game trajectory.

### Pros

* Available in many Lichess PGNs.
* Strong behavioral signal.
* Useful for Blitz games.
* Does not require external engine evaluation.

### Cons

* Clock comments may be missing or inconsistent in some PGNs.
* Time usage can be noisy.
* Must carefully avoid future-clock leakage.

---

## 7. Causal Player History

Player-history features are highly informative, especially for Elo prediction. The key rule is causality:

1. Compute current-game history features using earlier eligible games only.
2. Extract features and make the current-game row.
3. Update the player-history store with the current game.

This prevents future-game leakage while allowing the model to use realistic information that would have been available before a later game in the same chronological stream.

### History features considered

* prior game count,
* prior score rate,
* side-specific win rates,
* prior average opponent Elo,
* prior average observed Elo,
* recent score rate over the last 10 games,
* recent score rate over the last 30 games,
* differences between White and Black history statistics.

### Why history helps

Chess ratings change slowly over time. If a player appears multiple times in the same monthly data stream, their earlier games provide strong evidence about their later rating and playing strength.

### Pros

* Strong signal for Elo prediction.
* Causal if computed correctly.
* Useful in a time-aware pipeline.
* Similar to lagged features in financial time series or user-behavior modeling.

### Cons

* Strongly benefits repeat players.
* May generalize less well to unseen players.
* Must be audited carefully to ensure current-game and future-game information is not used.
* Headline validation metrics can look too optimistic if validation contains many repeat players.

---

## 8. Player Identity Features

Player usernames are known before the game starts. The project tested hashed identity text such as:

```text
white=<white_name> black=<black_name>
```

This can help models learn repeat-player effects. For Elo prediction, identity can be especially powerful because the same player’s rating changes slowly across games.

### Pros

* Known before the game.
* Useful for repeat-player prediction.
* Lightweight when hashed.
* No learned vocabulary artifact required.

### Cons

* May not generalize to completely unseen players.
* Hashed identity dimensions are not human-readable.
* Can look like memorization even when leakage-free.
* Must be disclosed and evaluated with repeat/unseen-player diagnostics.

### Research interpretation

In a Quantitative Research setting, player identity is analogous to an entity identifier or instrument identifier. It can be valid, but the model must be stress-tested under different entity-overlap regimes. A model that performs well only when entities repeat may still be useful, but its limitations must be disclosed.

---

## 9. Gradient Boosting and Tree-Based Models

## 9.1 Scikit-learn Gradient Boosting

Tree-based boosting models can capture nonlinear interactions between Elo, board features, clock features, and player-history features. Scikit-learn provides `GradientBoostingClassifier` and `HistGradientBoostingClassifier`. The histogram-based version is designed to be faster on larger datasets, especially when the number of samples is at least 10,000. [R5]

### Pros

* Captures nonlinear feature interactions.
* Works well with tabular numeric features.
* Available through scikit-learn.
* No extra dependency required if using scikit-learn implementations.

### Cons

* Can overfit noisy history features.
* Less straightforward to interpret than linear models.
* Requires careful regularization.
* Does not naturally handle sparse hashed text as simply as logistic regression.

### Practical use in this project

Regularized boosting is useful for after-3 and after-10 outcome models when using numeric board, clock, and optional Stockfish features. Tree depth, leaf size, subsampling, and feature subsampling are important overfitting controls.

---

## 9.2 LightGBM

LightGBM is a gradient boosting framework using tree-based learning algorithms. It is designed for efficiency, lower memory usage, and large-scale data. The original LightGBM paper reports large speedups over conventional GBDT while maintaining similar accuracy. [R6]

### Pros

* Strong tabular learning performance.
* Efficient training.
* Works well on large numeric feature sets.
* Common in data science and quantitative modeling.

### Cons

* Extra dependency beyond the strict lightweight profile.
* Installation can be more fragile than pure scikit-learn in some environments.
* Can overfit if not regularized.
* Requires careful documentation in a coding assessment.

### Practical use in this project

LightGBM was considered for optional no-Stockfish boosting experiments. It can improve performance on enhanced tabular features, especially for after-10 prediction and Elo regression. However, to keep the default assessment pipeline lightweight, LightGBM belongs in an optional experiment profile rather than the strict default requirements.

---

## 9.3 XGBoost

XGBoost is an optimized distributed gradient boosting library designed to be efficient, flexible, and portable. It implements gradient-boosted trees and is widely used for classification, regression, and ranking tasks. [R7]

### Pros

* Strong general-purpose tabular model.
* Supports regularization and scalable training.
* Common in applied ML competitions and production workflows.

### Cons

* Extra dependency.
* Larger installation footprint than scikit-learn-only models.
* Can overfit if not tuned.
* Not necessary if scikit-learn or LightGBM already provides sufficient performance.

### Practical use in this project

XGBoost was treated as an optional experimental model. It is useful as a benchmark for boosted-tree performance, but it is not required for the lightweight reproducible pipeline.

---

## 9.4 CatBoost

CatBoost is an open-source gradient boosting library designed to handle categorical features effectively. CatBoost papers describe ordered boosting and categorical feature handling methods intended to reduce prediction shift and target leakage risks in categorical encodings. [R8]

### Pros

* Strong support for categorical features.
* Useful when raw categorical variables are important.
* Can reduce leakage risks from naive target encoding.

### Cons

* Extra dependency.
* Heavier than scikit-learn-only models.
* Less necessary when the project already uses hashing for text/player identity.
* Needs additional evaluation before being included in a submission pipeline.

### Practical use in this project

CatBoost is a plausible future direction for identity and categorical metadata. However, it was not selected for the strict pipeline because hashed identity plus scikit-learn/Ridge/boosting models were sufficient and simpler to audit.

---

## 10. Stockfish Engine Evaluation

Stockfish is a free, open-source UCI chess engine that analyzes chess positions and computes strong moves or position evaluations. [R9]

In this project, Stockfish is used as an optional source of engine-evaluation features. For example:

```text
position after 10 full moves -> Stockfish evaluation -> numeric feature
```

The feature can be a centipawn evaluation or a normalized score. A positive value usually indicates an advantage for White; a negative value indicates an advantage for Black.

### Why Stockfish helps

Stockfish directly evaluates the chess position. This is especially valuable after 10 full moves because the board state already contains meaningful strategic and tactical information. Research on large-scale chess-engine analysis also notes that engine evaluations can support applications such as skill assessment, cheating detection, and human decision-making studies, although computing engine evaluations at scale is expensive. [R10]

### Pros

* Strong positional signal.
* Particularly useful for after-10 outcome prediction.
* Captures tactical and strategic features that hand-crafted features miss.

### Cons

* Requires an external engine binary.
* Can make reproducibility more fragile.
* Runtime can be high for 100,000 games.
* Should not be required for the default pipeline unless the environment is guaranteed.
* Must not evaluate positions beyond the prediction point.

### Practical use in this project

Stockfish-based models are strong enhanced experiments, especially for after-10 outcome prediction. However, the default pipeline should remain runnable without Stockfish. The recommended design is:

```text
default mode: no Stockfish required
optional enhanced mode: use Stockfish if available
```

---

## 11. Deep Learning and Human Chess Modeling

Deep learning models are highly relevant to chess modeling, but they were not selected for the submission pipeline because of runtime, dependency, and implementation complexity.

### 11.1 RatingNet

RatingNet estimates chess ratings from moves and clock times using a CNN-LSTM architecture. It uses over one million Lichess games and reports an MAE of 182 rating points on test data. This supports the idea that move sequences and clock time are valid signals for rating estimation, but the approach is much heavier than the lightweight tabular pipeline used here. [R4]

### 11.2 ChessMimic

ChessMimic uses small encoder-only transformers for move, clock, and outcome prediction in online Blitz chess. It conditions on position, recent move history, player rating, and clock state. Its outcome model reportedly achieves AUC 0.78 out of sample. This suggests that substantially stronger outcome prediction is possible with richer neural architectures and more specialized modeling. [R11]

### 11.3 Maia

Maia is a human-like neural chess engine trained to predict human moves rather than optimal engine moves. Maia models are designed around human move prediction and skill-level-specific behavior, not directly the same as this assessment’s White-win and Elo regression tasks. Still, Maia demonstrates that human chess behavior can be modeled effectively from large Lichess data. [R12]

### Why deep learning was not selected

Deep learning was not selected for the assessment pipeline because:

* it would add heavy dependencies,
* it would require longer training time,
* it would make the solution harder to reproduce,
* it would complicate the submission under file-size constraints,
* and simple tabular models already provide meaningful signal.

Deep learning is therefore treated as future work rather than a submission approach.

---

## 12. Approach Comparison

| Approach                    |           Expected Performance | Reproducibility | Dependency Risk |          Interpretability | Selected Role                 |
| --------------------------- | -----------------------------: | --------------: | --------------: | ------------------------: | ----------------------------- |
| Elo expected-score baseline |                         Medium |       Very high |        Very low |                      High | Required baseline             |
| Mean Elo baseline           |                            Low |       Very high |        Very low |                      High | Required baseline             |
| Logistic Regression         |                         Medium |            High |             Low | High for numeric features | Lightweight classifier        |
| Ridge Regression            |                    Medium/High |            High |             Low | High for numeric features | Conservative Elo model        |
| Move text hashing           |                         Medium |            High |             Low | Low for hashed dimensions | Lightweight sequence signal   |
| Basic board features        |                         Medium |            High |             Low |                      High | Default feature group         |
| Enhanced board features     |                    Medium/High |            High |             Low |                    Medium | Optional feature group        |
| Clock features              |                    Medium/High |            High |             Low |               Medium/High | Useful for after-10           |
| Causal player history       |                   High for Elo |     Medium/High |             Low |                    Medium | Important but caveated        |
| Player identity hashing     |        High for repeat players |     Medium/High |             Low |                       Low | Optional/repeat-player signal |
| Gradient Boosting / HistGB  |                           High |            High |  Low if sklearn |                    Medium | Strong tabular model          |
| LightGBM/XGBoost/CatBoost   |                           High |          Medium |          Medium |                    Medium | Optional experiments          |
| Stockfish                   | Very high for position outcome |          Medium |            High |                    Medium | Optional enhanced mode        |
| Deep learning               |            Potentially highest |      Low/Medium |            High |                Low/Medium | Future work                   |

---

## 13. Selected Direction

The selected strategy is dual-profile:

### 13.1 Lightweight Default Profile

The lightweight default profile prioritizes reproducibility and dependency safety. It uses only standard Python ML dependencies and avoids requiring external engines.

Typical components:

* scikit-learn models,
* logistic regression for White-win probability,
* Ridge or controlled tree-based regression for Elo depending on final audit results,
* HashingVectorizer for move text and identity,
* clock features when available,
* causal history features for Elo,
* no required Stockfish dependency.

This profile is best for the official submission run because it is easy for a reviewer to execute from scratch.

### 13.2 Enhanced Experiment Profile

The enhanced profile is used to demonstrate additional research effort and upper-bound performance.

It may include:

* regularized gradient boosting,
* HistGradientBoosting,
* LightGBM/XGBoost/CatBoost experiments,
* Stockfish evaluation features,
* repeat/unseen diagnostics,
* calibration analysis,
* XAI/feature-importance reports.

This profile is useful for research reporting, but not all enhanced components should be required for the default execution path.

---

## 14. Final Research Conclusions

The research process supports the following conclusions:

1. **Elo expected score is the correct natural baseline** for White-win prediction because it uses the strongest simple pre-game signal.

2. **Linear models are strong lightweight defaults** because they are reproducible, stable, and easy to audit.

3. **Move text and board features add early-game signal**, especially after 10 full moves.

4. **Clock features are valuable for Blitz games**, particularly for after-10 outcome prediction.

5. **Causal player history and identity are powerful for Elo prediction**, but they require repeat/unseen diagnostics because they are sensitive to player overlap.

6. **Gradient boosting can improve tabular performance**, but it must be regularized to avoid overfitting.

7. **Stockfish provides strong positional signal**, especially after 10 moves, but it should remain optional unless the runtime environment is controlled.

8. **Deep learning methods are promising but not suitable for the main submission pipeline** due to complexity, runtime, and dependency constraints.

The final project therefore balances empirical performance with reproducibility, auditability, and hidden-test robustness.

## Research Sources

* **[R1]** Lichess Open Database — public monthly rated PGN archives used as the main data source.
  https://database.lichess.org/

* **[R2]** `python-chess` PGN documentation — used for parsing PGN headers, moves, comments, and board states.
  https://python-chess.readthedocs.io/en/latest/pgn.html

* **[R3]** scikit-learn `HashingVectorizer` — used for lightweight move-text and player-identity feature encoding without saving a vocabulary.
  https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html

* **[R4]** Omori and Tadepalli, *Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM* — used to motivate move-sequence and clock-time features for Elo estimation.
  https://arxiv.org/abs/2409.11506

* **[R5]** scikit-learn `HistGradientBoostingClassifier` — used as a no-extra-dependency boosted-tree candidate for tabular classification.
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html

* **[R6]** LightGBM documentation — used as a reference for optional high-performance gradient boosting experiments.
  https://lightgbm.readthedocs.io/

* **[R7]** XGBoost documentation — used as a reference for optional boosted-tree classification and regression experiments.
  https://xgboost.readthedocs.io/

* **[R8]** Prokhorenkova et al., *CatBoost: unbiased boosting with categorical features* — used as a reference for categorical-aware boosting and identity-feature modeling.
  https://arxiv.org/abs/1706.09516

* **[R9]** Stockfish GitHub repository — used as a reference for optional chess-engine evaluation features.
  https://github.com/official-stockfish/Stockfish

* **[R10]** Acher and Esnault, *Large-scale Analysis of Chess Games with Chess Engines* — used to motivate engine-evaluation features and discuss computational cost.
  https://arxiv.org/abs/1607.04186

* **[R11]** Johnson, *ChessMimic: Per-Rating Transformer Models for Human Move, Clock, and Outcome Prediction in Online Blitz Chess* — used as a deep-learning reference for human chess outcome and behavior modeling.
  https://arxiv.org/abs/2606.04473

* **[R12]** Maia Chess GitHub repository — used as a reference for human-like neural chess modeling.
  https://github.com/CSSLab/maia-chess

* **[R13]** scikit-learn `LogisticRegression` — used as the main lightweight probabilistic classification model.
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

* **[R14]** scikit-learn `Ridge` — used as a regularized linear baseline for Elo regression.
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

* **[R15]** scikit-learn `RandomForestRegressor` — used as a high-performance Elo regression candidate and repeat-player sensitivity benchmark.
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

* **[R16]** scikit-learn model evaluation documentation — used for ROC-AUC, log loss, Brier score, accuracy, MAE, RMSE, and R² evaluation.
  https://scikit-learn.org/stable/modules/model_evaluation.html