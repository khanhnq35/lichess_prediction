# Results Report

## 1. Executive Summary

This report summarizes the final validation results of the Lichess Blitz prediction pipeline.

The final reported run uses the **portable no-Stockfish `report_best` profile**:

```bash
python solution.py --target-games 100000 --output-dir outputs_full --model-profile report_best
```

The pipeline was evaluated on the first `100,000` eligible Blitz games from the selected monthly Lichess archive. The split is chronological:

```text
First 80,000 eligible games  -> training set
Last 20,000 eligible games   -> validation set
```

This validation design simulates a realistic setting where past games are used to predict later games.

The final model configuration is:

| Task                          | Selected model       | Main feature groups                                                     |
| ----------------------------- | -------------------- | ----------------------------------------------------------------------- |
| White win before game         | Logistic Regression + causal history | Pre-game Elo, time-control, causal history                  |
| White win after 3 full moves  | LogisticRegression(C=0.5) | Pre-game, after-3 board, move behavior, enhanced board, first-6-ply clock features |
| White win after 10 full moves | sklearn HistGradientBoostingClassifier | Pre-game, after-10 board, move behavior, enhanced board, clock features |
| Elo after 10 full moves       | sklearn RandomForestRegressor | Time-control, causal history, after-10 board, enhanced board features |

The final results show three important findings:

1. **Before-game prediction is mostly an Elo problem.** The learned model performs very close to the Elo expected-score baseline.

2. **After-10 prediction adds meaningful signal beyond Elo.** The after-10 model improves ROC-AUC, log loss, and Brier score over both the before-game model and the Elo expected-score baseline.

3. **Elo prediction is very strong in the same-month chronological stream.** The model reduces Elo MAE from around `300` points under the mean baseline to around `29` points, mainly because causal player history is highly informative.

The headline metrics should always be read against baselines:

| Task | Final model | Baseline | Main takeaway |
| --- | ---: | ---: | --- |
| White win before game | ROC-AUC `0.579185` | Elo expected-score ROC-AUC `0.578497` | Pre-game prediction is essentially Elo-driven. |
| White win after 3 full moves | ROC-AUC `0.579614` | Elo expected-score ROC-AUC `0.578497` | First 6 plies add little extra signal and should not be overclaimed. |
| White win after 10 full moves | ROC-AUC `0.621742` | Elo expected-score ROC-AUC `0.578497` | First 20 plies plus clock/board features add meaningful signal. |
| Elo after 10 full moves | White/Black MAE `28.719 / 28.935` | Mean baseline MAE `300.224 / 300.586` | Causal same-stream player history makes rating reconstruction much stronger than a global mean baseline. |

The low Elo MAE is not a leakage claim and should not be interpreted as pure cold-start rating inference. The Elo model is best understood as **same-stream rating reconstruction**: it uses causal player-history features computed from earlier eligible games before the current game is processed. This is valid because current Elo, rating diff, result, future games, and validation fitting are excluded from the Elo model inputs.

---

## 2. Dataset and Run Summary

The final run used the public Lichess standard rated monthly archive for `2023-11` and filtered Blitz games.

| Item                     |            Value |
| ------------------------ | ---------------: |
| Runtime                  | `769.13 seconds` |
| Selected month           |        `2023-11` |
| Time-control             |          `Blitz` |
| Parsed games             |        `213,463` |
| Header-eligible games    |        `104,005` |
| Final eligible games     |        `100,000` |
| Train games              |         `80,000` |
| Validation games         |         `20,000` |
| Train positive rate      |       `0.493950` |
| Validation positive rate |       `0.496400` |

The pipeline parsed `213,463` games to collect `100,000` final eligible games. This means that less than half of parsed games entered the final dataset after filtering by time-control, valid Elo, valid result, legal PGN replay, and minimum length.

The training and validation positive rates are close:

```text
Train positive rate      = 0.493950
Validation positive rate = 0.496400
```

This is useful because it shows that the chronological split does not create a large class-balance shift between train and validation.

---

## 3. Result Distribution

The final eligible game set has the following result distribution:

| Result            |    Count |    Share |
| ----------------- | -------: | -------: |
| White win (`1-0`) | `49,444` | `49.44%` |
| Black win (`0-1`) | `46,373` | `46.37%` |
| Draw (`1/2-1/2`)  |  `4,183` |  `4.18%` |

The binary target is defined as:

```text
white_win = 1 if Result == "1-0"
white_win = 0 otherwise
```

Therefore, both Black wins and draws are treated as non-White-wins.

Because the validation set is close to balanced, majority-class accuracy is not a meaningful target by itself. A model that always predicts the majority class reaches only:

```text
Majority baseline validation accuracy = 0.503600
```

This is why the report focuses primarily on:

* ROC-AUC,
* log loss,
* Brier score,
* lift analysis,
* and Elo MAE.

---

## 4. White-Win Classification Results

Final classification results:

| Model                         |    ROC-AUC |   Log loss | Brier score |   Accuracy |
| ----------------------------- | ---------: | ---------: | ----------: | ---------: |
| White win before game         | `0.579185` | `0.678708` |  `0.243225` | `0.552200` |
| White win after 3 full moves  | `0.579614` | `0.678700` |  `0.243145` | `0.555150` |
| White win after 10 full moves | `0.621742` | `0.665223` |  `0.236834` | `0.583950` |
| Elo expected-score baseline   | `0.578497` | `0.680803` |  `0.243974` |        n/a |
| Majority baseline             |        n/a |        n/a |         n/a | `0.503600` |

The after-10 model is the strongest White-win classifier. It improves both ranking quality and probability quality against the main Elo expected-score baseline.

### Why Classification Metrics Are In The 0.5x-0.6x Range

The White-win ROC-AUC values are intentionally interpreted as realistic lightweight chess-outcome results, not as engine-level prediction scores. A ROC-AUC around `0.58` before the game and `0.62` after 10 moves is expected for this setup.

There are several reasons.

First, the validation set is close to balanced:

```text
Validation positive rate = 0.496400
Majority-class accuracy  = 0.503600
```

This means a naive classifier is already near `50%` accuracy, and raw accuracy has limited headroom. The goal is not to reach very high accuracy by guessing the majority class; the goal is to improve ranking and probability quality over strong baselines.

Second, Lichess Blitz pairings are often between similarly rated players. When two players have similar Elo, the true pre-game outcome is naturally close to uncertain. The Elo expected-score baseline itself reaches only:

```text
Elo expected-score ROC-AUC = 0.578497
```

This shows that even the strongest pre-game signal, rating difference, only separates White wins from non-White-wins moderately under this binary target.

Third, this project predicts **binary White win**, not chess expected score. Draws are counted as non-White-wins:

```text
White win = 1
Black win or draw = 0
```

This is stricter than Elo expected score, where draws receive `0.5`. Because draws are grouped with Black wins, some games that are “good outcomes for White” under expected-score logic become negative examples under the project target.

Fourth, the after-3 model only observes the first 6 plies. Most games are still in opening territory at that point. Material is usually equal, king safety has not fully developed, and the decisive tactical or time-pressure events often happen much later.

Fifth, even the after-10 model only observes the first 20 plies. Blitz games can swing sharply after move 10 due to blunders, tactical misses, time pressure, mouse slips, resignations, or endgame mistakes. Those future events are deliberately unavailable to the model because using them would be leakage.

Finally, the final profile does not use Stockfish or deep learning. Stockfish experiments achieved higher after-10 ROC-AUC, but they require an external engine or cached engine evaluations. The submitted model is designed to be portable, reproducible, and lightweight, so its classification performance should be judged against the Elo baseline and majority baseline rather than against engine-strength systems.

The key interpretation is therefore:

> ROC-AUC in the `0.58-0.62` range is not a sign that the pipeline is broken. It reflects a noisy early-game prediction problem, a strict binary target, balanced validation data, and an intentionally lightweight no-engine feature set.

The meaningful result is the improvement pattern:

```text
Before-game ROC-AUC       = 0.579185
After-3 ROC-AUC           = 0.579614
After-10 ROC-AUC          = 0.621742
Elo expected-score AUC    = 0.578497
Majority baseline accuracy = 0.503600
```

The after-10 model improves meaningfully over both the before-game model and the Elo expected-score baseline, while the after-3 model remains close to pre-game expectation, which is exactly what should be expected from only 6 plies.

The Elo expected-score baseline is:

```text
p_white = 1 / (1 + 10 ** (-(WhiteElo - BlackElo) / 400))
```

This is the right baseline for White-win prediction because Elo ratings are known before the game. It is not identical to the binary target because Elo expected score treats draws as half-points, while this project treats draws as non-White-wins. Even so, it is a strong and useful benchmark.

Compared with the Elo expected-score baseline:

```text
ROC-AUC improvement = 0.621742 - 0.578497 = 0.043245
Log-loss improvement = 0.680803 - 0.665223 = 0.015580
Brier improvement = 0.243974 - 0.236834 = 0.007140
```

Compared with the before-game model:

```text
ROC-AUC improvement = 0.621742 - 0.579185 = 0.042557
Accuracy improvement = 0.583950 - 0.552200 = 0.031750
```

This shows that the first 10 full moves contain useful predictive information beyond pre-game Elo.

---

## 5. Interpretation by Prediction Horizon

## 5.1 Before-Game Prediction

The before-game model reaches:

| Metric      |      Value |
| ----------- | ---------: |
| ROC-AUC     | `0.579185` |
| Log loss    | `0.678708` |
| Brier score | `0.243225` |
| Accuracy    | `0.552200` |

This performance is very close to the Elo expected-score baseline:

| Model                 |    ROC-AUC |   Log loss |      Brier |
| --------------------- | ---------: | ---------: | ---------: |
| Before-game model     | `0.579185` | `0.678708` | `0.243225` |
| Elo expected baseline | `0.578497` | `0.680803` | `0.243974` |

This is expected. Before the game starts, the strongest available information is the rating difference between players. The learned model mainly acts as a calibrated version of Elo-derived information.

The practical interpretation is:

> Before the first move, the model can rank White-win probability better than random, but there is limited information beyond player strength.

This is a reasonable result because chess outcomes are noisy. Even a large Elo advantage does not guarantee a win, especially in Blitz.

---

## 5.2 After-3 Prediction

The after-3 model reaches:

| Metric      |      Value |
| ----------- | ---------: |
| ROC-AUC     | `0.579614` |
| Log loss    | `0.678700` |
| Brier score | `0.243145` |
| Accuracy    | `0.555150` |

This is almost the same as the before-game model and only slightly above the Elo expected-score ROC-AUC baseline. That should not be overclaimed; it means the first 6 plies add only weak non-engine signal beyond rating information for this model profile.

After 3 full moves, only 6 plies have been played. Most games are still in common opening structures, and many decisive events have not happened yet. At this point:

* material is usually equal,
* kings are often not yet exposed,
* tactical mistakes may not have occurred,
* future blunders are not observable,
* and many opening choices transpose into similar positions.

The after-3 model adds early board-state and first-6-ply clock information, but the signal is still weak. It is useful as an early-game diagnostic model, but it is not expected to be dramatically stronger than a pre-game Elo model.

The practical interpretation is:

> The first 3 moves do not usually contain enough information to substantially change the pre-game expectation in a non-engine model.

This is also supported by the Stockfish A/B experiment: Stockfish features only gave very small improvement at the after-3 horizon.

---

## 5.3 After-10 Prediction

The after-10 model is the strongest White-win classifier:

| Metric      |      Value |
| ----------- | ---------: |
| ROC-AUC     | `0.621742` |
| Log loss    | `0.665223` |
| Brier score | `0.236834` |
| Accuracy    | `0.583950` |

This is the first prediction horizon where the model gains clear signal beyond Elo.

At 10 full moves, the game position contains more meaningful information:

* material imbalance may appear,
* development quality becomes visible,
* castling status matters,
* king-safety differences emerge,
* center control is more informative,
* mobility differences are more stable,
* clock usage starts to reflect pressure or uncertainty.

The improvement is visible across multiple metrics, not just one:

| Comparison               | ROC-AUC gain | Log-loss gain |  Brier gain |
| ------------------------ | -----------: | ------------: | ----------: |
| After-10 vs Elo baseline |  `+0.043245` |   `+0.015580` | `+0.007140` |
| After-10 vs before-game  |  `+0.042557` |   `+0.014853` | `+0.006916` |

This matters because:

* ROC-AUC improvement shows better ranking of games by White-win likelihood.
* Log-loss improvement shows better probabilistic prediction.
* Brier improvement shows better squared-error probability quality.

The practical interpretation is:

> By move 10, the model has learned meaningful early-game signals beyond pre-game Elo, especially from board state and clock behavior.

This is the key classification result: after-10 prediction is not merely reproducing Elo. It improves the Elo expected-score baseline by `+0.043245` ROC-AUC and also improves log loss and Brier score.

---

## 6. Probability Diagnostics

Probability diagnostics help check whether the models produce sensible probability distributions.

| Model        |        Min |       Mean |        Std |        P05 |        P50 |        P95 |        Max |
| ------------ | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| Before       | `0.006219` | `0.494660` | `0.083084` | `0.382329` | `0.493952` | `0.611631` | `0.986186` |
| After 3      | `0.002040` | `0.494366` | `0.086476` | `0.373407` | `0.493711` | `0.619123` | `0.999324` |
| After 10     | `0.035544` | `0.493420` | `0.114195` | `0.310773` | `0.494405` | `0.685055` | `0.955196` |
| Elo baseline | `0.000855` | `0.500040` | `0.106923` | `0.341521` | `0.500000` | `0.660999` | `0.997727` |

The mean predicted probabilities are close to the validation positive rate:

```text
Validation positive rate = 0.496400
Before mean probability  = 0.494660
After-3 mean probability = 0.494366
After-10 mean probability = 0.493420
```

This suggests there is no obvious global class-balance bias.

The after-10 model has a larger probability spread:

```text
Before std  = 0.083084
After-3 std = 0.086476
After-10 std = 0.114195
```

This is desirable. It means the after-10 model is more willing to separate high-probability and low-probability games after observing more information. The spread is not just noise: it is supported by better ROC-AUC, better log loss, better Brier score, and strong lift analysis.

---

## 7. Calibration and Lift Analysis

Lift analysis evaluates whether the model is useful for ranking games.

The after-10 model produces the following validation segments:

| Segment                                      | Actual White-win rate |
| -------------------------------------------- | --------------------: |
| Top 10% by predicted after-10 probability    |              `0.7170` |
| Bottom 10% by predicted after-10 probability |              `0.2825` |
| Top 20% by predicted after-10 probability    |              `0.6620` |
| Bottom 20% by predicted after-10 probability |              `0.3463` |

Lift:

```text
Top 10% - Bottom 10% = 0.7170 - 0.2825 = 0.4345
Top 20% - Bottom 20% = 0.6620 - 0.3463 = 0.3157
```

This is one of the most practically useful results in the project.

It means:

* The top decile of predictions has a White-win rate of `71.70%`.
* The bottom decile has a White-win rate of only `28.25%`.
* The model separates high-probability and low-probability White-win games clearly.

This gives the after-10 model practical value as a ranking model, even though raw accuracy is only around `58%`.

In probabilistic tasks, this is important. Accuracy alone hides the fact that the model is much better at ranking games than a majority baseline.

---

## 8. Calibration Bins

Calibration bins for the after-10 model:

| Bin |   Count | Mean predicted | Actual White-win rate |         Gap |
| --: | ------: | -------------: | --------------------: | ----------: |
|   1 | `2,000` |     `0.275829` |              `0.2825` | `-0.006671` |
|   2 | `2,000` |     `0.388055` |              `0.4100` | `-0.021945` |
|   3 | `2,000` |     `0.429363` |              `0.4475` | `-0.018137` |
|   4 | `2,000` |     `0.458072` |              `0.4590` | `-0.000928` |
|   5 | `2,000` |     `0.481453` |              `0.4805` |  `0.000953` |
|   6 | `2,000` |     `0.503539` |              `0.4915` |  `0.012039` |
|   7 | `2,000` |     `0.526503` |              `0.5195` |  `0.007003` |
|   8 | `2,000` |     `0.554357` |              `0.5495` |  `0.004857` |
|   9 | `2,000` |     `0.597572` |              `0.6070` | `-0.009428` |
|  10 | `2,000` |     `0.720125` |              `0.7170` |  `0.003125` |

The calibration gaps are small in most bins. The largest visible gaps are in bins 2 and 3, where the model slightly underpredicts the observed White-win rate. The top bin is well calibrated:

```text
Top bin mean predicted = 0.720125
Top bin actual rate    = 0.7170
Gap                    = 0.003125
```

This is useful because the top-score group is often the group where users care most about whether high-confidence predictions are reliable.

Overall interpretation:

> The after-10 model is not perfectly calibrated, but its probability estimates are directionally meaningful and reasonably aligned with observed validation frequencies.

---

## 9. Elo Regression Results

Elo regression is the strongest quantitative result in the project.

| Model         | White MAE | White RMSE |    White R² | Black MAE | Black RMSE |    Black R² |
| ------------- | --------: | ---------: | ----------: | --------: | ---------: | ----------: |
| Elo after 10  |  `28.719` |   `83.351` |  `0.948696` |  `28.935` |   `84.081` |  `0.947937` |
| Mean baseline | `300.224` |  `368.031` | `-0.000222` | `300.586` |  `368.529` | `-0.000195` |

Compared with the mean baseline:

```text
White MAE reduction = 300.224 - 28.719 = 270.983
Black MAE reduction = 300.586 - 28.935 = 271.210
```

Relative reduction:

```text
White MAE reduction rate ≈ 90.26%
Black MAE reduction rate ≈ 90.23%
```

This is a large improvement.

## 9.1 Why the Elo MAE is Low Without Leakage

The result should not be interpreted as “the model can always infer a player’s Elo from only 10 moves.” The model uses causal player-history features. If a player has appeared earlier in the monthly chronological stream, their earlier games provide strong information about their current rating.

This makes the task closer to **same-stream rating reconstruction** than pure cold-start Elo estimation. In the validation period, many players have prior appearances in the earlier chronological stream. For those players, features such as prior observed Elo, prior opponent strength, prior game count, and recent score rate are strong rating anchors.

This is valid under the project’s leakage rules because:

1. history is computed before the current game,
2. only earlier eligible games are used,
3. validation rows are not used for fitting,
4. current-game Elo is excluded from Elo input features,
5. `elo_diff`, `mean_elo`, rating-diff fields, result, termination, total game length, and future moves are excluded,
6. player histories are updated only after current-game features are extracted.

The practical interpretation is:

> The Elo model should be interpreted as a same-stream rating reconstruction model rather than a pure cold-start Elo estimator. Its strongest signal comes from causal player history and repeat-player structure. This is leakage-safe because the history is computed only from earlier games, but the headline MAE is most reliable when the validation/test stream has similar player-overlap patterns.

This distinction is important. A reviewer should not read MAE `29` as evidence that the model can estimate an arbitrary unseen player's rating from the first 10 moves alone. It means the pipeline can reconstruct ratings very accurately in a chronological Lichess stream where prior player observations are available causally.

---

## 10. Elo Error by Rating Band

Elo error varies strongly by rating band.

| Elo band    |   Count | White MAE | Black MAE |  Avg MAE | Avg error P50 | Avg error P90 | Avg error P99 |
| ----------- | ------: | --------: | --------: | -------: | ------------: | ------------: | ------------: |
| `<1000`     |   `975` |   `70.50` |   `74.51` |  `72.50` |       `17.02` |      `163.31` |      `746.44` |
| `1000-1399` | `4,212` |   `37.82` |   `37.94` |  `37.88` |       `10.99` |       `71.06` |      `459.43` |
| `1400-1799` | `7,572` |   `23.63` |   `23.44` |  `23.53` |       `10.67` |       `49.71` |      `230.28` |
| `1800-2199` | `6,302` |   `21.62` |   `22.09` |  `21.86` |        `6.16` |       `34.46` |      `367.38` |
| `2200-2599` |   `900` |   `39.85` |   `36.37` |  `38.11` |       `10.57` |       `61.55` |      `705.35` |
| `2600+`     |    `39` |  `148.86` |  `143.66` | `146.26` |       `79.11` |      `204.29` |     `1114.07` |

The model performs best in the dense middle of the Lichess Blitz distribution:

```text
1400-1799 Avg MAE = 23.53
1800-2199 Avg MAE = 21.86
```

It performs worse at the extremes:

```text
<1000 Avg MAE = 72.50
2600+ Avg MAE = 146.26
```

This is expected for three reasons:

1. Extreme rating bands have fewer examples.
2. Very high-rated players are rare, so there is less training coverage.
3. Very low or very high ratings can be more volatile and more sensitive to sparse history.

The `2600+` band has only `39` validation examples, so its error estimate is also less stable.

Practical interpretation:

> The Elo model is very strong for common rating ranges, but tail-rating predictions should be treated with caution.

---

## 11. Prediction-Level Examples

Representative prediction examples:

| Example                   | Game index | Result | P before | P after 10 | Elo error summary                            |
| ------------------------- | ---------: | ------ | -------: | ---------: | -------------------------------------------- |
| High-confidence correct   |   `206351` | `1-0`  | `0.9001` |   `0.9815` | White error `7.57`, Black error `4.86`       |
| High-confidence miss      |   `176737` | `1-0`  | `0.1128` |   `0.0678` | White error `356.43`, Black error `7.83`     |
| Largest after-10 increase |   `206812` | `1-0`  | `0.4917` |   `0.9786` | Large Elo error on both players              |
| Largest after-10 decrease |   `205020` | `0-1`  | `0.6574` |   `0.1690` | Large Elo error on both players              |
| Lowest Elo error          |   `172641` | `0-1`  | `0.4463` |   `0.4271` | White error `0.13`, Black error `0.07`       |
| Highest Elo error         |   `186670` | `1-0`  | `0.6434` |   `0.6586` | White error `1327.98`, Black error `1135.95` |

These examples reveal several important behaviors.

### High-Confidence Correct Case

The high-confidence correct case shows that when pre-game strength, board state, and early-game dynamics align, the model can produce a very high White-win probability and be correct.

This is the ideal use case for the model: ranking games where evidence strongly points in one direction.

### High-Confidence Miss

The high-confidence miss is more interesting. The model predicted a low White-win probability after 10 moves, but White eventually won.

This is not necessarily a model bug. Chess games can change dramatically after move 10 because of:

* tactical blunders,
* time-pressure mistakes,
* sacrifices,
* endgame errors,
* or resignation patterns not visible at ply 20.

The model predicts from limited information, not from the full game.

### Largest Probability Changes

The largest after-10 increase and decrease examples show that the model is sensitive to early board and clock signals. This is desirable because after-10 prediction should not simply copy the pre-game probability.

However, large probability swings can also correspond to cases with large Elo uncertainty. This suggests that unusual games or players with weak history can produce more volatile predictions.

### Elo Extremes

The lowest Elo error case shows that the model can reconstruct ratings almost exactly for some players, usually when history is strong.

The highest Elo error case shows the limitation: when player history is weak, unusual, or misleading, the model can fail badly. This reinforces the need to report Elo results with repeat-player and rating-band caveats.

---

## 12. Practical Value of the White-Win Models

The White-win models should be interpreted as lightweight probabilistic ranking models, not as engine-level predictors.

### What the models are useful for

They are useful for:

* ranking games by White-win probability,
* measuring how much signal is gained from early board state,
* comparing pre-game vs after-3 vs after-10 information,
* building a baseline for engine-enhanced or neural chess models,
* identifying high-probability and low-probability game segments.

The after-10 lift result is particularly useful:

```text
Top 10% White-win rate    = 71.70%
Bottom 10% White-win rate = 28.25%
```

This means that although accuracy is only around `58%`, the model provides meaningful ranking separation.

### What the models are not useful for

They are not suitable for:

* engine-level evaluation,
* move recommendation,
* precise tactical analysis,
* high-stakes betting or trading decisions,
* or replacing Stockfish/deep-learning chess models.

The model is intentionally lightweight and no-Stockfish. It is designed for reproducible ML assessment, not for maximum chess strength.

---

## 13. Practical Value of the Elo Model

The Elo model is practically useful for same-stream player-strength reconstruction.

It can answer:

> Given a player's previous games in this monthly stream and the first 10 full moves of the current game, can we estimate their current rating?

The answer is yes, with strong validation performance:

```text
White MAE = 28.719
Black MAE = 28.935
```

However, the model’s value depends on the data setting.

### Strong use case

The model is strongest when:

* players appear repeatedly,
* the validation stream has similar player overlap to training,
* ratings are in common ranges,
* causal history is available,
* and the target environment resembles the same Lichess monthly stream.

### Weak use case

The model is weaker when:

* players are completely unseen,
* rating bands are rare,
* history is sparse or misleading,
* players have rapidly changing ratings,
* or the test data distribution changes substantially.

Therefore, the correct interpretation is:

> The Elo model is strong for chronological same-month prediction with repeat-player structure, but it should not be presented as a universal cold-start Elo estimator.

---

## 14. XAI and Behavioral Interpretation

This submission uses output-level XAI and diagnostic analysis rather than SHAP-level model introspection.

The reason is practical: the final `report_best` pipeline is designed for reproducible execution and does not require saving heavy model artifacts. Instead, explainability is provided through:

* calibration bins,
* lift analysis,
* Elo error segmentation,
* prediction examples,
* probability diagnostics,
* and feature-group interpretation from experiments.

XAI assets:

```text
Results/xai/calibration_bins_after10.csv
Results/xai/lift_analysis_after10.json
Results/xai/elo_error_segments.csv
Results/xai/prediction_examples.json
Results/xai/xai_summary.md
```

Feature-group interpretation:

| Task         | Main interpretation                                                                              |
| ------------ | ------------------------------------------------------------------------------------------------ |
| Before-game  | Dominated by Elo difference                                                                      |
| After-3      | Still mostly Elo-driven; early board signal is limited                                           |
| After-10     | Board state, material, mobility, development, castling, and clock features add useful signal     |
| Elo after-10 | Driven primarily by causal history, prior observed Elo, player overlap, and rating-band patterns |

This level of explainability is appropriate for the project because the task is not only to maximize performance, but also to show that the pipeline is auditable and leakage-aware.

---

## 15. Leakage and Reproducibility Assessment

The final pipeline is leakage-safe under the project assumptions.

Key controls:

* Validation rows are not used for fitting.
* The split is chronological, not random.
* No result field is used as an input.
* No termination field is used.
* No rating-difference field is used.
* No total game length feature is used.
* No future moves are used.
* No future clock comments are used.
* Elo regression excludes current-game Elo and Elo-derived fields.
* Causal player history is computed before current-game history updates.
* Raw PGN and compressed database files are not included in the submission.

This matters because many apparent gains in chess prediction can come from leakage, especially:

* using final game length,
* using rating differences after the game,
* using future moves,
* fitting preprocessing on validation data,
* or computing player history with future games.

The reported results are meaningful because the pipeline explicitly avoids these leakage sources.

---

## 16. Limitations

The final results are strong for the assessment setting, but they have clear limitations.

### 16.1 Classification Accuracy Is Naturally Limited

Even after 10 moves, White-win prediction is noisy. A model only sees the first 20 plies, while many games are decided later by tactics, blunders, time pressure, or endgame mistakes.

Therefore, an after-10 ROC-AUC of around `0.62` is realistic for a no-engine lightweight model.

### 16.2 Draws Are Simplified

Draws are treated as non-White-wins. This makes the classification task binary, but it loses information compared with a three-class setup:

```text
White win / Draw / Black win
```

Future work could model expected score instead:

```text
White win = 1.0
Draw = 0.5
Black win = 0.0
```

### 16.3 Elo Results Are Repeat-Player Sensitive

The Elo model’s headline MAE is very strong, but the result is partly driven by repeated players in the chronological stream.

This is valid under the project setup, but the result should be reported carefully:

> The model is best interpreted as a same-stream Elo reconstruction model, not a pure cold-start rating estimator. Its low MAE is leakage-safe because history features are computed only from earlier games, but the metric depends on similar repeat-player/history overlap in validation.

### 16.4 No Stockfish in Final Profile

The no-Stockfish profile improves portability but does not maximize possible after-10 classification performance. Stockfish-heavy experiments achieved higher after-10 ROC-AUC, but were not selected because of dependency and reproducibility cost.

---

## 17. Final Conclusion

The final pipeline achieves a strong balance between predictive performance, reproducibility, and leakage safety.

The most important result is the after-10 White-win model:

```text
After-10 ROC-AUC = 0.621742
Elo baseline ROC-AUC = 0.578497
Improvement = +0.043245
```

This shows that early board state and clock information add meaningful predictive signal beyond pre-game Elo.

The second major result is Elo prediction:

```text
White MAE = 28.719
Black MAE = 28.935
Mean baseline MAE ≈ 300
```

This shows that causal player history and early-game features can reconstruct ratings well in a same-month chronological stream.

The Elo result should be reported with its baseline and caveat together: the model improves dramatically over the mean Elo baseline, but the improvement comes primarily from causal repeat-player history rather than from inferring skill solely from 20 plies of chess moves.

Overall, the selected no-Stockfish `report_best` profile is a defensible final solution because it:

* keeps after-10 prediction clearly above the Elo expected-score baseline,
* improves after-10 prediction meaningfully,
* produces useful probability ranking and lift,
* predicts Elo accurately under the chronological setup,
* avoids Stockfish dependency,
* avoids mandatory LightGBM/XGBoost dependencies in the final submitted profile,
* and remains leakage-safe and reproducible.

The final results are not engine-level chess predictions, but they are strong and practically meaningful for a reproducible quantitative ML assessment.
