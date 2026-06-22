# AI Workflow

## AI Tools Used

ChatGPT and Codex/Antigravity, based on GPT-5.5 and gemini 3.5 flash models, were used as AI engineering assistants during the project.

The AI tools were used in a human-in-the-loop workflow. They supported research, design, implementation, debugging, experiment planning, result interpretation, and documentation. However, all technical decisions, final design choices, code acceptance, experiment selection, and submission decisions were reviewed and made by the human developer.

## Role of AI in the Project

AI was used as a productivity and reasoning assistant across the project lifecycle. Its role was to help accelerate exploration, reduce implementation friction, identify risks, and improve documentation quality.

The workflow was intentionally designed so that AI could provide suggestions, alternatives, drafts, and diagnostic hypotheses, while the human developer remained responsible for evaluating those suggestions against the assessment requirements, implementation constraints, reproducibility expectations, and empirical results.

In practice, AI acted as a technical collaborator in the following ways:

* Research assistant for understanding the Lichess dataset and related modeling approaches.
* Design assistant for structuring the data ingestion, feature engineering, training, and prediction pipeline.
* Implementation assistant for drafting and refactoring Python code.
* Debugging assistant for identifying possible causes of parsing, leakage, dependency, or validation issues.
* Experiment assistant for proposing model variants and feature ablations.
* Documentation assistant for writing reproducibility notes, limitations, and workflow descriptions.
* Review assistant for checking whether the final solution remained aligned with the assessment requirements.

AI outputs were treated as proposals, not as final truth. Each important suggestion was reviewed, adapted, tested, or rejected by the human developer.

## How AI Was Used

AI support was used for the following project stages.

### 1. Requirement Clarification

AI was used to help interpret ambiguous parts of the assessment, including:

* How to define an eligible Lichess game.
* How to interpret “after 3 moves” and “after 10 moves”.
* How to separate pre-game, after-3, and after-10 prediction settings.
* How to avoid using validation games during training.
* How to keep the submitted package under the required size limit.
* How to document the selected month, time-control filter, and reproducibility assumptions.

The final interpretation used in the solution was selected by the human developer and documented in the code and report.

### 2. Dataset Research and Pipeline Planning

AI was used to summarize relevant information about the Lichess open database, including PGN structure, metadata fields, time-control definitions, clock comments, evaluation comments, and practical constraints related to streaming large `.pgn.zst` files.

AI also helped compare possible data processing strategies, such as:

* Full download versus streaming decompression.
* Header-only filtering before full PGN parsing.
* Chronological split versus random split.
* Strict eligibility filtering versus more permissive parsing.
* Compact derived features versus large raw artifacts.

The final pipeline design was chosen to prioritize reproducibility, portability, and leakage-safe validation.

### 3. Leakage Risk Analysis

AI was used to identify potential leakage risks in the dataset and feature design. Examples include:

* Avoiding `Result` as an input feature.
* Excluding `WhiteRatingDiff` and `BlackRatingDiff`.
* Avoiding final game length, final board state, final clock state, and termination-based features.
* Preventing moves after the prediction cutoff from entering after-3 or after-10 features.
* Computing player-history features chronologically, before updating the current game.
* Keeping validation games out of training and feature-history construction.

These risks were reviewed by the human developer and incorporated into the final pipeline design. Leakage prevention was treated as a core design requirement rather than a post-hoc cleanup step.

### 4. Feature Engineering Design

AI was used to brainstorm and organize feature groups for each prediction task.

The final feature engineering design separates features by prediction time:

* Pre-game features for White-win prediction before the game.
* Early-game features after 3 full moves.
* Early-game features after 10 full moves.
* Elo-prediction features after 10 full moves.

AI helped propose feature families such as:

* Elo and rating-difference features.
* Time-control features.
* Move-prefix features.
* Board-state features.
* Material-balance features.
* Castling and development features.
* Clock-usage features.
* Causal player-history features.
* Missing-value indicators.

The human developer selected the final feature set based on feasibility, reproducibility, empirical validation performance, and leakage safety.

### 5. Model Selection and Experiment Planning

AI was used to propose candidate models and experiment variants.

The model-selection process considered:

* Simple baselines such as logistic regression and ridge regression.
* Tree-based models for tabular features.
* Gradient-boosted models such as LightGBM and XGBoost.
* Strict fallback models for environments where optional boosting dependencies are unavailable.
* Separate models for White-win prediction and Elo prediction.
* Ablation experiments to compare feature groups.

The final model profile was chosen by balancing validation performance, runtime, dependency risk, solution size, and reproducibility.

### 6. Implementation Support

AI was used to assist with code drafting and refactoring. This included support for:

* PGN streaming and parsing logic.
* Feature extraction functions.
* Chronological train/validation splitting.
* Model training wrappers.
* Metric computation.
* Prediction CSV generation.
* Experiment configuration.
* Error handling and fallback behavior.
* Documentation strings and comments.

All AI-generated code was reviewed before being accepted. The human developer remained responsible for running the code locally, checking outputs, validating assumptions, and ensuring that the final submission was reproducible.

### 7. Diagnosis and Debugging

AI was used as a debugging assistant when investigating pipeline issues. It helped generate hypotheses about possible problems such as:

* PGN parsing failures.
* Missing or malformed headers.
* Incorrect time-control filtering.
* Incorrect cutoff interpretation.
* Feature leakage.
* Validation split mistakes.
* Dependency incompatibilities.
* Unexpected metric changes.
* Overly strong Elo prediction caused by potential leakage.

AI suggestions were not accepted automatically. Each diagnosis was checked against code behavior, logs, intermediate artifacts, and local experiment results.

### 8. Tuning and Evaluation

AI helped plan tuning experiments and feature ablations, but model training and metric computation were performed locally.

AI support was used to decide which experiments were worth running under the time constraint, including:

* Comparing baseline and boosting models.
* Comparing feature groups with and without player-history features.
* Comparing strict and full model profiles.
* Checking whether after-10 features improved over pre-game and after-3 features.
* Reviewing whether Elo-prediction metrics were plausible or suspiciously strong.

The final reported metrics come from local execution of the pipeline, not from AI-generated estimates.

### 9. Documentation and Reporting

AI was used to draft and polish documentation, including:

* Dataset description.
* Feature engineering explanation.
* Model-selection rationale.
* Reproducibility notes.
* Limitations and future work.
* AI workflow disclosure.
* Packaging and dependency notes.

The documentation was reviewed and edited by the human developer to ensure that it accurately reflected the implemented solution and did not overclaim the model’s capabilities.

## How AI Was Not Used

AI was not used as an autonomous system that independently completed the project.

Specifically, AI did not:

* Train the final models independently.
* Access hidden validation data.
* Access private Lichess data.
* Modify the chronological validation split without review.
* Decide the final feature set without human approval.
* Decide the final model profile without human approval.
* Generate final reported metrics without local execution.
* Replace manual review of the final code.
* Replace final responsibility for the submission.

All reported metrics come from local runs of `Solution.py` or related local experiment scripts. The AI tools only assisted with reasoning, implementation support, review, and documentation.

## Human Review and Final Responsibility

The human developer remained responsible for the final project decisions.

This included responsibility for:

* Interpreting the assessment requirements.
* Choosing the selected Lichess month and time control.
* Defining eligible games.
* Reviewing the data-processing pipeline.
* Verifying that validation games were not used for training.
* Reviewing leakage risks.
* Selecting the final model profile.
* Running the pipeline locally.
* Checking generated metrics and prediction files.
* Verifying dependency installation.
* Reviewing the final code and documentation.
* Preparing the final submission package.

AI suggestions were used only after human review. When AI proposed multiple possible directions, the human developer selected the direction that best matched the time limit, reproducibility requirements, performance goals, and assessment constraints.

## Responsible AI Usage Principles

The project followed a responsible AI-assisted development workflow.

The main principles were:

1. **Human ownership**
   AI assisted the project, but the human developer retained ownership of design decisions, implementation acceptance, and final submission quality.

2. **Transparency**
   The use of AI is explicitly documented. The report describes where AI helped and where it did not.

3. **Verification before acceptance**
   AI-generated suggestions were checked through code review, local execution, validation metrics, and reproducibility checks.

4. **No hidden data access**
   AI was not used to access hidden validation data or private datasets. The solution uses the public Lichess open database.

5. **Leakage awareness**
   AI was used to help identify leakage risks, but leakage prevention was implemented and verified in the actual local pipeline.

6. **Reproducibility first**
   AI-assisted code and documentation were reviewed to ensure that another user can reproduce the data download, processing, training, and prediction workflow.

7. **No metric fabrication**
   AI was not used to invent final results. Reported scores were generated by local experiments.

8. **Practical constraint awareness**
   AI recommendations were filtered through project constraints, including the 24-hour time limit, dependency availability, runtime, and submission size limit.

## AI-Assisted Workflow Summary

The overall AI-assisted workflow can be summarized as follows:

1. The human developer reviewed the assessment requirements and identified the project goals.
2. AI helped research the dataset, modeling options, leakage risks, and feasible implementation strategies.
3. The human developer selected a practical solution direction focused on a reproducible tabular ML pipeline.
4. AI helped draft code, suggest feature groups, and propose model experiments.
5. The human developer ran the pipeline locally and inspected intermediate outputs.
6. AI helped diagnose issues and suggest fixes when results or logs indicated possible problems.
7. The human developer selected the final models and accepted only the code paths that passed local validation.
8. AI helped draft the final documentation.
9. The human developer reviewed the final package for correctness, reproducibility, and assessment alignment.

This workflow reflects professional AI-assisted engineering: AI was used broadly as a support tool, but the human developer remained the decision-maker and final reviewer throughout the project.

## Reproducibility Notes

The workflow is intentionally transparent and reproducible:

* The data source is public.
* The selected month is fixed and documented.
* The time-control filter is implemented in code.
* Games are streamed and parsed directly from the Lichess PGN archive.
* Eligibility filtering is deterministic.
* The chronological split is fixed.
* Validation games are not used for training.
* Feature extraction is performed from information available only at the relevant prediction cutoff.
* Metrics are computed by local scripts.
* Prediction CSV files are saved as artifacts.
* No raw PGN data or large model binaries are included in the submission.
* Optional dependencies are documented, and a fallback profile is available for stricter environments.

These design choices make the project auditable and reduce the risk that AI-assisted development introduces hidden, non-reproducible, or unverifiable behavior.
