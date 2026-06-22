# AI Workflow

## AI Tools Used

ChatGPT / Codex based on GPT-5 was used as an engineering assistant during the project.

## How AI Was Used

AI support was used for:

- Clarifying ambiguous assessment requirements.
- Designing the data pipeline.
- Identifying leakage risks.
- Planning feature engineering.
- Drafting and reviewing model-selection experiments.
- Writing and refactoring Python code.
- Summarizing experiment results.
- Drafting documentation and reports.
- Auditing reproducibility and packaging risks.

## How AI Was Not Used

AI did not:

- Train the models independently.
- Access hidden validation data.
- Change the chronological validation split.
- Provide private Lichess data.
- Replace the final execution of the pipeline.

All reported metrics come from local runs of `Solution.py` or related local experiment scripts.

## Human Review Responsibility

The user remains responsible for:

- Reviewing the final code.
- Running the pipeline in the target environment.
- Verifying dependency installation.
- Confirming the final submission package.

## Reproducibility Notes

The workflow is intentionally transparent:

- The data source is public.
- The selected month is reproducible.
- The code streams and parses games directly.
- Metrics and prediction CSV files are saved.
- No raw PGN data or large model binaries are included.

