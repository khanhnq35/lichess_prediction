# Submission Package

This folder contains the final organized submission materials for the Lichess Blitz prediction assessment.

## Contents

- `Solution.py`: reproducible Python pipeline.
- `requirements.txt`: strict lightweight dependencies.
- `requirements-experiments.txt`: optional LightGBM/XGBoost dependencies for the best no-Stockfish profile.
- `Results/`: final full 100k metrics and validation predictions.
- `Results/xai/`: generated calibration, lift, Elo error, and prediction-example analysis.
- `Results/xai/generate_xai_report.py`: reproducible output-level XAI helper.
- `docs/00_quick_overview_report.md`: shortest overview and reading guide.
- `docs/01_Problem_definition.md`: task interpretation, assumptions, and leakage constraints.
- `docs/02_Research.md`: modeling options considered.
- `docs/03_Experiment.md`: experiments and results.
- `docs/04_Proposal_pipeline.md`: final selected pipeline.
- `docs/05_results_report.md`: detailed final result analysis.
- `docs/06_Limitation_and_future_work.md`: limitations and next steps.
- `docs/07_AI_workflow.md`: AI usage disclosure.

## Run Strict Lightweight Profile

```bash
pip install -r requirements.txt
python Solution.py --target-games 100000 --output-dir outputs_lightweight
```

## Run Final Boosting Profile

```bash
pip install -r requirements.txt
pip install -r requirements-experiments.txt
python Solution.py --target-games 100000 --output-dir outputs_full --model-profile boosting
```

The boosting profile is the reported best no-Stockfish configuration.

## Regenerate XAI Outputs

```bash
python Results/xai/generate_xai_report.py
```

The script uses only `Results/metrics.json` and `Results/validation_predictions.csv`.
