# Final Submission Checklist

Use this checklist to perform final sanity checks and build the clean upload package for the quantitative research assessment.

## 1. Required Files and Directory Structure

The final submission folder should be named `final_submission/` and structured as follows:

```text
final_submission/
├── solution.py                 # Self-contained reproducible data streaming + model pipeline
├── requirements.txt            # Minimal python dependencies
├── README.md                   # Full research report, business value analysis, and XAI findings
└── outputs_full_safe_default/  # Selected model outputs (from 100K games run)
    ├── metrics.json            # Final evaluations and parameters JSON
    ├── validation_predictions.csv  # Chronological validation predictions (row-by-row)
    ├── model_explainability.json   # Model coefficients (optional but recommended)
    └── prediction_explanation_examples.json # Row-level deterministic explanations (optional)
```

---

## 2. Forbidden Files (Exclude from Zip/Upload)

Ensure the following files are **not** present in your upload directory (check your `.gitignore` or zip command):

*   **Raw Chess Data**: Any `.pgn`, `.pgn.zst`, or decompressed chess text files (too large!).
*   **Virtual Environments**: The `.venv/` or any local virtual environment folder.
*   **Python Caches**: All `__pycache__/` directories and `.pyc` files.
*   **OS Artifacts**: `.DS_Store` (Mac) or `Thumbs.db` (Windows) files.
*   **Temporary/Scratch Files**: Development scratch scripts or intermediate cache files (like raw datasets cached during experimentation).

---

## 3. Pre-Upload Sanity Checks

### Size and Format Checks
*   [ ] **Zip File Size**: The final compressed archive should be under 5MB (since raw data is excluded).
*   [ ] **Validation Predictions CSV**: Ensure `validation_predictions.csv` contains predictions for exactly `20,000` games (20% of the 100K dataset).
*   [ ] **Predictions Column Headers**: Verify that columns include: `game_index`, `white_player`, `black_player`, `result`, `white_win_true`, `white_elo`, `black_elo`, `p_white_win_elo_baseline`, `p_white_win_before`, `p_white_win_after_3`, `p_white_win_after_10`, `white_elo_pred_after_10`, `black_elo_pred_after_10`, `split`.

### Rerun Command Verification
Run the pipeline from a fresh directory using the following terminal command to ensure it runs to completion without errors:

```bash
# Setup clean environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run pipeline on default settings (fits models and writes outputs)
python3 solution.py --target-games 100000 --selected-month 2023-11 --output-dir outputs_full_safe_default
```

*   [ ] The script completes successfully and creates the output directory.
*   [ ] No external API tokens are required.
*   [ ] Running with `--help` prints option usage correctly.

### Output Verification
*   [ ] **No Leakage**: Check that `white_elo` or `black_elo` are not being passed as features in `solution.py` to the Elo regression model (`ValueError` guard should not trigger).
*   [ ] **Valid Probabilities**: Check that all classifier output probabilities lie strictly in the range $(0, 1)$ and are clipped to avoid log-loss calculation crashes on boundary conditions.
