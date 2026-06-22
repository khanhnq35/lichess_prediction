#!/usr/bin/env python3
"""Generate output-level XAI artifacts for the submission package.

This script intentionally uses only files inside `submission/Results`.
It does not require raw PGN data, fitted model binaries, Stockfish, or refitting.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name == "xai" and SCRIPT_PATH.parent.parent.name == "Results":
    RESULTS_DIR = SCRIPT_PATH.parent.parent
    ROOT = RESULTS_DIR.parent
else:
    ROOT = SCRIPT_PATH.parents[1]
    RESULTS_DIR = ROOT / "Results"
XAI_DIR = RESULTS_DIR / "xai"
METRICS_PATH = RESULTS_DIR / "metrics.json"
PREDICTIONS_PATH = RESULTS_DIR / "validation_predictions.csv"


def rmse(values: pd.Series) -> float:
    return float(math.sqrt(np.mean(np.square(values.astype(float)))))


def calibration_bins(df: pd.DataFrame, prob_col: str, target_col: str, bins: int = 10) -> pd.DataFrame:
    work = df[[prob_col, target_col]].copy()
    work["bin"] = pd.qcut(work[prob_col], q=bins, labels=False, duplicates="drop")
    rows = []
    for bin_id in sorted(work["bin"].dropna().unique()):
        part = work[work["bin"] == bin_id]
        rows.append(
            {
                "bin": int(bin_id) + 1,
                "count": int(len(part)),
                "mean_predicted_prob": float(part[prob_col].mean()),
                "actual_white_win_rate": float(part[target_col].mean()),
                "calibration_gap_pred_minus_actual": float(part[prob_col].mean() - part[target_col].mean()),
            }
        )
    return pd.DataFrame(rows)


def lift_analysis(df: pd.DataFrame, prob_col: str, target_col: str) -> dict[str, float]:
    sorted_df = df.sort_values(prob_col, ascending=False).reset_index(drop=True)
    n = len(sorted_df)
    n10 = max(1, int(n * 0.10))
    n20 = max(1, int(n * 0.20))
    top10 = float(sorted_df.iloc[:n10][target_col].mean())
    bottom10 = float(sorted_df.iloc[-n10:][target_col].mean())
    top20 = float(sorted_df.iloc[:n20][target_col].mean())
    bottom20 = float(sorted_df.iloc[-n20:][target_col].mean())
    return {
        "rows": int(n),
        "top_10_pct_white_win_rate": top10,
        "bottom_10_pct_white_win_rate": bottom10,
        "top_minus_bottom_10_pct_lift": top10 - bottom10,
        "top_20_pct_white_win_rate": top20,
        "bottom_20_pct_white_win_rate": bottom20,
        "top_minus_bottom_20_pct_lift": top20 - bottom20,
    }


def elo_error_segments(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["white_abs_error"] = (work["white_elo_pred_after_10"] - work["white_elo"]).abs()
    work["black_abs_error"] = (work["black_elo_pred_after_10"] - work["black_elo"]).abs()
    work["avg_abs_error"] = (work["white_abs_error"] + work["black_abs_error"]) / 2.0
    work["mean_true_elo"] = (work["white_elo"] + work["black_elo"]) / 2.0
    bins = [0, 1000, 1400, 1800, 2200, 2600, 4000]
    labels = ["<1000", "1000-1399", "1400-1799", "1800-2199", "2200-2599", "2600+"]
    work["elo_band"] = pd.cut(work["mean_true_elo"], bins=bins, labels=labels, include_lowest=True, right=False)
    rows = []
    for band, part in work.groupby("elo_band", observed=False):
        if len(part) == 0:
            continue
        rows.append(
            {
                "elo_band": str(band),
                "count": int(len(part)),
                "white_mae": float(part["white_abs_error"].mean()),
                "black_mae": float(part["black_abs_error"].mean()),
                "avg_mae": float(part["avg_abs_error"].mean()),
                "white_rmse": rmse(part["white_elo_pred_after_10"] - part["white_elo"]),
                "black_rmse": rmse(part["black_elo_pred_after_10"] - part["black_elo"]),
                "avg_error_p50": float(part["avg_abs_error"].quantile(0.50)),
                "avg_error_p90": float(part["avg_abs_error"].quantile(0.90)),
                "avg_error_p99": float(part["avg_abs_error"].quantile(0.99)),
            }
        )
    return pd.DataFrame(rows)


def row_to_example(row: pd.Series, note: str) -> dict[str, object]:
    return {
        "note": note,
        "game_index": int(row["game_index"]),
        "white_player": str(row["white_player"]),
        "black_player": str(row["black_player"]),
        "result": str(row["result"]),
        "white_win_true": int(row["white_win_true"]),
        "p_white_win_before": float(row["p_white_win_before"]),
        "p_white_win_after_3": float(row["p_white_win_after_3"]),
        "p_white_win_after_10": float(row["p_white_win_after_10"]),
        "white_elo": int(row["white_elo"]),
        "black_elo": int(row["black_elo"]),
        "white_elo_pred_after_10": float(row["white_elo_pred_after_10"]),
        "black_elo_pred_after_10": float(row["black_elo_pred_after_10"]),
        "white_elo_abs_error": float(abs(row["white_elo_pred_after_10"] - row["white_elo"])),
        "black_elo_abs_error": float(abs(row["black_elo_pred_after_10"] - row["black_elo"])),
    }


def prediction_examples(df: pd.DataFrame) -> list[dict[str, object]]:
    work = df.copy()
    work["after10_binary_pred"] = (work["p_white_win_after_10"] >= 0.5).astype(int)
    work["prob_delta_after10_vs_before"] = work["p_white_win_after_10"] - work["p_white_win_before"]
    work["avg_elo_abs_error"] = (
        (work["white_elo_pred_after_10"] - work["white_elo"]).abs()
        + (work["black_elo_pred_after_10"] - work["black_elo"]).abs()
    ) / 2.0

    examples = []
    high_correct = work[work["after10_binary_pred"] == work["white_win_true"]].sort_values(
        "p_white_win_after_10", ascending=False
    )
    if not high_correct.empty:
        examples.append(row_to_example(high_correct.iloc[0], "High-confidence correct after-10 prediction"))

    high_wrong = work[work["after10_binary_pred"] != work["white_win_true"]].copy()
    if not high_wrong.empty:
        high_wrong["confidence"] = (high_wrong["p_white_win_after_10"] - 0.5).abs()
        examples.append(row_to_example(high_wrong.sort_values("confidence", ascending=False).iloc[0], "High-confidence miss"))

    examples.append(
        row_to_example(
            work.sort_values("prob_delta_after10_vs_before", ascending=False).iloc[0],
            "Largest after-10 probability increase relative to before-game",
        )
    )
    examples.append(
        row_to_example(
            work.sort_values("prob_delta_after10_vs_before", ascending=True).iloc[0],
            "Largest after-10 probability decrease relative to before-game",
        )
    )
    examples.append(row_to_example(work.sort_values("avg_elo_abs_error", ascending=True).iloc[0], "Lowest Elo error example"))
    examples.append(row_to_example(work.sort_values("avg_elo_abs_error", ascending=False).iloc[0], "Highest Elo error example"))
    return examples


def examples_to_markdown(examples: list[dict[str, object]]) -> str:
    rows = [
        "| note | game_index | result | p_white_win_before | p_white_win_after_10 | white_elo_abs_error | black_elo_abs_error |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for example in examples:
        rows.append(
            "| {note} | {game_index} | {result} | {p_before:.4f} | {p_after10:.4f} | {w_err:.2f} | {b_err:.2f} |".format(
                note=str(example["note"]).replace("|", "\\|"),
                game_index=int(example["game_index"]),
                result=str(example["result"]),
                p_before=float(example["p_white_win_before"]),
                p_after10=float(example["p_white_win_after_10"]),
                w_err=float(example["white_elo_abs_error"]),
                b_err=float(example["black_elo_abs_error"]),
            )
        )
    return "\n".join(rows)


def write_markdown_summary(
    metrics: dict,
    cal: pd.DataFrame,
    lift: dict,
    elo_segments: pd.DataFrame,
    examples: list[dict[str, object]],
) -> None:
    models = metrics["models"]
    baselines = metrics["baselines"]
    dataset = metrics["dataset_summary"]

    lines = [
        "# Output-Level XAI Summary",
        "",
        "This report explains the final validation outputs using only submission-local files.",
        "It is not a SHAP report and does not require fitted model objects.",
        "",
        "## Scope and Limitations",
        "",
        "This is output-level XAI, not SHAP, LIME, or exact feature-attribution XAI. It analyzes the behavior of saved validation predictions rather than introspecting fitted model internals.",
        "",
        "The report can analyze probability calibration, lift/ranking behavior, Elo error segments, representative success/failure examples, and validation-level behavior. It cannot provide exact per-feature attribution because the submitted package does not require fitted model objects, SHAP/LIME dependencies, Stockfish, raw PGN files, or model refitting.",
        "",
        "## Dataset Context",
        "",
        f"- Validation games: `{dataset['validation_games']:,}`.",
        f"- Validation positive rate: `{dataset['validation_positive_rate']:.4f}`.",
        f"- Final profile: `{metrics['run_config'].get('model_profile', 'unknown')}`.",
        "",
        "## Key Model Metrics",
        "",
        "| Task | Primary metric | Value |",
        "|---|---:|---:|",
        f"| White win before | ROC-AUC | {models['white_win_before_game']['roc_auc']:.6f} |",
        f"| White win after 3 | ROC-AUC | {models['white_win_after_3_moves']['roc_auc']:.6f} |",
        f"| White win after 10 | ROC-AUC | {models['white_win_after_10_moves']['roc_auc']:.6f} |",
        f"| Elo after 10 | Avg MAE | {(models['elo_after_10_moves']['white_elo_mae'] + models['elo_after_10_moves']['black_elo_mae']) / 2.0:.3f} |",
        f"| Elo expected-score baseline | ROC-AUC | {baselines['elo_expected_score_baseline']['roc_auc']:.6f} |",
        "",
        "## Calibration And Lift",
        "",
        f"- Top 10% predicted after-10 White-win rate: `{lift['top_10_pct_white_win_rate']:.4f}`.",
        f"- Bottom 10% predicted after-10 White-win rate: `{lift['bottom_10_pct_white_win_rate']:.4f}`.",
        f"- Top-minus-bottom 10% lift: `{lift['top_minus_bottom_10_pct_lift']:.4f}`.",
        f"- Top 20% predicted after-10 White-win rate: `{lift['top_20_pct_white_win_rate']:.4f}`.",
        f"- Bottom 20% predicted after-10 White-win rate: `{lift['bottom_20_pct_white_win_rate']:.4f}`.",
        "",
        "The after-10 model is useful for ranking games: the highest-probability bucket has a much higher actual White-win rate than the lowest bucket.",
        "",
        "## Calibration Bins",
        "",
        cal.to_markdown(index=False),
        "",
        "## Elo Error By Rating Band",
        "",
        elo_segments.to_markdown(index=False),
        "",
        "## Representative Prediction Examples",
        "",
        "The script also exports the full examples to `prediction_examples.json`. The table below summarizes the same representative validation rows.",
        "",
        examples_to_markdown(examples),
        "",
        "These examples show that confidence can be useful but is not perfect. Future mistakes, tactical swings, or time-pressure collapses after move 10 are not visible to the model. Elo prediction can be extremely accurate for repeat or history-rich players, but it can fail badly for sparse-history players or unusual rating-band cases.",
        "",
        "## Interpretation",
        "",
        "- Pre-game prediction is dominated by Elo difference and stays close to the Elo expected-score baseline.",
        "- After-10 prediction gains signal from observed board structure and clock/time-pressure features.",
        "- Elo prediction is very strong because causal player-history features are highly predictive for repeat players.",
        "- The Elo model should be interpreted as a same-stream rating reconstruction model rather than a pure cold-start Elo estimator. Its strongest signal comes from causal player history and repeat-player structure. This is leakage-safe because the history is computed only from earlier games, but the headline MAE is most reliable when the validation/test stream has similar player-overlap patterns.",
    ]
    (XAI_DIR / "xai_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    XAI_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    df = pd.read_csv(PREDICTIONS_PATH)

    cal = calibration_bins(df, "p_white_win_after_10", "white_win_true")
    lift = lift_analysis(df, "p_white_win_after_10", "white_win_true")
    elo_segments = elo_error_segments(df)
    examples = prediction_examples(df)

    cal.to_csv(XAI_DIR / "calibration_bins_after10.csv", index=False)
    elo_segments.to_csv(XAI_DIR / "elo_error_segments.csv", index=False)
    (XAI_DIR / "lift_analysis_after10.json").write_text(json.dumps(lift, indent=2), encoding="utf-8")
    (XAI_DIR / "prediction_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    write_markdown_summary(metrics, cal, lift, elo_segments, examples)

    print(f"Wrote XAI artifacts to {XAI_DIR}")


if __name__ == "__main__":
    main()
