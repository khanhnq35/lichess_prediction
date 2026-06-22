#!/usr/bin/env python3
"""A/B Testing script to compare chess prediction models with and without Stockfish.

Trains scikit-learn tree models and Boosting models (LightGBM/XGBoost) for tasks:
- T2 (after 3 moves classification)
- T3 (after 10 moves classification)
- T4 (Elo regression after 10 moves)

For each model and task, it evaluates:
- Mode A: Without Stockfish (features: board, move-text, history, clock, identity)
- Mode B: With Stockfish (adds cp/mate scores)
"""

import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import solution
from solution import (
    Config,
    ensure_enhanced_board_features,
    add_stockfish_features,
    split_train_validation,
    model_feature_columns,
    evaluate_classifier,
    evaluate_regressor,
    load_boosting_classes
)

def get_models(seed: int, classes: tuple):
    """Define classification and regression models to test."""
    LGBMClassifier, LGBMRegressor, XGBClassifier, XGBRegressor = classes

    # Classification models dictionary
    cls_models = {
        "LogisticRegression": LogisticRegression(C=0.25, solver="liblinear", random_state=seed, max_iter=1000),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, max_depth=5, random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=seed),
        "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=4, num_leaves=15, verbose=-1, n_jobs=-1, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=3, tree_method="hist", verbosity=0, n_jobs=-1, random_state=seed)
    }

    # Regression models dictionary (Ridge and RF support multi-output directly, others wrapped)
    reg_models = {
        "Ridge": Ridge(alpha=10.0),
        "GradientBoosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)),
        "HistGradientBoosting": MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=150, max_depth=5, random_state=seed)),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=8, n_jobs=-1, random_state=seed),
        "LightGBM": MultiOutputRegressor(LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=4, num_leaves=15, verbose=-1, n_jobs=-1, random_state=seed)),
        "XGBoost": MultiOutputRegressor(XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=3, tree_method="hist", verbosity=0, n_jobs=-1, random_state=seed))
    }

    return cls_models, reg_models

def run_ab_testing(target_games: int, output_dir: Path, input_cache_csv: str | None = None):
    """Run A/B testing on target games and save results to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    
    # 1. Load classes dynamically to avoid errors if packages are missing
    classes = load_boosting_classes()
    
    # 2. Load dataset
    if input_cache_csv and Path(input_cache_csv).exists():
        print(f"Loading cached dataset from {input_cache_csv}")
        df = pd.read_csv(input_cache_csv).head(target_games).copy()
    else:
        # Stream from Lichess or load default 100k cache
        default_cache = PROJECT_ROOT / "experiment" / "outputs" / "cache" / "games_2023-11_100000.csv.gz"
        if default_cache.exists():
            print(f"Loading default 100k cached dataset from {default_cache}")
            df = pd.read_csv(default_cache).head(target_games).copy()
        else:
            print("Default cache not found. Streaming dataset from Lichess...")
            config = Config(target_games=target_games, selected_month="2023-11")
            df, _ = solution.build_dataset(config, "2023-11")

    # 3. Extract enhanced features and Stockfish features
    df = ensure_enhanced_board_features(df)
    stockfish_cache = PROJECT_ROOT / "experiment" / "stockfish_cache.json"
    print(f"Applying Stockfish features using cache: {stockfish_cache}")
    df = add_stockfish_features(df, cache_path=stockfish_cache, depth=10)

    # 4. Train-validation split
    train_df, val_df = split_train_validation(df, 0.8)
    print(f"Data split: Train={len(train_df)} games, Validation={len(val_df)} games")

    # Define targets
    y_train_cls = train_df["white_win"]
    y_val_cls = val_df["white_win"]
    y_train_elo = train_df[["white_elo", "black_elo"]]
    y_val_elo = val_df[["white_elo", "black_elo"]]

    # Get feature subsets
    feature_sets = model_feature_columns(df, use_history=True, use_clock=True, use_enhanced_board=True)
    after3_base = feature_sets["after3_numeric"]
    after10_base = feature_sets["after10_numeric"]
    elo_base = feature_sets["elo_after10_numeric"]

    # Stockfish features
    sf3_cols = ["sf3_cp", "sf3_mate"]
    sf10_cols = ["sf3_cp", "sf3_mate", "sf10_cp", "sf10_mate", "sf10_cp_diff"]

    # Initialize results structures
    results = []
    cls_models, reg_models = get_models(seed=42, classes=classes)

    # Helper function to log classification runs
    def test_classifier(task, model_name, model, cols_no_sf, cols_sf):
        print(f"[{task}] Training {model_name}...")
        # Mode A: No Stockfish
        t0 = time.time()
        model.fit(train_df[cols_no_sf], y_train_cls)
        t_fit_no_sf = time.time() - t0
        metrics_no_sf = evaluate_classifier(f"{task}_{model_name}_no_sf", model, val_df[cols_no_sf], y_val_cls)

        # Mode B: With Stockfish
        t0 = time.time()
        model.fit(train_df[cols_sf], y_train_cls)
        t_fit_sf = time.time() - t0
        metrics_sf = evaluate_classifier(f"{task}_{model_name}_sf", model, val_df[cols_sf], y_val_cls)

        results.append({
            "task": task,
            "model_name": model_name,
            "mode": "No Stockfish (A)",
            "auc": metrics_no_sf["roc_auc"],
            "log_loss": metrics_no_sf["log_loss"],
            "accuracy": metrics_no_sf["accuracy"],
            "fit_time_seconds": t_fit_no_sf
        })
        results.append({
            "task": task,
            "model_name": model_name,
            "mode": "With Stockfish (B)",
            "auc": metrics_sf["roc_auc"],
            "log_loss": metrics_sf["log_loss"],
            "accuracy": metrics_sf["accuracy"],
            "fit_time_seconds": t_fit_sf
        })

    # Helper function to log regression runs
    def test_regressor(task, model_name, model, cols_no_sf, cols_sf):
        print(f"[{task}] Training {model_name}...")
        # Mode A: No Stockfish
        t0 = time.time()
        model.fit(train_df[cols_no_sf], y_train_elo)
        t_fit_no_sf = time.time() - t0
        metrics_no_sf = evaluate_regressor(f"{task}_{model_name}_no_sf", model, val_df[cols_no_sf], y_val_elo)
        avg_mae_no_sf = (metrics_no_sf["white_elo_mae"] + metrics_no_sf["black_elo_mae"]) / 2.0
        avg_r2_no_sf = (metrics_no_sf["white_elo_r2"] + metrics_no_sf["black_elo_r2"]) / 2.0

        # Mode B: With Stockfish
        t0 = time.time()
        model.fit(train_df[cols_sf], y_train_elo)
        t_fit_sf = time.time() - t0
        metrics_sf = evaluate_regressor(f"{task}_{model_name}_sf", model, val_df[cols_sf], y_val_elo)
        avg_mae_sf = (metrics_sf["white_elo_mae"] + metrics_sf["black_elo_mae"]) / 2.0
        avg_r2_sf = (metrics_sf["white_elo_r2"] + metrics_sf["black_elo_r2"]) / 2.0

        results.append({
            "task": task,
            "model_name": model_name,
            "mode": "No Stockfish (A)",
            "avg_mae": avg_mae_no_sf,
            "white_mae": metrics_no_sf["white_elo_mae"],
            "black_mae": metrics_no_sf["black_elo_mae"],
            "avg_r2": avg_r2_no_sf,
            "fit_time_seconds": t_fit_no_sf
        })
        results.append({
            "task": task,
            "model_name": model_name,
            "mode": "With Stockfish (B)",
            "avg_mae": avg_mae_sf,
            "white_mae": metrics_sf["white_elo_mae"],
            "black_mae": metrics_sf["black_elo_mae"],
            "avg_r2": avg_r2_sf,
            "fit_time_seconds": t_fit_sf
        })

    # Task T2: After 3 Moves Classification
    for name, model in cls_models.items():
        test_classifier("T2_after3", name, model, after3_base, after3_base + sf3_cols)

    # Task T3: After 10 Moves Classification
    for name, model in cls_models.items():
        test_classifier("T3_after10", name, model, after10_base, after10_base + sf10_cols)

    # Task T4: Elo Regression
    for name, model in reg_models.items():
        test_regressor("T4_elo", name, model, elo_base, elo_base + sf10_cols)

    # 5. Process and Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "ab_testing_results.csv", index=False)
    
    # Create final metrics JSON object
    metrics_summary = {
        "target_games": target_games,
        "runtime_seconds": time.time() - start_time,
        "results": results
    }
    with open(output_dir / "ab_testing_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # Generate Markdown Report
    generate_report(results_df, target_games, output_dir)
    print(f"\nCompleted A/B testing on {target_games} games. Output saved to {output_dir}")


def generate_report(df: pd.DataFrame, target_games: int, output_dir: Path):
    """Generate Markdown report for A/B Testing results."""
    report_lines = [
        f"# Báo Cáo A/B Testing: Stockfish vs No-Stockfish ({target_games:,} Games)",
        "",
        "Báo cáo này trình bày kết quả đánh giá A/B Testing giữa việc **Sử dụng Stockfish** (Mode B) và **Không sử dụng Stockfish** (Mode A) trên các mô hình scikit-learn và Boosting.",
        "",
        "## 1. Kết Quả Chi Tiết Từng Task",
        ""
    ]

    # Task T2 Classification Table
    report_lines.extend([
        "### 1.1 Task T2: Dự Đoán Kết Quả Thắng/Thua Sau 3 Nước Đi (After-3 classification)",
        "",
        "| Mô hình | Mode A (No-SF) AUC | Mode B (With-SF) AUC | Thay đổi (Δ AUC) | Mode A Loss | Mode B Loss |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ])
    t2_df = df[df["task"] == "T2_after3"]
    for model_name in t2_df["model_name"].unique():
        m_df = t2_df[t2_df["model_name"] == model_name]
        no_sf = m_df[m_df["mode"] == "No Stockfish (A)"].iloc[0]
        with_sf = m_df[m_df["mode"] == "With Stockfish (B)"].iloc[0]
        diff_auc = with_sf["auc"] - no_sf["auc"]
        report_lines.append(
            f"| {model_name} | {no_sf['auc']:.4f} | {with_sf['auc']:.4f} | **{diff_auc:+.4f}** | {no_sf['log_loss']:.4f} | {with_sf['log_loss']:.4f} |"
        )
    report_lines.append("")

    # Task T3 Classification Table
    report_lines.extend([
        "### 1.2 Task T3: Dự Đoán Kết Quả Thắng/Thua Sau 10 Nước Đi (After-10 classification)",
        "",
        "| Mô hình | Mode A (No-SF) AUC | Mode B (With-SF) AUC | Thay đổi (Δ AUC) | Mode A Loss | Mode B Loss |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ])
    t3_df = df[df["task"] == "T3_after10"]
    for model_name in t3_df["model_name"].unique():
        m_df = t3_df[t3_df["model_name"] == model_name]
        no_sf = m_df[m_df["mode"] == "No Stockfish (A)"].iloc[0]
        with_sf = m_df[m_df["mode"] == "With Stockfish (B)"].iloc[0]
        diff_auc = with_sf["auc"] - no_sf["auc"]
        report_lines.append(
            f"| {model_name} | {no_sf['auc']:.4f} | {with_sf['auc']:.4f} | **{diff_auc:+.4f}** | {no_sf['log_loss']:.4f} | {with_sf['log_loss']:.4f} |"
        )
    report_lines.append("")

    # Task T4 Regression Table
    report_lines.extend([
        "### 1.3 Task T4: Dự Đoán ELO Kỳ Thủ Sau 10 Nước Đi (Elo regression)",
        "",
        "| Mô hình | Mode A (No-SF) MAE | Mode B (With-SF) MAE | Thay đổi (Δ MAE) | Mode A $R^2$ | Mode B $R^2$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ])
    t4_df = df[df["task"] == "T4_elo"]
    for model_name in t4_df["model_name"].unique():
        m_df = t4_df[t4_df["model_name"] == model_name]
        no_sf = m_df[m_df["mode"] == "No Stockfish (A)"].iloc[0]
        with_sf = m_df[m_df["mode"] == "With Stockfish (B)"].iloc[0]
        diff_mae = with_sf["avg_mae"] - no_sf["avg_mae"]
        report_lines.append(
            f"| {model_name} | {no_sf['avg_mae']:.2f} | {with_sf['avg_mae']:.2f} | **{diff_mae:+.2f}** | {no_sf['avg_r2']:.4f} | {with_sf['avg_r2']:.4f} |"
        )
    report_lines.extend([
        "",
        "## 2. Kết Luận & Phân Tích A/B Testing",
        "",
        "- **Ảnh hưởng của Stockfish lên Classification (T2/T3)**: Đánh giá xem Stockfish mang lại cải thiện nhiều hơn ở giai đoạn Move 3 hay Move 10, và mô hình nào tận dụng tốt nhất thông tin ưu thế bàn cờ phi tuyến tính này.",
        "- **Ảnh hưởng của Stockfish lên Elo Regression (T4)**: Đánh giá xem Stockfish có thực sự giúp ích cho việc dự đoán Elo hay không. (Lịch sử các lần chạy trước cho thấy Stockfish hầu như không làm giảm MAE Elo vì thông tin Elo đã được ước lượng rất chính xác qua history features).",
        "- **So sánh tốc độ huấn luyện**: Đối chiếu thời gian huấn luyện giữa scikit-learn models và Boosting models."
    ])

    (output_dir / "ab_testing_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B Testing Chess Models with and without Stockfish")
    parser.add_argument("--target-games", type=int, default=10000, help="Number of games to use for experiment")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--input-cache-csv", type=str, default=None, help="Optional local cache CSV file path")
    args = parser.parse_args()

    run_ab_testing(
        target_games=args.target_games,
        output_dir=Path(args.output_dir),
        input_cache_csv=args.input_cache_csv
    )
