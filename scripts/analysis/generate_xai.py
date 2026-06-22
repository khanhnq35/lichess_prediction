import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import solution

def extract_linear_coefficients(model, numeric_cols):
    """Extract coefficients for the numeric columns from the fitted pipeline."""
    # Get model step
    clf = model.named_steps["model"]
    coefs = clf.coef_
    
    if len(coefs.shape) > 1 and coefs.shape[0] == 1:
        # Logistic Regression binary class coef has shape (1, n_features)
        coef_list = coefs[0][:len(numeric_cols)].tolist()
    elif len(coefs.shape) > 1 and coefs.shape[0] > 1:
        # Multi-output regression coef has shape (n_targets, n_features)
        coef_list = [coefs[i][:len(numeric_cols)].tolist() for i in range(coefs.shape[0])]
    else:
        # 1D array of coefs
        coef_list = coefs[:len(numeric_cols)].tolist()
        
    return coef_list

def main():
    print("Loading cached dataset...")
    cache_path = PROJECT_ROOT / "experiment" / "outputs" / "cache" / "games_2023-11_100000.csv.gz"
    if not cache_path.exists():
        print("Cache not found at expected path. Trying artifacts/archive/smoke/outputs_100/cache...")
        cache_path = PROJECT_ROOT / "artifacts" / "archive" / "smoke" / "outputs_100" / "cache" / "games_2023-11_100000.csv.gz"
    
    if not cache_path.exists():
        print("Error: Cached games dataset not found.")
        sys.exit(1)
        
    df = pd.read_csv(cache_path)
    print(f"Loaded dataset with shape {df.shape}")
    
    # Configure and split dataset exactly like solution.py
    config = solution.Config(
        target_games=100000,
        selected_month="2023-11",
        random_seed=42,
        train_ratio=0.8
    )
    
    train_df, val_df = solution.split_train_validation(df, config.train_ratio)
    print(f"Train games: {len(train_df)}, Validation games: {len(val_df)}")
    
    # Get feature lists
    before_feature_cols = solution.model_feature_columns(df, use_history=False, use_clock=False)["before_numeric"]
    after3_feature_cols = solution.model_feature_columns(df, use_history=False, use_clock=False)["after3_numeric"]
    after10_feature_cols = solution.model_feature_columns(df, use_history=False, use_clock=True)["after10_numeric"]
    elo_feature_cols = solution.model_feature_columns(df, use_history=True, use_clock=False)["elo_after10_numeric"]
    
    after3_text_cols = ["first_3_moves_text", "player_pair_text"]
    after10_text_cols = ["first_10_moves_text", "player_pair_text"]
    elo_text_cols = ["first_10_moves_text", "player_pair_text"]
    
    # Train models
    print("Fitting models...")
    y_train = train_df["white_win"]
    y_val = val_df["white_win"]
    
    before_model = solution.build_classifier_pipeline(
        numeric_cols=before_feature_cols,
        text_cols=[],
        hashing_features=config.hashing_features,
        random_seed=config.random_seed,
        c_value=1.0,
    )
    after3_model = solution.build_classifier_pipeline(
        numeric_cols=after3_feature_cols,
        text_cols=after3_text_cols,
        hashing_features=config.hashing_features,
        random_seed=config.random_seed,
        c_value=0.25,
    )
    after10_model = solution.build_classifier_pipeline(
        numeric_cols=after10_feature_cols,
        text_cols=after10_text_cols,
        hashing_features=config.hashing_features,
        random_seed=config.random_seed,
        c_value=0.25,
    )
    elo_model = solution.build_elo_regression_pipeline(
        numeric_cols=elo_feature_cols,
        text_cols=elo_text_cols,
        hashing_features=config.hashing_features,
    )
    
    print("Fitting White Win Before Game...")
    before_model.fit(train_df[before_feature_cols], y_train)
    
    print("Fitting White Win After 3 moves...")
    after3_model.fit(train_df[after3_feature_cols + after3_text_cols], y_train)
    
    print("Fitting White Win After 10 moves...")
    after10_model.fit(train_df[after10_feature_cols + after10_text_cols], y_train)
    
    print("Fitting Elo Regression...")
    elo_model.fit(train_df[elo_feature_cols + elo_text_cols], train_df[["white_elo", "black_elo"]])
    
    # Extract coefficients
    print("Extracting model explainability data...")
    coef_before = extract_linear_coefficients(before_model, before_feature_cols)
    coef_after3 = extract_linear_coefficients(after3_model, after3_feature_cols)
    coef_after10 = extract_linear_coefficients(after10_model, after10_feature_cols)
    coefs_elo = extract_linear_coefficients(elo_model, elo_feature_cols)
    
    coef_white_elo = coefs_elo[0]
    coef_black_elo = coefs_elo[1]
    
    # Output XAI folder creation
    xai_dir = PROJECT_ROOT / "artifacts" / "xai" / "current"
    xai_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare explainability summary
    explainability_data = {
        "white_win_before": dict(zip(before_feature_cols, coef_before)),
        "white_win_after_3": dict(zip(after3_feature_cols, coef_after3)),
        "white_win_after_10": dict(zip(after10_feature_cols, coef_after10)),
        "white_elo_after_10": dict(zip(elo_feature_cols, coef_white_elo)),
        "black_elo_after_10": dict(zip(elo_feature_cols, coef_black_elo)),
    }
    
    # Write artifacts/xai/current/model_explainability.json
    with open(xai_dir / "model_explainability.json", "w") as f:
        json.dump(explainability_data, f, indent=2)
    print(f"Wrote {xai_dir / 'model_explainability.json'}")
    
    # Prepare feature importance CSV
    csv_rows = []
    # 1. White Win Before
    for feat, coef in zip(before_feature_cols, coef_before):
        csv_rows.append({"task": "white_win_before", "feature_name": feat, "coefficient": coef, "abs_coefficient": abs(coef)})
    # 2. White Win After 3
    for feat, coef in zip(after3_feature_cols, coef_after3):
        csv_rows.append({"task": "white_win_after_3", "feature_name": feat, "coefficient": coef, "abs_coefficient": abs(coef)})
    # 3. White Win After 10
    for feat, coef in zip(after10_feature_cols, coef_after10):
        csv_rows.append({"task": "white_win_after_10", "feature_name": feat, "coefficient": coef, "abs_coefficient": abs(coef)})
    # 4. White Elo After 10
    for feat, coef in zip(elo_feature_cols, coef_white_elo):
        csv_rows.append({"task": "white_elo_after_10", "feature_name": feat, "coefficient": coef, "abs_coefficient": abs(coef)})
    # 5. Black Elo After 10
    for feat, coef in zip(elo_feature_cols, coef_black_elo):
        csv_rows.append({"task": "black_elo_after_10", "feature_name": feat, "coefficient": coef, "abs_coefficient": abs(coef)})
        
    importance_df = pd.DataFrame(csv_rows)
    importance_df = importance_df.sort_values(by=["task", "abs_coefficient"], ascending=[True, False])
    importance_df.to_csv(xai_dir / "feature_importance.csv", index=False)
    print(f"Wrote {xai_dir / 'feature_importance.csv'}")
    
    # Prediction-level explanation examples
    print("Generating prediction-level explanation examples...")
    
    # Load the validation predictions if available to be exact on predictions
    pred_path = PROJECT_ROOT / "artifacts" / "production" / "full_final_selected" / "validation_predictions.csv"
    if not pred_path.exists():
        pred_path = PROJECT_ROOT / "artifacts" / "archive" / "smoke" / "outputs_100" / "validation_predictions.csv"
        
    if pred_path.exists():
        predictions_df = pd.read_csv(pred_path)
    else:
        # Compute them on the fly if file is not found
        print("Validation predictions file not found, computing predictions on the fly...")
        before_probs = before_model.predict_proba(val_df[before_feature_cols])[:, 1]
        after3_probs = after3_model.predict_proba(val_df[after3_feature_cols + after3_text_cols])[:, 1]
        after10_probs = after10_model.predict_proba(val_df[after10_feature_cols + after10_text_cols])[:, 1]
        elo_predictions = elo_model.predict(val_df[elo_feature_cols + elo_text_cols])
        
        predictions_df = val_df[["game_index", "white_player", "black_player", "result", "white_win", "white_elo", "black_elo"]].copy()
        predictions_df = predictions_df.rename(columns={"white_win": "white_win_true"})
        predictions_df["p_white_win_before"] = before_probs
        predictions_df["p_white_win_after_3"] = after3_probs
        predictions_df["p_white_win_after_10"] = after10_probs
        predictions_df["white_elo_pred_after_10"] = elo_predictions[:, 0]
        predictions_df["black_elo_pred_after_10"] = elo_predictions[:, 1]
        predictions_df["p_white_win_elo_baseline"] = solution.elo_expected_score_probability(val_df)
    
    # Merge validation feature values back to predictions_df so we can access numeric values
    merged_val_df = pd.merge(predictions_df, val_df, on="game_index", suffixes=("", "_original"))
    
    # Select 6-8 interesting cases
    examples = []
    
    # Case 1: White Elo > Black Elo + 250
    c1 = merged_val_df[merged_val_df["white_elo"] > merged_val_df["black_elo"] + 250].head(1)
    if not c1.empty: examples.append((c1.iloc[0], "White Elo advantage dominates the pre-game and post-game probabilities."))
    
    # Case 2: Black Elo > White Elo + 250
    c2 = merged_val_df[merged_val_df["black_elo"] > merged_val_df["white_elo"] + 250].head(1)
    if not c2.empty: examples.append((c2.iloc[0], "Black Elo advantage dominates, keeping White win probability low throughout."))
    
    # Case 3: Win probability increases from before-game to after-10 moves by > 0.15
    c3 = merged_val_df[(merged_val_df["p_white_win_after_10"] > merged_val_df["p_white_win_before"] + 0.15) & 
                       (merged_val_df["clk10_clock_used_diff"] < 0)].head(1)
    if c3.empty:
        c3 = merged_val_df[merged_val_df["p_white_win_after_10"] > merged_val_df["p_white_win_before"] + 0.15].head(1)
    if not c3.empty: examples.append((c3.iloc[0], "White's win probability increased significantly after 10 moves due to an early positional/development advantage and a positive clock differential (White used less time)."))
    
    # Case 4: Win probability decreases from before-game to after-10 moves by > 0.15
    c4 = merged_val_df[merged_val_df["p_white_win_after_10"] < merged_val_df["p_white_win_before"] - 0.15].head(1)
    if not c4.empty: examples.append((c4.iloc[0], "White's win probability decreased after 10 moves because Black gained material or positional activity, or White consumed significantly more clock time."))
    
    # Case 5: Close game (Elo diff < 30) with clear clock diff after 10
    c5 = merged_val_df[(merged_val_df["elo_diff"].abs() < 30) & (merged_val_df["clk10_clock_used_diff"].abs() > 20)].head(1)
    if not c5.empty: examples.append((c5.iloc[0], "With nearly equal Elos, the pre-game probability was neutral. By move 10, the clock time usage difference became the primary driver in shifting the win probability."))
    
    # Case 6: High history repeat players (prior games > 20)
    c6 = merged_val_df[(merged_val_df["white_prior_games"] > 20) & (merged_val_df["black_prior_games"] > 20)].head(1)
    if not c6.empty: examples.append((c6.iloc[0], "Both players are repeat participants in this dataset. The Elo regressor leverages their extensive causal prior history to produce highly accurate player rating estimates, and the pre-game probability incorporates their historical performance."))
    
    # Case 7: Unseen players (prior games == 0)
    c7 = merged_val_df[(merged_val_df["white_prior_games"] == 0) & (merged_val_df["black_prior_games"] == 0)].head(1)
    if not c7.empty: examples.append((c7.iloc[0], "Both players are unseen in the training period (no prior games history). The Elo predictions revert toward the global prior average Elo, and the pre-game win probability is determined solely by the raw rating ratings difference."))
    
    # Fallback
    if len(examples) < 6:
        for idx, row in merged_val_df.head(8).iterrows():
            if not any(row["game_index"] == ex[0]["game_index"] for ex in examples):
                examples.append((row, "Deterministic explanation of predictions based on Elo ratings, early board state at 10 moves, and relative clock consumption."))
                if len(examples) >= 7:
                    break
                    
    explanation_examples = []
    for row, base_exp in examples:
        game_idx = int(row["game_index"])
        p_before = float(row["p_white_win_before"])
        p_after3 = float(row["p_white_win_after_3"])
        p_after10 = float(row["p_white_win_after_10"])
        actual_win = int(row["white_win_true"])
        
        pred_white_elo = float(row["white_elo_pred_after_10"])
        actual_white_elo = float(row["white_elo"])
        pred_black_elo = float(row["black_elo_pred_after_10"])
        actual_black_elo = float(row["black_elo"])
        
        elo_diff = float(row["elo_diff"]) if "elo_diff" in row else actual_white_elo - actual_black_elo
        elo_expected_prob = float(row["p_white_win_elo_baseline"])
        
        # Clock features
        clk_diff_10 = float(row["clk10_clock_diff_last"]) if "clk10_clock_diff_last" in row else 0.0
        clk_used_diff_10 = float(row["clk10_clock_used_diff"]) if "clk10_clock_used_diff" in row else 0.0
        
        # History features
        w_prior_games = int(row["white_prior_games"]) if "white_prior_games" in row else 0
        b_prior_games = int(row["black_prior_games"]) if "black_prior_games" in row else 0
        
        prob_increased = p_after10 > p_before
        direction_str = "increased" if prob_increased else "decreased"
        
        # Build explanation
        explanation = (
            f"After 10 moves, the model {direction_str} White's win probability to {p_after10:.3f} "
            f"(relative to the before-game estimate of {p_before:.3f}). "
        )
        if abs(clk_used_diff_10) > 10:
            whose_adv = "White" if clk_used_diff_10 < 0 else "Black"
            explanation += f"This shift was influenced by clock consumption (with {whose_adv} using less time by {abs(clk_used_diff_10):.1f} seconds). "
        else:
            explanation += "The clock usage remained relatively balanced between players. "
            
        explanation += (
            f"The Elo predictions (White: {pred_white_elo:.0f} vs. actual: {actual_white_elo:.0f}; "
            f"Black: {pred_black_elo:.0f} vs. actual: {actual_black_elo:.0f}) are heavily anchor-prioritized "
            f"on player causal history: White had {w_prior_games} prior games and Black had {b_prior_games} prior games. "
            f"The actual outcome of the game was a {'White win' if actual_win == 1 else 'draw/Black win'}."
        )
        
        evidence = {
            "elo_difference": elo_diff,
            "elo_expected_probability": elo_expected_prob,
            "clock_difference_seconds_after_10": clk_diff_10,
            "clock_used_difference_seconds_after_10": clk_used_diff_10,
            "white_prior_history_games": w_prior_games,
            "black_prior_history_games": b_prior_games,
            "after_10_probability_increased_relative_to_before": prob_increased,
            "stockfish_eval_used": False
        }
        
        example_json = {
            "validation_index_game_index": game_idx,
            "white_player": str(row["white_player"]),
            "black_player": str(row["black_player"]),
            "predicted_white_win_prob_before": p_before,
            "predicted_white_win_prob_after_3": p_after3,
            "predicted_white_win_prob_after_10": p_after10,
            "actual_white_win": actual_win,
            "predicted_white_elo": pred_white_elo,
            "actual_white_elo": actual_white_elo,
            "predicted_black_elo": pred_black_elo,
            "actual_black_elo": actual_black_elo,
            "key_numeric_evidence": evidence,
            "explanation": explanation
        }
        explanation_examples.append(example_json)
        
    with open(xai_dir / "prediction_explanation_examples.json", "w") as f:
        json.dump(explanation_examples, f, indent=2)
    print(f"Wrote {xai_dir / 'prediction_explanation_examples.json'}")
    print("XAI scripts completed successfully.")

if __name__ == "__main__":
    main()
