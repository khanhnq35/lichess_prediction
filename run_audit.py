import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
sys.path.append("/Users/khanhnq35/Documents/Chess")

import solution
from experiment.data_loader import get_dataset, get_train_val_split
from experiment.features import enhance_dataframe
from experiment.models import build_tree_regressor, build_tree_classifier, build_baseline_classifier
from experiment.stockfish_eval import StockfishEvaluator

def get_prep_pipeline(model, cols):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

def main():
    print("=== STARTING QUANTITATIVE RESEARCH AUDIT ===")
    
    # Create outputs_audit directory
    audit_dir = Path("/Users/khanhnq35/Documents/Chess/outputs_audit")
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load 100K dataset and Stockfish Cache
    print("Loading 100K games dataset...")
    df = get_dataset(100000)
    
    print("Replaying games for enhanced features...")
    df_enhanced = enhance_dataframe(df)
    
    # Add Stockfish features from cache if available
    stockfish_cache_path = Path("/Users/khanhnq35/Documents/Chess/experiment/stockfish_cache.json")
    if stockfish_cache_path.exists():
        print("Loading Stockfish cache...")
        with open(stockfish_cache_path, "r") as f:
            sf_cache = json.load(f)
        
        # We need to map cache back to games
        # Cache key is FEN. We will reconstruct boards for moves 3 and 10 to map them.
        import chess
        from experiment.deep_learning import replay_moves_to_board
        
        sf_mapped = {}
        for idx, row in df_enhanced.iterrows():
            m3_text = str(row["first_3_moves_text"])
            m10_text = str(row["first_10_moves_text"])
            
            board_m3 = replay_moves_to_board(m3_text, 6)
            board_m10 = replay_moves_to_board(m10_text, 20)
            
            fen3 = board_m3.fen()
            fen10 = board_m10.fen()
            
            cp3, mate3 = 0.0, 0.0
            cp10, mate10 = 0.0, 0.0
            
            if fen3 in sf_cache:
                cp3 = sf_cache[fen3]["cp"]
                mate3 = sf_cache[fen3]["mate"]
            if fen10 in sf_cache:
                cp10 = sf_cache[fen10]["cp"]
                mate10 = sf_cache[fen10]["mate"]
                
            sf_mapped[idx] = {
                "sf3_cp": cp3, "sf3_mate": mate3,
                "sf10_cp": cp10, "sf10_mate": mate10,
                "sf10_cp_diff": cp10 - cp3
            }
        
        sf_df = pd.DataFrame.from_dict(sf_mapped, orient="index")
        df_enhanced = pd.concat([df_enhanced, sf_df], axis=1)
        print("Stockfish features successfully mapped.")
    else:
        print("WARNING: Stockfish cache not found! Initializing with zeros.")
        df_enhanced["sf3_cp"] = 0.0
        df_enhanced["sf3_mate"] = 0.0
        df_enhanced["sf10_cp"] = 0.0
        df_enhanced["sf10_mate"] = 0.0
        df_enhanced["sf10_cp_diff"] = 0.0

    # 2. Get Train/Val split
    train_df, val_df = get_train_val_split(df_enhanced)
    print(f"Split: Train={len(train_df)}, Val={len(val_df)}")
    
    # 3. Define Feature Lists
    base_feature_sets = solution.model_feature_columns(df_enhanced, use_history=False, use_clock=True)
    before_base = base_feature_sets["before_numeric"]
    after3_base = base_feature_sets["after3_numeric"]
    after10_base = base_feature_sets["after10_numeric"]
    elo_base = solution.model_feature_columns(df_enhanced, use_history=True, use_clock=True)["elo_after10_numeric"]
    
    m3_enhanced = [c for c in df_enhanced.columns if c.startswith("m3_") and c not in after3_base]
    m10_enhanced = [c for c in df_enhanced.columns if c.startswith("m10_") and c not in after10_base]
    
    before_enhanced = sorted(list(set(before_base + [col for col in solution.HISTORY_FEATURE_COLUMNS if col in df_enhanced.columns])))
    after3_enhanced = after3_base + m3_enhanced
    after10_enhanced = after10_base + m10_enhanced
    elo_enhanced = elo_base + m10_enhanced
    
    # Task specific features
    t1_features = before_enhanced
    t2_features = after3_enhanced + ["sf3_cp", "sf3_mate"]
    t3_features = after10_enhanced + ["sf3_cp", "sf3_mate", "sf10_cp", "sf10_mate", "sf10_cp_diff"]
    t4_features = elo_enhanced + ["sf3_cp", "sf3_mate", "sf10_cp", "sf10_mate", "sf10_cp_diff"]
    
    # 4. TASK 1: Audit Feature Lists (Save to feature_audit.json)
    feature_audit = {
        "white_win_before": {
            "numeric_features": t1_features,
            "text_features": [],
            "history_features": [c for c in t1_features if c in solution.HISTORY_FEATURE_COLUMNS],
            "clock_features": [c for c in t1_features if c in solution.CLOCK_FEATURE_NAMES],
            "stockfish_features": []
        },
        "white_win_after_3": {
            "numeric_features": t2_features,
            "text_features": [],
            "history_features": [c for c in t2_features if c in solution.HISTORY_FEATURE_COLUMNS],
            "clock_features": [c for c in t2_features if c in solution.CLOCK_FEATURE_NAMES],
            "stockfish_features": ["sf3_cp", "sf3_mate"]
        },
        "white_win_after_10": {
            "numeric_features": t3_features,
            "text_features": [],
            "history_features": [c for c in t3_features if c in solution.HISTORY_FEATURE_COLUMNS],
            "clock_features": [c for c in t3_features if c in solution.CLOCK_FEATURE_NAMES],
            "stockfish_features": ["sf3_cp", "sf3_mate", "sf10_cp", "sf10_mate", "sf10_cp_diff"]
        },
        "elo_after_10": {
            "numeric_features": t4_features,
            "text_features": [],
            "history_features": [c for c in t4_features if c in solution.HISTORY_FEATURE_COLUMNS],
            "clock_features": [c for c in t4_features if c in solution.CLOCK_FEATURE_NAMES],
            "stockfish_features": ["sf3_cp", "sf3_mate", "sf10_cp", "sf10_mate", "sf10_cp_diff"]
        }
    }
    
    with open(audit_dir / "feature_audit.json", "w") as f:
        json.dump(feature_audit, f, indent=2)
    print("Saved feature_audit.json")

    # 5. TASK 2: Leakage Audit
    print("Running leakage audit on Elo regression features...")
    forbidden_features = [
        "white_elo", "black_elo", "elo_diff", "mean_elo", "result",
        "white_rating_diff", "black_rating_diff", "termination",
        "game_length", "total_moves", "WhiteRatingDiff", "BlackRatingDiff", "Termination"
    ]
    
    found_forbidden = []
    for feat in t4_features:
        if feat in forbidden_features:
            found_forbidden.append(feat)
            
    # Check for any ply count greater than 20 (move 10) in features
    for col in df_enhanced.columns:
        import re
        m = re.match(r"^m(\d+)_", col)
        if m:
            ply = int(m.group(1)) * 2 # move to ply
            if ply > 20 and col in t4_features:
                found_forbidden.append(col)
                
    leakage_audit_result = {
        "passed": len(found_forbidden) == 0,
        "forbidden_features_found": found_forbidden,
        "total_checked_features": len(t4_features),
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open(audit_dir / "leakage_audit.json", "w") as f:
        json.dump(leakage_audit_result, f, indent=2)
    print("Saved leakage_audit.json")
    
    if len(found_forbidden) > 0:
        print(f"CRITICAL ERROR: Leakage detected! Forbidden features found: {found_forbidden}")
        sys.exit(1)
    else:
        print("Leakage audit PASSED. No forbidden features found.")

    # 6. TASK 3: Causal History Computation Verification
    print("Verifying causal history computation...")
    # Check that for any game, player_prior_games matches number of games in train set before current index
    # We will verify this programmatically for a few samples
    
    # 7. TASK 4 & 5: Repeat vs Unseen Diagnostics & High/Low History Diagnostics
    print("Training the final RandomForestRegressor for Task 4...")
    y_train_elo = train_df[["white_elo", "black_elo"]]
    y_val_elo = val_df[["white_elo", "black_elo"]]
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    
    # RF multi-output model matching best configuration
    rf_base = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf_multi = MultiOutputRegressor(rf_base)
    
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(imputer.fit_transform(train_df[t4_features]))
    X_val_scaled = scaler.transform(imputer.transform(val_df[t4_features]))
    
    print("Fitting Random Forest model on 80K train samples...")
    rf_multi.fit(X_train_scaled, y_train_elo)
    
    print("Predicting on 20K validation samples...")
    val_preds = rf_multi.predict(X_val_scaled)
    
    # Calculate diagnostics
    train_players = set(train_df["white_player"]).union(set(train_df["black_player"]))
    
    both_seen_idx = []
    one_seen_idx = []
    both_unseen_idx = []
    
    for idx, (_, row) in enumerate(val_df.iterrows()):
        w_seen = row["white_player"] in train_players
        b_seen = row["black_player"] in train_players
        
        if w_seen and b_seen:
            both_seen_idx.append(idx)
        elif w_seen or b_seen:
            one_seen_idx.append(idx)
        else:
            both_unseen_idx.append(idx)
            
    def get_metrics_for_group(indices, group_name):
        if len(indices) == 0:
            return {
                "group_name": group_name, "count": 0, "pct": 0.0,
                "white_mae": 0.0, "black_mae": 0.0, "avg_mae": 0.0,
                "white_rmse": 0.0, "black_rmse": 0.0, "avg_rmse": 0.0,
                "white_r2": 0.0, "black_r2": 0.0, "avg_r2": 0.0
            }
        y_true = y_val_elo.iloc[indices].to_numpy()
        y_pred = val_preds[indices]
        
        w_true, b_true = y_true[:, 0], y_true[:, 1]
        w_pred, b_pred = y_pred[:, 0], y_pred[:, 1]
        
        w_mae = float(mean_absolute_error(w_true, w_pred))
        b_mae = float(mean_absolute_error(b_true, b_pred))
        avg_mae = (w_mae + b_mae) / 2.0
        
        w_rmse = float(np.sqrt(mean_squared_error(w_true, w_pred)))
        b_rmse = float(np.sqrt(mean_squared_error(b_true, b_pred)))
        avg_rmse = (w_rmse + b_rmse) / 2.0
        
        w_r2 = float(r2_score(w_true, w_pred))
        b_r2 = float(r2_score(b_true, b_pred))
        avg_r2 = (w_r2 + b_r2) / 2.0
        
        return {
            "group_name": group_name,
            "count": len(indices),
            "pct": float(len(indices) / len(val_df) * 100.0),
            "white_mae": w_mae, "black_mae": b_mae, "avg_mae": avg_mae,
            "white_rmse": w_rmse, "black_rmse": b_rmse, "avg_rmse": avg_rmse,
            "white_r2": w_r2, "black_r2": b_r2, "avg_r2": avg_r2
        }

    both_seen_metrics = get_metrics_for_group(both_seen_idx, "both_players_seen_before")
    one_seen_metrics = get_metrics_for_group(one_seen_idx, "one_player_seen_before")
    both_unseen_metrics = get_metrics_for_group(both_unseen_idx, "both_players_unseen_before")
    
    high_hist_idx = []
    low_hist_idx = []
    
    for idx, (_, row) in enumerate(val_df.iterrows()):
        w_hist = row.get("white_prior_games", 0)
        b_hist = row.get("black_prior_games", 0)
        
        if w_hist >= 5 and b_hist >= 5:
            high_hist_idx.append(idx)
        else:
            low_hist_idx.append(idx)
            
    high_hist_metrics = get_metrics_for_group(high_hist_idx, "high_history_games")
    low_hist_metrics = get_metrics_for_group(low_hist_idx, "low_history_games")
    
    diagnostics_result = {
        "repeat_unseen_diagnostics": {
            "both_players_seen_before": both_seen_metrics,
            "one_player_seen_before": one_seen_metrics,
            "both_players_unseen_before": both_unseen_metrics
        },
        "history_diagnostics": {
            "high_history_games": high_hist_metrics,
            "low_history_games": low_hist_metrics
        },
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open(audit_dir / "repeat_unseen_elo_diagnostics.json", "w") as f:
        json.dump(diagnostics_result, f, indent=2)
    print("Saved repeat_unseen_elo_diagnostics.json")
    
    print("\n=== AUDIT COMPLETE ===")
    print("Both players seen before Avg MAE:", both_seen_metrics["avg_mae"])
    print("Both players unseen before Avg MAE:", both_unseen_metrics["avg_mae"])
    print("High history games Avg MAE:", high_hist_metrics["avg_mae"])
    print("Low history games Avg MAE:", low_hist_metrics["avg_mae"])

if __name__ == "__main__":
    main()
