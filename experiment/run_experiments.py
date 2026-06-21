"""Main experiment runner script.

Executes all 7 phases of chess prediction experiments:
1. Baseline Recap
2. Enhanced Features
3. Tree-based Models & Tuning
4. Stockfish Evaluation
5. Deep Learning (MLP & CNN)
6. Ensemble & Stacking
7. Final 100K validation of selected best models

Saves all results to experiment_results.csv and best_models.json.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

# Add parent directory to path to import solution.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

import solution
from experiment.config import (
    OUTPUT_DIR, DEFAULT_RANDOM_SEED, DEFAULT_MONTH, 
    STOCKFISH_CACHE_FILE, STOCKFISH_DEPTH
)
from experiment.data_loader import get_dataset, get_train_val_split
from experiment.features import enhance_dataframe
from lightgbm import LGBMClassifier

from experiment.stockfish_eval import StockfishEvaluator
from experiment.models import (
    build_baseline_classifier, build_baseline_regressor,
    build_tree_classifier, build_tree_regressor,
    build_ensemble_classifier, build_ensemble_regressor,
    build_stacking_classifier, build_stacking_regressor,
    get_numeric_preprocessor
)
from experiment.deep_learning import (
    PyTorchMLPClassifier, PyTorchMLPRegressor,
    PyTorchCNNClassifier, PyTorchCNNRegressor,
    replay_moves_to_board
)

def evaluate_cls(y_true, y_prob):
    """Calculate classification metrics."""
    # Clip prob to avoid log loss infinity
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    auc = float(roc_auc_score(y_true, y_prob))
    loss = float(log_loss(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    # Accuracy at 0.5 threshold
    acc = float(np.mean(y_true == (y_prob >= 0.5)))
    return {"auc": auc, "log_loss": loss, "brier": brier, "accuracy": acc}

def evaluate_reg(y_true, y_pred):
    """Calculate regression metrics for both players."""
    w_true, b_true = y_true.iloc[:, 0].to_numpy(), y_true.iloc[:, 1].to_numpy()
    w_pred, b_pred = y_pred[:, 0], y_pred[:, 1]
    
    w_mae = float(mean_absolute_error(w_true, w_pred))
    b_mae = float(mean_absolute_error(b_true, b_pred))
    w_rmse = float(np.sqrt(mean_squared_error(w_true, w_pred)))
    b_rmse = float(np.sqrt(mean_squared_error(b_true, b_pred)))
    w_r2 = float(r2_score(w_true, w_pred))
    b_r2 = float(r2_score(b_true, b_pred))
    
    avg_mae = (w_mae + b_mae) / 2.0
    avg_rmse = (w_rmse + b_rmse) / 2.0
    avg_r2 = (w_r2 + b_r2) / 2.0
    
    return {
        "white_mae": w_mae, "black_mae": b_mae, "avg_mae": avg_mae,
        "white_rmse": w_rmse, "black_rmse": b_rmse, "avg_rmse": avg_rmse,
        "white_r2": w_r2, "black_r2": b_r2, "avg_r2": avg_r2
    }

def print_result_row(exp_id, task, model_name, metrics):
    """Helper to format-print a result row."""
    if "auc" in metrics:
        print(f"| {exp_id:<5} | {task:<4} | {model_name:<20} | AUC: {metrics['auc']:.4f} | Loss: {metrics['log_loss']:.4f} | Acc: {metrics['accuracy']:.4f} |")
    else:
        print(f"| {exp_id:<5} | {task:<4} | {model_name:<20} | MAE: {metrics['avg_mae']:.1f} | RMSE: {metrics['avg_rmse']:.1f} | R2: {metrics['avg_r2']:.4f} |")

def main():
    parser = argparse.ArgumentParser(description="Run Chess Prediction Experiments")
    parser.add_argument("--target-games", type=int, default=10000, help="Number of games to use for development")
    parser.add_argument("--skip-stockfish", action="store_true", help="Skip Stockfish features")
    parser.add_argument("--skip-dl", action="store_true", help="Skip PyTorch deep learning models")
    parser.add_argument("--final-only", action="store_true", help="Run only the final validation on 100K")
    args = parser.parse_args()
    
    print(f"=== Chess Prediction Experiments Started (Games: {args.target_games}) ===")
    
    # 1. Load data
    df = get_dataset(args.target_games)
    
    # 2. Extract enhanced features
    df_enhanced = enhance_dataframe(df)
    
    # 3. Train/Val split
    train_df, val_df = get_train_val_split(df_enhanced)
    
    print(f"Dataset Split: Train={len(train_df)} games, Val={len(val_df)} games")
    
    # Identify feature columns
    # We will get baseline features first (reusing solution.py logic)
    base_feature_sets = solution.model_feature_columns(df_enhanced, use_history=False, use_clock=True)
    before_base = base_feature_sets["before_numeric"]
    after3_base = base_feature_sets["after3_numeric"]
    after10_base = base_feature_sets["after10_numeric"]
    
    # For Elo prediction (T4), use history (but exclude forbidden Elo columns)
    elo_base = solution.model_feature_columns(df_enhanced, use_history=True, use_clock=True)["elo_after10_numeric"]
    
    # Enhanced feature lists
    m3_enhanced = [c for c in df_enhanced.columns if c.startswith("m3_") and c not in after3_base]
    m10_enhanced = [c for c in df_enhanced.columns if c.startswith("m10_") and c not in after10_base]
    
    # Define text features
    after3_text = ["first_3_moves_text", "player_pair_text"]
    after10_text = ["first_10_moves_text", "player_pair_text"]
    
    # Target variables
    y_train_cls = train_df["white_win"]
    y_val_cls = val_df["white_win"]
    y_train_elo = train_df[["white_elo", "black_elo"]]
    y_val_elo = val_df[["white_elo", "black_elo"]]
    
    # Dictionary to keep track of results
    results_list = []
    
    def log_result(exp_id, phase, task, model_name, metrics, features_used):
        res = {
            "exp_id": exp_id,
            "phase": phase,
            "task": task,
            "model_name": model_name,
            "features": features_used,
            "auc": metrics.get("auc"),
            "log_loss": metrics.get("log_loss"),
            "brier": metrics.get("brier"),
            "accuracy": metrics.get("accuracy"),
            "avg_mae": metrics.get("avg_mae"),
            "avg_rmse": metrics.get("avg_rmse"),
            "avg_r2": metrics.get("avg_r2"),
            "white_mae": metrics.get("white_mae"),
            "black_mae": metrics.get("black_mae"),
            "white_r2": metrics.get("white_r2"),
            "black_r2": metrics.get("black_r2")
        }
        results_list.append(res)
        print_result_row(exp_id, task, model_name, metrics)
        
    print("\n--- Phase 1: Baseline Recap ---")
    # T1: Win before
    m_t1 = build_baseline_classifier(before_base, c_value=1.0)
    m_t1.fit(train_df, y_train_cls)
    t1_prob = m_t1.predict_proba(val_df)[:, 1]
    log_result("B1", "P1_Baseline", "T1", "LogReg(C=1.0)", evaluate_cls(y_val_cls, t1_prob), "base_before")
    
    # T2: Win after 3
    m_t2 = build_baseline_classifier(after3_base, after3_text, c_value=0.25)
    m_t2.fit(train_df, y_train_cls)
    t2_prob = m_t2.predict_proba(val_df)[:, 1]
    log_result("B2", "P1_Baseline", "T2", "LogReg(C=0.25)+Hash", evaluate_cls(y_val_cls, t2_prob), "base_after3+text")
    
    # T3: Win after 10
    m_t3 = build_baseline_classifier(after10_base, after10_text, c_value=0.25)
    m_t3.fit(train_df, y_train_cls)
    t3_prob = m_t3.predict_proba(val_df)[:, 1]
    log_result("B3", "P1_Baseline", "T3", "LogReg(C=0.25)+Hash", evaluate_cls(y_val_cls, t3_prob), "base_after10+text")
    
    # T4: Elo after 10
    m_t4 = build_baseline_regressor(elo_base, after10_text, alpha=10.0)
    m_t4.fit(train_df, y_train_elo)
    t4_pred = m_t4.predict(val_df)
    log_result("B4", "P1_Baseline", "T4", "Ridge(α=10.0)+Hash", evaluate_reg(y_val_elo, t4_pred), "base_elo+text")
    
    print("\n--- Phase 2: Enhanced Feature Engineering ---")
    # F1: Win before with history features (baseline didn't use history!)
    before_enhanced = before_base + [col for col in solution.HISTORY_FEATURE_COLUMNS if col in df_enhanced.columns]
    before_enhanced = sorted(list(set(before_enhanced)))
    m_f1 = build_baseline_classifier(before_enhanced, c_value=1.0)
    m_f1.fit(train_df, y_train_cls)
    f1_prob = m_f1.predict_proba(val_df)[:, 1]
    log_result("F1", "P2_Enhanced", "T1", "LogReg(C=1.0)+Hist", evaluate_cls(y_val_cls, f1_prob), "before+history")
    
    # F2: Win after 3 with enhanced board features (no text)
    after3_enhanced = after3_base + m3_enhanced
    m_f2 = build_baseline_classifier(after3_enhanced, c_value=0.5)
    m_f2.fit(train_df, y_train_cls)
    f2_prob = m_f2.predict_proba(val_df)[:, 1]
    log_result("F2", "P2_Enhanced", "T2", "LogReg(C=0.5)", evaluate_cls(y_val_cls, f2_prob), "after3+enhanced")
    
    # F3: Win after 3 with enhanced board + text
    m_f3 = build_baseline_classifier(after3_enhanced, after3_text, c_value=0.25)
    m_f3.fit(train_df, y_train_cls)
    f3_prob = m_f3.predict_proba(val_df)[:, 1]
    log_result("F3", "P2_Enhanced", "T2", "LogReg(C=0.25)+Hash", evaluate_cls(y_val_cls, f3_prob), "after3+enhanced+text")
    
    # F4: Win after 10 with enhanced board features
    after10_enhanced = after10_base + m10_enhanced
    m_f4 = build_baseline_classifier(after10_enhanced, after10_text, c_value=0.25)
    m_f4.fit(train_df, y_train_cls)
    f4_prob = m_f4.predict_proba(val_df)[:, 1]
    log_result("F4", "P2_Enhanced", "T3", "LogReg(C=0.25)+Hash", evaluate_cls(y_val_cls, f4_prob), "after10+enhanced+text")
    
    # F5: Elo after 10 with enhanced board features
    elo_enhanced = elo_base + m10_enhanced
    m_f5 = build_baseline_regressor(elo_enhanced, after10_text, alpha=10.0)
    m_f5.fit(train_df, y_train_elo)
    f5_pred = m_f5.predict(val_df)
    log_result("F5", "P2_Enhanced", "T4", "Ridge(α=10.0)+Hash", evaluate_reg(y_val_elo, f5_pred), "elo+enhanced+text")
    
    print("\n--- Phase 3: Tree-Based Models ---")
    tree_models = ["lightgbm", "xgboost", "histgb", "randomforest", "gradientboosting"]
    
    # 3.1 Task 1: Win before
    best_t1_model = None
    best_t1_auc = 0.0
    for name in tree_models:
        exp_id = f"T1_{name[:3]}"
        clf = build_tree_classifier(name, before_enhanced, random_state=DEFAULT_RANDOM_SEED)
        clf.fit(train_df, y_train_cls)
        prob = clf.predict_proba(val_df)[:, 1]
        metrics = evaluate_cls(y_val_cls, prob)
        log_result(exp_id, "P3_Trees", "T1", name.capitalize(), metrics, "before+history")
        if metrics["auc"] > best_t1_auc:
            best_t1_auc = metrics["auc"]
            best_t1_model = name
            
    # 3.2 Task 2: Win after 3
    best_t2_model = None
    best_t2_auc = 0.0
    for name in tree_models:
        exp_id = f"T2_{name[:3]}"
        # Note: tree models work directly on numeric enhanced features, no HashingVectorizer
        clf = build_tree_classifier(name, after3_enhanced, random_state=DEFAULT_RANDOM_SEED)
        clf.fit(train_df, y_train_cls)
        prob = clf.predict_proba(val_df)[:, 1]
        metrics = evaluate_cls(y_val_cls, prob)
        log_result(exp_id, "P3_Trees", "T2", name.capitalize(), metrics, "after3+enhanced")
        if metrics["auc"] > best_t2_auc:
            best_t2_auc = metrics["auc"]
            best_t2_model = name
            
    # 3.3 Task 3: Win after 10
    best_t3_model = None
    best_t3_auc = 0.0
    for name in tree_models:
        exp_id = f"T3_{name[:3]}"
        clf = build_tree_classifier(name, after10_enhanced, random_state=DEFAULT_RANDOM_SEED)
        clf.fit(train_df, y_train_cls)
        prob = clf.predict_proba(val_df)[:, 1]
        metrics = evaluate_cls(y_val_cls, prob)
        log_result(exp_id, "P3_Trees", "T3", name.capitalize(), metrics, "after10+enhanced")
        if metrics["auc"] > best_t3_auc:
            best_t3_auc = metrics["auc"]
            best_t3_model = name
            
    # 3.4 Task 4: Elo after 10
    best_t4_model = None
    best_t4_mae = 9999.0
    for name in tree_models:
        exp_id = f"T4_{name[:3]}"
        reg = build_tree_regressor(name, elo_enhanced, random_state=DEFAULT_RANDOM_SEED)
        reg.fit(train_df, y_train_elo)
        pred = reg.predict(val_df)
        metrics = evaluate_reg(y_val_elo, pred)
        log_result(exp_id, "P3_Trees", "T4", name.capitalize(), metrics, "elo+enhanced")
        if metrics["avg_mae"] < best_t4_mae:
            best_t4_mae = metrics["avg_mae"]
            best_t4_model = name

    # Hyperparameter tuning for LightGBM on Task 3 (as SOTA candidate)
    print("\n--- Tuning top tree model (LightGBM) for Task 3 ---")
    lgb_pipeline = build_tree_classifier("lightgbm", after10_enhanced, random_state=DEFAULT_RANDOM_SEED)
    param_dist = {
        "classifier__n_estimators": [300, 500, 800],
        "classifier__max_depth": [4, 6, 8],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
        "classifier__subsample": [0.7, 0.8, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
    }
    
    # We must fit the preprocessor first to apply RandomizedSearchCV to the classifier step
    # Or tune on already preprocessed data to save time. Let's preprocess data manually first.
    preprocessor = get_numeric_preprocessor(after10_enhanced)
    X_train_prep = preprocessor.fit_transform(train_df)
    X_val_prep = preprocessor.transform(val_df)
    
    tune_clf = LGBMClassifier(random_state=DEFAULT_RANDOM_SEED)
    # Simplify search params for speed
    param_dist_clean = {k.replace("classifier__", ""): v for k, v in param_dist.items()}
    
    search = RandomizedSearchCV(
        tune_clf, param_distributions=param_dist_clean, 
        n_iter=8, cv=3, scoring="roc_auc", n_jobs=1, random_state=DEFAULT_RANDOM_SEED
    )
    search.fit(X_train_prep, y_train_cls)
    best_lgb_params = search.best_params_
    print(f"Best LightGBM params for T3: {best_lgb_params}")
    
    # Evaluate tuned LightGBM
    tuned_lgb = LGBMClassifier(**best_lgb_params, random_state=DEFAULT_RANDOM_SEED)
    tuned_lgb.fit(X_train_prep, y_train_cls)
    t3_tuned_prob = tuned_lgb.predict_proba(X_val_prep)[:, 1]
    log_result("T3_TUN", "P3_Trees", "T3", "LightGBM(Tuned)", evaluate_cls(y_val_cls, t3_tuned_prob), "after10+enhanced")

    # 4. Stockfish Evaluation
    sf_cache = {}
    if not args.skip_stockfish:
        print("\n--- Phase 4: Stockfish Evaluation ---")
        evaluator = StockfishEvaluator(STOCKFISH_CACHE_FILE, depth=STOCKFISH_DEPTH)
        
        # Evaluate move 3 and 10 positions
        # For efficiency, we only evaluate games in train/val sets
        # Let's extract board states for all games and run stockfish
        print("Extracting Stockfish evaluations (this may take a few minutes if cache is empty)...")
        
        sf3_cp_list = []
        sf3_mate_list = []
        sf10_cp_list = []
        sf10_mate_list = []
        
        # Combine train_df and val_df to evaluate all games
        full_df = pd.concat([train_df, val_df])
        
        # Check if cache is already populated
        evaluated_count = 0
        for idx, row in full_df.iterrows():
            # Reconstruct boards
            m10_text = str(row["first_10_moves_text"])
            m3_text = str(row["first_3_moves_text"])
            
            # Replay to get boards
            board_m3 = replay_moves_to_board(m3_text, 6)
            board_m10 = replay_moves_to_board(m10_text, 20)
            
            # Eval
            cp3, mate3 = evaluator.evaluate(board_m3)
            cp10, mate10 = evaluator.evaluate(board_m10)
            
            sf_cache[idx] = {
                "sf3_cp": cp3, "sf3_mate": mate3,
                "sf10_cp": cp10, "sf10_mate": mate10,
                "sf10_cp_diff": cp10 - cp3
            }
            evaluated_count += 1
            if evaluated_count % 1000 == 0:
                print(f"Evaluated {evaluated_count}/{len(full_df)} games...")
                # Intermediate save
                evaluator.save_cache()
                
        evaluator.close() # Save cache and close
        
        # Map features back
        sf_df = pd.DataFrame.from_dict(sf_cache, orient="index")
        
        train_df = pd.concat([train_df, sf_df.loc[train_df.index]], axis=1)
        val_df = pd.concat([val_df, sf_df.loc[val_df.index]], axis=1)
        
        sf_after3_cols = ["sf3_cp", "sf3_mate"]
        sf_after10_cols = ["sf3_cp", "sf3_mate", "sf10_cp", "sf10_mate", "sf10_cp_diff"]
        
        # T2 with Stockfish
        t2_sf_features = after3_enhanced + sf_after3_cols
        t2_sf_clf = build_tree_classifier(best_t2_model, t2_sf_features, random_state=DEFAULT_RANDOM_SEED)
        t2_sf_clf.fit(train_df, y_train_cls)
        t2_sf_prob = t2_sf_clf.predict_proba(val_df)[:, 1]
        log_result("S1", "P4_Stockfish", "T2", f"{best_t2_model.capitalize()}+SF", evaluate_cls(y_val_cls, t2_sf_prob), "after3+enhanced+SF")
        
        # T3 with Stockfish
        t3_sf_features = after10_enhanced + sf_after10_cols
        # Use tuned LightGBM params if LightGBM is best
        if best_t3_model == "lightgbm":
            t3_sf_clf = build_tree_classifier("lightgbm", t3_sf_features, random_state=DEFAULT_RANDOM_SEED, **best_lgb_params)
        else:
            t3_sf_clf = build_tree_classifier(best_t3_model, t3_sf_features, random_state=DEFAULT_RANDOM_SEED)
        t3_sf_clf.fit(train_df, y_train_cls)
        t3_sf_prob = t3_sf_clf.predict_proba(val_df)[:, 1]
        log_result("S2", "P4_Stockfish", "T3", f"{best_t3_model.capitalize()}+SF", evaluate_cls(y_val_cls, t3_sf_prob), "after10+enhanced+SF")
        
        # T4 with Stockfish
        t4_sf_features = elo_enhanced + sf_after10_cols
        t4_sf_reg = build_tree_regressor(best_t4_model, t4_sf_features, random_state=DEFAULT_RANDOM_SEED)
        t4_sf_reg.fit(train_df, y_train_elo)
        t4_sf_pred = t4_sf_reg.predict(val_df)
        log_result("S3", "P4_Stockfish", "T4", f"{best_t4_model.capitalize()}+SF", evaluate_reg(y_val_elo, t4_sf_pred), "elo+enhanced+SF")
    else:
        print("\n--- Phase 4: Stockfish Evaluation [SKIPPED] ---")
        
    # 5. Deep Learning Models
    if not args.skip_dl:
        print("\n--- Phase 5: Deep Learning ---")
        # D1: Win before MLP
        d1_mlp = PyTorchMLPClassifier(epochs=20, verbose=0)
        d1_mlp.fit(train_df[before_enhanced], y_train_cls)
        d1_prob = d1_mlp.predict_proba(val_df[before_enhanced])[:, 1]
        log_result("D1", "P5_DL", "T1", "MLP Classifier", evaluate_cls(y_val_cls, d1_prob), "before+history")
        
        # D2: Win after 3 MLP
        d2_mlp = PyTorchMLPClassifier(epochs=20, verbose=0)
        d2_mlp.fit(train_df[after3_enhanced], y_train_cls)
        d2_prob = d2_mlp.predict_proba(val_df[after3_enhanced])[:, 1]
        log_result("D2", "P5_DL", "T2", "MLP Classifier", evaluate_cls(y_val_cls, d2_prob), "after3+enhanced")
        
        # D3: Win after 10 MLP
        d3_mlp = PyTorchMLPClassifier(epochs=20, verbose=0)
        d3_mlp.fit(train_df[after10_enhanced], y_train_cls)
        d3_prob = d3_mlp.predict_proba(val_df[after10_enhanced])[:, 1]
        log_result("D3", "P5_DL", "T3", "MLP Classifier", evaluate_cls(y_val_cls, d3_prob), "after10+enhanced")
        
        # D4: Win after 10 CNN
        # PyTorchCNNClassifier expects the full DataFrame with move text
        d4_cnn = PyTorchCNNClassifier(ply_limit=10, epochs=15, verbose=0)
        d4_cnn.fit(train_df[after10_enhanced + ["first_10_moves_text"]], y_train_cls)
        d4_prob = d4_cnn.predict_proba(val_df[after10_enhanced + ["first_10_moves_text"]])[:, 1]
        log_result("D4", "P5_DL", "T3", "CNN Board Model", evaluate_cls(y_val_cls, d4_prob), "board_tensor+after10_enhanced")
        
        # D5: Elo after 10 MLP
        d5_mlp = PyTorchMLPRegressor(epochs=25, verbose=0)
        d5_mlp.fit(train_df[elo_enhanced], y_train_elo)
        d5_pred = d5_mlp.predict(val_df[elo_enhanced])
        log_result("D5", "P5_DL", "T4", "MLP Regressor", evaluate_reg(y_val_elo, d5_pred), "elo+enhanced")
        
        # D6: Elo after 10 CNN Regressor
        d6_cnn = PyTorchCNNRegressor(ply_limit=10, epochs=15, verbose=0)
        d6_cnn.fit(train_df[elo_enhanced + ["first_10_moves_text"]], y_train_elo)
        d6_pred = d6_cnn.predict(val_df[elo_enhanced + ["first_10_moves_text"]])
        log_result("D6", "P5_DL", "T4", "CNN Board Regressor", evaluate_reg(y_val_elo, d6_pred), "board_tensor+elo_enhanced")
    else:
        print("\n--- Phase 5: Deep Learning [SKIPPED] ---")
        
    print("\n--- Phase 6: Ensemble & Stacking ---")
    # Base estimators for T3 classification stacking
    t3_lgb = build_tree_classifier("lightgbm", after10_enhanced, random_state=DEFAULT_RANDOM_SEED)
    t3_xgb = build_tree_classifier("xgboost", after10_enhanced, random_state=DEFAULT_RANDOM_SEED)
    t3_rf = build_tree_classifier("randomforest", after10_enhanced, random_state=DEFAULT_RANDOM_SEED)
    
    t3_estimators = [
        ("lgb", t3_lgb),
        ("xgb", t3_xgb),
        ("rf", t3_rf)
    ]
    
    # E1: Win after 10 Voting
    t3_vote = build_ensemble_classifier(t3_estimators, voting="soft")
    t3_vote.fit(train_df, y_train_cls)
    t3_vote_prob = t3_vote.predict_proba(val_df)[:, 1]
    log_result("E1", "P6_Ensemble", "T3", "Voting(LGB+XGB+RF)", evaluate_cls(y_val_cls, t3_vote_prob), "after10+enhanced")
    
    # E2: Win after 10 Stacking
    t3_stack = build_stacking_classifier(t3_estimators, random_seed=DEFAULT_RANDOM_SEED)
    t3_stack.fit(train_df, y_train_cls)
    t3_stack_prob = t3_stack.predict_proba(val_df)[:, 1]
    log_result("E2", "P6_Ensemble", "T3", "Stacking(LGB+XGB+RF)", evaluate_cls(y_val_cls, t3_stack_prob), "after10+enhanced")
    
    # E3: Elo regression voting
    t4_lgb_1d = build_tree_regressor("lightgbm", elo_enhanced, random_state=DEFAULT_RANDOM_SEED, multi_output=False)
    t4_xgb_1d = build_tree_regressor("xgboost", elo_enhanced, random_state=DEFAULT_RANDOM_SEED, multi_output=False)
    t4_rf_1d = build_tree_regressor("randomforest", elo_enhanced, random_state=DEFAULT_RANDOM_SEED, multi_output=False)
    
    t4_estimators = [
        ("lgb", t4_lgb_1d),
        ("xgb", t4_xgb_1d),
        ("rf", t4_rf_1d)
    ]
    
    t4_vote = build_ensemble_regressor(t4_estimators)
    t4_vote.fit(train_df, y_train_elo)
    t4_vote_pred = t4_vote.predict(val_df)
    log_result("E3", "P6_Ensemble", "T4", "Voting Regressor", evaluate_reg(y_val_elo, t4_vote_pred), "elo+enhanced")
    
    # Save master results CSV
    res_df = pd.DataFrame(results_list)
    results_path = OUTPUT_DIR / "experiment_results.csv"
    res_df.to_csv(results_path, index=False)
    print(f"\nSaved master results table to {results_path}")
    
    # Select best config per task
    best_configs = {}
    for task in ["T1", "T2", "T3", "T4"]:
        task_rows = res_df[res_df["task"] == task]
        if task_rows.empty:
            continue
        if task in ["T1", "T2", "T3"]:
            # Classification: best is highest AUC
            best_idx = task_rows["auc"].idxmax()
            metric_val = task_rows.loc[best_idx, "auc"]
            metric_name = "AUC"
        else:
            # Regression: best is lowest avg MAE
            best_idx = task_rows["avg_mae"].idxmin()
            metric_val = task_rows.loc[best_idx, "avg_mae"]
            metric_name = "MAE"
            
        best_row = task_rows.loc[best_idx]
        best_configs[task] = {
            "exp_id": str(best_row["exp_id"]),
            "model_name": str(best_row["model_name"]),
            "phase": str(best_row["phase"]),
            "features": str(best_row["features"]),
            f"best_{metric_name.lower()}": float(metric_val)
        }
        print(f"Best model for {task}: {best_row['model_name']} from {best_row['phase']} ({metric_name}: {metric_val:.4f})")
        
    best_config_path = OUTPUT_DIR / "best_models.json"
    with open(best_config_path, "w") as f:
        json.dump(best_configs, f, indent=2)
    print(f"Saved best models metadata to {best_config_path}")
    
    print("\n=== Chess Prediction Experiments Completed ===")

if __name__ == "__main__":
    main()
