"""Model building module for baseline, tree-based, and ensemble models.

Provides helper functions to construct standard sklearn Pipelines with
appropriate imputers, scalers, and vectorizers for different models and tasks.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor
)
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from experiment.config import HASHING_FEATURES

def get_baseline_preprocessor(numeric_cols: list[str], text_cols: list[str] = None) -> ColumnTransformer:
    """Create ColumnTransformer for baseline models (imputer+scaler for numeric, hashing for text)."""
    transformers = []
    
    # Numeric transformer
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    transformers.append(("num", num_pipeline, numeric_cols))
    
    # Text transformers
    if text_cols:
        for t_col in text_cols:
            transformers.append((
                f"txt_{t_col}",
                HashingVectorizer(n_features=HASHING_FEATURES, alternate_sign=False),
                t_col
            ))
            
    return ColumnTransformer(transformers)

def get_numeric_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    """Create ColumnTransformer for numeric-only models (imputer + scaler)."""
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols)
    ])

def build_baseline_classifier(numeric_cols: list[str], text_cols: list[str] = None, c_value: float = 1.0, random_seed: int = 42) -> Pipeline:
    """Build a Logistic Regression classifier pipeline."""
    preprocessor = get_baseline_preprocessor(numeric_cols, text_cols)
    model = LogisticRegression(
        C=c_value,
        max_iter=5000,
        solver="liblinear",
        random_state=random_seed
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

def build_baseline_regressor(numeric_cols: list[str], text_cols: list[str] = None, alpha: float = 10.0) -> Pipeline:
    """Build a Ridge regressor pipeline."""
    preprocessor = get_baseline_preprocessor(numeric_cols, text_cols)
    # Note: Ridge natively supports multi-output regression (predicting both White and Black Elos)
    model = Ridge(alpha=alpha)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

def build_tree_classifier(model_name: str, numeric_cols: list[str], random_seed: int = 42, **kwargs) -> Pipeline:
    """Build a tree-based classifier pipeline (LightGBM, XGBoost, HistGB, RF) on numeric features."""
    preprocessor = get_numeric_preprocessor(numeric_cols)
    
    # Default parameters if not provided in kwargs
    params = {"random_state": random_seed}
    params.update(kwargs)
    
    if model_name.lower() == "lightgbm":
        # Force single thread to avoid multi-threading conflicts if needed, or let it default
        clf = LGBMClassifier(**params)
    elif model_name.lower() == "xgboost":
        clf = XGBClassifier(eval_metric="logloss", **params)
    elif model_name.lower() == "histgb":
        # Rename n_estimators to max_iter for HistGradientBoosting
        if "n_estimators" in params:
            params["max_iter"] = params.pop("n_estimators")
        clf = HistGradientBoostingClassifier(**params)
    elif model_name.lower() == "randomforest":
        clf = RandomForestClassifier(**params)
    elif model_name.lower() == "gradientboosting":
        clf = GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown tree model: {model_name}")
        
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

def build_tree_regressor(model_name: str, numeric_cols: list[str], random_seed: int = 42, multi_output: bool = True, **kwargs) -> Pipeline:
    """Build a tree-based regressor pipeline on numeric features."""
    preprocessor = get_numeric_preprocessor(numeric_cols)
    
    params = {"random_state": random_seed}
    params.update(kwargs)
    
    if model_name.lower() == "lightgbm":
        reg = LGBMRegressor(**params)
    elif model_name.lower() == "xgboost":
        reg = XGBRegressor(**params)
    elif model_name.lower() == "histgb":
        if "n_estimators" in params:
            params["max_iter"] = params.pop("n_estimators")
        reg = HistGradientBoostingRegressor(**params)
    elif model_name.lower() == "randomforest":
        reg = RandomForestRegressor(**params)
    elif model_name.lower() == "gradientboosting":
        reg = GradientBoostingRegressor(**params)
    else:
        raise ValueError(f"Unknown tree model: {model_name}")
        
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", reg)
    ])
    
    if multi_output:
        from sklearn.multioutput import MultiOutputRegressor
        return MultiOutputRegressor(pipeline)
    else:
        return pipeline

def build_ensemble_classifier(estimators: list[tuple[str, Pipeline]], voting: str = "soft") -> VotingClassifier:
    """Build a VotingClassifier ensemble."""
    # VotingClassifier takes estimators as list of (name, estimator)
    # We must ensure they all have predict_proba
    return VotingClassifier(estimators=estimators, voting=voting)

def build_ensemble_regressor(estimators: list[tuple[str, Pipeline]]) -> MultiOutputRegressor:
    """Build a VotingRegressor ensemble supporting multi-output targets."""
    from sklearn.multioutput import MultiOutputRegressor
    vote = VotingRegressor(estimators=estimators)
    return MultiOutputRegressor(vote)

def build_stacking_classifier(estimators: list[tuple[str, Pipeline]], final_estimator=None, random_seed: int = 42) -> StackingClassifier:
    """Build a StackingClassifier ensemble."""
    if final_estimator is None:
        final_estimator = LogisticRegression(random_state=random_seed)
    return StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=3, n_jobs=1)

def build_stacking_regressor(estimators: list[tuple[str, Pipeline]], final_estimator=None) -> MultiOutputRegressor:
    """Build a StackingRegressor ensemble supporting multi-output targets."""
    from sklearn.multioutput import MultiOutputRegressor
    if final_estimator is None:
        final_estimator = Ridge()
    stack = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=3, n_jobs=1)
    return MultiOutputRegressor(stack)


