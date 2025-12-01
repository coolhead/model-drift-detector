import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Literal

Task = Literal["classification", "regression"]

def train_model_and_shap(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    task: Task,
):
    X = df[feature_cols]
    y = df[target_col]

    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )

    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For classification, shap_values is list per class
    if task == "classification":
        shap_arr = np.mean(np.abs(shap_values), axis=0)  # average over classes
    else:
        shap_arr = np.abs(shap_values)

    mean_abs = np.mean(np.abs(shap_arr), axis=0)
    return pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})


def compute_shap_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    task: Task,
) -> pd.DataFrame:
    base_shap = train_model_and_shap(baseline_df, feature_cols, target_col, task)
    curr_shap = train_model_and_shap(current_df, feature_cols, target_col, task)

    merged = base_shap.merge(curr_shap, on="feature", suffixes=("_baseline", "_current"))
    merged["shap_ratio"] = (
        merged["mean_abs_shap_current"] / (merged["mean_abs_shap_baseline"] + 1e-6)
    )

    return merged.sort_values("shap_ratio", ascending=False)
