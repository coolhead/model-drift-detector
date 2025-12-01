import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

def compute_perf_metrics(df: pd.DataFrame) -> dict:
    y_true = df["y_true"]
    out = {}

    if "y_pred" in df.columns:
        out["accuracy"] = accuracy_score(y_true, df["y_pred"])

    if "y_pred_proba" in df.columns:
        try:
            out["auc"] = roc_auc_score(y_true, df["y_pred_proba"])
        except ValueError:
            pass

    return out


def compare_perf(baseline_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    base = compute_perf_metrics(baseline_df)
    curr = compute_perf_metrics(current_df)

    diff = {}
    for k in base.keys():
        if k in curr:
            diff[k] = {
                "baseline": base[k],
                "current": curr[k],
                "delta": curr[k] - base[k],
            }
    return diff
