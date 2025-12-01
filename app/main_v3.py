import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from drift_metrics import psi, ks_stat


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Model Drift Detector Pro",
    layout="wide",
    page_icon="📊",
)


# -------------------------
# Helper functions
# -------------------------
def categorize_psi(value: float) -> str:
    if value < 0.1:
        return "Stable"
    elif value < 0.25:
        return "Moderate"
    else:
        return "Severe"


def compute_drift_summary(baseline_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    common_cols = [c for c in baseline_df.columns if c in current_df.columns]

    rows = []
    for col in common_cols:
        if not (
            pd.api.types.is_numeric_dtype(baseline_df[col])
            and pd.api.types.is_numeric_dtype(current_df[col])
        ):
            continue

        base_col = baseline_df[col]
        curr_col = current_df[col]

        psi_value = psi(base_col, curr_col)
        ks_value = ks_stat(base_col, curr_col)
        severity = categorize_psi(psi_value)

        rows.append(
            {
                "feature": col,
                "psi": psi_value,
                "ks": ks_value,
                "severity": severity,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["feature", "psi", "ks", "severity"])

    df = pd.DataFrame(rows)
    df = df.sort_values("psi", ascending=False).reset_index(drop=True)
    return df


# -------------------------
# Sidebar – data inputs
# -------------------------
st.sidebar.header("⚙️ Configuration")

baseline_file = st.sidebar.file_uploader(
    "Baseline dataset (CSV)", type=["csv"], key="baseline"
)
current_file = st.sidebar.file_uploader(
    "Current dataset (CSV)", type=["csv"], key="current"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: use the same schema for baseline & current. "
    "Numeric columns are used for drift analysis."
)

st.title("📊 Model Drift Detector – Pro Edition")
st.write(
    "Upload **baseline** and **current** datasets to analyse data drift, "
    "time-series drift, SHAP drift, embedding drift, performance drift, and generate reports."
)

if not (baseline_file and current_file):
    st.info("Upload both baseline and current CSVs from the sidebar to see drift analysis.")
    st.stop()

baseline_df = pd.read_csv(baseline_file)
current_df = pd.read_csv(current_file)

drift_df = compute_drift_summary(baseline_df, current_df)

if drift_df.empty:
    st.warning("No numeric columns in common between the two datasets. Nothing to analyse.")
    st.stop()

total_features = len(drift_df)
severe_count = int((drift_df["severity"] == "Severe").sum())
moderate_count = int((drift_df["severity"] == "Moderate").sum())
stable_count = int((drift_df["severity"] == "Stable").sum())


# -------------------------
# Tabs
# -------------------------
tab_overview, tab_time, tab_shap, tab_embed, tab_perf, tab_report = st.tabs(
    ["Overview", "Time-series drift", "SHAP drift", "Embedding drift",
     "Performance drift", "Report"]
)

# -------------------------
# OVERVIEW TAB  (v2 layout)
# -------------------------
with tab_overview:
    st.subheader("📈 Drift overview")

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Numeric features analysed", total_features)
    mcol2.metric("Severe drift (PSI ≥ 0.25)", severe_count)
    mcol3.metric("Moderate drift (0.10 ≤ PSI < 0.25)", moderate_count)
    mcol4.metric("Stable (PSI < 0.10)", stable_count)

    st.caption(
        "Rule of thumb: PSI > 0.2–0.25 usually indicates significant drift worth investigating."
    )

    tcol1, tcol2 = st.columns([3, 2])

    with tcol1:
        st.markdown("### 🔍 Per-feature drift table")

        styled = (
            drift_df.style
            .format({"psi": "{:.3f}", "ks": "{:.3f}"})
            .apply(
                lambda s: [
                    "background-color: #2e7d32; color: white" if v == "Stable"
                    else "background-color: #f9a825; color: black" if v == "Moderate"
                    else "background-color: #c62828; color: white"
                    for v in s
                ],
                subset=["severity"],
            )
        )

        st.dataframe(styled, width="stretch")

    with tcol2:
        st.markdown("### 🌞 Drift severity sunburst")

        fig_sun = px.sunburst(
            drift_df,
            path=["severity", "feature"],
            values="psi",
            title="PSI contribution by feature & severity",
        )
        st.plotly_chart(fig_sun, width="stretch")

    st.markdown("### 🧬 Feature-level deep dive")

    feature_choices = drift_df["feature"].tolist()
    selected_feature = st.selectbox("Select a feature to inspect distribution shift", feature_choices)

    base_col = baseline_df[selected_feature]
    curr_col = current_df[selected_feature]

    dd_col1, dd_col2 = st.columns(2)

    with dd_col1:
        st.write("Baseline vs current distribution (histogram)")
        hist_df = pd.DataFrame(
            {
                selected_feature: pd.concat([base_col, curr_col], ignore_index=True),
                "dataset": ["baseline"] * len(base_col) + ["current"] * len(curr_col),
            }
        )
        fig_hist = px.histogram(
            hist_df,
            x=selected_feature,
            color="dataset",
            barmode="overlay",
            opacity=0.6,
            marginal="box",
        )
        st.plotly_chart(fig_hist, width="stretch")

    with dd_col2:
        psi_value = drift_df.loc[drift_df["feature"] == selected_feature, "psi"].iloc[0]
        ks_value = drift_df.loc[drift_df["feature"] == selected_feature, "ks"].iloc[0]
        severity = drift_df.loc[drift_df["feature"] == selected_feature, "severity"].iloc[0]

        st.markdown(f"#### Metrics for `{selected_feature}`")
        st.metric("PSI", f"{psi_value:.3f}")
        st.metric("KS statistic", f"{ks_value:.3f}")
        st.write(f"**Severity bucket:** `{severity}`")
        st.markdown(
            """
            **Interpretation:**
            - Higher PSI → larger population shift between baseline and current.
            - Higher KS → stronger difference in distributions.
            """
        )


# -------------------------
# TIME-SERIES DRIFT TAB
# -------------------------
with tab_time:
    st.subheader("⏱ Time-series drift")

    # detect datetime-like columns
    datetime_candidates = [
        c for c in baseline_df.columns
        if np.issubdtype(baseline_df[c].dtype, np.datetime64)
        or "date" in c.lower()
        or "time" in c.lower()
    ]
    if not datetime_candidates:
        st.info("No obvious datetime columns found. Add/convert a datetime column to use time-series drift.")
    else:
        dt_col = st.selectbox("Select datetime column", datetime_candidates)
        freq = st.selectbox("Aggregation frequency", ["D", "W", "M"], index=2)

        from time_drift.time_drift import compute_time_drift

        ts_df = compute_time_drift(baseline_df, current_df, dt_col, freq)
        st.dataframe(ts_df.head(), width="stretch")

        if not ts_df.empty:
            feat = st.selectbox("Feature to plot", ts_df["feature"].unique())
            sub = ts_df[ts_df["feature"] == feat]
            fig = px.line(sub, x="period", y="psi", title=f"PSI over time for {feat}")
            st.plotly_chart(fig, width="stretch")


# -------------------------
# SHAP DRIFT TAB
# -------------------------
with tab_shap:
    st.subheader("🧠 SHAP explainability drift")

    from time_shap.shap_drift import compute_shap_drift

    target_col = st.selectbox("Target column", baseline_df.columns)
    feature_cols = [
        c for c in baseline_df.columns
        if c != target_col and pd.api.types.is_numeric_dtype(baseline_df[c])
    ]

    if not feature_cols:
        st.info("No numeric feature columns found for SHAP. Add some numeric features.")
    else:
        task = st.selectbox("Task type", ["classification", "regression"])

        # Simple guard: warn if user picks classification on a clearly continuous target
        looks_continuous = (
            pd.api.types.is_float_dtype(baseline_df[target_col])
            and baseline_df[target_col].nunique() > 10
        )

        if task == "classification" and looks_continuous:
            st.warning(
                "Selected target looks continuous. "
                "Use task type 'regression' for this column, "
                "or choose a discrete label column (e.g. 0/1) for classification."
            )
        else:
            try:
                shap_df = compute_shap_drift(
                    baseline_df, current_df, feature_cols, target_col, task
                )

                st.dataframe(shap_df, width="stretch")

                if not shap_df.empty:
                    fig = px.bar(
                        shap_df,
                        x="feature",
                        y="shap_ratio",
                        title="Change in feature importance (mean |SHAP| current / baseline)",
                    )
                    st.plotly_chart(fig, width="stretch")
            except ValueError as e:
                st.error(f"Could not compute SHAP drift for this configuration: {e}")



# -------------------------
# EMBEDDING DRIFT TAB
# -------------------------
with tab_embed:
    st.subheader("🌀 Embedding drift (LLMOps)")

    text_cols = [
        c for c in baseline_df.columns
        if pd.api.types.is_string_dtype(baseline_df[c])
    ]
    if not text_cols:
        st.info("No text columns detected. Add a text field to use embedding drift.")
    else:
        text_col = st.selectbox("Text column", text_cols)

        from embeddings.embed_drift import compute_embedding_drift

        emb_dist = compute_embedding_drift(baseline_df, current_df, text_col)

        st.metric(
            "Cosine distance between baseline and current text embeddings",
            f"{emb_dist:.3f}",
        )
        st.caption("Higher distance → stronger semantic drift in text inputs.")


# -------------------------
# PERFORMANCE DRIFT TAB
# -------------------------
with tab_perf:
    st.subheader("📉 Model performance drift")

    perf_cols_ok = (
        {"y_true"} <= set(baseline_df.columns)
        and {"y_true"} <= set(current_df.columns)
    )

    if not perf_cols_ok:
        st.info(
            "To use this, include 'y_true' (and optionally 'y_pred' / 'y_pred_proba') "
            "columns in both datasets."
        )
    else:
        from performance.perf_drift import compare_perf

        perf = compare_perf(baseline_df, current_df)

        if not perf:
            st.info("No usable performance metrics found in the provided columns.")
        else:
            for metric, vals in perf.items():
                st.metric(
                    metric.upper(),
                    f"{vals['current']:.3f}",
                    delta=f"{vals['delta']:+.3f}",
                )


# -------------------------
# REPORT TAB
# -------------------------
with tab_report:
    st.subheader("📄 Generate drift report")

    from reporting.report_builder import build_report_tempfile

    perf_summary = {
        "Num features": total_features,
        "Severe drift features": severe_count,
        "Moderate drift features": moderate_count,
        "Stable features": stable_count,
    }

    if st.button("Build PDF report"):
        pdf_path = build_report_tempfile(drift_df, perf_summary)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download drift report PDF",
                data=f,
                file_name="drift_report.pdf",
                mime="application/pdf",
            )
