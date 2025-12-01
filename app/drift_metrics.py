import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def psi(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for a single feature."""
    # Drop NaNs
    baseline = baseline.dropna()
    current = current.dropna()

    # Use same bin edges from baseline
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(baseline, quantiles))

    baseline_counts, _ = np.histogram(baseline, bins=cuts)
    current_counts, _ = np.histogram(current, bins=cuts)

    baseline_perc = baseline_counts / max(baseline_counts.sum(), 1)
    current_perc = current_counts / max(current_counts.sum(), 1)

    # Avoid divide-by-zero
    baseline_perc = np.where(baseline_perc == 0, 1e-6, baseline_perc)
    current_perc = np.where(current_perc == 0, 1e-6, current_perc)

    psi_value = np.sum((current_perc - baseline_perc) * np.log(current_perc / baseline_perc))
    return float(psi_value)


def ks_stat(baseline: pd.Series, current: pd.Series) -> float:
    """Kolmogorov–Smirnov statistic for drift."""
    baseline = baseline.dropna()
    current = current.dropna()
    if len(baseline) == 0 or len(current) == 0:
        return 0.0
    return float(ks_2samp(baseline, current).statistic)
