"""
src/evaluation/stats_testing.py
Statistical testing tools for comparing models and policies.

Includes Paired t-test, Wilcoxon signed-rank test, and Bootstrap CIs.
"""

from __future__ import annotations

import numpy as np
import scipy.stats as stats
from dataclasses import dataclass
from typing import Optional

@dataclass
class StatTestResult:
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    effect_size: Optional[float] = None # Cohen's d for t-test

def paired_ttest(sample1: np.ndarray, sample2: np.ndarray, alpha: float = 0.05) -> StatTestResult:
    """
    Perform a paired t-test for normally distributed differences.
    """
    diff = sample1 - sample2
    stat, pval = stats.ttest_rel(sample1, sample2)
    
    # Cohen's d for paired samples
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    if std_diff > 0:
        d = mean_diff / std_diff
    elif mean_diff != 0:
        d = float('inf') if mean_diff > 0 else float('-inf')
    else:
        d = 0.0
    
    return StatTestResult(
        test_name="Paired t-test",
        statistic=float(stat),
        p_value=float(pval),
        significant=bool(pval < alpha),
        alpha=alpha,
        effect_size=float(d)
    )

def wilcoxon_test(sample1: np.ndarray, sample2: np.ndarray, alpha: float = 0.05) -> StatTestResult:
    """
    Perform Wilcoxon signed-rank test (non-parametric).
    Use when normality assumption fails.
    """
    try:
        stat, pval = stats.wilcoxon(sample1, sample2, zero_method='wilcox')
    except ValueError:
        # Handles case where differences are all zero
        stat, pval = 0.0, 1.0
        
    return StatTestResult(
        test_name="Wilcoxon Signed-Rank",
        statistic=float(stat),
        p_value=float(pval),
        significant=bool(pval < alpha),
        alpha=alpha
    )

def bootstrap_ci(data: np.ndarray, confidence: float = 0.95, n_resamples: int = 1000, seed: int = 42) -> tuple[float, float]:
    """
    Compute Bootstrap Confidence Interval for the mean.
    """
    rng = np.random.default_rng(seed)
    means = []
    n = len(data)
    for _ in range(n_resamples):
        sample = rng.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    
    lower_pct = (1.0 - confidence) / 2.0 * 100
    upper_pct = (1.0 + confidence) / 2.0 * 100
    
    lower = float(np.percentile(means, lower_pct))
    upper = float(np.percentile(means, upper_pct))
    
    return lower, upper
