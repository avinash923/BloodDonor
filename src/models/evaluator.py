"""
src/models/evaluator.py
Cross-validated evaluation engine for all ML classifiers.

Computes the following metrics per fold and reports mean ± std:
  - ROC-AUC    : Discrimination ability (threshold-independent)
  - PR-AUC     : Precision-Recall AUC (critical for 24% imbalance)
  - F1 (binary): Harmonic mean of precision and recall (positive class)
  - F1 (macro) : Unweighted average F1 across both classes
  - Brier Score: Mean squared error of probability predictions (calibration)
  - Accuracy   : Overall classification accuracy

DATA STREAM: UCI_PILOT (Phase 2 ML benchmark only)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
    accuracy_score,
    roc_curve,
    precision_recall_curve,
)

from src.models.base_model import BaseClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =================================================================
# Data Structures
# =================================================================

@dataclass
class FoldResult:
    """Metrics from a single CV fold."""
    fold: int
    roc_auc: float
    pr_auc: float
    f1_binary: float
    f1_macro: float
    brier_score: float
    accuracy: float
    fit_time_seconds: float
    n_train: int
    n_test: int


@dataclass
class ModelBenchmarkResult:
    """Aggregated benchmark result for one model across all CV folds."""
    model_name: str
    fold_results: list[FoldResult] = field(default_factory=list)

    # Aggregated statistics (computed after all folds)
    mean_roc_auc: float = 0.0
    std_roc_auc: float = 0.0
    mean_pr_auc: float = 0.0
    std_pr_auc: float = 0.0
    mean_f1_binary: float = 0.0
    std_f1_binary: float = 0.0
    mean_f1_macro: float = 0.0
    std_f1_macro: float = 0.0
    mean_brier_score: float = 0.0
    std_brier_score: float = 0.0
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    mean_fit_time: float = 0.0

    # Pooled probability outputs (for overall calibration curves)
    all_y_true: list = field(default_factory=list)
    all_y_prob: list = field(default_factory=list)

    def aggregate(self) -> None:
        """Compute mean and std across all fold results."""
        metrics = ["roc_auc", "pr_auc", "f1_binary", "f1_macro", "brier_score",
                   "accuracy", "fit_time_seconds"]
        for m in metrics:
            vals = np.array([getattr(f, m) for f in self.fold_results])
            attr_base = m.replace("_seconds", "")
            setattr(self, f"mean_{attr_base}", float(vals.mean()))
            if m != "fit_time_seconds":
                setattr(self, f"std_{attr_base}", float(vals.std()))
        self.mean_fit_time = float(
            np.mean([f.fit_time_seconds for f in self.fold_results])
        )

    def summary_dict(self) -> dict:
        """Return a flat dictionary suitable for DataFrame construction."""
        return {
            "Model": self.model_name,
            "ROC-AUC": f"{self.mean_roc_auc:.4f} ± {self.std_roc_auc:.4f}",
            "PR-AUC": f"{self.mean_pr_auc:.4f} ± {self.std_pr_auc:.4f}",
            "F1 (Binary)": f"{self.mean_f1_binary:.4f} ± {self.std_f1_binary:.4f}",
            "F1 (Macro)": f"{self.mean_f1_macro:.4f} ± {self.std_f1_macro:.4f}",
            "Brier Score": f"{self.mean_brier_score:.4f} ± {self.std_brier_score:.4f}",
            "Accuracy": f"{self.mean_accuracy:.4f} ± {self.std_accuracy:.4f}",
            "Mean Fit Time (s)": f"{self.mean_fit_time:.3f}",
        }


# =================================================================
# Core Evaluator
# =================================================================

class CrossValidatedEvaluator:
    """
    Runs stratified k-fold cross-validation on any BaseClassifier and
    collects all performance metrics.

    Usage:
        evaluator = CrossValidatedEvaluator(splits, X, y)
        result = evaluator.evaluate(model)
    """

    def __init__(
        self,
        splits: list[tuple[np.ndarray, np.ndarray]],
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        self.splits = splits
        self.X = X
        self.y = y

    def evaluate(
        self,
        model: BaseClassifier,
        X_val_fraction: float = 0.1,
        seed: int = 42,
    ) -> ModelBenchmarkResult:
        """
        Run cross-validated evaluation for a single model.

        For each fold:
          1. Scale features (fit on train, apply to test — no leakage).
          2. Fit the model on training fold.
          3. Predict probabilities on test fold.
          4. Compute all metrics.

        Args:
            model:           An unfitted BaseClassifier instance.
            X_val_fraction:  Fraction of training data to hold out for
                             XGBoost early stopping. Ignored for other models.
            seed:            Random seed for train/val split.

        Returns:
            ModelBenchmarkResult with per-fold results and aggregated stats.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        result = ModelBenchmarkResult(model_name=model.model_name)
        logger.info("Evaluating: %s (%d folds)", model.model_name, len(self.splits))

        for fold_idx, (train_idx, test_idx) in enumerate(self.splits):
            X_train_raw, X_test_raw = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            # Feature scaling: fit only on training fold
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            # XGBoost early stopping: carve out inner validation set
            fit_kwargs: dict = {}
            if hasattr(model, '_early_stopping_rounds'):
                X_tr, X_es, y_tr, y_es = train_test_split(
                    X_train, y_train,
                    test_size=X_val_fraction,
                    stratify=y_train,
                    random_state=seed,
                )
                fit_kwargs = {"X_val": X_es, "y_val": y_es}
            else:
                X_tr, y_tr = X_train, y_train

            # Fit and time
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr, **fit_kwargs)
            fit_time = time.perf_counter() - t0

            # Predict
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba_positive(X_test)

            # Metrics
            fold_result = FoldResult(
                fold=fold_idx + 1,
                roc_auc=roc_auc_score(y_test, y_prob),
                pr_auc=average_precision_score(y_test, y_prob),
                f1_binary=f1_score(y_test, y_pred, pos_label=1, average="binary", zero_division=0),
                f1_macro=f1_score(y_test, y_pred, average="macro", zero_division=0),
                brier_score=brier_score_loss(y_test, y_prob),
                accuracy=accuracy_score(y_test, y_pred),
                fit_time_seconds=fit_time,
                n_train=len(y_train),
                n_test=len(y_test),
            )
            result.fold_results.append(fold_result)
            result.all_y_true.extend(y_test.tolist())
            result.all_y_prob.extend(y_prob.tolist())

            logger.debug(
                "  Fold %d/%d | ROC-AUC=%.4f | PR-AUC=%.4f | Brier=%.4f",
                fold_idx + 1, len(self.splits),
                fold_result.roc_auc, fold_result.pr_auc, fold_result.brier_score,
            )

        result.aggregate()
        logger.info(
            "%s — ROC-AUC: %.4f±%.4f | PR-AUC: %.4f±%.4f | F1: %.4f±%.4f | Brier: %.4f±%.4f",
            result.model_name,
            result.mean_roc_auc, result.std_roc_auc,
            result.mean_pr_auc, result.std_pr_auc,
            result.mean_f1_binary, result.std_f1_binary,
            result.mean_brier_score, result.std_brier_score,
        )
        return result


# =================================================================
# Plotting Utilities
# =================================================================

def plot_roc_curves(
    results: list[ModelBenchmarkResult],
    output_path: str = "reports/figures/roc_curves.png",
    dpi: int = 150,
) -> None:
    """Generate ROC curves (pooled across folds) for all evaluated models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2E86AB", "#E84855", "#3BB273", "#F7B731"]

    for result, color in zip(results, colors):
        y_true = np.array(result.all_y_true)
        y_prob = np.array(result.all_y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{result.model_name} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Donor Propensity Classification (UCI)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("ROC curves saved to %s", output_path)


def plot_pr_curves(
    results: list[ModelBenchmarkResult],
    baseline_positive_rate: float = 0.24,
    output_path: str = "reports/figures/pr_curves.png",
    dpi: int = 150,
) -> None:
    """Generate Precision-Recall curves (pooled across folds) for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2E86AB", "#E84855", "#3BB273", "#F7B731"]

    for result, color in zip(results, colors):
        y_true = np.array(result.all_y_true)
        y_prob = np.array(result.all_y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, lw=2, color=color,
                label=f"{result.model_name} (PR-AUC={pr_auc:.4f})")

    ax.axhline(y=baseline_positive_rate, color="gray", linestyle="--", lw=1,
               label=f"No-skill baseline ({baseline_positive_rate:.2f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Donor Propensity (UCI)", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("PR curves saved to %s", output_path)


def plot_reliability_diagram(
    results: list[ModelBenchmarkResult],
    n_bins: int = 10,
    output_path: str = "reports/figures/calibration_plot.png",
    dpi: int = 150,
) -> None:
    """Plot reliability diagrams (calibration curves) for all models."""
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2E86AB", "#E84855", "#3BB273", "#F7B731"]

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    for result, color in zip(results, colors):
        y_true = np.array(result.all_y_true)
        y_prob = np.array(result.all_y_prob)
        fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        ax.plot(mean_pred, fraction_pos, "s-", lw=2, color=color,
                label=f"{result.model_name} (Brier={result.mean_brier_score:.4f})")

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Reliability Diagram — Probability Calibration (UCI)", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Reliability diagram saved to %s", output_path)


def save_summary_table(
    results: list[ModelBenchmarkResult],
    output_path: str = "reports/tables/ml_benchmark_summary.md",
) -> None:
    """Save a markdown-formatted comparison table of all model results."""
    import pandas as pd

    rows = [r.summary_dict() for r in results]
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Markdown table
    md_lines = ["# ML Benchmark: Donor Propensity Classification (UCI Dataset)\n",
                "**Data Stream**: UCI_PILOT | **Evaluation**: 5-Fold Stratified CV\n",
                df.to_markdown(index=False), "\n"]
    Path(output_path).write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Benchmark summary table saved to %s", output_path)
