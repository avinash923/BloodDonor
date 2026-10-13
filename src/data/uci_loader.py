"""
src/data/uci_loader.py
UCI Blood Transfusion Service Center Dataset loader and preprocessor.

Dataset origin:
  Yeh, I-Cheng, Yang, King-Jang, and Ting, Tao-Ming, "Knowledge discovery
  on RFM model using Bernoulli sequence," Expert Systems with Applications,
  2008, doi:10.1016/j.eswa.2008.07.018

UCI Repository URL:
  https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data

DATA STREAM TAG: UCI_PILOT (real-world behavioral data)
This module exclusively handles UCI Stream A data.
Never mix outputs from this loader with synthetic or empirical streams.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------
# Column names in the UCI transfusion.data file
# -----------------------------------------------------------------
_UCI_COLUMNS = [
    "recency",    # R: months since last donation
    "frequency",  # F: total number of donations
    "monetary",   # M: total blood donated (c.c.) — always 250 * F
    "time",       # T: months since first donation
    "donated",    # Target: 1 = donated March 2007, 0 = did not
]

_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "blood-transfusion/transfusion.data"
)

# Mirrors tried in order when the UCI host is unreachable. A mirror copy is
# only accepted if it reproduces the canonical content checksum below.
_MIRROR_URLS = (
    _UCI_URL,
    # PyCaret's public dataset repository ("blood" = UCI transfusion.data).
    "https://raw.githubusercontent.com/pycaret/datasets/main/data/common/blood.csv",
)

# Canonical content of transfusion.data: 748 rows, 178 positives. The checksum
# is over the parsed integer values (little-endian int64, row-major), so it is
# independent of header text and CRLF/LF line endings.
EXPECTED_N_ROWS = 748
EXPECTED_N_POSITIVE = 178
EXPECTED_CONTENT_SHA256 = "0c5bfc6d287ced28e723cf1485d9d73f3c38c7c26e524b756a5b6ac2b29274be"

_FEATURE_COLS = ["recency", "frequency", "monetary", "time"]
_TARGET_COL = "donated"


# =================================================================
# Download
# =================================================================

def download_uci_dataset(save_dir: str = "data/raw/uci") -> Path:
    """
    Download the UCI Blood Transfusion dataset if not already present.

    Args:
        save_dir: Directory to save transfusion.data.

    Returns:
        Path to the downloaded file.
    """
    dest = Path(save_dir)
    dest.mkdir(parents=True, exist_ok=True)
    filepath = dest / "transfusion.data"

    if filepath.exists():
        logger.info("UCI dataset already present at %s — skipping download.", filepath)
        return filepath

    errors = []
    for url in _MIRROR_URLS:
        logger.info("Downloading UCI Blood Transfusion dataset from %s ...", url)
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read()
            df = pd.read_csv(io.BytesIO(payload), header=0)
            verify_uci_content(df)
        except Exception as exc:  # network failure or content mismatch: try next mirror
            logger.warning("Source %s rejected: %s", url, exc)
            errors.append(f"{url}: {exc}")
            continue
        filepath.write_bytes(payload.replace(b"\r\n", b"\n"))
        logger.info("Download verified and saved to %s.", filepath)
        return filepath

    raise RuntimeError(
        "Could not obtain a verified copy of the UCI dataset. "
        f"Place transfusion.data manually in {save_dir}. Attempts: " + "; ".join(errors)
    )


def content_sha256(df: pd.DataFrame) -> str:
    """Checksum of the parsed 5-column integer table (format independent)."""
    values = np.ascontiguousarray(df.iloc[:, :5].to_numpy(dtype="<i8"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def verify_uci_content(df: pd.DataFrame) -> None:
    """Raise ValueError unless df is exactly the published UCI transfusion table."""
    if df.shape != (EXPECTED_N_ROWS, 5):
        raise ValueError(f"Expected shape ({EXPECTED_N_ROWS}, 5), got {df.shape}")
    if int(df.iloc[:, 4].sum()) != EXPECTED_N_POSITIVE:
        raise ValueError(f"Expected {EXPECTED_N_POSITIVE} positives, got {int(df.iloc[:, 4].sum())}")
    digest = content_sha256(df)
    if digest != EXPECTED_CONTENT_SHA256:
        raise ValueError(f"Content checksum mismatch: {digest}")


# =================================================================
# Loading & Validation
# =================================================================

def load_uci_dataframe(filepath: str | Path, verify: bool = False) -> pd.DataFrame:
    """
    Load and validate the UCI transfusion.data CSV file.

    Performs:
      - Column renaming to standardized RFMT names.
      - Dtype enforcement (all integer).
      - Consistency check: M == 250 * F for every row.
      - Missing value check.
      - Optional canonical-content verification (row count, positives, checksum).

    Args:
        filepath: Path to transfusion.data.
        verify:   If True, require the exact published UCI content.

    Returns:
        Validated pandas DataFrame with columns:
        ['recency', 'frequency', 'monetary', 'time', 'donated']

    Raises:
        ValueError: If the file fails schema validation.
    """
    filepath = Path(filepath)
    logger.info("Loading UCI dataset from %s ...", filepath)

    df = pd.read_csv(filepath, header=0)

    # The downloaded file already has a header row; rename to canonical names
    if len(df.columns) == 5:
        df.columns = _UCI_COLUMNS
    else:
        raise ValueError(
            f"Expected 5 columns in UCI dataset, got {len(df.columns)}. "
            f"Columns found: {list(df.columns)}"
        )

    # Dtype enforcement
    for col in _UCI_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Missing value check
    if df.isnull().any().any():
        n_missing = df.isnull().sum().sum()
        raise ValueError(
            f"UCI dataset contains {n_missing} missing values — "
            "dataset corruption or format mismatch."
        )

    if verify:
        verify_uci_content(df)

    # Monetary consistency check (M == 250 * F)
    inconsistent = (df["monetary"] != df["frequency"] * 250).sum()
    if inconsistent > 0:
        logger.warning(
            "%d rows have monetary != 250 * frequency. "
            "This deviates from the UCI specification — check dataset integrity.",
            inconsistent,
        )

    logger.info(
        "Dataset loaded: %d instances, %d features, positive rate = %.2f%%",
        len(df),
        len(_FEATURE_COLS),
        df[_TARGET_COL].mean() * 100,
    )
    return df


# =================================================================
# Feature Engineering & Preprocessing
# =================================================================

def extract_features_and_target(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and target vector y from the DataFrame.

    Args:
        df: Validated UCI DataFrame.

    Returns:
        Tuple of (X: np.ndarray of shape [n, 4], y: np.ndarray of shape [n])
    """
    X = df[_FEATURE_COLS].to_numpy(dtype=np.float64)
    y = df[_TARGET_COL].to_numpy(dtype=np.int32)
    return X, y


def get_feature_names() -> list[str]:
    """Return the canonical RFMT feature names."""
    return list(_FEATURE_COLS)


# =================================================================
# Cross-Validation Split Generation
# =================================================================

def get_stratified_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate stratified k-fold cross-validation index splits.

    Stratification preserves the ~24% positive class ratio in every fold.
    Splits are deterministic under the given seed.

    Args:
        X:        Feature matrix.
        y:        Target vector.
        n_splits: Number of folds (default 5).
        seed:     Random seed for reproducibility.

    Returns:
        List of (train_indices, test_indices) tuples, length = n_splits.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))
    logger.info(
        "Generated %d stratified CV splits (seed=%d). "
        "Class distribution in first test fold: %.2f%% positive.",
        n_splits,
        seed,
        y[splits[0][1]].mean() * 100,
    )
    return splits


# =================================================================
# Standard Scaling (fit on train, apply to test — no leakage)
# =================================================================

def scale_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on X_train and apply it to X_test.

    Ensures zero leakage: scaler statistics computed exclusively
    from training fold, then applied to held-out test fold.

    Args:
        X_train: Training feature matrix.
        X_test:  Test feature matrix.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# =================================================================
# Convenience: Full Pipeline
# =================================================================

def prepare_uci_data(
    data_dir: str = "data/raw/uci",
    n_splits: int = 5,
    seed: int = 42,
    download: bool = True,
) -> dict:
    """
    End-to-end preparation of the UCI dataset for cross-validated benchmarking.

    Args:
        data_dir:  Directory containing (or to receive) transfusion.data.
        n_splits:  Number of CV folds.
        seed:      Global random seed.
        download:  If True, attempt to download the dataset if absent.

    Returns:
        Dictionary with keys:
          'X'      : np.ndarray — full unscaled feature matrix
          'y'      : np.ndarray — full target vector
          'splits' : list of (train_idx, test_idx) tuples
          'feature_names' : list of str
          'filepath': Path to raw data file
          'class_weight_positive' : float — reciprocal positive rate (imbalance weight)
    """
    filepath = download_uci_dataset(data_dir) if download else Path(data_dir) / "transfusion.data"
    df = load_uci_dataframe(filepath)
    X, y = extract_features_and_target(df)
    splits = get_stratified_cv_splits(X, y, n_splits=n_splits, seed=seed)

    positive_rate = y.mean()
    class_weight_positive = (1.0 - positive_rate) / positive_rate if positive_rate > 0 else 1.0

    return {
        "X": X,
        "y": y,
        "splits": splits,
        "feature_names": get_feature_names(),
        "filepath": filepath,
        "class_weight_positive": class_weight_positive,
        "positive_rate": positive_rate,
        "n_samples": len(y),
    }
