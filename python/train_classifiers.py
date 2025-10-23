"""Training script for logistic regression and LightGBM classifiers.

This module prepares simple sample data, fits the models, and exports the
results as ONNX graphs alongside metadata that records the feature ordering
and label mapping used by the models.

The implementation mirrors the quick-start recipe described in the project
specification. Replace the synthetic data generation logic with the pipeline
that loads your curated feature table.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "onnx"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = MODELS_DIR / "meta.joblib"

FEATURE_COLUMNS = ["feature_temp", "feature_rain", "feature_is_holiday"]
LABELS = {0: "Green", 1: "Yellow", 2: "Red"}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def generate_demo_frame() -> pd.DataFrame:
    """Create a synthetic feature table for demonstration purposes."""

    rng = np.random.default_rng(42)
    size = 5_000
    return pd.DataFrame(
        {
            "station_id": rng.choice(["S001", "S002", "S003"], size=size),
            "slot_time": pd.date_range("2025-09-01", periods=size, freq="30min"),
            "feature_temp": rng.normal(20, 8, size=size),
            "feature_rain": rng.random(size=size),
            "feature_is_holiday": rng.choice([0, 1], size=size, p=[0.9, 0.1]),
            "y": rng.choice([0, 1, 2], size=size, p=[0.6, 0.3, 0.1]),
        }
    )


def load_training_frame() -> pd.DataFrame:
    """Load the training data frame.

    Replace this helper with your real data loader (e.g. CSV, database read,
    feature store query). The demo loader falls back to the synthetic frame
    so the script runs out of the box.
    """

    data_path = BASE_DIR / "features.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    return generate_demo_frame()


# ---------------------------------------------------------------------------
# Model training routines
# ---------------------------------------------------------------------------
def train_logistic_classifier(X: pd.DataFrame, y: pd.Series) -> tuple[StandardScaler, LogisticRegression]:
    """Fit a standardised one-vs-rest logistic regression classifier."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(
        max_iter=1_000,
        multi_class="ovr",
        class_weight="balanced",
    )
    lr.fit(X_scaled, y)
    return scaler, lr


def train_calibrated_logistic(X: pd.DataFrame, y: pd.Series) -> CalibratedClassifierCV:
    """Fit an isotonic calibrated logistic regression model."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base = LogisticRegression(
        max_iter=1_000,
        multi_class="ovr",
        class_weight="balanced",
    )
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calibrated.fit(X_scaled, y)
    joblib.dump({"scaler": scaler, "model": calibrated}, MODELS_DIR / "lr_calibrated.joblib")
    return calibrated


def train_lightgbm_classifier(X: pd.DataFrame, y: pd.Series) -> LGBMClassifier:
    """Fit the LightGBM multi-class classifier."""

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multiclass",
        class_weight="balanced",
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# ONNX export utilities
# ---------------------------------------------------------------------------
def export_sklearn_model(model, *, path: Path, input_dim: int) -> None:
    """Export a scikit-learn compatible estimator to ONNX."""

    initial_type = [("input", FloatTensorType([None, input_dim]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    path.write_bytes(onnx_model.SerializeToString())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_training_frame()

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[FEATURE_COLUMNS].astype(np.float32)
    y = df["y"].astype(int)

    scaler, lr = train_logistic_classifier(X, y)
    export_sklearn_model(lr, path=MODELS_DIR / "logistic_v1.onnx", input_dim=X.shape[1])

    train_calibrated_logistic(X, y)

    lgbm = train_lightgbm_classifier(X, y)
    export_sklearn_model(lgbm, path=MODELS_DIR / "lgbm_v1.onnx", input_dim=X.shape[1])

    meta = {
        "features": FEATURE_COLUMNS,
        "classes": list(LABELS.keys()),
        "label_map": LABELS,
    }
    joblib.dump(meta, META_PATH)

    print("Exported logistic_v1.onnx, lgbm_v1.onnx, and meta.joblib")


if __name__ == "__main__":
    main()
