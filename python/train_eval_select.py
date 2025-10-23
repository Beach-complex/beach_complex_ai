"""Train, evaluate, and select the best congestion classifier."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "onnx"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {0: "Green", 1: "Yellow", 2: "Red"}


def load_features() -> pd.DataFrame:
    path = BASE_DIR / "features.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["slot_time"])

    rng = np.random.default_rng(13)
    size = 4_000
    return pd.DataFrame(
        {
            "station_id": rng.choice(["S001", "S002", "S003"], size=size),
            "slot_time": pd.date_range("2025-06-01", periods=size, freq="30min"),
            "feature_temp": rng.normal(24, 6, size=size),
            "feature_rain": rng.random(size=size),
            "feature_is_holiday": rng.choice([0, 1], size=size, p=[0.9, 0.1]),
            "y": rng.choice([0, 1, 2], size=size, p=[0.65, 0.25, 0.1]),
        }
    )


def export_model(model, path: Path, input_dim: int) -> None:
    initial_type = [("input", FloatTensorType([None, input_dim]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    path.write_bytes(onnx_model.SerializeToString())


def evaluate_logistic(X_train, y_train, X_valid, y_valid):
    scaler = StandardScaler().fit(X_train)
    lr = LogisticRegression(
        max_iter=1_000,
        multi_class="ovr",
        class_weight="balanced",
    ).fit(scaler.transform(X_train), y_train)

    preds = lr.predict(scaler.transform(X_valid))
    score = f1_score(y_valid, preds, average="macro")

    export_model(lr, MODELS_DIR / "logistic_v1.onnx", X_train.shape[1])
    joblib.dump({"scaler": scaler}, MODELS_DIR / "logistic_scaler.joblib")
    return score


def evaluate_lightgbm(X_train, y_train, X_valid, y_valid):
    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multiclass",
        class_weight="balanced",
    ).fit(X_train, y_train)

    preds = model.predict(X_valid)
    score = f1_score(y_valid, preds, average="macro")

    export_model(model, MODELS_DIR / "lgbm_v1.onnx", X_train.shape[1])
    return score


def main() -> None:
    df = load_features()
    feature_cols = [c for c in df.columns if c.startswith("feature_")]

    if not feature_cols:
        raise ValueError("No feature_ columns found in the dataset")

    df = df.sort_values("slot_time")
    split_threshold = df["slot_time"].max() - pd.Timedelta(weeks=4)
    train_mask = df["slot_time"] < split_threshold

    X_train = df.loc[train_mask, feature_cols].astype(np.float32)
    y_train = df.loc[train_mask, "y"].astype(int)
    X_valid = df.loc[~train_mask, feature_cols].astype(np.float32)
    y_valid = df.loc[~train_mask, "y"].astype(int)

    reports = {}
    reports["logistic"] = {"macro_f1": evaluate_logistic(X_train, y_train, X_valid, y_valid)}
    reports["lightgbm"] = {"macro_f1": evaluate_lightgbm(X_train, y_train, X_valid, y_valid)}

    best_model = max(reports.items(), key=lambda item: item[1]["macro_f1"])[0]
    active_path = MODELS_DIR / "active.onnx"
    if best_model == "logistic":
        shutil.copy(MODELS_DIR / "logistic_v1.onnx", active_path)
    else:
        shutil.copy(MODELS_DIR / "lgbm_v1.onnx", active_path)

    meta = {
        "features": feature_cols,
        "label_map": LABELS,
        "selected": best_model,
        "reports": reports,
    }
    (BASE_DIR / "reports.metrics.json").write_text(json.dumps(meta, indent=2))
    joblib.dump(meta, MODELS_DIR / "meta.joblib")

    print(f"Selected model: {best_model}")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
