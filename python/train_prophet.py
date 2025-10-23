"""Offline training script for Prophet models per station."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from prophet import Prophet

BASE_DIR = Path(__file__).resolve().parent.parent
PROPHET_DIR = BASE_DIR / "models" / "prophet"
PROPHET_DIR.mkdir(parents=True, exist_ok=True)


def load_timeseries() -> pd.DataFrame:
    """Load the station-level time series frame.

    The script expects the columns `station_id`, `ds`, and `y`. If no
    user-provided file is present a small synthetic dataset is generated so the
    workflow can be exercised in development environments.
    """

    path = BASE_DIR / "timeseries.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["ds"])

    idx = pd.date_range("2025-01-01", periods=24 * 30, freq="H")
    df = pd.DataFrame({"ds": idx, "y": 70 + (idx.hour / 24 * 60)})
    df["station_id"] = "SYNTH"
    return df


def train_prophet_for_station(station_df: pd.DataFrame) -> Prophet:
    model = Prophet(
        weekly_seasonality=True,
        daily_seasonality=True,
        yearly_seasonality=False,
    )
    model.add_country_holidays(country_name="KR")
    model.fit(station_df[["ds", "y"]].sort_values("ds"))
    return model


def main() -> None:
    df = load_timeseries()
    required_cols = {"station_id", "ds", "y"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Timeseries frame missing columns: {sorted(missing)}")

    for station_id, group in df.groupby("station_id"):
        model = train_prophet_for_station(group)
        out_path = PROPHET_DIR / f"{station_id}.joblib"
        joblib.dump(model, out_path)
        print(f"Saved Prophet model for {station_id}: {out_path}")


if __name__ == "__main__":
    main()
