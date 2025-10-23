"""FastAPI sidecar that serves Prophet-based congestion forecasts."""

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "prophet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = {"green": 80.0, "red": 130.0}


class ForecastRequest(BaseModel):
    station_id: str = Field(..., description="Target station identifier")
    horizon_hours: int = Field(24, ge=1, le=72, description="Forecast horizon in hours")


class ForecastItem(BaseModel):
    ts: str
    yhat: float
    bucket: str


class ForecastResponse(BaseModel):
    station_id: str
    horizon: int
    items: List[ForecastItem]


app = FastAPI(title="Prophet Forecast API", version="1.0.0")


def to_bucket(value: float) -> str:
    if value <= THRESHOLDS["green"]:
        return "Green"
    if value >= THRESHOLDS["red"]:
        return "Red"
    return "Yellow"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    model_path = MODEL_DIR / f"{req.station_id}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="model_not_found")

    model = joblib.load(model_path)
    future = model.make_future_dataframe(
        periods=req.horizon_hours,
        freq="H",
        include_history=False,
    )
    frame = model.predict(future)[["ds", "yhat"]]

    items = [
        ForecastItem(
            ts=row.ds.isoformat(),
            yhat=float(row.yhat),
            bucket=to_bucket(float(row.yhat)),
        )
        for row in frame.itertuples()
    ]

    return ForecastResponse(
        station_id=req.station_id,
        horizon=req.horizon_hours,
        items=items,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
