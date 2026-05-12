from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel


class BeachInfo(BaseModel):
    id: str
    name: str
    status: str
    lat: float
    lon: float
    popularity_weight: float


class Weather(BaseModel):
    temp_c: float
    rain_mm: float
    wind_mps: float


class InputContext(BaseModel):
    timestamp: datetime
    weather: Weather
    is_weekend_or_holiday: bool


class RuleBasedOutput(BaseModel):
    score_raw: float
    score_pct: float
    level: str


class AiOutput(BaseModel):
    score_raw: float
    score_pct: float
    level: str
    model_version: Optional[str] = None


class CongestionResult(BaseModel):
    beach_id: str
    beach_name: str
    input: InputContext
    rule_based: RuleBasedOutput
    ai: Optional[AiOutput] = None


class HourlyPoint(BaseModel):
    hour: int
    score_raw: float
    score_pct: float
    level: str


class HourlyCongestionResult(BaseModel):
    beach_id: str
    beach_name: str
    date: date
    points: List[HourlyPoint]
