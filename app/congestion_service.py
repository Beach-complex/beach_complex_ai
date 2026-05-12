from datetime import date, datetime
from zoneinfo import ZoneInfo

from app.beach_repository import BeachRepository
from app.schemas import (
    BeachInfo,
    CongestionResult,
    HourlyCongestionResult,
    HourlyPoint,
    InputContext,
    RuleBasedOutput,
    Weather,
)
from app.weather_client import OpenWeatherClient


class BeachNotFoundError(Exception):
    def __init__(self, beach_id: str) -> None:
        self.beach_id = beach_id
        super().__init__(f"해수욕장을 찾을 수 없음: {beach_id}")


class CongestionCalculator:
    def raw_score(
        self,
        beach: BeachInfo,
        dt_kst: datetime,
        weather: Weather,
        is_holiday_or_weekend: bool,
    ) -> float:
        return (
            beach.popularity_weight
            * self.time_factor(dt_kst.hour)
            * self.season_factor(dt_kst.month)
            * self.weekday_factor(is_holiday_or_weekend)
            * self.weather_factor(weather)
        )

    @staticmethod
    def time_factor(hour: int) -> float:
        if 6 <= hour < 10:
            return 0.4
        if 10 <= hour < 13:
            return 0.8
        if 13 <= hour < 17:
            return 1.0
        if 17 <= hour < 20:
            return 0.7
        return 0.2

    @staticmethod
    def season_factor(month: int) -> float:
        if month in [7, 8]:
            return 1.0
        if month in [6, 9]:
            return 0.8
        if month in [5, 10]:
            return 0.6
        return 0.5

    @staticmethod
    def weekday_factor(is_weekend_or_holiday: bool) -> float:
        return 1.0 if is_weekend_or_holiday else 0.8

    @staticmethod
    def weather_factor(weather: Weather) -> float:
        if weather.temp_c < 10:
            temp_score = 0.5
        elif weather.temp_c < 20:
            temp_score = 0.7
        elif 20 <= weather.temp_c <= 27:
            temp_score = 1.0
        else:
            temp_score = 0.8

        rain_score = 0.3 if weather.rain_mm > 1 else 1.0
        wind_score = 0.7 if weather.wind_mps > 10 else 1.0

        return temp_score * rain_score * wind_score

    @staticmethod
    def normalize_to_0_100(score: float, max_score: float = 1.0) -> float:
        x = min(score / max_score, 1.5)
        return round(x * 100, 1)

    @staticmethod
    def level_from_pct(pct: float) -> str:
        if pct < 30:
            return "여유"
        if pct < 60:
            return "보통"
        if pct < 90:
            return "혼잡"
        return "매우 혼잡"


class CongestionService:
    def __init__(
        self,
        beach_repository: BeachRepository,
        weather_client: OpenWeatherClient,
        calculator: CongestionCalculator,
        timezone: ZoneInfo,
    ) -> None:
        self._beach_repository = beach_repository
        self._weather_client = weather_client
        self._calculator = calculator
        self._timezone = timezone

    def list_beaches(self) -> list[BeachInfo]:
        return self._beach_repository.list_all()

    def get_current(self, beach_id: str) -> CongestionResult:
        beach = self._get_beach_or_raise(beach_id)
        now_kst = datetime.now(tz=self._timezone)
        weather = self._weather_client.fetch_for_beach(beach)
        is_weekend = self.is_weekend_or_holiday(now_kst)
        return self._build_current_result(beach, now_kst, weather, is_weekend)

    def get_hourly(
        self,
        beach_id: str,
        target_date: date | None = None,
    ) -> HourlyCongestionResult:
        beach = self._get_beach_or_raise(beach_id)
        resolved_date = target_date or datetime.now(tz=self._timezone).date()
        weather = self._weather_client.fetch_for_beach(beach)

        points: list[HourlyPoint] = []
        for hour in range(24):
            dt_kst = datetime(
                year=resolved_date.year,
                month=resolved_date.month,
                day=resolved_date.day,
                hour=hour,
                minute=0,
                second=0,
                tzinfo=self._timezone,
            )
            raw, pct, level = self._calculate(beach, dt_kst, weather)
            points.append(
                HourlyPoint(
                    hour=hour,
                    score_raw=raw,
                    score_pct=pct,
                    level=level,
                )
            )

        return HourlyCongestionResult(
            beach_id=beach.id,
            beach_name=beach.name,
            date=resolved_date,
            points=points,
        )

    def _get_beach_or_raise(self, beach_id: str) -> BeachInfo:
        beach = self._beach_repository.get(beach_id)
        if beach is None:
            normalized = self._beach_repository.normalize_code(beach_id)
            raise BeachNotFoundError(normalized)
        return beach

    def _build_current_result(
        self,
        beach: BeachInfo,
        dt_kst: datetime,
        weather: Weather,
        is_weekend: bool,
    ) -> CongestionResult:
        raw, pct, level = self._calculate(beach, dt_kst, weather)

        return CongestionResult(
            beach_id=beach.id,
            beach_name=beach.name,
            input=InputContext(
                timestamp=dt_kst,
                weather=weather,
                is_weekend_or_holiday=is_weekend,
            ),
            rule_based=RuleBasedOutput(
                score_raw=raw,
                score_pct=pct,
                level=level,
            ),
            ai=None,
        )

    def _calculate(
        self,
        beach: BeachInfo,
        dt_kst: datetime,
        weather: Weather,
    ) -> tuple[float, float, str]:
        is_weekend = self.is_weekend_or_holiday(dt_kst)
        raw = self._calculator.raw_score(
            beach=beach,
            dt_kst=dt_kst,
            weather=weather,
            is_holiday_or_weekend=is_weekend,
        )
        pct = self._calculator.normalize_to_0_100(raw)
        level = self._calculator.level_from_pct(pct)
        return raw, pct, level

    @staticmethod
    def is_weekend_or_holiday(dt: datetime) -> bool:
        return dt.weekday() >= 5
