from datetime import date
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from mangum import Mangum

from app.beach_repository import BeachRepository
from app.config import settings
from app.congestion_service import (
    BeachNotFoundError,
    CongestionCalculator,
    CongestionService,
)
from app.schemas import BeachInfo, CongestionResult, HourlyCongestionResult
from app.secret_provider import OpenWeatherApiKeyError, OpenWeatherApiKeyProvider
from app.weather_client import OpenWeatherClient

KST = ZoneInfo("Asia/Seoul")
BEACHES_JSON_PATH = Path(__file__).parents[1] / "beaches.json"


def create_congestion_service() -> CongestionService:
    beach_repository = BeachRepository(BEACHES_JSON_PATH)
    api_key_provider = OpenWeatherApiKeyProvider(
        secret_name=settings.openweather_secret_name,
        aws_region=settings.aws_region,
    )
    weather_client = OpenWeatherClient(
        api_key_provider=api_key_provider,
        base_url=settings.openweather_base_url,
    )
    return CongestionService(
        beach_repository=beach_repository,
        weather_client=weather_client,
        calculator=CongestionCalculator(),
        timezone=KST,
    )


def create_app(service: CongestionService | None = None) -> FastAPI:
    app = FastAPI(title="Beach Congestion API", version="0.1.0")
    congestion_service = service or create_congestion_service()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/beaches", response_model=List[BeachInfo])
    def list_beaches():
        return congestion_service.list_beaches()

    @app.get("/congestion/current", response_model=CongestionResult)
    def get_current_congestion(beach_id: str):
        try:
            return congestion_service.get_current(beach_id)
        except BeachNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except OpenWeatherApiKeyError as e:
            raise HTTPException(
                status_code=500,
                detail={"msg": f"OpenWeather API 키 조회 실패: {str(e)}"},
            ) from e

    @app.get("/congestion/hourly", response_model=HourlyCongestionResult)
    def get_hourly_congestion(
        beach_id: str,
        target_date: Optional[date] = None,
    ):
        try:
            return congestion_service.get_hourly(beach_id, target_date)
        except BeachNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except OpenWeatherApiKeyError as e:
            raise HTTPException(
                status_code=500,
                detail={"msg": f"OpenWeather API 키 조회 실패: {str(e)}"},
            ) from e

    return app


app = create_app()
handler = Mangum(app, lifespan="off")
