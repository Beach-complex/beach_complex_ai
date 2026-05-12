from typing import Any

import requests

from app.schemas import BeachInfo, Weather
from app.secret_provider import OpenWeatherApiKeyProvider


class OpenWeatherClient:
    def __init__(
        self,
        api_key_provider: OpenWeatherApiKeyProvider,
        base_url: str,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key_provider = api_key_provider
        self._base_url = base_url
        self._session = session or requests.Session()
        self._session.trust_env = False

    def fetch_for_beach(self, beach: BeachInfo) -> Weather:
        api_key = self._api_key_provider.get_api_key()
        params = {
            "lat": beach.lat,
            "lon": beach.lon,
            "appid": api_key,
        }

        try:
            response = self._session.get(
                self._base_url,
                params=params,
                timeout=(5, 10),
            )
            print("[OpenWeather] called:", response.url, "status:", response.status_code)

            if response.status_code != 200:
                print("[OpenWeather] non-200 response body:", response.text[:500])
                return self._fallback_weather()

            return self._parse_weather(response.json())

        except Exception as e:
            print("[OpenWeather] exception:", repr(e))
            return self._fallback_weather()

    @staticmethod
    def _parse_weather(data: dict[str, Any]) -> Weather:
        temp_k = data["main"]["temp"]
        temp_c = temp_k - 273.15

        rain_mm = 0.0
        if "rain" in data:
            rain_mm = data["rain"].get("1h", data["rain"].get("3h", 0.0))

        wind_mps = data.get("wind", {}).get("speed", 0.0)

        return Weather(temp_c=temp_c, rain_mm=rain_mm, wind_mps=wind_mps)

    @staticmethod
    def _fallback_weather() -> Weather:
        return Weather(temp_c=26.0, rain_mm=0.0, wind_mps=3.0)
