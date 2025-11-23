from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, date
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional
import requests

KST = ZoneInfo("Asia/Seoul")

# ===== 설정 =====
OPENWEATHER_API_KEY = "1d5cc2adeb9583f7289f0d588a2f4964"  # 실제 키로 교체
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


# ===== 해수욕장 정보 =====
class BeachInfo(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    popularity_weight: float  # 인기도 가중치


BEACHES: Dict[str, BeachInfo] = {
    "haeundae": BeachInfo(
        id="haeundae",
        name="해운대해수욕장",
        lat=35.158698,
        lon=129.160384,
        popularity_weight=1.0,
    ),
    "gwangalli": BeachInfo(
        id="gwangalli",
        name="광안리해수욕장",
        lat=35.153208,
        lon=129.118386,
        popularity_weight=0.85,
    ),
}

# ===== 혼잡도 로직 =====
def time_factor(hour: int) -> float:
    # 시간대별 가중치
    if 6 <= hour < 10:
        return 0.4
    elif 10 <= hour < 13:
        return 0.8
    elif 13 <= hour < 17:
        return 1.0
    elif 17 <= hour < 20:
        return 0.7
    else:
        return 0.2


def season_factor(month: int) -> float:
    # 성수기: 7~8
    if month in [7, 8]:
        return 1.0      # 성수기
    # 준성수기: 6, 9
    elif month in [6, 9]:
        return 0.8
    # 어중간 시즌: 5, 10
    elif month in [5, 10]:
        return 0.6
    # 한겨울/초봄: 나머지
    else:
        return 0.5      # 기존 0.3 → 0.5로 완화


def weekday_factor(is_weekend_or_holiday: bool) -> float:
    # 공휴일 API 붙이기 전까지는 토/일만 주말 처리
    return 1.0 if is_weekend_or_holiday else 0.8


def weather_factor(temp_c: float, rain_mm: float, wind_mps: float) -> float:
    # 온도
    if temp_c < 10:          # 진짜 추울 때
        temp_score = 0.5
    elif temp_c < 20:        # 약간 쌀쌀
        temp_score = 0.7
    elif 20 <= temp_c <= 27: # 딱 좋음
        temp_score = 1.0
    else:                    # 더움
        temp_score = 0.8

    # 비
    rain_score = 0.3 if rain_mm > 1 else 1.0

    # 바람
    wind_score = 0.7 if wind_mps > 10 else 1.0

    return temp_score * rain_score * wind_score



def raw_congestion_score(
    beach: BeachInfo,
    dt_kst: datetime,
    temp_c: float,
    rain_mm: float,
    wind_mps: float,
    is_holiday_or_weekend: bool,
) -> float:
    base = beach.popularity_weight
    t = time_factor(dt_kst.hour)
    s = season_factor(dt_kst.month)
    w = weekday_factor(is_holiday_or_weekend)
    we = weather_factor(temp_c, rain_mm, wind_mps)
    return base * t * s * w * we


def normalize_to_0_100(score: float, max_score: float = 1.0) -> float:
    # 최대 이론치 기준 0~100 정규화
    x = min(score / max_score, 1.5)  # 상한 150%
    return round(x * 100, 1)


def level_from_pct(pct: float) -> str:
    if pct < 30:
        return "여유"
    elif pct < 60:
        return "보통"
    elif pct < 90:
        return "혼잡"
    else:
        return "매우 혼잡"


# ===== 날씨 조회 (선택지 B: 에러 내용 그대로 노출) =====
from fastapi import HTTPException
from pydantic import BaseModel
import requests

class Weather(BaseModel):
    temp_c: float
    rain_mm: float
    wind_mps: float


def fetch_weather_for_beach(beach: BeachInfo) -> Weather:
    """
    - OPENWEATHER_API_KEY만 검사해서 없으면 에러를 내고,
    - 실제 요청은:
        - 환경 프록시(HTTP_PROXY, HTTPS_PROXY 등) 무시
        - timeout 조금 넉넉하게 (연결 5초 → 10초)
    로 보낸다.
    """

    # 1) API 키 미설정이면 진짜 설정 에러니까 그대로 에러
    if OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY" or not OPENWEATHER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail={
                "msg": "OPENWEATHER_API_KEY가 설정되지 않았습니다. main.py 상단의 OPENWEATHER_API_KEY 값을 실제 키로 바꾸세요.",
            },
        )

    url = OPENWEATHER_BASE_URL
    params = {
        "lat": beach.lat,
        "lon": beach.lon,
        "appid": OPENWEATHER_API_KEY,
    }

    try:
        # ★ 핵심: 환경 프록시 무시
        session = requests.Session()
        session.trust_env = False  # HTTP_PROXY, HTTPS_PROXY 등 무시

        # connect/read 타임아웃을 분리해서 조금 넉넉하게
        resp = session.get(url, params=params, timeout=(5, 10))

        # 디버깅용 로그
        print("[OpenWeather] called:", resp.url, "status:", resp.status_code)

        if resp.status_code != 200:
            # 상태 코드가 이상하면 서버 응답을 콘솔에 찍고, 일단 더미값으로 진행
            print("[OpenWeather] non-200 response body:", resp.text[:500])
            return Weather(temp_c=26.0, rain_mm=0.0, wind_mps=3.0)

        data = resp.json()
        temp_k = data["main"]["temp"]
        temp_c = temp_k - 273.15

        rain_mm = 0.0
        if "rain" in data:
            rain_mm = data["rain"].get("1h", data["rain"].get("3h", 0.0))

        wind_mps = data.get("wind", {}).get("speed", 0.0)

        return Weather(temp_c=temp_c, rain_mm=rain_mm, wind_mps=wind_mps)

    except Exception as e:
        # requests 쪽에서 어떤 예외가 나도 서버 콘솔에 찍고 더미 날씨로 fallback
        print("[OpenWeather] exception:", repr(e))
        return Weather(temp_c=26.0, rain_mm=0.0, wind_mps=3.0)

# ===== 응답 스키마 =====
class InputContext(BaseModel):
    timestamp: datetime
    weather: "Weather"
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


def is_weekend_or_holiday(dt: datetime) -> bool:
    # 토/일만 true
    return dt.weekday() >= 5


# ===== FastAPI 앱 인스턴스 (이게 꼭 필요!!) =====
app = FastAPI(title="Beach Congestion API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/beaches", response_model=List[BeachInfo])
def list_beaches():
    return list(BEACHES.values())


@app.get("/congestion/current", response_model=CongestionResult)
def get_current_congestion(beach_id: str):
    if beach_id not in BEACHES:
        raise HTTPException(status_code=404, detail="해수욕장을 찾을 수 없음")
    beach = BEACHES[beach_id]

    now_kst = datetime.now(tz=KST)
    weather = fetch_weather_for_beach(beach)
    flag = is_weekend_or_holiday(now_kst)

    # 1) 룰 기반 혼잡도 계산
    raw = raw_congestion_score(
        beach=beach,
        dt_kst=now_kst,
        temp_c=weather.temp_c,
        rain_mm=weather.rain_mm,
        wind_mps=weather.wind_mps,
        is_holiday_or_weekend=flag,
    )
    pct = normalize_to_0_100(raw)
    level = level_from_pct(pct)

    # 2) input 블록 구성
    input_ctx = InputContext(
        timestamp=now_kst,
        weather=weather,
        is_weekend_or_holiday=flag,
    )

    # 3) rule_based 블록 구성
    rule_based_output = RuleBasedOutput(
        score_raw=raw,
        score_pct=pct,
        level=level,
    )

    # 4) ai 블록은 아직 없으니 None (나중에 모델 붙이면서 채우면 됨)
    ai_output = None
    # 예: 나중에는 이런 식으로:
    # ai_output = AiOutput(
    #     score_raw=ai_raw,
    #     score_pct=ai_pct,
    #     level=ai_level,
    #     model_version="ai-v1.0.0",
    # )

    return CongestionResult(
        beach_id=beach.id,
        beach_name=beach.name,
        input=input_ctx,
        rule_based=rule_based_output,
        ai=ai_output,
    )



@app.get("/congestion/hourly", response_model=HourlyCongestionResult)
def get_hourly_congestion(beach_id: str, target_date: Optional[date] = None):
    if beach_id not in BEACHES:
        raise HTTPException(status_code=404, detail="해수욕장을 찾을 수 없음")
    beach = BEACHES[beach_id]

    # 기본은 오늘 날짜
    if target_date is None:
        target_date = datetime.now(tz=KST).date()

    # 디버깅 단순화: target_date 기준으로, 날씨는 현재 값 하나만 사용
    weather = fetch_weather_for_beach(beach)

    points: List[HourlyPoint] = []
    for hour in range(0, 24):
        dt_kst = datetime(
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
            hour=hour,
            minute=0,
            second=0,
            tzinfo=KST,
        )
        flag = is_weekend_or_holiday(dt_kst)
        raw = raw_congestion_score(
            beach=beach,
            dt_kst=dt_kst,
            temp_c=weather.temp_c,
            rain_mm=weather.rain_mm,
            wind_mps=weather.wind_mps,
            is_holiday_or_weekend=flag,
        )
        pct = normalize_to_0_100(raw)
        level = level_from_pct(pct)

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
        date=target_date,
        points=points,
    )


if __name__ == "__main__":
    import uvicorn
    # 이렇게 실행해도 되고, 터미널에서 `uvicorn main:app --reload` 해도 된다.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
