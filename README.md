# Beach Congestion AI Server

부산 해수욕장(해운대, 광안리) 혼잡도를 예측하는 AI 서버입니다.
[메인 앱](https://github.com/Beach-complex/Beach_complex)(Spring Boot)의 `CongestionClient`가 이 서버에 HTTP GET 요청을 보내고, 응답을 `CongestionCurrentResponse`로 역직렬화하여 사용합니다.
현재는 룰 기반 가중치 계산으로 동작하며, 추후 AI 모델로 교체 예정입니다.

## 기술 스택

- Python 3.11+
- FastAPI
- Pydantic
- OpenWeatherMap API

## 설치 및 실행

```bash
# 패키지 설치
pip install fastapi uvicorn requests

# 서버 실행
uvicorn main:app --reload
```

서버 기본 주소: `http://127.0.0.1:8000`

## API 키 설정

`main.py` 상단의 `OPENWEATHER_API_KEY` 값을 실제 키로 교체하세요.

```python
OPENWEATHER_API_KEY = "your_api_key_here"
```

## 엔드포인트

### `GET /health`
서버 상태 확인

```json
{ "status": "ok" }
```

---

### `GET /beaches`
지원하는 해수욕장 목록 반환

```json
[
  { "id": "haeundae", "name": "해운대해수욕장", "lat": 35.158698, "lon": 129.160384, "popularity_weight": 1.0 },
  { "id": "gwangalli", "name": "광안리해수욕장", "lat": 35.153208, "lon": 129.118386, "popularity_weight": 0.85 }
]
```

---

### `GET /congestion/current?beach_id={id}`
현재 시각 기준 혼잡도 반환

| 파라미터 | 필수 | 설명 |
|---|---|---|
| `beach_id` | O | `haeundae` 또는 `gwangalli` |

```json
{
  "beach_id": "haeundae",
  "beach_name": "해운대해수욕장",
  "input": {
    "timestamp": "2024-08-01T14:00:00+09:00",
    "weather": { "temp_c": 29.0, "rain_mm": 0.0, "wind_mps": 2.5 },
    "is_weekend_or_holiday": true
  },
  "rule_based": {
    "score_raw": 0.85,
    "score_pct": 85.0,
    "level": "혼잡"
  },
  "ai": null
}
```

---

### `GET /congestion/hourly?beach_id={id}&target_date={date}`
특정 날짜의 시간대별(0~23시) 혼잡도 반환

| 파라미터 | 필수 | 설명 |
|---|---|---|
| `beach_id` | O | `haeundae` 또는 `gwangalli` |
| `target_date` | X | `YYYY-MM-DD` 형식, 생략 시 오늘 |

## 혼잡도 등급

| 범위 | 등급 |
|---|---|
| 0 ~ 29 | 여유 |
| 30 ~ 59 | 보통 |
| 60 ~ 89 | 혼잡 |
| 90 이상 | 매우 혼잡 |

## 혼잡도 계산 방식

룰 기반 가중치 곱셈 방식입니다.

```
혼잡도 = 인기도 × 시간대 × 계절 × 주말여부 × 날씨
```

AI 모델 기반 예측은 추후 `ai` 블록에 추가 예정입니다.

## Swagger UI

서버 실행 후 아래 주소에서 API 명세 확인 및 테스트 가능합니다.

```
http://127.0.0.1:8000/docs
```
