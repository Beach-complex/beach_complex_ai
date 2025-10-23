FROM python:3.11.9-slim

ENV PIP_NO_CACHE_DIR=1 \
    CMDSTANPY_CACHE_DIR=/root/.cmdstan \
    PROPHET_STARTUP_MSG=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    make \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Optional warmup compile for Prophet to build the Stan model during the image build
RUN python - <<'PY'
import pandas as pd
from prophet import Prophet

df = pd.DataFrame({'ds': pd.date_range('2024-01-01', periods=5, freq='D'), 'y': [0, 1, 0, 1, 0]})
model = Prophet()
model.fit(df)
PY

COPY . .
CMD ["python", "app_prophet.py"]
