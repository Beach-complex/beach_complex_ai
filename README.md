# Seoul/Busan Subway Congestion Prediction Stack

This repository packages a tri-model congestion prediction workflow together
with a Spring Boot integration layer. The implementation follows the recipe
outlined in the project brief and provides ready-to-run Docker images for the
Java API and Prophet sidecar.

## Repository layout

```
models/
  onnx/                 # ONNX classifiers + metadata (placeholders tracked via .gitkeep)
  prophet/              # Serialized Prophet models per station (joblib)
python/
  app_prophet.py        # FastAPI application serving Prophet forecasts
  train_classifiers.py  # Train logistic/LGBM models and export ONNX artifacts
  train_prophet.py      # Optional Prophet offline training script
  train_eval_select.py  # Train→evaluate→select workflow for the classifiers
  requirements.txt      # Pinned Python dependency set
java-api/
  build.gradle          # Spring Boot + ONNX Runtime configuration
  Dockerfile            # Multi-stage build producing the API container image
  src/main/java         # Java sources (controller, service, ONNX helper, main)
  src/main/resources    # Application configuration
```

## Quick start

1. **Install Python dependencies** and train/export the classification models:

   ```bash
   pip install -r python/requirements.txt
   python python/train_classifiers.py
   # or
   python python/train_eval_select.py
   ```

   The scripts emit ONNX artifacts and accompanying metadata under
   `models/onnx/`.

2. **(Optional) Train Prophet models** per station:

   ```bash
   python python/train_prophet.py
   ```

   Drop the resulting `{station_id}.joblib` files inside `models/prophet/`.

3. **Launch the services** via Docker Compose:

   ```bash
   docker-compose up --build
   ```

   * `api-java` exposes the classification endpoint on `:8080`.
   * `prophet-infer` exposes the Prophet forecasts on `:8000`.

4. **Sample requests**:

   ```bash
   curl -X POST http://localhost:8080/v1/classify \
     -H 'Content-Type: application/json' \
     -d '{
       "feature_temp": 27.4,
       "feature_rain": 0.2,
       "feature_is_holiday": 0
     }'

   curl "http://localhost:8080/v1/forecast?stationId=S001&hours=24"
   ```

## Notes

* The Java service expects an `active.onnx` file within `models/onnx/`.
  `train_eval_select.py` copies the best-performing classifier to this name
  automatically.
* The ONNX inference helper discovers the input node name dynamically, so the
  classifier export scripts can freely choose the signature as long as the
  feature ordering matches `meta.joblib`.
* The Prophet Dockerfile performs a warm-up fit during the build to pre-compile
  the Stan model and avoid first-request latency at runtime.
