# Pearls AQI Predictor — Project Analysis & Completion Guide

**Developed by: Maham Ahmed · Bachelor of Data Science · 2026**  
**City: Karachi, Pakistan | Model: Random Forest | Dataset: Aug 2022 – Dec 2024**

---

## 1. What You've Built (Completed Components)

### ✅ Data Pipeline (Complete)
| Script | What it does | Status |
|--------|-------------|--------|
| `historic_weather.py` | Pulls 2022-2024 daily weather from Open-Meteo archive API | ✅ Done |
| `historic_air_quality.py` | Loads air_quality_historical.csv, extracts pollutants + US AQI | ✅ Done |
| `merge_dataset.py` | Inner-joins weather + air quality on date → historical_karachi_dataset.csv | ✅ Done |
| `feature_engineering.py` | Adds time features, lag-1/2/3, rolling 3- and 7-day means, rain_flag, temp×humidity | ✅ Done |
| `forecast_weather.py` | Open-Meteo 4-day forecast (temperature, humidity, pressure, wind, rain) | ✅ Done |
| `forecast_air_quality.py` | OpenWeather current + forecast air pollution (PM2.5, PM10, O₃, CO, NO₂, SO₂) | ✅ Done |
| `forecast_prediction.py` | Recursive 3-day AQI forecast using trained model + lag propagation | ✅ Done |
| `refresh_data.py` | Orchestrates forecast_weather → forecast_air_quality → forecast_prediction | ✅ Done |

### ✅ ML Training Pipeline (Complete)
| Script | What it does | Status |
|--------|-------------|--------|
| `train_model.py` | RandomForest (200 trees, depth 12), 80/20 split, MAE/RMSE/R² eval, versioned model registry | ✅ Done |
| `eda.py` | AQI distribution, trend, temperature trend, PM2.5 scatter, correlation heatmap, boxplot | ✅ Done |
| `feature_importance.py` | Extracts and plots RF feature importances → data/eda/feature_importance.png | ✅ Done |
| `shap_explain.py` | SHAP TreeExplainer, summary plot, bar plot, waterfall → data/eda/shap_*.png | ✅ Done |

### ✅ Model Performance (Excellent)
```
MAE   :  5.55   (average prediction error is only ~5 AQI points)
RMSE  :  8.50
R²    :  0.943  (model explains 94.3% of AQI variance — very strong)
```

### ✅ Dataset
- **884 daily records** spanning Aug 2022 – Dec 2024 (~29 months)
- **26 features** after engineering (13 raw + 13 derived)
- 4 AQI nulls and 3 PM2.5 nulls present in raw merge (cleaned by `dropna` in feature_engineering.py)

### ✅ Streamlit Dashboard (Two versions)
- `app.py` — original dashboard (static SHAP/EDA PNG images, purple/pink theme)
- `h.py` — upgraded dashboard (live OpenWeather API, Plotly interactive charts, dark theme, model registry integration, real-time pollutant display)

**`h.py` is the production-ready version** and should be renamed `app.py`.

---

## 2. What's Missing vs Project Requirements

### ❌ MISSING: GitHub Actions CI/CD (High Priority)
**Requirement:** *"Feature pipeline runs every hour automatically. Training pipeline runs daily."*  
No `.github/workflows/` directory exists. You need two workflow files.

### ❌ MISSING: `requirements.txt`
No dependency file exists. Project cannot be reproduced or deployed without it.

### ❌ MISSING: Feature Store Integration (Hopsworks or Vertex AI)
**Requirement:** *"Store processed features in Feature Store (Hopsworks or Vertex AI)"*  
Currently features are stored as local CSV files only. For full marks, a lightweight Hopsworks free-tier integration should be added or clearly noted as a design decision.

### ❌ MISSING: Flask/FastAPI API Layer
**Requirement:** *"Display interactive dashboard with Streamlit/Gradio and Flask/FastAPI"*  
You have Streamlit but no Flask REST API. A simple Flask endpoint wrapping the model prediction is required.

### ⚠️ PARTIAL: TensorFlow / Deep Learning Model
**Requirement:** *"Experiment with various ML models (Random Forest, Ridge Regression, TensorFlow/PyTorch)"*  
Only Random Forest is implemented. Ridge Regression and a TensorFlow LSTM (suitable for time-series AQI) should be added and compared.

### ⚠️ PARTIAL: `merge_dataset.py` has a bug
`dropna(subset=["AQI"])` happens AFTER the CSV is already saved. The saved CSV still contains nulls. Fix: move `dropna` before `to_csv`.

### ⚠️ PARTIAL: `h.py` naming
`h.py` should be renamed to something descriptive (`app_v2.py` or replace `app.py`) so Streamlit knows which file to run.

### ⚠️ PARTIAL: AQICN API unused
Both API keys are in `.env` (OpenWeather + AQICN) but AQICN is never called. The requirement lists AQICN as a primary source.

---

## 3. Files to Create (Complete Code)

### File 1: `requirements.txt`
```
streamlit==1.35.0
pandas==2.2.2
scikit-learn==1.5.0
joblib==1.4.2
plotly==5.22.0
requests==2.32.3
python-dotenv==1.0.1
shap==0.45.1
matplotlib==3.9.0
numpy==1.26.4
tensorflow==2.16.1
flask==3.0.3
```

### File 2: `.github/workflows/feature_pipeline.yml`
*(Runs every hour — fetches live weather + air quality + produces forecast)*

### File 3: `.github/workflows/training_pipeline.yml`
*(Runs daily at midnight — retrains model on accumulated data)*

### File 4: `api.py`
*(Flask REST API exposing /predict and /forecast endpoints)*

### File 5: `train_model_comparison.py`
*(Trains RF + Ridge + TensorFlow LSTM, compares metrics, saves best to registry)*

---

## 4. Pipeline Execution Order (Full Run from Scratch)

```
Step 1 — Historic data
  python historic_weather.py         # → data/processed/daily_weather.csv
  python historic_air_quality.py     # → data/processed/daily_air_quality.csv
  python merge_dataset.py            # → data/processed/historical_karachi_dataset.csv

Step 2 — Feature engineering
  python feature_engineering.py      # → data/processed/final_features.csv

Step 3 — EDA + explainability
  python eda.py                      # → data/eda/*.png
  python train_model.py              # → data/models/random_forest_v*.pkl
  python feature_importance.py       # → data/eda/feature_importance.png
  python shap_explain.py             # → data/eda/shap_*.png

Step 4 — Live forecast (runs on refresh)
  python forecast_weather.py         # → data/processed/weather_forecast.csv
  python forecast_air_quality.py     # → data/processed/air_quality_forecast.csv
  python forecast_prediction.py      # → data/processed/aqi_forecast.csv

Step 5 — Dashboard
  streamlit run h.py                 # or: streamlit run app.py
```

---

## 5. Key Issues to Fix Before Submission

### Bug Fix 1 — `merge_dataset.py` (save after cleaning)
Move the `dropna` call BEFORE `to_csv` so the saved CSV doesn't contain null AQI rows:
```python
merged_df = pd.merge(weather, air, on="date", how="inner")
merged_df = merged_df.dropna(subset=["AQI"]).reset_index(drop=True)   # ← move here
merged_df.to_csv("data/processed/historical_karachi_dataset.csv", index=False)
```

### Bug Fix 2 — `h.py` handles 3-day forecast correctly
In `h.py`, the forecast skips index 0 (today) and shows indices 1–3:
```python
future_forecast = forecast.iloc[1:4]   # correct: days 2, 3, 4
```
But `app.py` shows indices 0–2 (today + 2 more). Decide which is correct and be consistent.

### Enhancement — Add AQI change rate feature
The project description mentions "AQI change rate" as a derived feature. Add to `feature_engineering.py`:
```python
df["AQI_change_rate"] = df["AQI"].diff()   # day-over-day delta
```

### Enhancement — Alert system in `h.py`
The alert logic already exists (`actual_aqi <= 50` etc.). Consider adding an email/webhook alert when AQI > 150, which can be triggered from the GitHub Actions hourly run.

---

## 6. Summary Table: Requirements vs Completion

| Requirement | Status | Location |
|---|---|---|
| Fetch weather + pollutant data from APIs | ✅ Complete | `forecast_weather.py`, `forecast_air_quality.py` |
| Time-based + derived features | ✅ Complete | `feature_engineering.py` |
| Historical data backfill (2022–2024) | ✅ Complete | `historic_weather.py`, `merge_dataset.py` |
| Random Forest model | ✅ Complete | `train_model.py` |
| RMSE / MAE / R² evaluation | ✅ Complete | `train_model.py` |
| Model registry (versioned) | ✅ Complete | `data/registry/model_registry.csv` |
| 3-day recursive AQI forecast | ✅ Complete | `forecast_prediction.py` |
| Streamlit dashboard | ✅ Complete | `h.py` (production), `app.py` (v1) |
| SHAP explainability | ✅ Complete | `shap_explain.py`, `h.py` inline Plotly |
| AQI health alerts | ✅ Complete | `h.py` |
| EDA (trends, distributions, heatmap) | ✅ Complete | `eda.py` |
| Feature importance | ✅ Complete | `feature_importance.py` |
| Feature Store (Hopsworks/Vertex AI) | ❌ Missing | Not implemented |
| GitHub Actions / Airflow CI-CD | ❌ Missing | No `.github/` folder |
| Flask / FastAPI REST endpoint | ❌ Missing | No `api.py` |
| `requirements.txt` | ❌ Missing | No file |
| TensorFlow / Ridge model comparison | ⚠️ Partial | Only RF trained |
| AQICN API usage | ⚠️ Partial | Key exists, not called |
| Dataset null-safety in merge | ⚠️ Bug | Fixed order needed |
