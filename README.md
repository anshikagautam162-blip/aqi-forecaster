# 🌬️ AQI Forecaster — Global Air Quality Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aqi-forecaster-3qxqouxeps5dwawf2wcfdc.streamlit.app)

Real-time global air quality monitoring + 72-hour PM2.5 forecasting using XGBoost, LSTM+Attention, and Causal ML (DoWhy).

**🔴 Live Demo:** [aqi-forecaster.streamlit.app](https://aqi-forecaster-3qxqouxeps5dwawf2wcfdc.streamlit.app)

---

## 🎯 Key Results

| Model | 24h MAE | 48h MAE | 72h MAE | R² |
|---|---|---|---|---|
| Persistence Baseline | 9.584 | 11.195 | 12.304 | — |
| XGBoost (7 cities) | 9.912 | 11.078 | 12.591 | 0.762 |
| **XGBoost 37 cities ⭐** | **8.659** | **9.794** | **10.195** | **0.721** |
| LSTM+Attention (37 cities) | 9.320 | 9.736 | 10.201 | 0.645 |

---

## 🏗️ Architecture
Raw Data (WAQI + Open-Meteo)
↓
Feature Engineering (129 features: lags, rolling, cyclical)
↓
┌────────────┐ ┌──────────────────┐
│ XGBoost │ │ LSTM+Attention │
│ (primary) │ │ (secondary) │
└────────────┘ └──────────────────┘
↓ ↓
24h/48h/72h PM2.5 Forecast
↓
SHAP Explainability + DoWhy Causal ML
↓
Streamlit Dashboard (Live)

---

## 📊 Dataset

- **37 global cities** — Delhi, Mumbai, Beijing, London, Tokyo, Cairo, Dubai + 30 more
- **325,008 hourly records** — 1 year (Oct 2025 – Oct 2026)
- **129 features** — weather, pollution lags, rolling averages, cyclical time encoding
- **Sources:** WAQI API, Open-Meteo, CPCB India (official govt data)
- **Split:** 83% train / 8% val / 8% test — time-based, no data leakage

---

## 🔬 Causal ML Findings (DoWhy)

| Factor | Causal Effect | Significance |
|---|---|---|
| 🌧️ Precipitation | **-6.645 µg/m³ per mm** | p<0.05 ✅ Real |
| ⚗️ NO2 | **+1.079 µg/m³** | p<0.05 ✅ Real |
| 🌡️ Temperature | +0.042 µg/m³ per °C | Indirect |
| 💨 Wind speed | +3.107 µg/m³ | ⚠️ Confounded |

**Key insight:** Standard ML shows correlation. Causal ML (DoWhy + PC Algorithm + Refutation) proves **what actually causes PM2.5 to change.** Wind speed appears positive globally due to sea salt confounding — city-stratified analysis reveals Geneva shows true negative effect (-0.20).

---

## 🛠️ Tech Stack
Data: WAQI API, Open-Meteo, CPCB India
ML: XGBoost (Optuna-tuned), PyTorch LSTM+Attention
Causal: DoWhy, PC Algorithm, Refutation Tests
Explainability: SHAP (TreeSHAP)
Dashboard: Streamlit, Plotly, Folium
Training: Kaggle GPU T4 (LSTM)

---

## 📁 Project Structure
aqi_forecaster/
app.py ← Streamlit dashboard
requirements.txt
notebooks/
01_data_pipeline.ipynb ← WAQI live data
02_data_collection.ipynb← Weather + AQI historical
03_eda.ipynb ← EDA + feature engineering
04_models.ipynb ← XGBoost + baselines
05_causal_shap.ipynb ← DoWhy + SHAP
models/
xgb_24h_37cities.pkl
xgb_48h_37cities.pkl
xgb_72h_37cities.pkl
results/
causal_effects.json
causal_graph.png
shap_plots/
model_comparison_final.csv
docs/
causal_findings.md

---

## 🚀 Run Locally

```bash
git clone https://github.com/anshikagautam162-blip/aqi-forecaster
cd aqi-forecaster
pip install -r requirements.txt
# Add WAQI_TOKEN to .env file
streamlit run app.py
```

---

## 📄 Research Report

See [`docs/causal_findings.md`](docs/causal_findings.md) for detailed causal analysis findings.

---

## 👩‍💻 Author

Demonstrates: end-to-end ML pipeline, causal inference, deep learning, production deployment.