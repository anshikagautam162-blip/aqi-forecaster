# Real-Time Global PM2.5 Forecasting with Causal ML
## AQI Forecaster — Mini Research Report

**Author:** Anshika Gautam  
**Date:** October 2026  
**Live Demo:** https://aqi-forecaster-3qxqouxeps5dwawf2wcfdc.streamlit.app

---

## Abstract

I present AQI Forecaster, an end-to-end machine learning system 
for real-time PM2.5 forecasting across 37 global cities. Using 
325,008 hourly records with 129 engineered features, we train 
XGBoost (MAE=8.66 µg/m³) and LSTM+Attention models for 24/48/72h 
horizons. Beyond prediction, I apply DoWhy causal inference and 
PC Algorithm to identify true causal drivers of PM2.5, finding 
precipitation (−6.6 µg/m³/mm, p<0.05) and NO2 (+1.1 µg/m³) 
as verified causal factors. Results are deployed as a live 
Streamlit dashboard with real-time API integration.

---

## 1. Introduction

Air pollution causes 7 million deaths annually (WHO, 2022). 
Accurate PM2.5 forecasting enables early warnings and policy 
interventions. Existing systems focus on prediction but ignore 
causal structure — limiting scientific interpretability.

This work contributes:
1. Multi-horizon PM2.5 forecasting (24/48/72h) across 37 cities
2. Causal discovery using DoWhy + PC Algorithm
3. SHAP explainability integrated into live dashboard
4. Production deployment with real-time data pipeline

---

## 2. Data & Methods

### 2.1 Dataset
- **Sources:** WAQI API (live AQI), Open-Meteo (weather), 
  CPCB India (official government data)
- **Coverage:** 37 global cities, 1 year hourly (Oct 2025–Oct 2026)
- **Size:** 325,008 rows, 129 features
- **Split:** Time-based — Train 83% / Val 8% / Test 8%
- **Leakage prevention:** Strict temporal ordering, 
  scaler fit on train only, no future information in features

### 2.2 Feature Engineering
- Lag features: PM2.5 at 1h, 3h, 6h, 12h, 24h, 48h
- Rolling averages: 3h, 6h, 12h, 24h, 48h, 168h windows
- Cyclical encoding: hour, month, day-of-week (sin/cos)
- City one-hot encoding: 37 binary columns
- Missing value indicators

### 2.3 Models
**XGBoost:** Gradient boosted trees with Optuna hyperparameter 
optimization (50 Bayesian trials). HuberLoss objective for 
outlier robustness. Early stopping on validation MAE.

**LSTM+Attention:** 3-layer LSTM (hidden=256), 4-head 
MultiheadAttention, dropout=0.3, AdamW optimizer, 
ReduceLROnPlateau scheduler. Trained on Kaggle T4 GPU, 
75 epochs, batch size 512.

### 2.4 Causal ML
DoWhy linear regression estimator on domain-knowledge DAG 
(20 causal edges). PC Algorithm for data-driven structure 
discovery. Placebo refutation tests verify non-spurious effects.

---

## 3. Results

### 3.1 Forecast Accuracy (Validation Set)

| Model | 24h MAE | 48h MAE | 72h MAE | 24h R² |
|---|---|---|---|---|
| Persistence | 9.584 | 11.195 | 12.304 | — |
| Climatology | 15.001 | 14.978 | 14.962 | — |
| XGBoost 7 cities | 9.912 | 11.078 | 12.591 | 0.762 |
| LSTM 7 cities | 11.456 | 12.050 | 12.744 | 0.684 |
| **XGBoost 37 cities** | **8.659** | **9.794** | **10.195** | **0.721** |
| LSTM 37 cities | 9.320 | 9.736 | 10.201 | 0.645 |

Scaling from 7 to 37 cities improved XGBoost 24h MAE by 12.6%, 
demonstrating strong generalization across diverse pollution regimes.

### 3.2 SHAP Analysis
Top features by mean |SHAP|: pm25 current (16.22), 
pm25_rolling_168h (5.97), pm25_rolling_3h (3.99). 
Current PM2.5 dominates short-term forecasts; 
weekly rolling average captures seasonal memory.

---

## 4. Causal Analysis

### 4.1 Verified Causal Effects

| Factor | Effect | Refutation |
|---|---|---|
| Precipitation | −6.645 µg/m³/mm | 120% change ✅ |
| NO2 | +1.079 µg/m³ | 99.6% change ✅ |
| Temperature | +0.042 µg/m³/°C | Indirect |
| Wind speed | +3.107 µg/m³ | Confounded ⚠️ |

### 4.2 Key Discovery: Wind Speed Confounding
Global analysis shows wind speed positively associated with PM2.5 
(+3.1 µg/m³) — seemingly contradicting atmospheric physics. 
City-stratified analysis reveals confounding: coastal cities 
(Mumbai, LA) have both high wind and high PM2.5 from sea salt. 
Geneva (clean European city) shows true negative effect (−0.20), 
consistent with dispersion physics. Standard ML cannot detect 
this confounding — causal inference identifies it.

### 4.3 PC Algorithm Validation
16 edges discovered; 3 confirmed by domain knowledge, 
4 new valid atmospheric chemistry edges (NO2→CO, 
temperature→O3, humidity→O3, NO2→O3), 
9 spurious edges correctly rejected via correlation thresholding.

---

## 5. Conclusion

AQI Forecaster demonstrates that production ML systems benefit 
from causal reasoning beyond predictive accuracy. Our XGBoost 
model achieves MAE=8.66 µg/m³ across 37 cities with 
interpretable SHAP explanations. Causal analysis identifies 
precipitation as the strongest actionable driver 
(−6.6 µg/m³/mm), with implications for cloud seeding 
interventions in extreme pollution events.

**Live system:** https://aqi-forecaster-3qxqouxeps5dwawf2wcfdc.streamlit.app  
**Code:** https://github.com/anshikagautam162-blip/aqi-forecaster

---

*Keywords: PM2.5 forecasting, XGBoost, LSTM, causal inference, 
DoWhy, SHAP, air quality, real-time ML*