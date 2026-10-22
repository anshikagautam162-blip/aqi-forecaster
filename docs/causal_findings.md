# AQI Forecaster — Causal Analysis Findings

**Dataset:** 7 global cities | 1 year | 61,488 hourly records
**Method:** DoWhy Linear Regression Estimator + PC Algorithm

---

## Finding 1: Precipitation is the Strongest Causal Driver
**Effect:** -6.645 µg/m³ per mm of rainfall
**Explanation:** Rain removes PM2.5 via wet deposition.
**Policy implication:** Cloud seeding could reduce extreme pollution.

---

## Finding 2: NO2 is a Causal Precursor of PM2.5
**Effect:** +1.079 µg/m³ per µg/m³ of NO2
**Explanation:** NO2 forms secondary particulate via photochemical reactions.
**Policy implication:** Reducing vehicular emissions causally reduces PM2.5.

---

## Finding 3: Temperature has Small but Real Causal Effect
**Effect:** +0.042 µg/m³ per °C
**Explanation:** Higher temperature accelerates photochemical reactions.
Effect is indirect — acts through atmospheric mixing and chemistry.

---

## Finding 4: Humidity Net Effect is Near-Zero
**Effect:** -0.001 µg/m³ per % humidity
**Explanation:** Two mechanisms cancel each other:
  - High humidity increases PM2.5 (hygroscopic growth)
  - High humidity decreases PM2.5 (precipitation)

---

## Finding 5: Wind Speed Confounding Discovery
**Global effect:** +3.107 µg/m³ (spurious — confounded)
**City-stratified:**
  Delhi: +2.036 | Beijing: +3.147 | Lahore: +3.545
  Berlin: +0.139 | Geneva: -0.201 (true negative effect!)

**Contribution:** Standard ML misses this confounding.
Geneva shows true negative effect — wind disperses pollution.

---

## SHAP vs Causal Analysis

| Feature | SHAP | Causal Effect |
|---|---|---|
| pm25 current | 19.44 | N/A (it IS outcome) |
| precipitation | Low | -6.645 (strongest!) |
| wind_speed | Low | Confounded (+3.1) |
| no2 | 0.94 | +1.079 (causal) |

---

## Limitations
1. Wind speed needs city-specific causal graphs
2. 5+ years data would improve estimates
3. Traffic data would strengthen NO2 → PM2.5 path

---
*AQI Forecaster | DoWhy + PC Algorithm | 7 cities | 2025-2026*
