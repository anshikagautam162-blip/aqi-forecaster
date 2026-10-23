import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
WAQI_TOKEN_DEFAULT = os.getenv('WAQI_TOKEN', '')

import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import json
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AQI Forecaster — Global Air Quality",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background-color: #0a0e1a; }
.stApp > header { background-color: transparent; }
[data-testid="stSidebar"] {
    background-color: #0f1117;
    border-right: 0.5px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #94a3b8; }
[data-testid="stMetric"] {
    background: #0f1117;
    border: 0.5px solid #1e293b;
    border-radius: 8px;
    padding: 0.75rem;
}
[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-size: 11px !important;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.4rem !important;
    white-space: nowrap !important;
    overflow: visible !important;
}
[data-testid="stMetricDelta"] { font-size: 11px !important; }
h1 {
    color: #f8fafc !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}
h2 { color: #e2e8f0 !important; font-size: 1.2rem !important; }
h3 { color: #cbd5e1 !important; font-size: 1rem !important; }
h4 { color: #94a3b8 !important; }
p, .stMarkdown { color: #94a3b8; }
hr { border-color: #1e293b !important; }
[data-testid="stDataFrame"] {
    background: #0f1117;
    border: 0.5px solid #1e293b;
    border-radius: 8px;
}
.stButton button {
    background: #1e293b;
    color: #94a3b8;
    border: 0.5px solid #334155;
    border-radius: 6px;
}
.stButton button:hover {
    background: #334155;
    color: #f1f5f9;
}
.stDownloadButton button {
    background: #1e293b;
    color: #818cf8;
    border: 0.5px solid #4f46e5;
    border-radius: 6px;
    font-size: 12px;
}
.stAlert { border-radius: 8px !important; }
.stCaption { color: #475569 !important; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent
MOD_DIR  = BASE_DIR / 'models'
RES_DIR  = BASE_DIR / 'results'

# High MAE cities
HIGH_MAE_CITIES = [
    'Riyadh','Beijing','Jakarta',
    'Dhaka','Kolkata','Hanoi','Shanghai'
]

CITY_COORDS = {
    'Delhi':       (28.61,  77.21),
    'Mumbai':      (19.08,  72.88),
    'Beijing':     (39.90, 116.41),
    'Berlin':      (52.52,  13.41),
    'Geneva':      (46.20,   6.14),
    'Los Angeles': (34.05,-118.24),
    'Lahore':      (31.52,  74.36),
    'Tokyo':       (35.68, 139.65),
    'London':      (51.51,  -0.13),
    'Paris':       (48.86,   2.35),
    'Cairo':       (30.04,  31.24),
    'Dubai':       (25.20,  55.27),
    'Shanghai':    (31.23, 121.47),
    'Seoul':       (37.57, 126.98),
    'Bangkok':     (13.76, 100.50),
    'Riyadh':      (24.71,  46.68),
    'Sydney':      (-33.87,151.21),
    'New York':    (40.71, -74.01),
}

FEATURE_COLS = [
    'apparent_temp','cloud_cover','co','day_of_week',
    'dew_point','dow_cos','dow_sin','dust',
    'hour','hour_cos','hour_sin','humidity',
    'humidity_lag_12h','humidity_lag_1h','humidity_lag_24h',
    'humidity_lag_3h','humidity_lag_6h',
    'is_weekend','month','month_cos','month_sin',
    'no2','no2_missing','o3','o3_missing',
    'pm10','pm10_missing','pm25',
    'pm25_lag_12h','pm25_lag_1h','pm25_lag_24h',
    'pm25_lag_3h','pm25_lag_48h','pm25_lag_6h',
    'pm25_missing',
    'pm25_rolling_12h','pm25_rolling_168h','pm25_rolling_24h',
    'pm25_rolling_3h','pm25_rolling_48h','pm25_rolling_6h',
    'precip_lag_12h','precip_lag_1h','precip_lag_24h',
    'precip_lag_3h','precip_lag_6h',
    'precip_rolling_12h','precip_rolling_168h','precip_rolling_24h',
    'precip_rolling_3h','precip_rolling_48h','precip_rolling_6h',
    'precipitation','pressure',
    'pressure_lag_12h','pressure_lag_1h','pressure_lag_24h',
    'pressure_lag_3h','pressure_lag_6h','pressure_missing',
    'rain','season','snowfall','so2',
    'temp_lag_12h','temp_lag_1h','temp_lag_24h',
    'temp_lag_3h','temp_lag_6h',
    'temp_rolling_12h','temp_rolling_168h','temp_rolling_24h',
    'temp_rolling_3h','temp_rolling_48h','temp_rolling_6h',
    'temperature','uv_index','vapour_pressure',
    'wind_dir_lag_12h','wind_dir_lag_1h','wind_dir_lag_24h',
    'wind_dir_lag_3h','wind_dir_lag_6h',
    'wind_direction','wind_gusts',
    'wind_lag_12h','wind_lag_1h','wind_lag_24h',
    'wind_lag_3h','wind_lag_6h',
    'wind_speed','wind_speed_missing',
]

CITY_COLS = [
    'city_Amsterdam','city_Bangkok','city_Beijing',
    'city_Berlin','city_Cairo','city_Chennai',
    'city_Chicago','city_Delhi','city_Dhaka',
    'city_Dubai','city_Geneva','city_Hanoi',
    'city_Jakarta','city_Karachi','city_Kathmandu',
    'city_Kolkata','city_Lagos','city_Lahore',
    'city_London','city_Los Angeles','city_Madrid',
    'city_Melbourne','city_Mexico City','city_Mumbai',
    'city_Nairobi','city_New York','city_Paris',
    'city_Prague','city_Riyadh','city_Rome',
    'city_Sao Paulo','city_Seoul','city_Shanghai',
    'city_Sydney','city_Tokyo','city_Toronto',
    'city_Warsaw',
]
ALL_FEATURES = FEATURE_COLS + CITY_COLS

def get_aqi_category(pm25):
    if pm25 is None: return 'Unknown','⚪','#808080'
    pm25 = float(pm25)
    if pm25 <= 12:    return 'Good',               '🟢','#22c55e'
    elif pm25 <= 35:  return 'Moderate',           '🟡','#f59e0b'
    elif pm25 <= 55:  return 'Unhealthy Sensitive','🟠','#f97316'
    elif pm25 <= 150: return 'Unhealthy',          '🔴','#ef4444'
    elif pm25 <= 250: return 'Very Unhealthy',     '🟣','#9333ea'
    else:             return 'Hazardous',          '⚫','#dc2626'

def get_map_color(pm25):
    if pm25 is None: return '#808080'
    pm25 = float(pm25)
    if pm25 <= 12:    return '#22c55e'
    elif pm25 <= 35:  return '#f59e0b'
    elif pm25 <= 55:  return '#f97316'
    elif pm25 <= 150: return '#ef4444'
    elif pm25 <= 250: return '#9333ea'
    else:             return '#dc2626'

@st.cache_data(ttl=3600)
def fetch_live_aqi(city, token):
    try:
        resp = requests.get(
            f"https://api.waqi.info/feed/{city}/",
            params={'token': token}, timeout=15)
        data = resp.json()
        if data['status'] == 'ok':
            d    = data['data']
            iaqi = d.get('iaqi', {})
            fc   = d.get('forecast',{}).get('daily',{})
            return {
                'aqi':           d.get('aqi'),
                'pm25':          iaqi.get('pm25',{}).get('v'),
                'pm10':          iaqi.get('pm10',{}).get('v'),
                'no2':           iaqi.get('no2', {}).get('v'),
                'o3':            iaqi.get('o3',  {}).get('v'),
                'co':            iaqi.get('co',  {}).get('v'),
                'humidity':      iaqi.get('h',   {}).get('v'),
                'temperature':   iaqi.get('t',   {}).get('v'),
                'wind':          iaqi.get('w',   {}).get('v'),
                'pressure':      iaqi.get('p',   {}).get('v'),
                'datetime':      d.get('time',{}).get('iso'),
                'station':       d.get('city',{}).get('name',''),
                'latitude':      d.get('city',{}).get('geo',[None,None])[0],
                'longitude':     d.get('city',{}).get('geo',[None,None])[1],
                'forecast_pm25': fc.get('pm25',[]),
            }
    except:
        return None

@st.cache_data(ttl=3600)
def fetch_weather_forecast(lat, lon):
    try:
        resp = requests.get(
            "https://forecast-api.open-meteo.com/v1/forecast",
            params={
                'latitude': lat,'longitude': lon,
                'hourly': [
                    'temperature_2m','relative_humidity_2m',
                    'precipitation','wind_speed_10m',
                    'wind_direction_10m','surface_pressure',
                    'cloud_cover','dew_point_2m',
                    'apparent_temperature','wind_gusts_10m',
                    'vapour_pressure_deficit','rain','snowfall',
                ],
                'forecast_days': 4,'timezone': 'auto',
            }, timeout=30)
        hourly = resp.json().get('hourly', {})
        times  = hourly.get('time', [])
        records = []
        for i, t in enumerate(times):
            records.append({
                'datetime':       t,
                'temperature':    hourly.get('temperature_2m',         [None]*len(times))[i],
                'humidity':       hourly.get('relative_humidity_2m',   [None]*len(times))[i],
                'precipitation':  hourly.get('precipitation',          [None]*len(times))[i],
                'wind_speed':     hourly.get('wind_speed_10m',         [None]*len(times))[i],
                'wind_direction': hourly.get('wind_direction_10m',     [None]*len(times))[i],
                'pressure':       hourly.get('surface_pressure',       [None]*len(times))[i],
                'cloud_cover':    hourly.get('cloud_cover',            [None]*len(times))[i],
                'dew_point':      hourly.get('dew_point_2m',           [None]*len(times))[i],
                'apparent_temp':  hourly.get('apparent_temperature',   [None]*len(times))[i],
                'wind_gusts':     hourly.get('wind_gusts_10m',         [None]*len(times))[i],
                'vapour_pressure':hourly.get('vapour_pressure_deficit',[None]*len(times))[i],
                'rain':           hourly.get('rain',                   [None]*len(times))[i],
                'snowfall':       hourly.get('snowfall',               [None]*len(times))[i],
            })
        return pd.DataFrame(records)
    except:
        return pd.DataFrame()

def prepare_features(live_data, weather_df, city_name):
    now = pd.Timestamp.now()
    row = {}
    pm25_now  = float(live_data.get('pm25') or 0)
    pm10_now  = float(live_data.get('pm10') or 0)
    no2_now   = float(live_data.get('no2')  or 0)
    o3_now    = float(live_data.get('o3')   or 0)
    co_now    = float(live_data.get('co')   or 0)
    row.update({
        'pm25':pm25_now,'pm10':pm10_now,'no2':no2_now,
        'o3':o3_now,'co':co_now,'so2':0.0,
        'dust':0.0,'uv_index':0.0,
        'pm25_missing':0,'pm10_missing':0,
        'no2_missing':0,'o3_missing':0,
        'pressure_missing':0,'wind_speed_missing':0,
    })
    if not weather_df.empty:
        w = weather_df.iloc[0]
        temp     = float(w.get('temperature',    20) or 20)
        humidity = float(w.get('humidity',       60) or 60)
        precip   = float(w.get('precipitation',   0) or 0)
        wspeed   = float(w.get('wind_speed',      5) or 5)
        wdir     = float(w.get('wind_direction', 180) or 180)
        pressure = float(w.get('pressure',      1013) or 1013)
        cloud    = float(w.get('cloud_cover',    50) or 50)
        dew      = float(w.get('dew_point',      10) or 10)
        gusts    = float(w.get('wind_gusts',     10) or 10)
        vpress   = float(w.get('vapour_pressure', 1) or 1)
        rain     = float(w.get('rain',            0) or 0)
        snow     = float(w.get('snowfall',        0) or 0)
        atemp    = float(w.get('apparent_temp',  20) or 20)
    else:
        temp=float(live_data.get('temperature') or 20)
        humidity=float(live_data.get('humidity') or 60)
        precip=0.0; wspeed=float(live_data.get('wind') or 5)
        wdir=180.0; pressure=float(live_data.get('pressure') or 1013)
        cloud=50.0; dew=10.0; gusts=10.0
        vpress=1.0; rain=0.0; snow=0.0; atemp=temp

    row.update({
        'temperature':temp,'humidity':humidity,
        'precipitation':precip,'wind_speed':wspeed,
        'wind_direction':wdir,'pressure':pressure,
        'cloud_cover':cloud,'dew_point':dew,
        'wind_gusts':gusts,'vapour_pressure':vpress,
        'rain':rain,'snowfall':snow,'apparent_temp':atemp,
    })
    row.update({
        'hour':now.hour,'day_of_week':now.dayofweek,
        'month':now.month,
        'season':{12:0,1:0,2:0,3:1,4:1,5:1,
                  6:2,7:2,8:2,9:3,10:3,11:3}[now.month],
        'is_weekend':int(now.dayofweek in [5,6]),
        'hour_sin':np.sin(2*np.pi*now.hour/24),
        'hour_cos':np.cos(2*np.pi*now.hour/24),
        'month_sin':np.sin(2*np.pi*now.month/12),
        'month_cos':np.cos(2*np.pi*now.month/12),
        'dow_sin':np.sin(2*np.pi*now.dayofweek/7),
        'dow_cos':np.cos(2*np.pi*now.dayofweek/7),
    })
    for lag in [1,3,6,12,24,48]:
        row[f'pm25_lag_{lag}h'] = pm25_now
    for w_ in [3,6,12,24,48,168]:
        row[f'pm25_rolling_{w_}h'] = pm25_now
    for lag in [1,3,6,12,24]:
        row[f'temp_lag_{lag}h']     = temp
        row[f'wind_lag_{lag}h']     = wspeed
        row[f'wind_dir_lag_{lag}h'] = wdir
        row[f'humidity_lag_{lag}h'] = humidity
        row[f'pressure_lag_{lag}h'] = pressure
        row[f'precip_lag_{lag}h']   = precip
    for w_ in [3,6,12,24,48,168]:
        row[f'precip_rolling_{w_}h'] = precip
    for w_ in [3,6,12,24,48,168]:
        row[f'temp_rolling_{w_}h'] = temp

    city_clean = city_name.strip().title()
    for cc in CITY_COLS:
        row[cc] = 1 if cc == f'city_{city_clean}' else 0

    df_row = pd.DataFrame([row])
    for col in ALL_FEATURES:
        if col not in df_row.columns:
            df_row[col] = 0.0
    return df_row[ALL_FEATURES].fillna(0)

@st.cache_resource
def load_models():
    models = {}
    for h in ['24h','48h','72h']:
        path = MOD_DIR / f'xgb_{h}_37cities.pkl'
        if path.exists():
            with open(path,'rb') as f:
                models[h] = pickle.load(f)
    return models

@st.cache_data
def load_causal():
    path = RES_DIR / 'causal_effects.json'
    if path.exists():
        with open(path,'r') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_comparison():
    path = RES_DIR / 'model_comparison_final.csv'
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌬️ AQI Forecaster")
    st.caption("Global Air Quality Intelligence")
    st.divider()
    WAQI_TOKEN = st.text_input(
        "WAQI API Token",
        value=WAQI_TOKEN_DEFAULT,
        type="password")
    st.divider()
    st.markdown("### 🔍 City Search")
    city_input = st.text_input(
        "Enter any city worldwide",
        value="Delhi",
        placeholder="Tokyo, Paris, Cairo...")
    st.caption("Supports 10,000+ cities via WAQI API")
    st.divider()
    st.markdown("### ⏱️ Forecast Horizon")
    horizon = st.select_slider(
        "Hours ahead",options=[24,48,72],value=24)
    st.divider()
    st.markdown("**Data sources:**")
    st.caption("• WAQI API — live AQI")
    st.caption("• Open-Meteo — weather forecast")
    st.caption("• XGBoost 37 cities — primary model")
    st.caption("• LSTM+Attention 37 cities — secondary")
    st.caption("• DoWhy — causal inference")
    st.divider()
    if st.button("🔄 Refresh Data",use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    auto_refresh = st.toggle("⏱️ Auto-refresh (1hr)",value=False)
    st.divider()
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ══════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════
models     = load_models()
causal_res = load_causal()
df_compare = load_comparison()

with st.spinner(f"Fetching live AQI for {city_input}..."):
    live_data = fetch_live_aqi(city_input, WAQI_TOKEN)

city_not_found = live_data is None
if city_not_found:
    live_data = {
        'pm25':None,'pm10':None,'no2':None,'o3':None,
        'co':None,'humidity':None,'temperature':None,
        'wind':None,'pressure':None,
        'datetime':datetime.now().isoformat(),
        'station':city_input,'latitude':20.0,
        'longitude':0.0,'forecast_pm25':[],'aqi':None,
    }
    st.warning(f"⚠️ '{city_input}' not in WAQI — "
               f"showing XGBoost forecast only.")

# ══════════════════════════════════════════
# XGBoost PREDICTIONS
# ══════════════════════════════════════════
lat = live_data.get('latitude') or 28.6
lon = live_data.get('longitude') or 77.2

with st.spinner("Computing XGBoost forecast..."):
    weather_df = fetch_weather_forecast(lat, lon)
    X_pred     = prepare_features(live_data, weather_df, city_input)

xgb_preds = {}
if models:
    for h, model in models.items():
        try:
            xgb_preds[h] = max(0, float(
                model.predict(X_pred)[0]))
        except:
            xgb_preds[h] = None

# ══════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════
col_h1, col_h2 = st.columns([3,1])
with col_h1:
    st.markdown("# 🌬️ AQI Forecaster — Global Air Quality")
    st.caption(
        f"Real-time monitoring + 72h forecast · "
        f"37 cities trained · XGBoost + LSTM + Causal ML · "
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
with col_h2:
    st.markdown(f"**📍 {city_input}**")
    st.caption(f"Station: {live_data.get('station','')}")

st.divider()

# ══════════════════════════════════════════
# SECTION 1: ALERT + METRICS
# ══════════════════════════════════════════
pm25 = live_data.get('pm25')
pm25_val = float(pm25 or 0)
cat, emoji, color = get_aqi_category(pm25)
who_ratio = pm25_val/15 if pm25_val else 0

if city_not_found:
    st.info(f"ℹ️ Live AQI unavailable for '{city_input}' — XGBoost forecast shown")
elif pm25_val > 200:
    st.error(f"🚨 **DANGER** — PM2.5 = {pm25_val:.0f} µg/m³ · {who_ratio:.1f}× WHO limit · Avoid all outdoor activity!")
elif pm25_val > 150:
    st.error(f"🔴 **UNHEALTHY** — PM2.5 = {pm25_val:.0f} µg/m³ · {who_ratio:.1f}× WHO limit · Sensitive groups must stay indoors")
elif pm25_val > 100:
    st.warning(f"🟠 **UNHEALTHY FOR SENSITIVE GROUPS** — PM2.5 = {pm25_val:.0f} µg/m³")
elif pm25_val > 50:
    st.warning(f"🟡 **MODERATE** — PM2.5 = {pm25_val:.0f} µg/m³")
else:
    st.success(f"✅ **GOOD** — PM2.5 = {pm25_val:.0f} µg/m³ · Air quality satisfactory in {city_input}")

# City accuracy disclaimer
city_clean = city_input.strip().title()
if city_clean in HIGH_MAE_CITIES:
    st.warning(
        f"⚠️ **Forecast accuracy note:** {city_clean} has higher "
        f"uncertainty (MAE ~30-60 µg/m³) due to extreme pollution "
        f"events like dust storms. Use forecast as a guide only.")

st.markdown("#### 📊 Current Conditions")
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("🌫️ PM2.5",
          f"{pm25_val:.0f}" if pm25 else "N/A",
          f"{cat} {emoji}")
c2.metric("💨 PM10",
          f"{float(live_data.get('pm10') or 0):.0f}","µg/m³")
c3.metric("🌡️ Temp",
          f"{live_data.get('temperature') or 'N/A'}","°C")
c4.metric("💧 Humidity",
          f"{live_data.get('humidity') or 'N/A'}","%")
c5.metric("💨 Wind",
          f"{float(live_data.get('wind') or 0):.1f}",
          "km/h")
c6.metric("🔵 NO2",
          f"{live_data.get('no2') or 'N/A'}","µg/m³")

st.caption(
    f"📍 {live_data.get('station','Unknown')} · "
    f"🕐 {live_data.get('datetime','N/A')} · "
    f"WHO limit: 15 µg/m³"
    + (f" · **{who_ratio:.1f}× exceeded**"
       if who_ratio > 1 else " · ✅ Within limit"))
st.divider()

# ══════════════════════════════════════════
# SECTION 2: FORECAST CHART
# ══════════════════════════════════════════
st.markdown("#### 📈 72-Hour PM2.5 Forecast")
col_chart, col_pred = st.columns([1.5,1])

with col_chart:
    forecast_pm25 = live_data.get('forecast_pm25',[])
    fig = go.Figure()

    # AQI zones
    fig.add_hrect(y0=0,  y1=12,  fillcolor='#22c55e',opacity=0.05,line_width=0)
    fig.add_hrect(y0=12, y1=35,  fillcolor='#f59e0b',opacity=0.05,line_width=0)
    fig.add_hrect(y0=35, y1=55,  fillcolor='#f97316',opacity=0.05,line_width=0)
    fig.add_hrect(y0=55, y1=150, fillcolor='#ef4444',opacity=0.05,line_width=0)
    fig.add_hrect(y0=150,y1=300, fillcolor='#9333ea',opacity=0.05,line_width=0)

    # WHO line
    fig.add_hline(
        y=15,line_dash='dash',line_color='#22c55e',
        line_width=1.5,
        annotation_text='WHO 15 µg/m³',
        annotation_font_color='#22c55e',
        annotation_font_size=10)

    # Today's point — historical anchor
    today_str = datetime.now().strftime('%Y-%m-%d')
    if pm25_val > 0:
        fig.add_trace(go.Scatter(
            x=[today_str], y=[pm25_val],
            mode='markers',
            name='Current PM2.5',
            marker=dict(size=12,color='#f1f5f9',
                       symbol='circle',
                       line=dict(color='#818cf8',width=2)),
            hovertemplate='Today: %{y:.0f} µg/m³'))

    # WAQI forecast
    if forecast_pm25:
        dates = [f.get('day','') for f in forecast_pm25]
        avg   = [f.get('avg', 0) for f in forecast_pm25]
        low   = [f.get('min', 0) for f in forecast_pm25]
        high  = [f.get('max', 0) for f in forecast_pm25]

        fig.add_trace(go.Scatter(
            x=dates+dates[::-1],
            y=high+low[::-1],
            fill='toself',
            fillcolor='rgba(129,140,248,0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            name='WAQI confidence range'))
        fig.add_trace(go.Scatter(
            x=dates,y=avg,
            mode='lines+markers',
            name='WAQI forecast',
            line=dict(color='#818cf8',width=2),
            marker=dict(size=6,color='#818cf8')))

    # XGBoost predictions
    xgb_dates = [
        (pd.Timestamp.now()+pd.Timedelta(hours=24)).strftime('%Y-%m-%d'),
        (pd.Timestamp.now()+pd.Timedelta(hours=48)).strftime('%Y-%m-%d'),
        (pd.Timestamp.now()+pd.Timedelta(hours=72)).strftime('%Y-%m-%d'),
    ]
    xgb_vals = [xgb_preds.get('24h'),
                xgb_preds.get('48h'),
                xgb_preds.get('72h')]
    xgb_clean  = [v for v in xgb_vals if v is not None]
    dates_clean = xgb_dates[:len(xgb_clean)]

    if xgb_clean:
        fig.add_trace(go.Scatter(
            x=dates_clean,y=xgb_clean,
            mode='lines+markers',
            name='XGBoost forecast',
            line=dict(color='#f87171',width=2.5,dash='dash'),
            marker=dict(size=10,symbol='star',color='#f87171')))

    fig.update_layout(
        title=dict(
            text=f"PM2.5 Forecast — {city_input}",
            font=dict(color='#e2e8f0',size=13)),
        paper_bgcolor='#0f1117',
        plot_bgcolor='#0a0e1a',
        font=dict(color='#64748b',size=10),
        xaxis=dict(title='Date',gridcolor='#1e293b',color='#64748b'),
        yaxis=dict(title='PM2.5 (µg/m³)',gridcolor='#1e293b',color='#64748b'),
        legend=dict(orientation='h',yanchor='bottom',y=1.02,
                    font=dict(color='#94a3b8',size=10),
                    bgcolor='rgba(0,0,0,0)'),
        height=320,hovermode='x unified',
        margin=dict(l=50,r=20,t=60,b=40))
    st.plotly_chart(fig,use_container_width=True)

    # CSV Download
    if xgb_clean:
        fc_df = pd.DataFrame({
            'horizon':  ['24h','48h','72h'],
            'datetime': xgb_dates,
            'xgb_pm25': xgb_vals,
            'city':     city_input,
        })
        st.download_button(
            "📥 Download Forecast CSV",
            fc_df.to_csv(index=False),
            f"forecast_{city_input}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv')

with col_pred:
    st.markdown("**🤖 XGBoost Model**")
    st.caption("37 cities · MAE 8.66 µg/m³")
    for h, label in [('24h','24 Hours'),
                     ('48h','48 Hours'),
                     ('72h','72 Hours')]:
        val = xgb_preds.get(h)
        if val is not None:
            c_,e_,_ = get_aqi_category(val)
            st.metric(f"⏱️ {label}",f"{val:.1f} µg/m³",f"{c_} {e_}")
        else:
            st.metric(f"⏱️ {label}","N/A","Loading...")
    st.markdown("---")
    st.markdown("**📡 WAQI Daily**")
    if forecast_pm25:
        for i, label in enumerate(['Tomorrow','Day 2','Day 3']):
            if i < len(forecast_pm25):
                avg_v = forecast_pm25[i].get('avg',0)
                c_,e_,_ = get_aqi_category(avg_v)
                st.metric(f"📅 {label}",
                          f"{avg_v:.0f} µg/m³",
                          f"{c_} {e_}")
    else:
        st.info("No WAQI forecast available")
    st.markdown("---")
    st.markdown("**📊 Model Accuracy**")
    st.caption("Validation set · 37 cities")
    st.dataframe(pd.DataFrame({
        'Horizon':['24h','48h','72h'],
        'MAE':['8.659','9.794','10.195'],
        'R²':['0.721','0.668','0.640'],
    }),hide_index=True,use_container_width=True)

st.divider()

# ══════════════════════════════════════════
# SECTION 3: GLOBAL MAP
# ══════════════════════════════════════════
st.markdown("#### 🗺️ Global City AQI Map")

m = folium.Map(location=[25,30],zoom_start=2,
               tiles='OpenStreetMap')

for city_name,(lat_m,lon_m) in CITY_COORDS.items():
    cd = fetch_live_aqi(city_name,WAQI_TOKEN)
    pv = float(cd.get('pm25') or 0) if cd else 0
    col_m = get_map_color(pv)
    cat_m,em_m,_ = get_aqi_category(pv)
    radius = max(6,min(20,pv/6))
    folium.CircleMarker(
        location=[lat_m,lon_m],
        radius=radius,
        color=col_m,fill=True,
        fill_color=col_m,fill_opacity=0.8,weight=1.5,
        popup=folium.Popup(
            f"<div style='font-family:sans-serif'>"
            f"<b>{city_name}</b><br>"
            f"PM2.5: <b>{pv:.0f} µg/m³</b><br>"
            f"{cat_m} {em_m}</div>",max_width=180),
        tooltip=folium.Tooltip(
            f"{city_name}: {pv:.0f} µg/m³ {em_m}")
    ).add_to(m)
    if pv > 150:
        folium.CircleMarker(
            location=[lat_m,lon_m],
            radius=radius+5,color=col_m,
            fill=False,weight=1,opacity=0.3
        ).add_to(m)

st_folium(m,height=420,use_container_width=True)
lc1,lc2,lc3,lc4,lc5 = st.columns(5)
lc1.markdown("🟢 **Good** ≤12")
lc2.markdown("🟡 **Moderate** ≤35")
lc3.markdown("🟠 **Sensitive** ≤55")
lc4.markdown("🔴 **Unhealthy** ≤150")
lc5.markdown("🟣 **Hazardous** >150")
st.divider()

# ══════════════════════════════════════════
# SECTION 4: SHAP + CAUSAL + ALERTS
# ══════════════════════════════════════════
st.markdown("#### 🔬 Scientific Analysis")
col_shap,col_causal,col_alerts = st.columns(3)

with col_shap:
    st.markdown("**🔍 SHAP Feature Importance**")
    st.caption("Why is PM2.5 changing? · XGBoost 37 cities")
    shap_img = RES_DIR/'shap_plots'/'shap_bar_37cities.png'
    if shap_img.exists():
        st.image(str(shap_img),use_container_width=True)
    wf_map = {
        'Delhi':  'shap_waterfall_delhi.png',
        'Riyadh': 'shap_waterfall_riyadh.png',
        'Rome':   'shap_waterfall_rome.png',
    }
    wf_file = wf_map.get(city_clean)
    if wf_file:
        wf_path = RES_DIR/'shap_plots'/wf_file
        if wf_path.exists():
            st.caption(f"Waterfall — {city_clean} worst day")
            st.image(str(wf_path),use_container_width=True)
    st.caption("TreeSHAP · test set n=2000 · global analysis")

with col_causal:
    st.markdown("**🔗 Causal Effects on PM2.5**")
    st.caption("DoWhy · PC Algorithm · refutation verified")
    causal_display = {
        'precipitation':{'label':'🌧️ Precipitation','unit':'per mm'},
        'no2':          {'label':'⚗️ NO2',          'unit':'per µg/m³'},
        'temperature':  {'label':'🌡️ Temperature',  'unit':'per °C'},
        'humidity':     {'label':'💧 Humidity',     'unit':'per %'},
        'wind_speed':   {'label':'💨 Wind speed',   'unit':'confounded ⚠️'},
    }
    for key,info in causal_display.items():
        result = causal_res.get(key,{})
        if isinstance(result,dict):
            effect = result.get('effect')
            if effect is not None:
                sign  = "📉" if effect < 0 else "📈"
                col_  = "🟢" if effect < 0 else "🔴"
                st.markdown(
                    f"{col_} **{info['label']}**  \n"
                    f"`{effect:+.3f} µg/m³` {sign} · "
                    f"*{info['unit']}*")
    st.markdown("---")
    st.caption(
        "💡 Precipitation strongest causal driver "
        "(-6.6 µg/m³/mm). NO2 causally forms PM2.5 "
        "(+1.1 µg/m³). Refutation verified >99%.")
    causal_img = RES_DIR/'causal_graph.png'
    if causal_img.exists():
        st.image(str(causal_img),
                 caption="Causal DAG — DoWhy",
                 use_container_width=True)

with col_alerts:
    st.markdown("**🚨 Alert System**")
    st.caption("Current + XGBoost forecast alerts")
    if city_not_found:
        st.info("📡 No live data")
    elif pm25_val > 250:
        st.error("🚨 HAZARDOUS — Stay indoors NOW!")
    elif pm25_val > 200:
        st.error("🔴 VERY UNHEALTHY")
    elif pm25_val > 150:
        st.warning("🟠 UNHEALTHY — Limit outdoor exposure")
    elif pm25_val > 100:
        st.warning("🟡 Unhealthy for sensitive groups")
    elif pm25_val > 50:
        st.info("🟢 Moderate")
    else:
        st.success("✅ GOOD — Air quality satisfactory")

    st.markdown("**Forecast alerts:**")
    for h,label in [('24h','Tomorrow'),
                    ('48h','Day 2'),
                    ('72h','Day 3')]:
        val = xgb_preds.get(h)
        if val is not None:
            c_,e_,_ = get_aqi_category(val)
            if val > 150:
                st.error(f"⚠️ {label}: {val:.0f} — {c_} {e_}")
            elif val > 55:
                st.warning(f"⚠️ {label}: {val:.0f} — {c_} {e_}")
            else:
                st.success(f"✅ {label}: {val:.0f} — {c_} {e_}")

    precip = causal_res.get('precipitation',{})
    if isinstance(precip,dict) and precip.get('effect'):
        st.markdown("**Causal insight:**")
        st.info(f"🌧️ 1mm rain → PM2.5 drops "
                f"{abs(precip['effect']):.1f} µg/m³")

st.divider()

# ══════════════════════════════════════════
# SECTION 5: MODEL COMPARISON
# ══════════════════════════════════════════
st.markdown("#### 📊 Model Performance Comparison")
if not df_compare.empty:
    df_show = df_compare[
        ['model','cities','mae_24h','mae_48h',
         'mae_72h','r2_24h']].copy()
    df_show.columns = ['Model','Cities',
                       '24h MAE','48h MAE',
                       '72h MAE','24h R²']
    df_show = df_show.astype(str).replace('nan','—').replace('None','—')
    st.dataframe(df_show, hide_index=True, use_container_width=True)
    st.caption(
        "⭐ XGBoost 37 cities = best 24h & 72h · "
        "LSTM 37 cities = best 48h · "
        "No data leakage · Time-based split · "
        "325,008 training records")
st.divider()

# ══════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════
st.markdown(
    f"<div style='text-align:center;color:#334155;"
    f"font-size:11px;padding:1rem 0'>"
    f"🌬️ AQI Forecaster · XGBoost + LSTM + Causal ML · "
    f"37 Global Cities · 325,008 training records · "
    f"WAQI API + Open-Meteo · © {datetime.now().year}"
    f"</div>",
    unsafe_allow_html=True)

# Auto refresh
if auto_refresh:
    time.sleep(3600)
    st.rerun()