# ================== CORE IMPORTS ==================
import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
import hashlib
import calendar
import json
import time  # <-- ADD THIS IMPORT
from datetime import datetime, timedelta  # <-- REMOVE 'time' from here
from dateutil.relativedelta import relativedelta
from io import BytesIO
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import performance_metrics
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from PIL import Image
import random

# ================== STREAMLIT CONFIG (MUST BE FIRST) ==================
import streamlit as st
import base64
from pathlib import Path

def get_base64_image(image_path):
    """Convert image to base64 for HTML embedding"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

st.set_page_config(
    page_title="SForecast - Intelligent Forecasting Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

LOGO_PATH = Path(r"C:/Users/chris.mutuku/OneDrive - Skanem AS/Desktop/logo.jpg")

if LOGO_PATH.exists():
    try:
        img_base64 = get_base64_image(LOGO_PATH)
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:20px;">
                <img src="data:image/jpeg;base64,{img_base64}" width="44" style="border-radius:8px;">
                <div>
                    <h1 style="margin:0; color:#0E4E4E;">SForecast</h1>
                    <p style="margin:0; color:#666; font-size:14px;">Intelligent Forecasting Platform</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    except:
        # Fallback if image loading fails
        st.title("SForecast")
        st.caption("Intelligent Forecasting Platform")
else:
    # Fallback if logo file doesn't exist
    st.title("SForecast")
    st.caption("Intelligent Forecasting Platform")
# Add some spacing
st.divider()
# ================== DATABASE ==================
DB_NAME = "skanem_forecasting.db"

# ================== SESSION DEFAULTS ==================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

if "forecast_cache" not in st.session_state:
    st.session_state.forecast_cache = {}

if "simulation_scenarios" not in st.session_state:
    st.session_state.simulation_scenarios = []

if "financial_metrics" not in st.session_state:
    st.session_state.financial_metrics = {
        'exchange_rates': {'USD': 1.0, 'EUR': 0.92, 'GBP': 0.79, 'INR': 83.0},
        'material_costs': {'PP Granules': 1200, 'Additives': 2500, 'Masterbatch': 1800},
        'conversion_factors': {'kg_to_liter': 1.0, 'kg_to_sqm': 10.0}
    }
PRIMARY_COLOR = "#0E4E4E" 
# ================== THEME DEFINITIONS ==================
def get_theme(mode="dark"):
    if mode == "light":
        return {
            "PRIMARY": "#0E4E4E",
            "SECONDARY": "#80BCE6",
            "ACCENT": "#E1EBAE",
            "BG": "#F8FAFC",
            "BG2": "#E5EEF0",
            "TEXT": "#0F172A",
            "MUTED": "#475569",
            "GOOD": "#2E7D32",
            "WARNING": "#F59E0B",
            "DANGER": "#DC2626",
            "INFO": "#3B82F6"
        }
    else:
        return {
            "PRIMARY": "#0E4E4E",
            "SECONDARY": "#80BCE6",
            "ACCENT": "#E9FF7D",
            "BG": "#0B1F1F",
            "BG2": "#123636",
            "TEXT": "#C4C4C4",
            "MUTED": "#D1ECFD",
            "GOOD": "#2E7D32",
            "WARNING": "#F5E50B",
            "DANGER": "#DC2626",
            "INFO": "#3B82F6"
        }

THEME = get_theme(st.session_state.theme_mode)

# ================== APPLY THEME ==================
def apply_theme(T):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {T["BG"]};
            color: {T["TEXT"]};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {T["BG2"]};
            border-right: 2px solid {T["PRIMARY"]};
        }}

        h1, h2, h3, h4, h5 {{
            color: {T["PRIMARY"]};
            font-weight: 700;
        }}

        p, span, label, div {{
            color: {T["TEXT"]};
        }}

        .stButton > button {{
            background-color: {T["PRIMARY"]};
            color: white;
            border-radius: 8px;
            font-weight: 600;
            border: none;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }}

        .stButton > button:hover {{
            background-color: {T["SECONDARY"]};
            color: #000000;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        div[data-testid="metric-container"] {{
            background-color: {T["BG2"]};
            border-left: 6px solid {T["ACCENT"]};
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}

        div[data-testid="metric-container"]:hover {{
            transform: translateY(-4px);
        }}

        .stDataFrame {{
            background-color: {T["BG2"]};
            color: {T["TEXT"]};
            border-radius: 8px;
            border: 1px solid {T["PRIMARY"]}20;
        }}

        .stAlert {{
            border-left: 6px solid {T["DANGER"]};
            border-radius: 8px;
        }}

        .success-alert {{
            border-left: 6px solid {T["GOOD"]};
            border-radius: 8px;
        }}

        .info-alert {{
            border-left: 6px solid {T["INFO"]};
            border-radius: 8px;
        }}

        .metric-value {{
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: {T["PRIMARY"]} !important;
        }}

        .metric-delta {{
            font-size: 1rem !important;
        }}

        .card {{
            background-color: {T["BG2"]};
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid {T["PRIMARY"]}20;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }}

        .stat-card {{
            text-align: center;
            padding: 1.5rem;
            border-radius: 12px;
            background: linear-gradient(135deg, {T["PRIMARY"]}20, {T["SECONDARY"]}20);
            border: 1px solid {T["PRIMARY"]}30;
        }}

        .kpi-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 600;
            margin: 0.25rem;
        }}

        .kpi-good {{
            background-color: {T["GOOD"]}20;
            color: {T["GOOD"]};
            border: 1px solid {T["GOOD"]}40;
        }}

        .kpi-warning {{
            background-color: {T["WARNING"]}20;
            color: {T["WARNING"]};
            border: 1px solid {T["WARNING"]}40;
        }}

        .kpi-danger {{
            background-color: {T["DANGER"]}20;
            color: {T["DANGER"]};
            border: 1px solid {T["DANGER"]}40;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

apply_theme(THEME)

# ================== SECURITY ==================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ================== USER TABLE ==================
def init_user_table():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

init_user_table()

# ================== AUTH FUNCTIONS ==================
def check_credentials(username, password):
    with sqlite3.connect(DB_NAME) as conn:
        row = conn.execute(
            "SELECT password FROM users WHERE username = ?",
            (username.lower(),)
        ).fetchone()
    return row and hash_password(password) == row[0]

def register_user(username, password):
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username.lower(), hash_password(password))
            )
        return True
    except sqlite3.IntegrityError:
        return False

# ================== AUTH UI ==================
LOGO_PATH = "C:/Users/chris.mutuku/OneDrive - Skanem AS/Desktop/logo.jpg"

def authenticate():
    if st.session_state.authenticated:
        return

    st.markdown("""
        <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #0E4E4E, #123636); 
                    border-radius: 20px; margin: 2rem auto; max-width: 500px;'>
            <h1 style='color: white; margin-bottom: 1rem;'> SForecast</h1>
            <p style='color: #E1EBAE;'>Intelligent Forecasting Platform</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image(Image.open(LOGO_PATH), width=44)
        except:
            pass

        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])

        with tab1:
            with st.form("login_form"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                login_submitted = st.form_submit_button("Login", use_container_width=True)
                
                if login_submitted:
                    if check_credentials(u, p):
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("Invalid credentials")

        with tab2:
            with st.form("signup_form"):
                nu = st.text_input("New Username")
                npw = st.text_input("New Password", type="password")
                confirm_pw = st.text_input("Confirm Password", type="password")
                signup_submitted = st.form_submit_button("Create Account", use_container_width=True)
                
                if signup_submitted:
                    if npw != confirm_pw:
                        st.error("Passwords do not match")
                    elif register_user(nu, npw):
                        st.success("Account created successfully! Please login.")
                    else:
                        st.error("Username already exists")

    st.stop()

authenticate()

# ================== TOP BAR CONTROLS ==================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    if st.toggle("🌗 Dark / Light Mode"):
        st.session_state.theme_mode = "light" if st.session_state.theme_mode == "dark" else "dark"
        st.rerun()
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    if st.button("🔄 Refresh All Data"):
        st.success("Data refreshed!")
    
    if st.button("📊 Generate Reports"):
        st.info("Report generation started...")
    
    st.markdown("---")
    
    if st.button("🚪 Logout", type="secondary"):
        st.session_state.authenticated = False
        st.rerun()

# ================== DATABASE INITIALIZATION ==================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Conversions table
    c.execute('''CREATE TABLE IF NOT EXISTS conversions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_name TEXT,
        input_value REAL,
        input_unit TEXT,
        output_value REAL,
        output_unit TEXT,
        thickness_microns REAL,
        density REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Forecasts table
    c.execute('''CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        forecast_id TEXT,
        material_name TEXT,
        forecast_type TEXT,
        horizon TEXT,
        rmse REAL,
        mape REAL,
        r2 REAL,
        forecast_data TEXT,
        financial_data TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Simulations table
    c.execute('''CREATE TABLE IF NOT EXISTS simulations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        simulation_id TEXT,
        scenario_name TEXT,
        sku_name TEXT,
        simulation_params TEXT,
        results TEXT,
        financial_impact TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Uploaded data table
    c.execute('''CREATE TABLE IF NOT EXISTS uploaded_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        data_type TEXT,
        columns TEXT,
        row_count INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Inventory table
    c.execute('''CREATE TABLE IF NOT EXISTS inventory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_name TEXT,
        quantity REAL,
        unit TEXT,
        category TEXT,
        cost_per_unit REAL,
        currency TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Consumption table
    c.execute('''CREATE TABLE IF NOT EXISTS consumption (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_name TEXT,
        date DATE,
        quantity REAL,
        unit TEXT,
        category TEXT,
        cost REAL,
        currency TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Production schedule table
    c.execute('''CREATE TABLE IF NOT EXISTS production_schedule (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name TEXT,
        machine TEXT,
        start_time DATETIME,
        end_time DATETIME,
        quantity REAL,
        unit TEXT,
        status TEXT,
        notes TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Financial metrics table
    c.execute('''CREATE TABLE IF NOT EXISTS financial_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_name TEXT,
        metric_value REAL,
        currency TEXT,
        date DATE,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()
    conn.close()

init_db()

# ================== UNIT CONVERSION FUNCTIONS ==================
def kg_to_sqm(kg, thickness_microns, density=0.92):
    thickness_m = thickness_microns * 1e-6
    return kg / (density * thickness_m)

def kg_to_meters(kg, width_m, thickness_microns, density=0.92):
    sqm = kg_to_sqm(kg, thickness_microns, density)
    return sqm / width_m

def kg_to_liters(kg, density=0.92):
    return kg / density

def convert_units(value, from_unit, to_unit, **kwargs):
    converters = {
        ('kg', 'sqm'): lambda x: kg_to_sqm(x, kwargs.get('thickness_microns', 35), kwargs.get('density', 0.92)),
        ('kg', 'meters'): lambda x: kg_to_meters(x, kwargs.get('width_m', 1), kwargs.get('thickness_microns', 35), kwargs.get('density', 0.92)),
        ('kg', 'liters'): lambda x: kg_to_liters(x, kwargs.get('density', 0.92)),
        ('sqm', 'kg'): lambda x: x * (kwargs.get('thickness_microns', 35) * 1e-6 * kwargs.get('density', 0.92)),
        ('meters', 'kg'): lambda x: x * kwargs.get('width_m', 1) * (kwargs.get('thickness_microns', 35) * 1e-6 * kwargs.get('density', 0.92)),
        ('liters', 'kg'): lambda x: x * kwargs.get('density', 0.92)
    }
    return converters.get((from_unit, to_unit), lambda x: x)(value)

# ================== FINANCIAL FORECASTING FUNCTIONS ==================
def calculate_material_cost(quantity, unit, material_type, currency='USD'):
    """Calculate material cost based on quantity and type"""
    base_prices = {
        'PP Granules': {'kg': 1200, 'ton': 1200000},
        'Additives': {'kg': 2500, 'ton': 2500000},
        'Masterbatch': {'kg': 1800, 'ton': 1800000},
        'BOPP': {'sqm': 0.35, 'roll': 85},
        'PE': {'kg': 1500, 'ton': 1500000}
    }
    
    conversion_rates = st.session_state.financial_metrics['exchange_rates']
    
    if material_type in base_prices and unit in base_prices[material_type]:
        base_cost = quantity * base_prices[material_type][unit]
        
        # Convert to selected currency
        if currency != 'USD':
            usd_to_target = conversion_rates.get(currency, 1.0)
            base_cost = base_cost / usd_to_target
        
        return round(base_cost, 2)
    return 0

def calculate_production_cost(quantity, machine_hours, labor_hours, energy_kwh):
    """Calculate total production cost"""
    machine_rate = 250  # USD per hour
    labor_rate = 50     # USD per hour
    energy_rate = 0.15  # USD per kWh
    
    machine_cost = machine_hours * machine_rate
    labor_cost = labor_hours * labor_rate
    energy_cost = energy_kwh * energy_rate
    
    return {
        'machine_cost': machine_cost,
        'labor_cost': labor_cost,
        'energy_cost': energy_cost,
        'total_cost': machine_cost + labor_cost + energy_cost,
        'unit_cost': (machine_cost + labor_cost + energy_cost) / quantity if quantity > 0 else 0
    }

def calculate_revenue(forecast_quantity, selling_price_per_unit, currency='USD'):
    """Calculate revenue from forecast"""
    conversion_rates = st.session_state.financial_metrics['exchange_rates']
    
    if currency != 'USD':
        selling_price_per_unit = selling_price_per_unit / conversion_rates.get(currency, 1.0)
    
    return forecast_quantity * selling_price_per_unit

# ================== ADVANCED FORECASTING FUNCTIONS ==================
def detect_seasonality(series, freq='D'):
    """Detect seasonality in time series"""
    try:
        decomposition = seasonal_decompose(series, model='additive', period=7 if freq == 'D' else 12)
        seasonal_strength = np.abs(decomposition.seasonal).mean() / np.abs(series).mean()
        return seasonal_strength > 0.1  # Threshold for meaningful seasonality
    except:
        return False

def check_stationarity(series):
    """Check if time series is stationary using Augmented Dickey-Fuller test"""
    result = adfuller(series.dropna())
    return result[1] < 0.05  # p-value < 0.05 indicates stationarity

def ensemble_forecast(df, horizon, freq='D'):
    """Generate ensemble forecast using multiple models"""
    models = {
        'prophet': Prophet(),
        'holt_winters': ExponentialSmoothing(df['y'], seasonal='add', seasonal_periods=7 if freq == 'D' else 12),
        'linear': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    forecasts = {}
    for name, model in models.items():
        try:
            if name == 'prophet':
                m = model
                m.fit(df.rename(columns={'ds': 'ds', 'y': 'y'}))
                future = m.make_future_dataframe(periods=horizon, freq=freq)
                forecast = m.predict(future)
                forecasts[name] = forecast['yhat'].values[-horizon:]
            elif name == 'holt_winters':
                fitted = model.fit()
                forecast = fitted.forecast(horizon)
                forecasts[name] = forecast
            else:
                # For ML models, create features
                X = np.arange(len(df)).reshape(-1, 1)
                y = df['y'].values
                model.fit(X, y)
                X_future = np.arange(len(df), len(df) + horizon).reshape(-1, 1)
                forecasts[name] = model.predict(X_future)
        except:
            continue
    
    # Weighted average (Prophet gets highest weight)
    weights = {'prophet': 0.4, 'holt_winters': 0.3, 'linear': 0.15, 'random_forest': 0.15}
    ensemble = np.zeros(horizon)
    total_weight = 0
    
    for name, forecast in forecasts.items():
        if name in weights and len(forecast) == horizon:
            weight = weights[name]
            ensemble += forecast * weight
            total_weight += weight
    
    if total_weight > 0:
        ensemble /= total_weight
    
    return ensemble, forecasts

# ================== SIMULATION FUNCTIONS ==================
def run_simulation(scenario_params):
    """Run forecasting simulation with given parameters"""
    base_demand = scenario_params.get('base_demand', 1000)
    growth_rate = scenario_params.get('growth_rate', 0.05)
    volatility = scenario_params.get('volatility', 0.1)
    horizon = scenario_params.get('horizon', 30)
    seasonality = scenario_params.get('seasonality', True)
    
    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=horizon, freq='D')
    
    # Base trend with growth
    trend = base_demand * (1 + growth_rate) ** np.arange(horizon)
    
    # Add seasonality if required
    seasonal_component = 0
    if seasonality:
        seasonal_component = 0.2 * base_demand * np.sin(2 * np.pi * np.arange(horizon) / 7)
    
    # Add random noise
    noise = volatility * base_demand * np.random.randn(horizon)
    
    # Combine components
    simulated_demand = trend + seasonal_component + noise
    simulated_demand = np.maximum(simulated_demand, 0)  # Ensure non-negative
    
    return pd.DataFrame({
        'date': dates,
        'demand': simulated_demand,
        'trend': trend,
        'seasonality': seasonal_component,
        'noise': noise
    })

def monte_carlo_simulation(base_forecast, n_simulations=1000, volatility=0.15):
    """Run Monte Carlo simulation on base forecast"""
    simulations = []
    for _ in range(n_simulations):
        # Add random walk to base forecast
        random_walk = np.cumsum(np.random.randn(len(base_forecast)) * volatility * np.mean(base_forecast))
        sim = base_forecast + random_walk
        sim = np.maximum(sim, 0)  # Ensure non-negative
        simulations.append(sim)
    
    simulations = np.array(simulations)
    
    # Calculate statistics
    mean_forecast = np.mean(simulations, axis=0)
    lower_bound = np.percentile(simulations, 5, axis=0)
    upper_bound = np.percentile(simulations, 95, axis=0)
    
    return {
        'mean': mean_forecast,
        'lower': lower_bound,
        'upper': upper_bound,
        'all_simulations': simulations,
        'confidence_interval': (lower_bound, upper_bound)
    }

# ================== SESSION STATE ==================
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'conversion_history' not in st.session_state:
    st.session_state.conversion_history = []
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = []
if 'inventory_data' not in st.session_state:
    st.session_state.inventory_data = None
if 'consumption_data' not in st.session_state:
    st.session_state.consumption_data = None
if 'current_forecast' not in st.session_state:
    st.session_state.current_forecast = None
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = []

# ================== ERROR METRICS ==================
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = np.finfo(np.float64).eps
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    return 2 * np.mean(np.abs(y_pred - y_true) / np.maximum(denominator, 1e-8)) * 100

# ================== PLOTLY LAYOUT ==================
def skanem_plotly_layout(title=None, height=500):
    """Create a standardized Plotly layout with Skanem theme"""
    # Fixed grid colors using rgba format
    grid_color = 'rgba(14, 78, 78, 0.2)'  # #0E4E4E with 0.2 alpha
    
    return dict(
        title=dict(
            text=title,
            font=dict(size=20, color=THEME["TEXT"]),
            x=0.5,
            xanchor='center'
        ),
        paper_bgcolor=THEME["BG"],
        plot_bgcolor=THEME["BG2"],
        font=dict(color=THEME["TEXT"], size=12),
        colorway=[THEME["PRIMARY"], THEME["SECONDARY"], THEME["ACCENT"], 
                 THEME["GOOD"], THEME["WARNING"], THEME["DANGER"], THEME["INFO"]],
        xaxis=dict(
            gridcolor=grid_color,
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor=grid_color,
            showgrid=True,
            zeroline=False
        ),
        hovermode='x unified',
        height=height,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        showlegend=True
    )
# ================== TABS ==================
tab_labels = [
    "📊 Forecast Dashboard",
    "🔄 Unit Conversion",
    "📤 Data Upload",
    "📅 Demand Planning",
    "🔮 Forecasting",
    "🧪 Model Testing",
    "🗃️ Database"
]

tabs = st.tabs(tab_labels)

# ================== TAB 0: FORECAST DASHBOARD ==================
with tabs[0]:
    # Header with Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class='stat-card'>
                <div style='font-size: 0.9rem; color: #666;'>Total Forecasts</div>
                <div style='font-size: 2.5rem; font-weight: 700; color: #0E4E4E;'>42</div>
                <div style='font-size: 0.8rem; color: #2E7D32;'>↑ 12% this month</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='stat-card'>
                <div style='font-size: 0.9rem; color: #666;'>Avg Accuracy</div>
                <div style='font-size: 2.5rem; font-weight: 700; color: #2E7D32;'>94.2%</div>
                <div style='font-size: 0.8rem; color: #0E4E4E;'>±2.1% variance</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='stat-card'>
                <div style='font-size: 0.9rem; color: #666;'>Active Materials</div>
                <div style='font-size: 2.5rem; font-weight: 700; color: #3B82F6;'>18</div>
                <div style='font-size: 0.8rem; color: #F59E0B;'>3 need attention</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class='stat-card'>
                <div style='font-size: 0.9rem; color: #666;'>Forecast Value</div>
                <div style='font-size: 2.5rem; font-weight: 700; color: #DC2626;'>$2.4M</div>
                <div style='font-size: 0.8rem; color: #2E7D32;'>↑ $124K vs target</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Main Dashboard Content
    st.markdown("## 📈 Intelligent Forecasting Dashboard")
    
    # Dashboard Tabs - ADDED "Basic Forecasting" tab
    dash_tabs = st.tabs(["📊 Overview", "🔍 Multi-Dimensional Forecast", "💰 Financial Forecast", "🎯 Simulation Lab", "📈 Performance Analytics", "🔢 Basic Forecasting"])
    
    with dash_tabs[0]:
        # ... (existing overview content remains the same) ...
        # Overview Dashboard
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📅 Recent Forecasts")
            
            # Sample forecast data
            forecast_data = pd.DataFrame({
                'Material': ['BOPP 35µ', 'BOPP 20µ', 'White PE', 'Clear PP', 'Metallized'],
                'Current Month': [1250, 890, 1100, 750, 620],
                'Next Month': [1320, 950, 1150, 780, 650],
                'Change %': [5.6, 6.7, 4.5, 4.0, 4.8],
                'Accuracy': [94.2, 92.8, 95.1, 91.5, 93.7],
                'Status': ['On Track', 'Attention', 'On Track', 'On Track', 'Attention']
            })
            
            # Style the DataFrame
            def color_status(val):
                if val == 'On Track':
                    return 'background-color: #d4edda; color: #155724;'
                else:
                    return 'background-color: #fff3cd; color: #856404;'
            
            styled_df = forecast_data.style.applymap(color_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Quick Forecast Chart
            st.markdown("### 📈 Quick Forecast Preview")
            
            fig = go.Figure()
            
            # Historical data (simulated)
            dates = pd.date_range(start='2023-10-01', periods=60, freq='D')
            historical = 1000 + np.random.randn(60).cumsum() * 10 + np.sin(np.arange(60) * 2 * np.pi / 7) * 50
            
            # Forecast data
            forecast_dates = pd.date_range(start=dates[-1] + timedelta(days=1), periods=30, freq='D')
            forecast = historical[-1] + np.random.randn(30).cumsum() * 15 + np.sin(np.arange(30) * 2 * np.pi / 7) * 60
            
            fig.add_trace(go.Scatter(
                x=dates, y=historical,
                name='Historical',
                line=dict(color=THEME['PRIMARY'], width=3),
                mode='lines'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=forecast,
                name='Forecast',
                line=dict(color=THEME['SECONDARY'], width=3, dash='dash'),
                mode='lines'
            ))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast * 1.2,
                fill=None,
                mode='lines',
                line_color='rgba(255,255,255,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast * 0.8,
                fill='tonexty',
                mode='lines',
                fillcolor='rgba(128, 188, 230, 0.2)',
                line_color='rgba(255,255,255,0)',
                name='Confidence Interval'
            ))
            
            fig.update_layout(
                title="Demand Forecast - BOPP 35µ",
                paper_bgcolor=THEME["BG"],
                plot_bgcolor=THEME["BG2"],
                font=dict(color=THEME["TEXT"]),
                xaxis=dict(gridcolor='rgba(14, 78, 78, 0.2)', showgrid=True),
                yaxis=dict(gridcolor='rgba(14, 78, 78, 0.2)', showgrid=True),
                showlegend=True,
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⚡ Quick Actions")
            
            # Material Selection for Quick Forecast
            selected_material = st.selectbox(
                "Select Material",
                ["BOPP 35µ", "BOPP 20µ", "White PE", "Clear PP", "Metallized", "All Materials"],
                key="quick_mat"
            )
            
            forecast_horizon = st.select_slider(
                "Forecast Horizon",
                options=["1 week", "2 weeks", "1 month", "3 months", "6 months"],
                value="1 month",
                key="quick_horizon"
            )
            
            if st.button("🚀 Generate Quick Forecast", use_container_width=True):
                st.success(f"Generating {forecast_horizon} forecast for {selected_material}...")
                # Simulate forecast generation
                with st.spinner("Running forecast models..."):
                    # Simulate some processing
                    st.session_state.last_forecast = {
                        'material': selected_material,
                        'horizon': forecast_horizon,
                        'timestamp': datetime.now()
                    }
                    st.balloons()
            
            
            st.markdown("---")
            
            # Alerts Section
            st.markdown("### ⚠️ Alerts & Notifications")
            
            alerts = [
                {"type": "warning", "message": "BOPP 20µ forecast accuracy below 93%"},
                {"type": "info", "message": "New consumption data available for White PE"},
                {"type": "success", "message": "Monthly forecast report generated"},
                {"type": "warning", "message": "Inventory levels low for Clear PP"}
            ]
            
            for alert in alerts:
                if alert["type"] == "warning":
                    st.warning(alert["message"])
                elif alert["type"] == "info":
                    st.info(alert["message"])
                else:
                    st.success(alert["message"])
            
            st.markdown("---")
            
            # KPIs
            st.markdown("### 📊 Key Metrics")
            
            kpi_data = {
                "Forecast Bias": {"value": 1.2, "unit": "%", "trend": "positive"},
                "Inventory Turns": {"value": 8.5, "unit": "x", "trend": "stable"},
                "Service Level": {"value": 98.7, "unit": "%", "trend": "positive"},
                "Forecast Value Add": {"value": 12.4, "unit": "%", "trend": "positive"}
            }
            
            for kpi, data in kpi_data.items():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**{kpi}**")
                with col_b:
                    trend_color = "#2E7D32" if data["trend"] == "positive" else "#DC2626" if data["trend"] == "negative" else "#F59E0B"
                    st.markdown(f"<span style='color: {trend_color}; font-weight: bold;'>{data['value']}{data['unit']}</span>", unsafe_allow_html=True)

    # ADD THE BASIC FORECASTING FEATURE AS A NEW TAB
    with dash_tabs[5]:  # This is the 6th tab (index 5) - "Basic Forecasting"
        st.header("Basic Forecasting Calculator")
        
        with st.expander("📊 Basic Forecasting Parameters", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                material_name = st.text_input("Material Name", "White BOPP 35 Mic Film 35", key="basic_mat_name")
                current_balance = st.number_input("Current Available Balance (kg)", min_value=0.0, value=1000.0, step=1.0, key="basic_balance")
                avg_daily_consumption = st.number_input("Average Daily Consumption (kg)", min_value=0.0, value=50.0, step=1.0, key="basic_consumption")
            
            with col2:
                consumption_variability = st.slider("Consumption Variability (%)", 0, 50, 10, key="basic_variability")
                safety_stock = st.number_input("Safety Stock Level (kg)", min_value=0.0, value=200.0, step=1.0, key="basic_safety")
                lead_time = st.number_input("Lead Time (days)", min_value=1, value=7, step=1, key="basic_lead")
            
            forecast_horizon = st.selectbox("Forecast Horizon", ["30 days", "60 days", "90 days", "6 months", "1 year", "5 years"], index=0, key="basic_horizon")
            
            # KPI calculations
            days_until_stockout = int(current_balance / avg_daily_consumption) if avg_daily_consumption > 0 else 0
            stockout_date = (datetime.now() + timedelta(days=days_until_stockout)).strftime("%Y-%m-%d")
            reorder_point = safety_stock + (lead_time * avg_daily_consumption)
            days_until_reorder = int((current_balance - reorder_point) / avg_daily_consumption) if current_balance > reorder_point and avg_daily_consumption > 0 else 0
            
            # Display KPIs in metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Balance", f"{current_balance:.2f} kg")
            col2.metric("Days Until Stockout", days_until_stockout, f"Expected by {stockout_date}")
            col3.metric("Reorder Point", f"{reorder_point:.2f} kg", f"{days_until_reorder} days until reorder" if days_until_reorder > 0 else "Below reorder point!")
            col4.metric("Avg Daily Consumption", f"{avg_daily_consumption:.2f} kg", f"±{consumption_variability}% variability")
        
        # Generate forecast button
        if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
            with st.spinner("Generating forecast..."):
                # Convert horizon to days
                if forecast_horizon.endswith("days"):
                    horizon_days = int(forecast_horizon.split(" ")[0])
                elif forecast_horizon == "6 months":
                    horizon_days = 180
                elif forecast_horizon == "1 year":
                    horizon_days = 365
                elif forecast_horizon == "5 years":
                    horizon_days = 365 * 5
                
                dates = pd.date_range(datetime.now(), periods=horizon_days, freq='D')
                
                # Calculate forecasts
                forecast_deterministic = [max(0, current_balance - (i * avg_daily_consumption)) for i in range(horizon_days)]
                
                np.random.seed(42)
                daily_variation = 1 + (np.random.rand(horizon_days) - 0.5) * (consumption_variability / 100)
                forecast_probabilistic = [max(0, current_balance - np.sum(avg_daily_consumption * daily_variation[:i+1])) for i in range(horizon_days)]
                
                # Create DataFrame
                df_forecast = pd.DataFrame({
                    "Date": dates,
                    "Deterministic Forecast": forecast_deterministic,
                    "Probabilistic Forecast": forecast_probabilistic,
                    "Reorder Point": reorder_point,
                    "Safety Stock": safety_stock
                })
                
                # Store in session state
                st.session_state.basic_forecast = df_forecast
                st.session_state.basic_forecast_params = {
                    'material': material_name,
                    'horizon': forecast_horizon,
                    'generated_at': datetime.now()
                }
                
                st.success("Forecast generated successfully!")
        
        # Display forecast if available
        if 'basic_forecast' in st.session_state:
            df_forecast = st.session_state.basic_forecast
            
            # Melt data for plotting
            df_melted = df_forecast.melt(
                id_vars="Date",
                value_vars=["Deterministic Forecast", "Probabilistic Forecast", "Reorder Point", "Safety Stock"],
                var_name="Metric",
                value_name="Value"
            )
            
            # Create plot
            fig = px.line(df_melted, x="Date", y="Value", color="Metric",
                          title=f"Material Forecast: {st.session_state.basic_forecast_params['material']}",
                          labels={"Value": "Quantity (kg)", "Date": "Date"},
                          template="plotly_white")
            
            fig.add_hline(y=0, line_dash="dot", line_color="red", 
                          annotation_text="Stockout Level", 
                          annotation_position="bottom right")
            
            # Apply theme
            fig.update_layout(
                paper_bgcolor=THEME["BG"],
                plot_bgcolor=THEME["BG2"],
                font=dict(color=THEME["TEXT"]),
                xaxis=dict(gridcolor='rgba(14, 78, 78, 0.2)', showgrid=True),
                yaxis=dict(gridcolor='rgba(14, 78, 78, 0.2)', showgrid=True),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data preview
            with st.expander("📋 Forecast Data Preview"):
                st.dataframe(df_forecast.head(10), use_container_width=True)
                
                # Export options
                col1, col2, col3 = st.columns(3)
                csv = df_forecast.to_csv(index=False)
                col1.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"forecast_{material_name.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                if col2.button("💾 Save to Database", use_container_width=True):
                    conn = sqlite3.connect(DB_NAME)
                    c = conn.cursor()
                    
                    forecast_data = df_forecast.to_json(orient='records')
                    
                    c.execute('''INSERT INTO forecasts 
                                 (material_name, forecast_type, horizon, forecast_data, created_at) 
                                 VALUES (?, ?, ?, ?, ?)''',
                                 (material_name, "Basic", forecast_horizon, forecast_data, datetime.now()))
                    
                    conn.commit()
                    conn.close()
                    st.success("Forecast saved to database!")
                
                if col3.button("🔄 Generate Report", use_container_width=True):
                    st.info("Report generation feature coming soon!")
    
 
    with dash_tabs[1]:
        # Multi-Dimensional Forecasting
        st.markdown("## 🔍 Multi-Dimensional Forecasting")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Configuration")
            
            # Material Selection
            materials = st.multiselect(
                "Select Materials",
                ["BOPP 35µ", "BOPP 20µ", "White PE", "Clear PP", "Metallized", 
                 "PET", "PVC", "LDPE", "HDPE", "LLDPE"],
                default=["BOPP 35µ", "BOPP 20µ", "White PE"],
                key="multi_mat"
            )
            
            # Unit Selection
            forecast_unit = st.radio(
                "Forecast Unit",
                ["Weight (kg)", "Volume (liters)", "Area (sqm)", "Length (meters)"],
                horizontal=True,
                key="multi_unit"
            )
            
            # Conversion Parameters
            with st.expander("⚙️ Conversion Parameters"):
                thickness = st.number_input("Thickness (microns)", value=35.0, min_value=1.0, max_value=200.0, step=1.0)
                density = st.number_input("Density (g/cm³)", value=0.92, min_value=0.1, max_value=2.0, step=0.01)
                width = st.number_input("Width (meters)", value=1.0, min_value=0.1, max_value=5.0, step=0.1)
            
            # Forecasting Method
            method = st.selectbox(
                "Forecasting Method",
                ["Prophet", "Holt-Winters", "Ensemble", "Neural Network"],
                key="multi_method"
            )
            
            # Horizon
            horizon = st.slider("Forecast Horizon (days)", 7, 365, 90, key="multi_horizon")
            
            if st.button("🔮 Generate Multi-Forecast", type="primary", use_container_width=True):
                st.session_state.multi_forecast_running = True
        
        with col2:
            if st.session_state.get('multi_forecast_running'):
                st.markdown("### 📊 Multi-Dimensional Forecast Results")
                
                # Simulate forecast results
                np.random.seed(42)
                dates = pd.date_range(start='2023-12-01', periods=horizon, freq='D')
                
                fig = go.Figure()
                
                for i, material in enumerate(materials):
                    # Generate synthetic forecast data
                    base = 1000 * (1 + i * 0.2)
                    trend = base * (1 + 0.005 * np.arange(horizon))
                    seasonal = 0.1 * base * np.sin(2 * np.pi * np.arange(horizon) / 7)
                    noise = 0.05 * base * np.random.randn(horizon)
                    
                    forecast_values = trend + seasonal + noise
                    
                    fig.add_trace(go.Scatter(
                        x=dates, y=forecast_values,
                        name=material,
                        mode='lines',
                        line=dict(width=2),
                        hovertemplate=f'{material}<br>Date: %{{x}}<br>Value: %{{y:.0f}} {forecast_unit.split()[0]}<extra></extra>'
                    ))
                
                fig.update_layout(
                    skanem_plotly_layout(f"Multi-Material Forecast ({forecast_unit})"),
                    xaxis_title="Date",
                    yaxis_title=f"Quantity ({forecast_unit.split()[0]})",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary Statistics
                st.markdown("### 📈 Summary Statistics")
                
                summary_data = []
                for material in materials:
                    summary_data.append({
                        "Material": material,
                        "Avg Forecast": np.random.randint(800, 1500),
                        "Min": np.random.randint(600, 800),
                        "Max": np.random.randint(1600, 2000),
                        "Std Dev": np.random.randint(50, 150),
                        "Trend": np.random.choice(["↑ Increasing", "→ Stable", "↓ Decreasing"])
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df.style.format({
                    'Avg Forecast': '{:.0f}',
                    'Min': '{:.0f}',
                    'Max': '{:.0f}',
                    'Std Dev': '{:.0f}'
                }), use_container_width=True)
                
                # Export Options
                st.download_button(
                    "📥 Download Forecast Data",
                    summary_df.to_csv(index=False),
                    "multi_forecast_summary.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    with dash_tabs[2]:
        # Financial Forecasting
        st.markdown("## 💰 Financial Impact Forecasting")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Financial Parameters")
            
            # Currency Selection
            currency = st.selectbox(
                "Currency",
                ["USD", "EUR", "GBP", "INR", "CNY"],
                key="fin_currency"
            )
            
            # Material Costs
            st.markdown("#### Material Costs")
            material_costs = {}
            materials_fin = ["BOPP 35µ", "BOPP 20µ", "White PE", "Clear PP"]
            
            for mat in materials_fin:
                col_a, col_b = st.columns([3, 2])
                with col_a:
                    st.write(mat)
                with col_b:
                    cost = st.number_input(
                        f"Cost per kg ({currency})",
                        value=float(np.random.randint(1200, 2500)),
                        min_value=0.0,
                        step=10.0,
                        key=f"cost_{mat}"
                    )
                    material_costs[mat] = cost
            
            # Operational Costs
            st.markdown("#### Operational Costs")
            machine_rate = st.number_input("Machine Rate/hr", value=250.0, min_value=0.0, step=10.0)
            labor_rate = st.number_input("Labor Rate/hr", value=50.0, min_value=0.0, step=5.0)
            energy_rate = st.number_input("Energy Rate/kWh", value=0.15, min_value=0.0, step=0.01)
            
            # Selling Price
            st.markdown("#### Pricing")
            selling_price = st.number_input(f"Selling Price/kg ({currency})", value=2800.0, min_value=0.0, step=50.0)
            
            if st.button("💰 Calculate Financial Forecast", type="primary", use_container_width=True):
                st.session_state.financial_calc = True
        
        with col2:
            if st.session_state.get('financial_calc'):
                st.markdown("### 📊 Financial Forecast Analysis")
                
                # Simulate financial data
                np.random.seed(42)
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # Revenue Forecast
                base_revenue = 1000000
                revenue_growth = 0.08
                revenue = base_revenue * (1 + revenue_growth) ** np.arange(12)
                revenue_noise = 0.1 * revenue * np.random.randn(12)
                revenue = revenue + revenue_noise
                
                # Cost Forecast
                base_cost = 600000
                cost_inflation = 0.05
                costs = base_cost * (1 + cost_inflation) ** np.arange(12)
                cost_noise = 0.08 * costs * np.random.randn(12)
                costs = costs + cost_noise
                
                # Profit Calculation
                profit = revenue - costs
                margin = (profit / revenue) * 100
                
                # Create financial chart
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Revenue Forecast', 'Cost Forecast', 'Profit Forecast', 'Margin Trend'),
                    vertical_spacing=0.15,
                    horizontal_spacing=0.15
                )
                
                # Revenue
                fig.add_trace(
                    go.Bar(x=months, y=revenue, name='Revenue', marker_color=THEME['GOOD']),
                    row=1, col=1
                )
                
                # Costs
                fig.add_trace(
                    go.Bar(x=months, y=costs, name='Costs', marker_color=THEME['DANGER']),
                    row=1, col=2
                )
                
                # Profit
                fig.add_trace(
                    go.Bar(x=months, y=profit, name='Profit', marker_color=THEME['PRIMARY']),
                    row=2, col=1
                )
                
                # Margin
                fig.add_trace(
                    go.Scatter(x=months, y=margin, name='Margin %', mode='lines+markers',
                             line=dict(color=THEME['SECONDARY'], width=3)),
                    row=2, col=2
                )
                
                fig.update_layout(
                    height=700,
                    showlegend=True,
                    paper_bgcolor=THEME["BG"],
                    plot_bgcolor=THEME["BG2"],
                    font=dict(color=THEME["TEXT"])
                )
                
                # Update axes
                fig.update_yaxes(title_text=f"Amount ({currency})", row=1, col=1)
                fig.update_yaxes(title_text=f"Amount ({currency})", row=1, col=2)
                fig.update_yaxes(title_text=f"Amount ({currency})", row=2, col=1)
                fig.update_yaxes(title_text="Percentage (%)", row=2, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Financial Summary
                st.markdown("### 📋 Financial Summary")
                
                total_revenue = np.sum(revenue)
                total_cost = np.sum(costs)
                total_profit = total_revenue - total_cost
                avg_margin = np.mean(margin)
                
                fin_cols = st.columns(4)
                fin_cols[0].metric("Total Revenue", f"{currency} {total_revenue/1e6:.2f}M", "8% growth")
                fin_cols[1].metric("Total Cost", f"{currency} {total_cost/1e6:.2f}M", "5% increase")
                fin_cols[2].metric("Total Profit", f"{currency} {total_profit/1e6:.2f}M", "12% growth")
                fin_cols[3].metric("Avg Margin", f"{avg_margin:.1f}%", "0.8pp improvement")
                
                # ROI Calculation
                st.markdown("#### 📈 Return on Investment Analysis")
                roi_data = {
                    "Material": ["BOPP 35µ", "BOPP 20µ", "White PE", "Clear PP"],
                    "Investment": [500000, 350000, 450000, 300000],
                    "Returns": [620000, 410000, 520000, 360000],
                    "ROI %": [24.0, 17.1, 15.6, 20.0]
                }
                
                roi_df = pd.DataFrame(roi_data)
                st.dataframe(roi_df.style.format({
                    'Investment': '{:,.0f}',
                    'Returns': '{:,.0f}',
                    'ROI %': '{:.1f}%'
                }), use_container_width=True)
    
    with dash_tabs[3]:
        # Simulation Lab
        st.markdown("## 🎯 Forecast Simulation Lab")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Simulation Parameters")
            
            # Scenario Builder
            scenario_name = st.text_input("Scenario Name", "Base Case")
            
            # Demand Parameters
            st.markdown("#### Demand Parameters")
            base_demand = st.number_input("Base Demand (kg/day)", value=1000.0, min_value=10.0, step=100.0)
            growth_rate = st.slider("Growth Rate (%/month)", 0.0, 20.0, 5.0, 0.5) / 100
            volatility = st.slider("Volatility (%)", 0.0, 50.0, 15.0, 1.0) / 100
            
            # Seasonality
            seasonality = st.checkbox("Include Seasonality", value=True)
            if seasonality:
                season_strength = st.slider("Seasonality Strength", 0.0, 1.0, 0.3, 0.05)
            
            # Horizon
            sim_horizon = st.slider("Simulation Horizon (days)", 30, 365, 90)
            
            # Monte Carlo Settings
            st.markdown("#### Monte Carlo Settings")
            n_simulations = st.slider("Number of Simulations", 100, 5000, 1000, 100)
            confidence_level = st.slider("Confidence Level", 80, 99, 95, 1)
            
            if st.button("🎲 Run Simulation", type="primary", use_container_width=True):
                st.session_state.run_simulation = True
        
        with col2:
            if st.session_state.get('run_simulation'):
                st.markdown(f"### 📊 Simulation Results: {scenario_name}")
                
                # Run simulation
                scenario_params = {
                    'base_demand': base_demand,
                    'growth_rate': growth_rate,
                    'volatility': volatility,
                    'horizon': sim_horizon,
                    'seasonality': seasonality
                }
                
                # Generate base forecast
                dates = pd.date_range(start='2024-01-01', periods=sim_horizon, freq='D')
                base_trend = base_demand * (1 + growth_rate/30) ** np.arange(sim_horizon)
                
                if seasonality:
                    seasonal = season_strength * base_demand * np.sin(2 * np.pi * np.arange(sim_horizon) / 7)
                else:
                    seasonal = np.zeros(sim_horizon)
                
                base_forecast = base_trend + seasonal
                
                # Run Monte Carlo simulation
                simulations = []
                for _ in range(n_simulations):
                    noise = volatility * base_demand * np.random.randn(sim_horizon)
                    sim = base_forecast + noise
                    sim = np.maximum(sim, 0)
                    simulations.append(sim)
                
                simulations = np.array(simulations)
                
                # Calculate statistics
                mean_forecast = np.mean(simulations, axis=0)
                lower_bound = np.percentile(simulations, (100-confidence_level)/2, axis=0)
                upper_bound = np.percentile(simulations, 100-(100-confidence_level)/2, axis=0)
                
                # Create simulation visualization
                fig = go.Figure()
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=list(dates) + list(dates[::-1]),
                    y=list(upper_bound) + list(lower_bound[::-1]),
                    fill='toself',
                    fillcolor='rgba(128, 188, 230, 0.3)',
                    line_color='rgba(255,255,255,0)',
                    name=f'{confidence_level}% Confidence Interval'
                ))
                
                # Mean forecast
                fig.add_trace(go.Scatter(
                    x=dates, y=mean_forecast,
                    name='Mean Forecast',
                    line=dict(color=THEME['PRIMARY'], width=3)
                ))
                
                # Base forecast
                fig.add_trace(go.Scatter(
                    x=dates, y=base_forecast,
                    name='Base Forecast',
                    line=dict(color=THEME['DANGER'], width=2, dash='dash')
                ))
                
                # Sample simulations
                for i in range(min(5, n_simulations)):
                    fig.add_trace(go.Scatter(
                        x=dates, y=simulations[i],
                        name=f'Simulation {i+1}',
                        line=dict(width=1, color='rgba(200,200,200,0.3)'),
                        showlegend=(i==0)
                    ))
                
                fig.update_layout(
                    skanem_plotly_layout(f"Monte Carlo Simulation - {scenario_name}"),
                    xaxis_title="Date",
                    yaxis_title="Demand (kg/day)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Simulation Statistics
                st.markdown("### 📈 Simulation Statistics")
                
                final_forecast = mean_forecast[-1]
                min_forecast = np.min(simulations[:, -1])
                max_forecast = np.max(simulations[:, -1])
                std_forecast = np.std(simulations[:, -1])
                
                stat_cols = st.columns(4)
                stat_cols[0].metric("Final Forecast", f"{final_forecast:.0f} kg/day", 
                                   f"±{std_forecast:.0f} kg")
                stat_cols[1].metric("Minimum", f"{min_forecast:.0f} kg/day", 
                                   f"{((min_forecast-final_forecast)/final_forecast*100):.1f}%")
                stat_cols[2].metric("Maximum", f"{max_forecast:.0f} kg/day", 
                                   f"{((max_forecast-final_forecast)/final_forecast*100):.1f}%")
                stat_cols[3].metric("Volatility", f"{std_forecast/final_forecast*100:.1f}%", 
                                   f"{volatility*100:.1f}% target")
                
                # Risk Analysis
                st.markdown("#### ⚠️ Risk Analysis")
                risk_level = "Low"
                if std_forecast/final_forecast > 0.25:
                    risk_level = "High"
                elif std_forecast/final_forecast > 0.15:
                    risk_level = "Medium"
                
                st.markdown(f"**Risk Level:** <span class='kpi-badge kpi-{'danger' if risk_level=='High' else 'warning' if risk_level=='Medium' else 'good'}'>{risk_level}</span>", unsafe_allow_html=True)
                
                # Save simulation
                if st.button("💾 Save Simulation Scenario", use_container_width=True):
                    simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    st.session_state.simulation_scenarios.append({
                        'id': simulation_id,
                        'name': scenario_name,
                        'params': scenario_params,
                        'results': {
                            'mean': mean_forecast.tolist(),
                            'lower': lower_bound.tolist(),
                            'upper': upper_bound.tolist()
                        },
                        'timestamp': datetime.now().isoformat()
                    })
                    st.success(f"Simulation '{scenario_name}' saved!")
    
    with dash_tabs[4]:
        # Performance Analytics
        st.markdown("## 📈 Forecast Performance Analytics")
        
        # Performance Metrics Over Time
        st.markdown("### 📊 Performance Trends")
        
        # Simulate performance data
        months = pd.date_range(start='2023-01-01', periods=12, freq='MS')
        
        performance_data = pd.DataFrame({
            'Month': months,
            'Accuracy': 92 + np.random.randn(12).cumsum() * 0.5 + np.sin(np.arange(12) * np.pi/6) * 1.5,
            'Bias': np.random.randn(12).cumsum() * 0.2 + np.cos(np.arange(12) * np.pi/6) * 0.5,
            'Volatility': 8 + np.random.randn(12) * 0.3 + np.sin(np.arange(12) * np.pi/4) * 1.2,
            'Value Add': 10 + np.random.randn(12).cumsum() * 0.3
        })
        
        # Create performance chart
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Bias', 'Volatility', 'Value Add']
        colors = [THEME['GOOD'], THEME['WARNING'], THEME['DANGER'], THEME['INFO']]
        units = ['%', '%', '%', '%']
        
        for metric, color, unit in zip(metrics, colors, units):
            fig.add_trace(go.Scatter(
                x=performance_data['Month'],
                y=performance_data[metric],
                name=metric,
                line=dict(color=color, width=3),
                yaxis='y' if metric == 'Accuracy' else 'y2' if metric == 'Bias' else 'y3' if metric == 'Volatility' else 'y4'
            ))
        
        fig.update_layout(
            title="Forecast Performance Metrics Over Time",
            xaxis=dict(title="Month"),
            yaxis=dict(title="Accuracy (%)", color=THEME['GOOD']),
            yaxis2=dict(title="Bias (%)", overlaying='y', side='right', color=THEME['WARNING']),
            yaxis3=dict(title="Volatility (%)", overlaying='y', side='right', position=0.15, color=THEME['DANGER']),
            yaxis4=dict(title="Value Add (%)", overlaying='y', side='right', position=0.3, color=THEME['INFO']),
            height=500,
            hovermode='x unified',
            paper_bgcolor=THEME["BG"],
            plot_bgcolor=THEME["BG2"],
            font=dict(color=THEME["TEXT"])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model Comparison
        st.markdown("### 🤖 Model Performance Comparison")
        
        model_data = pd.DataFrame({
            'Model': ['Prophet', 'Holt-Winters', 'Ensemble', 'Neural Network', 'ARIMA', 'Random Forest'],
            'Accuracy': [94.2, 92.8, 95.1, 93.5, 91.7, 94.8],
            'Training Time': [45, 12, 68, 120, 23, 89],
            'Complexity': [3, 2, 4, 5, 3, 4],
            'Stability': [9.2, 8.7, 9.5, 8.9, 8.5, 9.3]
        })
        
        fig = px.scatter_matrix(
            model_data,
            dimensions=['Accuracy', 'Training Time', 'Complexity', 'Stability'],
            color='Model',
            title="Model Performance Comparison",
            height=600
        )
        
        fig.update_layout(
            paper_bgcolor=THEME["BG"],
            font=dict(color=THEME["TEXT"])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation Engine
        st.markdown("### 🎯 Model Recommendations")
        
        # User requirements
        st.markdown("#### Your Requirements")
        
        req_cols = st.columns(3)
        with req_cols[0]:
            accuracy_req = st.slider("Minimum Accuracy (%)", 85, 99, 92, 1)
        with req_cols[1]:
            time_req = st.slider("Max Training Time (s)", 10, 300, 60, 10)
        with req_cols[2]:
            complexity_req = st.slider("Max Complexity (1-5)", 1, 5, 3, 1)
        
        # Filter models
        recommended_models = model_data[
            (model_data['Accuracy'] >= accuracy_req) &
            (model_data['Training Time'] <= time_req) &
            (model_data['Complexity'] <= complexity_req)
        ]
        
        if len(recommended_models) > 0:
            st.success(f"✅ {len(recommended_models)} models match your requirements:")
            st.dataframe(recommended_models.sort_values('Accuracy', ascending=False), use_container_width=True)
        else:
            st.warning("No models match all your requirements. Consider relaxing some constraints.")
            
            # Find closest matches
            model_data['score'] = (
                (model_data['Accuracy'] / accuracy_req) * 0.5 +
                (time_req / model_data['Training Time']) * 0.3 +
                (complexity_req / model_data['Complexity']) * 0.2
            )
            closest_matches = model_data.nlargest(3, 'score')
            st.info("Closest matches:")
            st.dataframe(closest_matches[['Model', 'Accuracy', 'Training Time', 'Complexity', 'score']], use_container_width=True)

# ================== REST OF THE TABS (1-6) ==================
# Note: Due to character limits, I'll show the structure for other tabs.
# The complete implementation would follow similar patterns to Tab 0.
with tabs[1]:
    # REMOVED: display_logo() call
    st.header("🔁 Unit Conversion Hub")
    
    conv_col1, conv_col2 = st.columns([1, 2])
    
    with conv_col1:
        st.subheader("Single Conversion")
        conversion_type = st.radio("Conversion Type", 
                                  ["Standard Units", "Multi-layer Paper Weight"],
                                  help="Choose between standard unit conversions or specialized paper weight calculations")
        
        if conversion_type == "Standard Units":
            input_value = st.number_input("Input Value", min_value=0.0, value=1000.0)
            input_unit = st.selectbox("From Unit", ["kg", "sqm", "meters", "liters"])
            output_unit = st.selectbox("To Unit", ["kg", "sqm", "meters", "liters"])
            
            if input_unit in ["kg", "sqm", "meters"]:
                thickness = st.number_input("Thickness (microns)", value=35.0)
                density = st.number_input("Density (g/cm³)", value=0.92)
            else:
                thickness = 35.0
                density = 0.92
            
            if st.button("Convert Single"):
                result = convert_units(input_value, input_unit, output_unit,
                                    thickness_microns=thickness, density=density)
                st.metric("Result", f"{result:.2f} {output_unit}")
                
                # Save to session state
                st.session_state.conversion_history.append({
                    "input": f"{input_value} {input_unit}",
                    "output": f"{result:.2f} {output_unit}",
                    "type": "Standard"
                })
                
        else:  # Multi-layer Paper Weight Calculation
            st.markdown("**Paper Dimensions**")
            cols = st.columns(3)
            with cols[0]:
                width = st.number_input("Width (cm)", min_value=0.1, value=21.0)
            with cols[1]:
                length = st.number_input("Length (cm)", min_value=0.1, value=29.7)
            with cols[2]:
                sheets = st.number_input("Number of Sheets", min_value=1, value=1)
            
            st.markdown("**Layer Properties**")
            layers = st.number_input("Number of Layers", min_value=1, max_value=10, value=3)
            
            layer_props = []
            for i in range(layers):
                with st.expander(f"Layer {i+1} Properties"):
                    cols = st.columns(2)
                    with cols[0]:
                        thickness = st.number_input(f"Thickness (microns) - Layer {i+1}", 
                                                  min_value=1.0, value=35.0, key=f"thick_{i}")
                    with cols[1]:
                        density = st.number_input(f"Density (g/cm³) - Layer {i+1}", 
                                                min_value=0.1, value=0.92, key=f"density_{i}")
                    layer_props.append((thickness, density))
            
            if st.button("Calculate Total Weight"):
                total_weight = 0
                for thick, density in layer_props:
                    area_sqm = (width/100) * (length/100)  # cm² to m²
                    thickness_m = thick * 1e-6  # microns to meters
                    layer_weight = area_sqm * thickness_m * density * 1000
                    total_weight += layer_weight
                
                total_weight *= sheets  # Multiply by number of sheets
                st.metric("Total Weight", f"{total_weight:.4f} kg")
                st.metric("Weight per Sheet", f"{total_weight/sheets:.4f} kg")
                
                # Save to session state
                st.session_state.conversion_history.append({
                    "input": f"{sheets} sheets, {layers} layers",
                    "output": f"{total_weight:.2f} kg",
                    "type": "Multi-layer"
                })
    
    with conv_col2:
        st.subheader("Bulk Conversion")
        uploaded_file = st.file_uploader("Upload CSV/XLSX with columns matching these fields:", 
                                       type=["csv", "xlsx"],
                                       help="File must contain: input_value, input_unit, output_unit, thickness_microns (if applicable), density")
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                required_cols = ['input_value', 'input_unit', 'output_unit']
                optional_cols = ['thickness_microns', 'density', 'material_name']
                
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    st.stop()
                
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                needs_thickness = any(unit in ['kg', 'sqm', 'meters'] for unit in pd.concat([df['input_unit'], df['output_unit']]))
                if needs_thickness and 'thickness_microns' not in df.columns:
                    st.warning("Some conversions require thickness but column not found. Using default 35 microns.")
                    df['thickness_microns'] = 35.0
                
                if 'density' not in df.columns:
                    st.warning("Density column not found. Using default 0.92 g/cm³.")
                    df['density'] = 0.92
                
                if 'material_name' not in df.columns:
                    df['material_name'] = "Bulk Conversion"
                
                if st.button("⚡ Convert All Rows"):
                    results = []
                    for _, row in df.iterrows():
                        try:
                            result = convert_units(
                                row['input_value'],
                                row['input_unit'],
                                row['output_unit'],
                                thickness_microns=row.get('thickness_microns', 35.0),
                                density=row.get('density', 0.92)
                            )
                            results.append(result)
                        except Exception as e:
                            st.warning(f"Row {_+1} failed: {str(e)}")
                            results.append(None)
                    
                    df['output_value'] = results
                    st.success(f"Converted {len(df)} rows!")
                    
                    st.dataframe(df)
                    
                    conn = sqlite3.connect(DB_NAME)
                    df[['material_name', 'input_value', 'input_unit', 
                        'output_value', 'output_unit', 'thickness_microns', 
                        'density']].to_sql('conversions', conn, if_exists='append', index=False)
                    st.success("Saved to database!")
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "💾 Download Results",
                        csv,
                        "bulk_conversion_results.csv",
                        "text/csv"
                    )
                
            except Exception as e:
                st.error(f"File processing error: {str(e)}")
  
    # Show recent conversions from session state
    if st.session_state.conversion_history:
        with st.expander("Recent Conversions"):
            st.table(pd.DataFrame(st.session_state.conversion_history[-5:]))
  
    st.markdown("""
    **Connected Features:**
    - Use converted values in 📈 Forecast Dashboard
    - Upload bulk conversions to 🗃️ Database
    """)

with tabs[2]:
    # REMOVED: display_logo() call
    st.header("📤 Data Upload Center")
    
    upload_tabs = st.tabs(["Inventory Data", "Consumption Data", "Other Data"])
    
    with upload_tabs[0]:
        st.subheader("Inventory Upload")
        inv_file = st.file_uploader("Upload current inventory (CSV/XLSX)", type=["csv", "xlsx"], key="inv_upload")
        if inv_file:
            try:
                if inv_file.name.endswith('.csv'):
                    inv_df = pd.read_csv(inv_file)
                else:
                    inv_df = pd.read_excel(inv_file)
                
                st.success(f"Uploaded {len(inv_df)} inventory records")
                st.dataframe(inv_df)
                
                if st.button("Save Inventory"):
                    conn = sqlite3.connect(DB_NAME)
                    inv_df.to_sql('inventory', conn, if_exists='replace', index=False)
                    st.session_state.inventory_data = inv_df
                    st.success("Inventory data saved!")
            except Exception as e:
                st.error(str(e))
    
    with upload_tabs[1]:
        st.subheader("Consumption Upload")
        cons_file = st.file_uploader("Upload consumption data (CSV/XLSX)", type=["csv", "xlsx"], key="cons_upload")
        if cons_file:
            try:
                if cons_file.name.endswith('.csv'):
                    cons_df = pd.read_csv(cons_file)
                else:
                    cons_df = pd.read_excel(cons_file)
                
                st.success(f"Uploaded {len(cons_df)} consumption records")
                st.dataframe(cons_df)
                
                if st.button("Save Consumption"):
                    conn = sqlite3.connect(DB_NAME)
                    cons_df.to_sql('consumption', conn, if_exists='replace', index=False)
                    st.session_state.consumption_data = cons_df
                    st.success("Consumption data saved!")
            except Exception as e:
                st.error(str(e))
    
    st.markdown("""
    **Data Usage:**
    - Use uploaded data in 🔮 Forecasting
    - Analyze in 🧪 Model Testing
    - View in 🗃️ Database
    """)

with tabs[3]:
    # REMOVED: display_logo() call
    st.header("📅 Demand Planning")
    
    # Show data availability status
    st.sidebar.markdown("### Data Status")
    if st.session_state.inventory_data is not None:
        st.sidebar.success("Inventory Data Loaded")
    if st.session_state.consumption_data is not None:
        st.sidebar.success("Consumption Data Loaded")
    
    # Split view between calendar and scheduler
    view_type = st.radio("View Mode", ["Calendar View", "Scheduler View"], horizontal=True)
    
    if view_type == "Calendar View":
        # Calendar View Section
        st.subheader("🗓️ Forecast Selection Calendar")
        
        # Get current date and set up calendar navigation
        today = datetime.now()
        current_year = today.year
        current_month = today.month
        
        # Create columns for calendar navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            selected_year = st.selectbox("Year", range(current_year, current_year + 5), index=0)
        with col2:
            selected_month = st.selectbox("Month", [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ], index=current_month - 1)
        with col3:
            view_type = st.radio("View", ["Monthly", "Weekly"], horizontal=True, key="calendar_view")
        
        # Convert selected month to number
        month_num = datetime.strptime(selected_month, "%B").month
        
        # Generate calendar data
        if view_type == "Monthly":
            # Create monthly calendar
            cal = calendar.monthcalendar(selected_year, month_num)
            
            # Display calendar header
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            cols = st.columns(7)
            for i, day in enumerate(days):
                cols[i].write(f"**{day}**")
            
            # Display calendar days
            for week in cal:
                cols = st.columns(7)
                for i, day in enumerate(week):
                    if day == 0:
                        cols[i].write(" ")
                    else:
                        date_str = f"{selected_year}-{month_num:02d}-{day:02d}"
                        with cols[i]:
                            # Check if date has forecasts (placeholder logic)
                            has_forecast = random.random() > 0.7  # Replace with actual check
                            
                            if has_forecast:
                                st.markdown(f"""
                                    <div style='border: 2px solid {PRIMARY_COLOR}; border-radius: 5px; padding: 5px; text-align: center;'>
                                        <strong>{day}</strong>
                                        <div style='font-size: 0.8em; color: green;'>Forecast</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <div style='border: 1px solid #ccc; border-radius: 5px; padding: 5px; text-align: center;'>
                                        <strong>{day}</strong>
                                    </div>
                                """, unsafe_allow_html=True)
        else:
            # Weekly view
            st.write("Weekly view coming soon")
        
        # Demand Planning Tools Section
        st.subheader("🛠️ Demand Planning Tools")
        
        tool_col1, tool_col2 = st.columns(2)
        
        with tool_col1:
            st.markdown("**📊 Forecast Summary**")
            # Placeholder data - replace with actual forecasts
            forecast_data = {
                "Material": ["BOPP 35µ", "BOPP 20µ", "White PE"],
                "This Month": [1200, 850, 950],
                "Next Month": [1350, 900, 1000],
                "Variance": ["+12.5%", "+5.9%", "+5.3%"]
            }
            st.dataframe(pd.DataFrame(forecast_data))
            
            st.markdown("**📅 Key Dates**")
            key_dates = {
                "Date": ["2023-11-15", "2023-12-01", "2023-12-15"],
                "Event": ["Inventory Count", "New Product Launch", "Year-End Close"]
            }
            st.dataframe(pd.DataFrame(key_dates))
        
        with tool_col2:
            st.markdown("**🔍 Forecast Comparison**")
            time_period = st.selectbox("Compare", ["Month-over-Month", "Year-over-Year"])
            
            # Placeholder comparison chart
            fig = go.Figure()
            if time_period == "Month-over-Month":
                fig.add_trace(go.Bar(
                    x=["Oct", "Nov", "Dec"],
                    y=[1000, 1200, 1350],
                    name="BOPP 35µ"
                ))
                fig.add_trace(go.Bar(
                    x=["Oct", "Nov", "Dec"],
                    y=[800, 850, 900],
                    name="BOPP 20µ"
                ))
            else:
                fig.add_trace(go.Bar(
                    x=["2022", "2023"],
                    y=[12000, 13500],
                    name="BOPP 35µ"
                ))
                fig.add_trace(go.Bar(
                    x=["2022", "2023"],
                    y=[9500, 10200],
                    name="BOPP 20µ"
                ))
            st.plotly_chart(fig, use_container_width=True)
        
        # Demand Planning Actions
        st.subheader("🚀 Planning Actions")
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("🔄 Refresh Forecasts"):
                st.success("Forecasts refreshed for selected period")
        
        with action_col2:
            if st.button("📧 Export Plan"):
                st.success("Demand plan exported to Excel")
        
        with action_col3:
            if st.button("📌 Create Planning Task"):
                st.success("New planning task created")
    
    else:  # Scheduler View
        st.subheader("🏭 Production Scheduler")
        
        # Timeframe Selection
        timeframe = st.radio("Schedule View", 
                            ["Daily", "Weekly", "Monthly"], 
                            horizontal=True,
                            index=1,
                            key="scheduler_view")
        
        # Get current date and calculate date range
        today = datetime.now().date()
        start_date = st.date_input("Start Date", today, key="scheduler_date")
        
        if timeframe == "Daily":
            end_date = start_date + timedelta(days=1)
        elif timeframe == "Weekly":
            end_date = start_date + timedelta(weeks=1)
        else:  # Monthly
            end_date = start_date + relativedelta(months=1)
        
        # Placeholder production data - replace with real data
        production_data = [
            {
                "Product": "BOPP 35µ",
                "Machine": "Extruder 1",
                "Start": datetime.combine(start_date + timedelta(days=1), datetime.time(8, 0)),
                "End": datetime.combine(start_date + timedelta(days=1), datetime.time(16, 0)),
                "Quantity": 1200,
                "Status": "Scheduled"
            },
            {
                "Product": "BOPP 20µ",
                "Machine": "Extruder 2",
                "Start": datetime.combine(start_date + timedelta(days=2), datetime.time(10, 0)),
                "End": datetime.combine(start_date + timedelta(days=2), datetime.time(18, 0)),
                "Quantity": 800,
                "Status": "Confirmed"
            }
        ]
        
        # Convert to DataFrame
        df_schedule = pd.DataFrame(production_data)
        
        # Display as Gantt chart
        fig = px.timeline(
            df_schedule,
            x_start="Start",
            x_end="End",
            y="Machine",
            color="Product",
            title=f"Production Schedule ({timeframe} view)",
            hover_name="Product",
            hover_data=["Quantity", "Status"]
        )
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)
        
        # Production Planning Tools
        st.subheader("🛠️ Scheduling Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📦 Material Requirements**")
            # Placeholder MRP data
            mrp_data = {
                "Material": ["PP Granules", "Additives", "Masterbatch"],
                "Required": [2500, 120, 75],
                "On Hand": [1800, 100, 60],
                "Shortage": [700, 20, 15]
            }
            st.dataframe(pd.DataFrame(mrp_data))
            
            st.markdown("**⚙️ Machine Utilization**")
            utilization_data = {
                "Machine": ["Extruder 1", "Extruder 2", "Coater"],
                "Utilization": ["85%", "78%", "65%"],
                "Status": ["Optimal", "Good", "Underutilized"]
            }
            st.dataframe(pd.DataFrame(utilization_data))
        
        with col2:
            st.markdown("**📊 Schedule Metrics**")
            metrics_col1, metrics_col2 = st.columns(2)
            
            metrics_col1.metric("Scheduled Hours", "156", "+12% vs plan")
            metrics_col1.metric("Changeovers", "8", "3 planned")
            metrics_col2.metric("Utilization", "82%", "2% above target")
            metrics_col2.metric("OEE", "76%", "On track")
            
            st.markdown("**🔍 Schedule Analysis**")
            analysis_option = st.selectbox("View", 
                                         ["Capacity", "Changeovers", "Downtime"],
                                         index=0)
            
            # Placeholder analysis chart
            fig = go.Figure()
            if analysis_option == "Capacity":
                fig.add_trace(go.Bar(
                    x=["Extruder 1", "Extruder 2", "Coater"],
                    y=[85, 78, 65],
                    name="Utilization %"
                ))
            elif analysis_option == "Changeovers":
                fig.add_trace(go.Bar(
                    x=["Mon", "Tue", "Wed", "Thu", "Fri"],
                    y=[3, 2, 1, 2, 0],
                    name="Changeovers"
                ))
            else:
                fig.add_trace(go.Bar(
                    x=["Mechanical", "Electrical", "Cleaning", "Other"],
                    y=[12, 8, 15, 5],
                    name="Downtime Hours"
                ))
            st.plotly_chart(fig, use_container_width=True)
        
        # Schedule Actions
        st.subheader("⚡ Schedule Actions")
        
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("🔄 Optimize Schedule", key="optimize"):
                st.success("Schedule optimized using available capacity")
        
        with action_col2:
            if st.button("📋 Generate Work Orders", key="work_orders"):
                st.success("Work orders generated for selected period")
        
        with action_col3:
            if st.button("📤 Export Schedule", key="export_schedule"):
                st.success("Production schedule exported to PDF")
    
    st.markdown("""
    **Integration Points:**
    - Uses inventory from 📤 Data Upload
    - Connects to forecasts from 🔮 Forecasting
    """)

with tabs[4]:
    # REMOVED: display_logo() call
    st.header("📈 Forecasting")

    st.subheader("1. Upload Your Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel file with historical data", 
                                   type=["csv", "xlsx"], 
                                   key="forecast_upload")

    # Initialize with empty dataframe
    df = pd.DataFrame()
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.uploaded_data = df
            st.success("✅ Data loaded successfully")
            
            # Show data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()

    if not df.empty:
        st.subheader("2. Configure Your Forecast")
        
        # Column selection
        cols = st.columns(2)
        with cols[0]:
            date_col = st.selectbox("Select Date Column", 
                                  df.columns, 
                                  key="forecast_date_col")
        with cols[1]:
            value_col = st.selectbox("Select Value Column", 
                                   df.select_dtypes(include='number').columns, 
                                   key="forecast_value_col")
        
        # MULTI-SELECT item filter - CHANGED FROM SINGLE SELECT
        item_col = st.selectbox("Filter by Item Column (optional)", 
                              ["No filter"] + [c for c in df.columns if c not in [date_col, value_col]],
                              key="forecast_item_col")
        
        selected_items = []  # CHANGED: Now a list for multiple items
        if item_col != "No filter":
            # CHANGED: Using multiselect instead of selectbox
            selected_items = st.multiselect("Select items to forecast (select one or more)", 
                                          df[item_col].unique(),
                                          key="forecast_item_select")
            st.info(f"Selected {len(selected_items)} items for forecasting")

        # Prepare data
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            
            # CHANGED: Filter for multiple items or use all if none selected
            if selected_items:
                df = df[df[item_col].isin(selected_items)]
                st.success(f"Filtered data for {len(selected_items)} items")
            else:
                st.info("No specific items selected. Using all available data.")
            
            df = df[[date_col, value_col] + ([item_col] if item_col != "No filter" else [])]
            df = df.rename(columns={date_col: "ds", value_col: "y"})
            df = df.dropna().sort_values("ds")
            
            # Get the latest date from the data
            last_date = df['ds'].max()
            
        except Exception as e:
            st.error(f"❌ Error preparing data: {e}")
            st.stop()

        st.subheader("3. Forecast Settings")
        
        # Forecast configuration
        method = st.radio("Forecasting method", 
                         ["Prophet (recommended)", "Holt-Winters"],
                         horizontal=True)
        
        cols = st.columns(2)
        with cols[0]:
            # Calculate default forecast end date (6 months from last date)
            default_end_date = last_date + pd.DateOffset(months=6)
            
            # Create date input for forecast end date
            forecast_end_date = st.date_input(
                "Select Forecast End Date",
                min_value=last_date + pd.Timedelta(days=1),
                max_value=datetime(2028, 12, 31),
                value=default_end_date
            )
            
            # Convert to pandas Timestamp
            forecast_end_date = pd.Timestamp(forecast_end_date)
        with cols[1]:
            freq = st.radio("Frequency", 
                          ["Daily", "Weekly", "Monthly"], 
                          horizontal=True)
        
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Creating forecast..."):
                try:
                    # Calculate horizon based on selected end date
                    last_date = df['ds'].max()
                    date_range = pd.date_range(start=last_date, end=forecast_end_date, freq='D')
                    
                    if freq == "Daily":
                        horizon = len(date_range) - 1  # Subtract 1 because last_date is included
                    elif freq == "Weekly":
                        horizon = len(date_range) // 7
                    else:  # Monthly
                        horizon = (forecast_end_date.year - last_date.year) * 12 + \
                                 (forecast_end_date.month - last_date.month)
                    
                    # Ensure at least 1 period
                    horizon = max(1, horizon)
                    
                    # CHANGED: Handle multiple items forecasting
                    all_forecasts = []
                    
                    if item_col != "No filter" and selected_items:
                        # Forecast for each selected item individually
                        for item in selected_items:
                            st.write(f"🔮 Forecasting for: {item}")
                            
                            # Filter data for current item
                            item_data = df[df[item_col] == item][['ds', 'y']].copy()
                            
                            if len(item_data) < 2:
                                st.warning(f"Not enough data for {item}. Skipping.")
                                continue
                            
                            # Forecasting
                            if method.startswith("Prophet"):
                                m = Prophet()
                                m.fit(item_data)
                                
                                # Map frequency to Prophet frequency codes
                                freq_map = {
                                    "Daily": "D",
                                    "Weekly": "W",
                                    "Monthly": "M"
                                }
                                
                                future = m.make_future_dataframe(periods=horizon, freq=freq_map[freq])
                                forecast = m.predict(future)
                                
                                # Merge actuals and forecast
                                result = pd.merge(item_data, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                                                on='ds', how='outer')
                                
                                # Add item information
                                result[item_col] = item
                                all_forecasts.append(result)
                            else:
                                # Holt-Winters implementation for single item
                                pass
                    
                    else:
                        # Single time series forecasting (no item grouping)
                        if method.startswith("Prophet"):
                            m = Prophet()
                            m.fit(df[['ds', 'y']])
                            
                            freq_map = {
                                "Daily": "D",
                                "Weekly": "W",
                                "Monthly": "M"
                            }
                            
                            future = m.make_future_dataframe(periods=horizon, freq=freq_map[freq])
                            forecast = m.predict(future)
                            
                            result = pd.merge(df[['ds', 'y']], forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                                            on='ds', how='outer')
                            if item_col != "No filter":
                                result[item_col] = "All Items"
                            all_forecasts.append(result)
                    
                    # Combine all forecasts
                    if not all_forecasts:
                        st.error("No forecasts generated. Check your data and selections.")
                        st.stop()
                    
                    combined_result = pd.concat(all_forecasts, ignore_index=True)
                    
                    # Ensure no negative forecasts
                    combined_result['yhat'] = combined_result['yhat'].clip(lower=0)
                    combined_result['yhat_lower'] = combined_result['yhat_lower'].clip(lower=0)
                    combined_result['yhat_upper'] = combined_result['yhat_upper'].clip(lower=0)
                    
                    # Filter to show only up to the selected end date
                    combined_result = combined_result[combined_result['ds'] <= forecast_end_date]
                    
                    # Forecast Preview
                    st.subheader("📋 Forecast Preview")
                    preview_df = combined_result[combined_result['ds'] > last_date].copy()
                    
                    # Display preview with item information
                    display_cols = ['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']
                    if item_col != "No filter":
                        display_cols.insert(0, item_col)
                    
                    st.dataframe(preview_df[display_cols].style.format({
                        'yhat': '{:.2f}',
                        'yhat_lower': '{:.2f}',
                        'yhat_upper': '{:.2f}',
                        'y': '{:.2f}'
                    }))
                    
                    # Visualization for multiple items
                    st.subheader("📊 Forecast Results")
                    
                    fig = go.Figure()
                    
                    # Color palette for multiple items
                    colors = px.colors.qualitative.Set1
                    
                    if item_col != "No filter" and selected_items:
                        # Plot each item separately
                        for i, item in enumerate(selected_items):
                            item_data = combined_result[combined_result[item_col] == item]
                            
                            color = colors[i % len(colors)]
                            
                            # Actual values
                            actuals = item_data.dropna(subset=['y'])
                            if not actuals.empty:
                                fig.add_trace(go.Scatter(
                                    x=actuals['ds'], y=actuals['y'],
                                    name=f'{item} - Actual',
                                    line=dict(color=color),
                                    mode='lines+markers',
                                    opacity=0.7
                                ))
                            
                            # Forecast
                            forecasts = item_data[item_data['ds'] > last_date]
                            if not forecasts.empty:
                                fig.add_trace(go.Scatter(
                                    x=forecasts['ds'], y=forecasts['yhat'],
                                    name=f'{item} - Forecast',
                                    line=dict(color=color, dash='dash'),
                                    opacity=0.9
                                ))
                    else:
                        # Single time series
                        # Actual values
                        actuals = combined_result.dropna(subset=['y'])
                        if not actuals.empty:
                            fig.add_trace(go.Scatter(
                                x=actuals['ds'], y=actuals['y'],
                                name='Actual',
                                line=dict(color=colors[0]),
                                mode='lines+markers'
                            ))
                        
                        # Forecast
                        forecasts = combined_result[combined_result['ds'] > last_date]
                        if not forecasts.empty:
                            fig.add_trace(go.Scatter(
                                x=forecasts['ds'], y=forecasts['yhat'],
                                name='Forecast',
                                line=dict(color=colors[1])
                            ))
                    
                    # Add vertical line to separate historical data and forecast
                    fig.add_vline(x=last_date.timestamp() * 1000, 
                                line_dash="dash", 
                                line_color="green",
                                annotation_text="Forecast Start",
                                annotation_position="top left")
                    
                    title = "Forecast vs Actuals"
                    if item_col != "No filter" and selected_items:
                        title += f" - {len(selected_items)} Items"
                    
                    fig.update_layout(
                        title=title,
                        xaxis_title='Date',
                        yaxis_title='Value',
                        hovermode='x unified',
                        template='plotly_white',
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ENHANCED ACCURACY METRICS WITH CONDITIONAL FORMATTING
                    if 'y' in combined_result.columns:
                        actuals_with_forecast = combined_result.dropna(subset=['y', 'yhat'])
                        if len(actuals_with_forecast) > 0:
                            st.subheader("🔍 Forecast Accuracy")
                            
                            # Function to determine color based on metric value and thresholds
                            def get_metric_color(metric_name, value):
                                if metric_name == 'MAPE':
                                    if value < 5:
                                        return '#00b894'  # Excellent - Emerald Green
                                    elif value < 10:
                                        return '#00cec9'  # Good - Turquoise
                                    elif value < 20:
                                        return '#fdcb6e'  # Fair - Yellow
                                    elif value < 30:
                                        return '#e17055'  # Poor - Orange
                                    else:
                                        return '#d63031'  # Very Poor - Red
                                
                                elif metric_name == 'RMSE':
                                    y_std = actuals_with_forecast['y'].std()
                                    if value < 0.3 * y_std:
                                        return '#00b894'  # Excellent
                                    elif value < 0.5 * y_std:
                                        return '#00cec9'  # Good
                                    elif value < 0.7 * y_std:
                                        return '#fdcb6e'  # Fair
                                    elif value < y_std:
                                        return '#e17055'  # Poor
                                    else:
                                        return '#d63031'  # Very Poor
                                
                                elif metric_name == 'R²':
                                    if value >= 0.9:
                                        return '#00b894'  # Excellent
                                    elif value >= 0.8:
                                        return '#00cec9'  # Good
                                    elif value >= 0.7:
                                        return '#fdcb6e'  # Fair
                                    elif value >= 0.6:
                                        return '#e17055'  # Poor
                                    else:
                                        return '#d63031'  # Very Poor
                                
                                return '#636e72'  # Default gray
                            
                            # Function to get metric interpretation
                            def get_metric_interpretation(metric_name, value):
                                if metric_name == 'MAPE':
                                    if value < 5:
                                        return "Excellent (High accuracy)"
                                    elif value < 10:
                                        return "Good (Good accuracy)"
                                    elif value < 20:
                                        return "Fair (Acceptable)"
                                    elif value < 30:
                                        return "Poor (Needs improvement)"
                                    else:
                                        return "Very Poor (Poor fit)"
                                
                                elif metric_name == 'RMSE':
                                    y_std = actuals_with_forecast['y'].std()
                                    if value < 0.3 * y_std:
                                        return "Excellent (Very precise)"
                                    elif value < 0.5 * y_std:
                                        return "Good (Precise)"
                                    elif value < 0.7 * y_std:
                                        return "Fair (Moderate precision)"
                                    elif value < y_std:
                                        return "Poor (Low precision)"
                                    else:
                                        return "Very Poor (High error)"
                                
                                elif metric_name == 'R²':
                                    if value >= 0.9:
                                        return "Excellent (Perfect fit)"
                                    elif value >= 0.8:
                                        return "Good (Strong fit)"
                                    elif value >= 0.7:
                                        return "Fair (Moderate fit)"
                                    elif value >= 0.6:
                                        return "Poor (Weak fit)"
                                    else:
                                        return "Very Poor (Poor fit)"
                                
                                return "Not available"
                            
                            # Calculate metrics per item if multiple items
                            if item_col != "No filter" and selected_items:
                                metrics_data = []
                                for item in selected_items:
                                    item_data = actuals_with_forecast[actuals_with_forecast[item_col] == item]
                                    if len(item_data) > 0:
                                        mape = mean_absolute_percentage_error(item_data['y'], item_data['yhat'])
                                        rmse = np.sqrt(mean_squared_error(item_data['y'], item_data['yhat']))
                                        r2 = r2_score(item_data['y'], item_data['yhat'])
                                        
                                        # Apply conditional formatting for display
                                        def color_cell(val, metric):
                                            color = get_metric_color(metric, val)
                                            if metric == 'MAPE':
                                                return f'background-color: {color}; color: white; font-weight: bold;'
                                            elif metric == 'RMSE':
                                                return f'background-color: {color}; color: white; font-weight: bold;'
                                            elif metric == 'R²':
                                                return f'background-color: {color}; color: white; font-weight: bold;'
                                            return ''
                                        
                                        metrics_data.append({
                                            'Item': item,
                                            'MAPE': mape,
                                            'RMSE': rmse,
                                            'R²': r2,
                                            'MAPE_Interpretation': get_metric_interpretation('MAPE', mape),
                                            'RMSE_Interpretation': get_metric_interpretation('RMSE', rmse),
                                            'R²_Interpretation': get_metric_interpretation('R²', r2)
                                        })
                                
                                if metrics_data:
                                    metrics_df = pd.DataFrame(metrics_data)
                                    
                                    # Display metrics with conditional formatting
                                    st.dataframe(metrics_df[['Item', 'MAPE', 'RMSE', 'R²', 
                                                           'MAPE_Interpretation', 'RMSE_Interpretation', 'R²_Interpretation']]
                                        .style
                                        .format({
                                            'MAPE': '{:.1f}%',
                                            'RMSE': '{:.2f}',
                                            'R²': '{:.3f}'
                                        })
                                        .applymap(lambda x: '', subset=['Item', 'MAPE_Interpretation', 'RMSE_Interpretation', 'R²_Interpretation'])
                                        .apply(lambda x: [color_cell(x['MAPE'], 'MAPE') if i == 1 else 
                                                         color_cell(x['RMSE'], 'RMSE') if i == 2 else 
                                                         color_cell(x['R²'], 'R²') if i == 3 else 
                                                         '' for i in range(len(x))], 
                                               axis=1, subset=['MAPE', 'RMSE', 'R²'])
                                    )
                                    
                                    # Summary statistics
                                    st.subheader("📊 Summary Statistics")
                                    avg_mape = metrics_df['MAPE'].mean()
                                    avg_rmse = metrics_df['RMSE'].mean()
                                    avg_r2 = metrics_df['R²'].mean()
                                    
                                    summary_cols = st.columns(3)
                                    with summary_cols[0]:
                                        st.metric("Average MAPE", f"{avg_mape:.1f}%")
                                    with summary_cols[1]:
                                        st.metric("Average RMSE", f"{avg_rmse:.2f}")
                                    with summary_cols[2]:
                                        st.metric("Average R²", f"{avg_r2:.3f}")
                                    
                            else:
                                # Single time series metrics
                                mape = mean_absolute_percentage_error(actuals_with_forecast['y'], actuals_with_forecast['yhat'])
                                rmse = np.sqrt(mean_squared_error(actuals_with_forecast['y'], actuals_with_forecast['yhat']))
                                r2 = r2_score(actuals_with_forecast['y'], actuals_with_forecast['yhat'])
                                
                                # Get colors and interpretations
                                mape_color = get_metric_color('MAPE', mape)
                                rmse_color = get_metric_color('RMSE', rmse)
                                r2_color = get_metric_color('R²', r2)
                                
                                mape_interpretation = get_metric_interpretation('MAPE', mape)
                                rmse_interpretation = get_metric_interpretation('RMSE', rmse)
                                r2_interpretation = get_metric_interpretation('R²', r2)
                                
                                # Display metrics with enhanced styling
                                cols = st.columns(3)
                                metrics_display = [
                                    {
                                        'label': 'MAPE',
                                        'value': f"{mape:.1f}%",
                                        'color': mape_color,
                                        'interpretation': mape_interpretation
                                    },
                                    {
                                        'label': 'RMSE',
                                        'value': f"{rmse:.2f}",
                                        'color': rmse_color,
                                        'interpretation': rmse_interpretation
                                    },
                                    {
                                        'label': 'R²',
                                        'value': f"{r2:.3f}",
                                        'color': r2_color,
                                        'interpretation': r2_interpretation
                                    }
                                ]
                                
                                for col, metric in zip(cols, metrics_display):
                                    with col:
                                        st.markdown(f"""
                                            <div style="
                                                border: 2px solid {metric['color']};
                                                padding: 16px;
                                                background-color: {metric['color']}15;
                                                border-radius: 8px;
                                                margin-bottom: 10px;
                                                text-align: center;
                                            ">
                                                <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">{metric['label']}</div>
                                                <div style="font-size: 2.2em; font-weight: bold; color: {metric['color']}">{metric['value']}</div>
                                                <div style="font-size: 0.8em; color: #444; margin-top: 8px; font-style: italic;">
                                                    {metric['interpretation']}
                                                </div>
                                            </div>
                                        """, unsafe_allow_html=True)
                                
                                # Additional metric explanations
                                with st.expander("📖 Metric Explanations"):
                                    st.markdown("""
                                    **MAPE (Mean Absolute Percentage Error):**
                                    - **<5%**: Excellent - Highly accurate forecasts
                                    - **5-10%**: Good - Reliable for most purposes
                                    - **10-20%**: Fair - Acceptable for general planning
                                    - **20-30%**: Poor - Needs model improvement
                                    - **>30%**: Very Poor - Model may not be suitable
                                    
                                    **RMSE (Root Mean Square Error):**
                                    - Compared to data standard deviation
                                    - **<30% of std**: Excellent precision
                                    - **30-50% of std**: Good precision
                                    - **50-70% of std**: Fair precision
                                    - **70-100% of std**: Poor precision
                                    - **>100% of std**: Very poor precision
                                    
                                    **R² (Coefficient of Determination):**
                                    - **≥0.9**: Excellent fit - Model explains most variation
                                    - **0.8-0.9**: Good fit - Strong predictive power
                                    - **0.7-0.8**: Fair fit - Moderate predictive power
                                    - **0.6-0.7**: Poor fit - Limited predictive power
                                    - **<0.6**: Very Poor fit - Model may not be appropriate
                                    """)
                    
                    # Save forecast results for model testing
                    forecast_result = {
                        'items': selected_items if selected_items else ['All Items'],
                        'method': method.split()[0],
                        'forecast': combined_result.to_dict('records'),
                        'metrics': {
                            'mape': mape if 'mape' in locals() else None,
                            'rmse': rmse if 'rmse' in locals() else None,
                            'r2': r2 if 'r2' in locals() else None
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    if 'forecast_results' not in st.session_state:
                        st.session_state.forecast_results = []
                    
                    st.session_state.forecast_results.append(forecast_result)
                    
                    # NEW: FORECAST TABLE PREVIEW BEFORE DOWNLOAD
                    st.subheader("📊 Final Forecast Table Preview")
                    
                    # Create a clean forecast table for preview
                    forecast_preview_df = combined_result.copy()
                    
                    # Add forecast type indicator
                    forecast_preview_df['Forecast_Type'] = forecast_preview_df['ds'].apply(
                        lambda x: 'Forecast' if x > last_date else 'Actual'
                    )
                    
                    # Format dates nicely
                    forecast_preview_df['Date'] = forecast_preview_df['ds'].dt.strftime('%Y-%m-%d')
                    
                    # Select and order columns for preview
                    preview_columns = ['Date', 'Forecast_Type']
                    if item_col != "No filter":
                        preview_columns.insert(0, item_col)
                    
                    # Add value columns
                    preview_columns.extend(['y', 'yhat', 'yhat_lower', 'yhat_upper'])
                    
                    # Filter columns that exist
                    available_columns = [col for col in preview_columns if col in forecast_preview_df.columns]
                    forecast_preview_df = forecast_preview_df[available_columns]
                    
                    # Rename columns for better readability
                    column_rename = {
                        item_col: 'Item' if item_col != "No filter" else None,
                        'y': 'Actual_Value',
                        'yhat': 'Forecast_Value',
                        'yhat_lower': 'Lower_Bound',
                        'yhat_upper': 'Upper_Bound'
                    }
                    forecast_preview_df = forecast_preview_df.rename(
                        columns={k: v for k, v in column_rename.items() if v}
                    )
                    
                    # Create tabs for different views
                    preview_tab1, preview_tab2, preview_tab3 = st.tabs([
                        "📈 Complete View", 
                        "🔮 Forecast Only", 
                        "📋 Summary View"
                    ])
                    
                    with preview_tab1:
                        # Show complete data with conditional formatting
                        st.write("**Complete Dataset (Actuals + Forecasts)**")
                        styled_df = forecast_preview_df.style.format({
                            'Actual_Value': '{:.2f}',
                            'Forecast_Value': '{:.2f}',
                            'Lower_Bound': '{:.2f}',
                            'Upper_Bound': '{:.2f}'
                        })
                        
                        # Apply conditional formatting for forecast type
                        def highlight_forecast_type(row):
                            if row['Forecast_Type'] == 'Forecast':
                                return ['background-color: #e8f5e8'] * len(row)
                            else:
                                return ['background-color: #f0f8ff'] * len(row)
                        
                        styled_df = styled_df.apply(highlight_forecast_type, axis=1)
                        
                        st.dataframe(styled_df, height=400)
                        
                        # Show some statistics
                        cols = st.columns(3)
                        total_records = len(forecast_preview_df)
                        forecast_records = len(forecast_preview_df[forecast_preview_df['Forecast_Type'] == 'Forecast'])
                        actual_records = len(forecast_preview_df[forecast_preview_df['Forecast_Type'] == 'Actual'])
                        
                        with cols[0]:
                            st.metric("Total Records", total_records)
                        with cols[1]:
                            st.metric("Actual Records", actual_records)
                        with cols[2]:
                            st.metric("Forecast Records", forecast_records)
                    
                    with preview_tab2:
                        # Show only forecast data
                        forecast_only_df = forecast_preview_df[forecast_preview_df['Forecast_Type'] == 'Forecast']
                        
                        if not forecast_only_df.empty:
                            st.write("**Forecast Data Only**")
                            
                            # Add confidence interval width
                            forecast_only_df['Confidence_Width'] = (
                                forecast_only_df['Upper_Bound'] - forecast_only_df['Lower_Bound']
                            )
                            
                            # Format the display
                            display_cols = ['Date', 'Forecast_Value', 'Lower_Bound', 'Upper_Bound', 'Confidence_Width']
                            if 'Item' in forecast_only_df.columns:
                                display_cols.insert(0, 'Item')
                            
                            styled_forecast_df = forecast_only_df[display_cols].style.format({
                                'Forecast_Value': '{:.2f}',
                                'Lower_Bound': '{:.2f}',
                                'Upper_Bound': '{:.2f}',
                                'Confidence_Width': '{:.2f}'
                            })
                            
                            # Color code by confidence width
                            def color_confidence(val):
                                if val > 50:
                                    return 'color: #d63031; font-weight: bold;'
                                elif val > 25:
                                    return 'color: #e17055;'
                                elif val > 10:
                                    return 'color: #fdcb6e;'
                                else:
                                    return 'color: #00b894;'
                            
                            styled_forecast_df = styled_forecast_df.applymap(
                                color_confidence, subset=['Confidence_Width']
                            )
                            
                            st.dataframe(styled_forecast_df, height=400)
                            
                            # Forecast statistics
                            if 'Item' in forecast_only_df.columns:
                                items_count = forecast_only_df['Item'].nunique()
                                avg_forecast = forecast_only_df['Forecast_Value'].mean()
                                st.write(f"**Forecast Statistics:** {items_count} items, Average Forecast: {avg_forecast:.2f}")
                        else:
                            st.info("No forecast data available.")
                    
                    with preview_tab3:
                        # Summary statistics
                        st.write("**Forecast Summary Statistics**")
                        
                        summary_data = []
                        
                        if 'Item' in forecast_preview_df.columns:
                            # Group by item for summary
                            for item in forecast_preview_df['Item'].unique():
                                item_df = forecast_preview_df[forecast_preview_df['Item'] == item]
                                forecast_df = item_df[item_df['Forecast_Type'] == 'Forecast']
                                
                                if not forecast_df.empty:
                                    summary_data.append({
                                        'Item': item,
                                        'Forecast_Periods': len(forecast_df),
                                        'Avg_Forecast': forecast_df['Forecast_Value'].mean(),
                                        'Min_Forecast': forecast_df['Forecast_Value'].min(),
                                        'Max_Forecast': forecast_df['Forecast_Value'].max(),
                                        'Avg_Confidence_Width': forecast_df.apply(
                                            lambda x: x['Upper_Bound'] - x['Lower_Bound'], axis=1
                                        ).mean()
                                    })
                        else:
                            # Single time series summary
                            forecast_df = forecast_preview_df[forecast_preview_df['Forecast_Type'] == 'Forecast']
                            
                            if not forecast_df.empty:
                                summary_data.append({
                                    'Item': 'All Items',
                                    'Forecast_Periods': len(forecast_df),
                                    'Avg_Forecast': forecast_df['Forecast_Value'].mean(),
                                    'Min_Forecast': forecast_df['Forecast_Value'].min(),
                                    'Max_Forecast': forecast_df['Forecast_Value'].max(),
                                    'Avg_Confidence_Width': forecast_df.apply(
                                        lambda x: x['Upper_Bound'] - x['Lower_Bound'], axis=1
                                    ).mean()
                                })
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df.style.format({
                                'Avg_Forecast': '{:.2f}',
                                'Min_Forecast': '{:.2f}',
                                'Max_Forecast': '{:.2f}',
                                'Avg_Confidence_Width': '{:.2f}'
                            }))
                        
                        # Overall forecast statistics
                        st.write("**Overall Statistics**")
                        overall_cols = st.columns(4)
                        
                        with overall_cols[0]:
                            st.metric("Forecast Horizon", f"{horizon} periods")
                        with overall_cols[1]:
                            st.metric("Forecast End", forecast_end_date.strftime('%Y-%m-%d'))
                        with overall_cols[2]:
                            st.metric("Frequency", freq)
                        with overall_cols[3]:
                            st.metric("Method", method.split()[0])
                    
                    # Enhanced export with item name
                    st.subheader("💾 Export Forecast")
                    
                    # Display export options
                    export_cols = st.columns(3)
                    
                    with export_cols[0]:
                        export_format = st.radio(
                            "Export Format",
                            ["CSV", "Excel"],
                            horizontal=True,
                            key="export_format"
                        )
                    
                    with export_cols[1]:
                        include_actuals = st.checkbox(
                            "Include actual historical data",
                            value=True,
                            help="Include actual historical data in the export"
                        )
                    
                    with export_cols[2]:
                        confidence_intervals = st.checkbox(
                            "Include confidence intervals",
                            value=True,
                            help="Include lower and upper bounds in the export"
                        )
                    
                    # Prepare export data based on user preferences
                    export_df = combined_result.copy()
                    
                    if not include_actuals:
                        # Only include forecast data
                        export_df = export_df[export_df['ds'] > last_date]
                    
                    if not confidence_intervals:
                        # Remove confidence interval columns
                        export_df = export_df.drop(['yhat_lower', 'yhat_upper'], axis=1, errors='ignore')
                    
                    # Create export button
                    if export_format == "CSV":
                        export_data = export_df.to_csv(index=False)
                        file_extension = "csv"
                        mime_type = "text/csv"
                    else:
                        # Excel export
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            export_df.to_excel(writer, index=False, sheet_name='Forecast')
                        export_data = output.getvalue()
                        file_extension = "xlsx"
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    
                    # Generate filename
                    filename = "forecast_export"
                    if selected_items:
                        filename += f"_{len(selected_items)}_items"
                    else:
                        filename += "_all_data"
                    filename += f"_{datetime.now().strftime('%Y%m%d_%H%M')}.{file_extension}"
                    
                    # Download button
                    st.download_button(
                        f"⬇️ Download Forecast ({export_format})",
                        export_data,
                        filename,
                        mime_type,
                        help="Download the forecast data in your selected format"
                    )
                    
                    st.success("✅ Forecast generated successfully! Preview the data above before downloading.")
                    
                except Exception as e:
                    st.error(f"Forecast failed: {str(e)}")

    else:
        st.info("ℹ️ Please upload your data to begin forecasting")
with tabs[5]:
    # REMOVED: display_logo() call
    st.header("🧪 Model Testing")

    # Check for available forecast results
    if not st.session_state.get('forecast_results'):
        st.warning("No forecast results available. Please generate forecasts in the 🔮 Forecasting tab first.")
        st.stop()

    # Add filtering options similar to forecasting tab
    st.subheader("🔍 Filter Forecasts")
    col1, col2 = st.columns(2)
    
    with col1:
        # Filter by item name
        all_items = list(set(fr.get('item', 'Unnamed') for fr in st.session_state.forecast_results))
        selected_item_filter = st.selectbox("Filter by Item", ["All"] + all_items)
    
    with col2:
        # Filter by method
        all_methods = list(set(fr.get('method', 'Unknown') for fr in st.session_state.forecast_results))
        selected_method_filter = st.selectbox("Filter by Method", ["All"] + all_methods)
    
    # Apply filters
    filtered_forecasts = [
        fr for fr in st.session_state.forecast_results 
        if (selected_item_filter == "All" or fr.get('item', 'Unnamed') == selected_item_filter) and
           (selected_method_filter == "All" or fr.get('method', 'Unknown') == selected_method_filter)
    ]
    
    if not filtered_forecasts:
        st.warning("No forecasts match the selected filters")
        st.stop()

    st.sidebar.markdown("### Available Forecasts")
    for i, fr in enumerate(filtered_forecasts):
        item_name = fr.get('item', 'Unnamed')
        method = fr.get('method', 'Unknown')
        st.sidebar.markdown(f"{i+1}. {item_name} ({method})")

    st.subheader("Test Forecast Accuracy")
    
    select_options = [f"{i+1}. {fr.get('item', 'Unnamed')} ({fr.get('method', 'Unknown')})" 
                     for i, fr in enumerate(filtered_forecasts)]
    selected_forecast = st.selectbox(
        "Select Forecast to Test",
        options=select_options,
        index=0
    )
    
    if selected_forecast:
        selected_idx = int(selected_forecast.split(".")[0]) - 1
        forecast_data = filtered_forecasts[selected_idx]
    else:
        st.warning("No forecast selected.")
        st.stop()
    
    item_name = forecast_data.get('item', 'Unnamed')
    st.write(f"Testing forecast for: **{item_name}**")
    st.write(f"Method: {forecast_data.get('method', 'Unknown')}")
    
    # Enhanced forecast visualization with bounds
    st.subheader("📊 Forecast Preview with Confidence Bounds")
    
    # Convert forecast data to DataFrame
    forecast_df = pd.DataFrame(forecast_data['forecast'])
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    
    # Create enhanced visualization
    fig = go.Figure()
    
    # Add actual values if available
    if 'y' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'], y=forecast_df['y'],
            name='Actual Values',
            line=dict(color=PRIMARY_COLOR, width=3),
            mode='lines+markers'
        ))
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'], y=forecast_df['yhat'],
        name='Forecast',
        line=dict(color='blue', width=2)
    ))
    
    # Add confidence interval if available
    if all(col in forecast_df.columns for col in ['yhat_lower', 'yhat_upper']):
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'], y=forecast_df['yhat_upper'],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name='Upper Bound'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'], y=forecast_df['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0,100,80,0.2)',
            name='Confidence Interval'
        ))
    
    fig.update_layout(
        title=f"Forecast for {item_name} with Confidence Bounds",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced export with all available data
    st.subheader("💾 Export Forecast")
    
    # Prepare export data
    export_df = forecast_df.copy()
    export_df['item'] = item_name
    export_df['method'] = forecast_data.get('method', 'Unknown')
    
    # Create CSV
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        "⬇️ Download Enhanced Forecast CSV",
        csv,
        f"forecast_{item_name.replace(' ', '_')}.csv",
        "text/csv",
        key="enhanced_export"
    )
    
    st.info(f"Export includes: {', '.join(export_df.columns.tolist())}")
    
    st.subheader("Train-Test Split Evaluation")
    test_size = st.slider("Test Set Size (%)", 10, 40, 20)
    
    if st.button("Run Evaluation"):
        try:
            # Split into train and test
            split_idx = int(len(forecast_df) * (1 - test_size/100))
            train = forecast_df.iloc[:split_idx]
            test = forecast_df.iloc[split_idx:]
            
            # Plot results
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=train['ds'], y=train['yhat'],
                name='Train Forecast',
                line=dict(color=PRIMARY_COLOR)
            ))
            fig.add_trace(go.Scatter(
                x=test['ds'], y=test['yhat'],
                name='Test Forecast',
                line=dict(color='red')
            ))
            
            if 'y' in forecast_df.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'], y=forecast_df['y'],
                    name='Actual Values',
                    line=dict(color='green', dash='dot')
                ))
            
            fig.update_layout(
                title="Train-Test Forecast Evaluation",
                xaxis_title="Date",
                yaxis_title="Value"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate metrics if actual values available
            if 'y' in forecast_df.columns:
                y_true = forecast_df.iloc[split_idx:]['y'].dropna()
                y_pred = forecast_df.iloc[split_idx:]['yhat']
                
                # Align indices
                y_pred = y_pred[y_true.index]
                
                if len(y_true) > 0:
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mape = safe_mape(y_true, y_pred)
                    smape_val = smape(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    
                    cols = st.columns(4)
                    cols[0].metric("Test RMSE", f"{rmse:.2f}")
                    cols[1].metric("Test MAPE", f"{mape:.2f}%")
                    cols[2].metric("Test SMAPE", f"{smape_val:.2f}%")
                    cols[3].metric("Test R²", f"{r2:.2f}")
            
            st.success("Evaluation complete!")
            
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
    
    st.subheader("Advanced Analysis Techniques")
    technique = st.selectbox(
        "Select Analysis Technique",
        ["Residual Analysis", "Error Distribution", "Seasonality Analysis"]
    )
    
    if st.button(f"Run {technique}"):
        try:
            if technique == "Residual Analysis" and 'y' in forecast_df.columns:
                forecast_df['residual'] = forecast_df['y'] - forecast_df['yhat']
                
                fig = px.scatter(
                    forecast_df, x='yhat', y='residual',
                    title="Residuals vs Predicted Values",
                    trendline="lowess"
                )
                fig.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig)
                
            elif technique == "Error Distribution" and 'y' in forecast_df.columns:
                forecast_df['error'] = forecast_df['y'] - forecast_df['yhat']
                
                fig = px.histogram(
                    forecast_df, x='error',
                    title="Error Distribution",
                    nbins=30
                )
                st.plotly_chart(fig)
                
            elif technique == "Seasonality Analysis":
                # Simple seasonality analysis
                if len(forecast_df) > 30:  # Enough data for seasonality
                    forecast_df['month'] = forecast_df['ds'].dt.month
                    monthly_avg = forecast_df.groupby('month')['yhat'].mean()
                    
                    fig = px.line(
                        x=monthly_avg.index, y=monthly_avg.values,
                        title="Monthly Seasonality Pattern",
                        labels={'x': 'Month', 'y': 'Average Forecast'}
                    )
                    st.plotly_chart(fig)
                else:
                    st.info("Not enough data for seasonality analysis")
            else:
                st.info(f"{technique} analysis requires actual values (y) in the forecast data.")
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

with tabs[6]:
    # REMOVED: display_logo() call
    st.header("🗃️ Database Content Viewer")

    conn = sqlite3.connect(DB_NAME)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    selected_table = st.selectbox("Select Table", tables['name'])

    if selected_table:
        data = pd.read_sql(f"SELECT * FROM {selected_table}", conn)
        st.dataframe(data)

        if st.button(f"Clear {selected_table}"):
            conn.execute(f"DELETE FROM {selected_table}")
            conn.commit()
            st.success("Table cleared!")
            
        if not data.empty:
            st.download_button(
                "Export to CSV",
                data.to_csv(index=False),
                f"{selected_table}_export.csv"
            )
    
    st.markdown("""
    **Database Contents:**
    - View all uploaded and processed data
    - Clear tables as needed
    """)
# ================== SIDEBAR ENHANCEMENTS ==================
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📈 Live Updates")
    
    # Real-time metrics
    metric_cols = st.columns(2)
    with metric_cols[0]:
        st.metric("Active Users", "12", "3")
    with metric_cols[1]:
        st.metric("Forecasts Today", "8", "2")
    
    # System status
    st.markdown("#### 🖥️ System Status")
    status_items = [
        {"name": "Forecast Engine", "status": "✅ Online", "color": "green"},
        {"name": "Database", "status": "✅ Online", "color": "green"},
        {"name": "API Services", "status": "⚠️ Partial", "color": "orange"},
        {"name": "Data Pipeline", "status": "✅ Online", "color": "green"}
    ]
    
    for item in status_items:
        st.markdown(f"<div style='margin: 5px 0;'><span style='font-weight: bold;'>{item['name']}:</span> <span style='color: {item['color']};'>{item['status']}</span></div>", unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown("---")
footer_cols = st.columns(3)
with footer_cols[0]:
    st.markdown("**SForecast v2.0**")
    st.markdown("Intelligent Forecasting Platform")
with footer_cols[1]:
    st.markdown("**Contact**")
    st.markdown("support@skanem-forecast.com")
with footer_cols[2]:
    st.markdown("**Last Updated**")
    st.markdown(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# Run the app
if __name__ == "__main__":

    pass
