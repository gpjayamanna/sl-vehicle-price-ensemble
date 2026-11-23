import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

st.set_page_config(
    page_title="Sri Lankan Vehicle Valuation System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean, centered navbar and elegant card layout
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    #MainMenu, header, footer, .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
    
    [data-testid="stHeaderActionElements"], a.anchorjs-link {
        display: none !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1100px;
        margin: 0 auto;
    }

    /* Centered Header */
    .app-hero {
        text-align: center;
        margin-bottom: 0.5rem;
        padding-bottom: 0rem;
    }
    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0;
    }

    /* Force Centered Minimal Navbar across all container versions */
    .stTabs, [data-testid="stTabs"] {
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    
    .stTabs > div:first-child, [data-testid="stTabs"] > div:first-child {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12) !important;
        margin-bottom: 3.5rem !important;
    }
    
    .stTabs [data-baseweb="tab-list"], [data-testid="stTabs"] [data-baseweb="tab-list"], div[role="tablist"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 0 auto !important;
        gap: 5rem !important;
        background: transparent !important;
        border: none !important;
        width: auto !important;
        max-width: 900px !important;
    }
    
    .stTabs [data-baseweb="tab"], [data-testid="stTabs"] [data-baseweb="tab"], button[role="tab"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.8rem 1.4rem 1rem 1.4rem !important;
        font-size: 1.05rem !important;
        font-weight: 500 !important;
        color: rgba(255, 255, 255, 0.55) !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        transition: all 0.25s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover, [data-testid="stTabs"] [data-baseweb="tab"]:hover, button[role="tab"]:hover {
        color: #ffffff !important;
        border-bottom: 2px solid rgba(245, 158, 11, 0.5) !important;
    }
    
    .stTabs [aria-selected="true"], [data-testid="stTabs"] [aria-selected="true"], button[role="tab"][aria-selected="true"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #f59e0b !important;
    }
    
    .stTabs [data-baseweb="tab-border"], [data-testid="stTabs"] [data-baseweb="tab-border"] {
        display: none !important;
    }
    
    .stTabs [data-baseweb="tab-panel"], [data-testid="stTabs"] [data-baseweb="tab-panel"] {
        width: 100% !important;
    }

    /* Primary Accent Styling (Amber / Warm Gold matched) */
    div.stButton > button {
        background: #f59e0b !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.4rem !important;
        transition: all 0.2s ease !important;
        margin-top: 1rem !important;
    }
    div.stButton > button:hover {
        background: #d97706 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.3) !important;
    }
    
    .res-range {
        background: rgba(245, 158, 11, 0.08);
        border-left: 3px solid #f59e0b;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        font-size: 0.9rem;
    }

    /* Clean Card Container */
    .clean-box {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .col-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        opacity: 0.6;
        margin-bottom: 1rem;
    }

    /* Result Card */
    .result-container {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 1.8rem;
        margin-top: 1.5rem;
        animation: fadeIn 0.4s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .res-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.6;
    }
    .res-price {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }
    .res-sub {
        font-size: 1rem;
        opacity: 0.75;
    }
    .res-range {
        background: rgba(255, 255, 255, 0.06);
        border-left: 3px solid #e2b170;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        font-size: 0.9rem;
    }

    /* Stats Grid */
    .stats-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.8rem;
        margin-top: 1.4rem;
        padding-top: 1.4rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stat-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 0.9rem 0.5rem;
        text-align: center;
    }
    .stat-val {
        font-size: 1.25rem;
        font-weight: 700;
    }
    .stat-lbl {
        font-size: 0.75rem;
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 0.2rem;
    }

    /* Button */
    div.stButton > button {
        background: #2563eb;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
        margin-top: 1rem;
    }
    div.stButton > button:hover {
        background: #1d4ed8;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "vehicle_price_pipeline.joblib")
BENCHMARK_PATH = os.path.join(MODELS_DIR, "benchmark_metrics.json")
METADATA_PATH = os.path.join(MODELS_DIR, "ui_metadata.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "car_price_dataset.csv")

@st.cache_resource
def load_model():
    return joblib.load(PIPELINE_PATH)

@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as f:
        return json.load(f)

@st.cache_data
def load_benchmark():
    with open(BENCHMARK_PATH, "r") as f:
        return pd.read_json(f)

@st.cache_data
def load_sample_data():
    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

try:
    pipeline = load_model()
    metadata = load_metadata()
    benchmark_df = load_benchmark()
except Exception as e:
    st.error(f"Error loading system models: {e}")
    st.stop()

# Header
st.markdown("""
<div class="app-hero">
    <div class="app-title">Sri Lankan Vehicle Valuation System</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "Vehicle Valuation", 
    "Model Evaluation", 
    "Market Analytics"
])

# ----------------- TAB 1: VALUATION -----------------
with tab1:
    left_col, right_col = st.columns([1.1, 1], gap="large")
    
    with left_col:
        st.markdown('<div class="col-title">Vehicle Specifications</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            brand = st.selectbox(
                "Brand", 
                options=metadata["brands"], 
                index=metadata["brands"].index("TOYOTA") if "TOYOTA" in metadata["brands"] else 0
            )
        with c2:
            available_models = metadata["models_by_brand"].get(brand, ["Other"])
            if not available_models:
                available_models = ["Other"]
            model = st.selectbox("Model", options=available_models)

        c3, c4 = st.columns(2)
        with c3:
            condition = st.selectbox("Condition", options=metadata["conditions"])
        with c4:
            town = st.selectbox(
                "Location", 
                options=metadata["towns"], 
                index=metadata["towns"].index("Colombo") if "Colombo" in metadata["towns"] else 0
            )

        yom = st.slider("Manufacture Year (YOM)", min_value=1990, max_value=2024, value=2017)

        c5, c6 = st.columns(2)
        with c5:
            engine_cc = st.number_input("Engine Capacity (cc)", min_value=600, max_value=4500, value=1500, step=100)
        with c6:
            mileage_km = st.number_input("Mileage (KM)", min_value=1000, max_value=400000, value=65000, step=5000)

        c7, c8 = st.columns(2)
        with c7:
            gear = st.selectbox("Transmission", options=metadata["gears"])
        with c8:
            fuel_type = st.selectbox("Fuel Type", options=metadata["fuel_types"])

        st.markdown('<div class="col-title" style="margin-top: 1rem;">Features & Equipment</div>', unsafe_allow_html=True)
        eq1, eq2 = st.columns(2)
        with eq1:
            ac = st.checkbox("Air Conditioning", value=True)
            ps = st.checkbox("Power Steering", value=True)
        with eq2:
            pm = st.checkbox("Power Mirrors", value=True)
            pw = st.checkbox("Power Windows", value=True)
            
        predict_btn = st.button("Calculate Valuation", use_container_width=True)

    with right_col:
        st.markdown('<div class="col-title">Valuation Outcome</div>', unsafe_allow_html=True)
        
        if predict_btn:
            vehicle_age = max(2025 - yom, 0)
            km_per_year = mileage_km / (vehicle_age + 1)
            feature_score = int(ac) + int(ps) + int(pm) + int(pw)
            
            input_data = pd.DataFrame([{
                "Brand_Cleaned": brand,
                "Model_Cleaned": model,
                "Gear": gear,
                "Fuel Type": fuel_type,
                "Town_Cleaned": town,
                "Condition": condition,
                "YOM": yom,
                "Vehicle_Age": vehicle_age,
                "Engine (cc)": engine_cc,
                "Millage(KM)": mileage_km,
                "Km_Per_Year": km_per_year,
                "AIR CONDITION": int(ac),
                "POWER STEERING": int(ps),
                "POWER MIRROR": int(pm),
                "POWER WINDOW": int(pw),
                "Feature_Score": feature_score
            }])
            
            pred_lakhs = pipeline.predict(input_data)[0]
            pred_lakhs = max(pred_lakhs, 2.5)
            pred_millions = pred_lakhs / 10.0
            
            low_range = max(pred_lakhs - 5.5, 2.0)
            high_range = pred_lakhs + 5.5
            
            st.markdown(f"""
            <div class="result-container" style="margin-top: 0;">
                <div>
                    <div class="res-label">Estimated Fair Market Price</div>
                    <div class="res-price">Rs. {pred_millions:.2f} Million</div>
                    <div class="res-sub">Equivalent to <strong>{pred_lakhs:.2f} Lakhs</strong> LKR</div>
                </div>
                <div style="margin-top: 1rem;">
                    <div class="res-range">
                        Valuation Range: <strong>{low_range:.2f} Lakhs</strong> – <strong>{high_range:.2f} Lakhs</strong>
                    </div>
                </div>
                <div class="stats-row" style="grid-template-columns: repeat(2, 1fr);">
                    <div class="stat-card">
                        <div class="stat-val">{vehicle_age} Years</div>
                        <div class="stat-lbl">Vehicle Age</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val">{km_per_year:,.0f} km</div>
                        <div class="stat-lbl">Annual Usage</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val">{feature_score} / 4</div>
                        <div class="stat-lbl">Options Score</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val">90.13%</div>
                        <div class="stat-lbl">Accuracy (R²)</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px dashed rgba(255, 255, 255, 0.12); border-radius: 12px; padding: 3rem 1.5rem; text-align: center;">
                <div style="font-size: 1rem; font-weight: 600; opacity: 0.8; margin-bottom: 0.5rem;">Awaiting Vehicle Input</div>
                <div style="font-size: 0.85rem; opacity: 0.5; max-width: 300px; margin: 0 auto;">Select specifications on the left and click Calculate Valuation to view the predicted market price.</div>
            </div>
            """, unsafe_allow_html=True)

# ----------------- TAB 2: MODEL BENCHMARKS -----------------
with tab2:
    st.markdown('<div class="col-title">Performance Comparison Table</div>', unsafe_allow_html=True)
    st.dataframe(
        benchmark_df.style.format({
            "MAE (LKR Lakhs)": "{:.2f}",
            "RMSE (LKR Lakhs)": "{:.2f}",
            "R2 Score": "{:.4f}",
            "MAPE (%)": "{:.2f}%"
        }),
        use_container_width=True
    )
    
    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)
    
    col_g1, col_g2 = st.columns(2, gap="medium")
    with col_g1:
        fig_r2 = px.bar(
            benchmark_df.sort_values(by="R2 Score"),
            x="R2 Score",
            y="Model",
            color="Paradigm",
            orientation="h",
            title="R² Score (Higher is better)",
            color_discrete_sequence=["#3b82f6", "#10b981", "#8b5cf6"]
        )
        fig_r2.update_layout(
            xaxis_range=[0.80, 0.95],
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=35, b=0),
            height=320
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with col_g2:
        fig_mae = px.bar(
            benchmark_df.sort_values(by="MAE (LKR Lakhs)", ascending=False),
            x="MAE (LKR Lakhs)",
            y="Model",
            color="Paradigm",
            orientation="h",
            title="Mean Absolute Error in Lakhs (Lower is better)",
            color_discrete_sequence=["#3b82f6", "#10b981", "#8b5cf6"]
        )
        fig_mae.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=35, b=0),
            height=320
        )
        st.plotly_chart(fig_mae, use_container_width=True)

# ----------------- TAB 3: MARKET ANALYTICS -----------------
with tab3:
    df_raw = load_sample_data()
    
    col_a1, col_a2 = st.columns(2, gap="medium")
    with col_a1:
        top_brands = df_raw["Brand"].value_counts().head(10).reset_index()
        top_brands.columns = ["Brand", "Count"]
        fig_b = px.bar(
            top_brands, 
            x="Brand", 
            y="Count", 
            title="Top 10 Vehicle Brands in Sri Lanka",
            color_discrete_sequence=["#38bdf8"]
        )
        fig_b.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=35, b=0),
            height=320
        )
        st.plotly_chart(fig_b, use_container_width=True)
        
    with col_a2:
        fuel_price = df_raw.groupby("Fuel Type")["Price"].median().reset_index()
        fig_f = px.bar(
            fuel_price, 
            x="Fuel Type", 
            y="Price", 
            title="Median Market Price by Fuel Type (Lakhs)",
            color_discrete_sequence=["#34d399"]
        )
        fig_f.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=35, b=0),
            height=320
        )
        st.plotly_chart(fig_f, use_container_width=True)
