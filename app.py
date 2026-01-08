import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(page_title="HydroGuard AI | Aditya Atmaram", page_icon="💧", layout="wide")

# 2. CSS Styling (Refined for the new Header)
st.markdown("""
    <style>
    .stApp { background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url("https://images.unsplash.com/photo-1518063319789-7217e6706b04?q=80&w=2000&auto=format&fit=crop"); background-size: cover; }
    [data-testid="stSidebar"] { background-color: #0e1117 !important; border-right: 2px solid #00d4ff; }
    .glass-card { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 20px; border: 1px solid rgba(0,212,255,0.3); margin-bottom: 20px;}
    
    /* Unique Header Styling */
    .hero-section {
        background: linear-gradient(90deg, rgba(0,212,255,0.2) 0%, rgba(0,0,0,0) 100%);
        padding: 30px;
        border-left: 5px solid #00d4ff;
        border-radius: 0px 15px 15px 0px;
        margin-bottom: 30px;
    }
    .system-badge {
        background-color: #00d4ff;
        color: #0e1117;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    div.stButton > button { background-color: #00d4ff !important; color: #0e1117 !important; font-weight: bold; width: 100%; height: 3em; border-radius: 10px; border: none; }
    h1, h2, h3, p, label, .stMarkdown { color: white !important; }
    .stSlider label { color: #00d4ff !important; font-weight: bold; }
    .stat-box { background: rgba(0, 212, 255, 0.1); padding: 10px; border-radius: 8px; border: 1px solid rgba(0, 212, 255, 0.3); margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Assets
@st.cache_resource
def load_assets():
    path = os.path.dirname(__file__)
    try:
        with open(os.path.join(path, 'water_model.pkl'), 'rb') as f1: m = pickle.load(f1)
        with open(os.path.join(path, 'scaler.pkl'), 'rb') as f2: s = pickle.load(f2)
        return m, s
    except: return None, None

model, scaler = load_assets()

# 4. SIDEBAR (Untouched as requested)
with st.sidebar:
    st.markdown("<h2 style='color:#00d4ff;'>💧 HydroGuard</h2>", unsafe_allow_html=True)
    st.write("👤 **Aditya Atmaram**")
    st.caption("B.Tech Mechatronics | MPSTME")
    st.caption("AI & Data Science | BIA")
    st.markdown("---")
    st.markdown("### 📊 Model Stats")
    st.markdown("""
    <div class="stat-box"><small>Algorithm</small><br><b>Random Forest</b></div>
    <div class="stat-box"><small>Test Accuracy</small><br><b>65%</b></div>
    <div class="stat-box"><small>Logic</small><br><b>AI + Safety Override</b></div>
    """, unsafe_allow_html=True)
    st.success("System: Operational")

# 5. UNIQUE HEADER
st.markdown("""
    <div class="hero-section">
        <span class="system-badge">System ID: HG-2026-AI</span>
        <h1 style='margin-top:10px; font-size: 42px; letter-spacing: -1px;'>HydroGuard <span style='color:#00d4ff;'>Intelligence</span></h1>
        <p style='font-size: 18px; opacity: 0.8;'>Neural-Network Assisted Water Potability Analysis & Sensor Diagnostics</p>
        <p style='font-size: 14px; color:#00d4ff;'>Developed by Aditya Atmaram | MPSTME Mechatronics Division</p>
    </div>
    """, unsafe_allow_html=True)

# 6. Sensor Inputs
st.markdown("### 🛰️ Real-Time Sensor Array")
c1, c2, c3 = st.columns(3)
with c1:
    v1 = st.slider("pH Level", 0.0, 14.0, 7.0)
    v2 = st.slider("Hardness (mg/L)", 50.0, 400.0, 196.0)
    v3 = st.slider("Solids (TDS)", 5000.0, 50000.0, 22000.0)
with c2:
    v4 = st.slider("Chloramines (ppm)", 0.0, 15.0, 7.0)
    v5 = st.slider("Sulfate (mg/L)", 100.0, 500.0, 333.0)
    v6 = st.slider("Conductivity (μS/cm)", 100.0, 800.0, 426.0)
with c3:
    v7 = st.slider("Organic Carbon (ppm)", 0.0, 30.0, 14.0)
    v8 = st.slider("Trihalomethanes (μg/L)", 0.0, 130.0, 66.0)
    v9 = st.slider("Turbidity (NTU)", 0.0, 7.0, 3.9)

# 7. Diagnostic Logic
if st.button("⚡ EXECUTE SYSTEM DIAGNOSTIC"):
    if model is not None and scaler is not None:
        input_array = np.array([[v1, v2, v3, v4, v5, v6, v7, v8, v9]])
        scaled_features = scaler.transform(input_array)
        
        ai_p = model.predict(scaled_features)[0]
        probs = model.predict_proba(scaled_features)[0]
        conf_val = round(probs
