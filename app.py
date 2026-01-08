import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="HydroGuard AI | Aditya Atmaram", 
    page_icon="💧", 
    layout="wide"
)

# 2. Advanced CSS for Visibility and Innovation
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)), 
        url("https://images.unsplash.com/photo-1518063319789-7217e6706b04?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
    }
    [data-testid="stSidebar"] {
        background-color: #0e1117 !important;
        border-right: 2px solid #00d4ff;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        margin-bottom: 20px;
    }
    div.stButton > button:first-child {
        background-color: #00d4ff !important;
        color: #0e1117 !important;
        font-weight: bold !important;
        width: 100%;
        height: 3.5em;
        border-radius: 10px;
        border: none;
    }
    h1, h2, h3, p, label, .stMarkdown { color: white !important; }
    .stSlider label { color: #00d4ff !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 3. Secure Asset Loading
@st.cache_resource
def load_assets():
    base = os.path.dirname(__file__)
    try:
        with open(os.path.join(base, 'water_model.pkl'), 'rb') as f1:
            m = pickle.load(f1)
        with open(os.path.join(base, 'scaler.pkl'), 'rb') as f2:
            s = pickle.load(f2)
        return m, s
    except Exception as e:
        st.error(f"Asset Error: {e}")
        return None, None

model, scaler = load_assets()

# 4. Professional Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00d4ff;'>💧 HydroGuard</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.write("👤 **Developer:** Aditya Atmaram")
    st.write("🎓 **Status:** B.Tech Mechatronics Candidate")
    st.caption("MPSTME | BIA (AI & Data Science)")
    st.markdown("---")
    st.success("System: Online")
    st.info("Model: Random Forest")

# 5. Dashboard Header
st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); padding: 20px; border-radius: 15px; border-left: 8px solid #00d4ff; margin-bottom: 20px;">
        <h1 style='margin:0;'>Intelligent Water Quality Monitor</h1>
        <p style='margin:0; color: #00d4ff;'>AI-Driven Diagnostic System</p>
    </div>
    """, unsafe_allow_html=True)

# 6. Interactive Sensor Sliders
st.markdown("### 🛰️ Sensor Inputs")
c1, c2, c3 = st.columns(3)
with c1:
    ph = st.slider("pH Level", 0.0, 14.0, 7.0)
    hardness = st.slider("Hardness (mg/L)", 50.0, 400.0, 196.3)
    solids = st.slider("Solids (ppm)", 5000.0, 50000.0, 22000.0)
with c2:
    chlor = st.slider("Chloramines (ppm)", 0.0, 15.0, 7.1)
    sulfate = st.slider("Sulfate (mg/L)", 100.0, 500.0, 333.6)
    cond = st.slider("Conductivity (μS/cm)", 100.0, 800.0, 426.2)
with c3:
    carb = st.slider("Organic Carbon (ppm)", 0.0, 30.0, 14.2)
    trihal = st.slider("Trihalomethanes (μg/L)", 0.0, 130.0, 66.4)
    turb = st.slider("Turbidity
