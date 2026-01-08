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

# 2. CSS: High Contrast & Glassmorphism
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
        padding: 25px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        margin-bottom: 20px;
    }
    div.stButton > button:first-child {
        background-color: #00d4ff !important;
        color: #0e1117 !important;
        font-size: 20px !important;
        font-weight: bold !important;
        height: 3.5em !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
    }
    h1, h2, h3, p, span, label, .stMarkdown {
        color: white !important;
    }
    .stSlider label {
        font-weight: bold !important;
        color: #00d4ff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Asset Loading (Fixed Syntax)
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'water_model.pkl')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    
    try:
        with open(model_path, 'rb') as m_file:
            model = pickle.load(m_file)
        with open(scaler_path, 'rb') as s_file:
            scaler = pickle.load(s_file)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

model, scaler = load_assets()

# 4. SIDEBAR
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #00d4ff;'>💧 HydroGuard</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 👨‍🎓 Project Developer")
    st.write("**Aditya Atmaram**")
    st.write("B.Tech Mechatronics Candidate")
    st.caption("MPSTME | BIA (AI & Data Science)")
    st.markdown("---")
    st.subheader("📡 System Status")
    st.success("Random Forest: Online")
    st.info("Accuracy: 65%")

# 5. MAIN CONTENT
st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); padding: 25px; border-radius: 15px; border-left: 10px solid #00d4ff; margin-bottom: 25px;">
        <h1 style='margin:0; color: white;'>Intelligent Water Quality Monitor</h1>
        <p style='margin:0; opacity: 0.9; color: #00d4ff;'>Mechatronics & Data Science Diagnostic Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

# 6. SLIDERS
st.markdown("### 🎛️ Digital Sensor Simulation")
c1, c2, c3 = st.columns(3)
with c1:
    ph = st.slider("pH Level", 0.0, 14.0, 7.0)
    hardness = st.slider("Hardness (mg/L)", 50.0, 400.0, 196.3)
    solids = st.slider("Solids (ppm)", 5000.0, 50000.0, 22000.0)
with c2:
    chloramines = st.slider("Chloramines (ppm)", 0.0, 15.0, 7.1)
    sulfate = st.slider("Sulfate (mg/L)", 100.0, 500.0, 333.6)
    conductivity = st.slider("Conductivity (μS/cm)", 100.0, 800.0,
