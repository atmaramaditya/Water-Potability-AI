import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(page_title="HydroGuard AI", page_icon="💧", layout="wide")

# 2. High-Contrast CSS
st.markdown("""
    <style>
    .stApp { background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url("https://images.unsplash.com/photo-1518063319789-7217e6706b04?q=80&w=2000&auto=format&fit=crop"); background-size: cover; }
    [data-testid="stSidebar"] { background-color: #0e1117 !important; border-right: 2px solid #00d4ff; }
    .glass-card { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 20px; border: 1px solid rgba(0,212,255,0.3); }
    div.stButton > button { background-color: #00d4ff !important; color: #0e1117 !important; font-weight: bold; width: 100%; height: 3em; border-radius: 10px; }
    h1, h2, h3, p, label, .stMarkdown { color: white !important; }
    .stSlider label { color: #00d4ff !important; font-weight: bold; }
    /* Sidebar Metric Styling */
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

# 4. Sidebar with Model Statistics
with st.sidebar:
    st.markdown("<h2 style='color:#00d4ff;'>💧 HydroGuard</h2>", unsafe_allow_html=True)
    st.write("👤 **Aditya Atmaram**")
    st.caption("B.Tech Mechatronics | MPSTME")
    st.caption("AI & Data Science | BIA")
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    
    # Clean Statistics Display
    st.markdown("""
    <div class="stat-box">
        <small>Algorithm</small><br><b>Random Forest</b>
    </div>
    <div class="stat-box">
        <small>Test Accuracy</small><br><b>65%</b>
    </div>
    <div class="stat-box">
        <small>F1-Score (Weighted)</small><br><b>0.64</b>
    </div>
    <div class="stat-box">
        <small>Precision (Non-Potable)</small><br><b>0.69</b>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("Performance evaluated on a 656-sample test set.")
    st.markdown("---")
    st.success("System: Operational")

# 5. Header
st.markdown("<div style='border-left:8px solid #00d4ff; padding:10px; background:rgba(0
