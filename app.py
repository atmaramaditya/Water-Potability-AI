import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import requests

# 1. Page Configuration
st.set_page_config(
    page_title="HydroGuard AI | Aditya Atmaram", 
    page_icon="💧", 
    layout="wide"
)

# 2. IMPROVED CSS: High Contrast & Visibility
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

# 3. Asset Loading
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        with open(os.path.join(base_path, 'water_model.pkl'), 'rb') as m_file:
            model = pickle.load(m_file)
        with open(os.path.join(base_path, 'scaler.pkl'), 'rb') as s
