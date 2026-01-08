import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
from streamlit_lottie import st_lottie

# 1. Page Configuration
st.set_page_config(
    page_title="Water Potability AI | Aditya Atmaram", 
    page_icon="💧", 
    layout="wide"
)

# Professional UI Styling
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Asset Loading
def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_water = load_lottieurl("https://lottie.host/677054f1-6718-47c0-8d54-1596541f92e8/4C0h0P8FPr.json")
lottie_warning = load_lottieurl("https://lottie.host/880a4b73-0f73-455b-8007-9f6874c7e627/7Z2LqO1L5L.json")

@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        with open(os.path.join(base_path, 'water_model.pkl'), 'rb') as m_file:
            model = pickle.load(m_file)
        with open(os.path.join(base_path, 'scaler.pkl'), 'rb') as s_file:
            scaler = pickle.load(s_file)
        return model, scaler
    except: return None, None

model, scaler = load_assets()

# 3. CLEAN & PROFESSIONAL SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80) # Generic Profile Icon
    st.title("Aditya Atmaram")
    st.caption("Mechatronics Engineer | AI & Data Science Specialist")
    st.markdown("---")
    
    st.subheader("📊 Model Performance")
    # Custom Metric Cards for a professional look
    st.markdown("""
        <div class="metric-card">
            <small>Winning Algorithm</small><br>
            <strong>Random Forest Classifier</strong>
        </div>
        <div class="metric-card">
            <small>Model Accuracy</small><br>
            <strong>65%</strong>
        </div>
        <div class="metric-card">
            <small>Weighted F1-Score</small><br>
            <strong>0.64</strong>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🏫 Education")
    st.markdown("""
    * **MPSTME** (B.Tech Mechatronics)
    * **BIA** (Diploma in AI & DS)
    """)

# 4. Main App Content
st.title("💧 Water Potability Analysis")
st.write("Determine water safety by analyzing chemical and physical parameters using machine learning.")
st.markdown("---")

# Input Section
col1, col2, col3 = st.columns(3)
with col1:
    ph = st.number_input("pH Level", 0.0, 14.0, 7.0)
    hardness = st.number_input("Hardness (mg/L)", value=196.36)
    solids = st.number_input("Solids (ppm)", value=22014.0)
with col2:
    chloramines = st.number_input("Chloramines (ppm)", value=7.12)
    sulfate = st.number_input("Sulfate (mg/L)", value=333.60)
    conductivity = st.number_input("Conductivity (μS/cm)", value=426.20)
with col3:
    organic_carbon = st.number_input("Organic Carbon (ppm)", value=14.28)
    trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=66.40)
    turbidity = st.number_input("Turbidity (NTU)", value=3.96)

# 5. Prediction & Logic
if st.button("Analyze Water Sample"):
    if model and scaler:
        input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                                conductivity, organic_carbon, trihalomethanes, turbidity]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        
        st.markdown("---")
        
        if prediction == 1:
            st.balloons()
            st.success("### ✅ Result: Potable (Safe for Consumption)")
            if lottie_water: st_lottie(lottie_water, height=150)
        else:
            st.error("### ❌ Result: Not Potable (Unsafe)")
            if lottie_warning: st_lottie(lottie_warning, height=150)
            
            # Parameter Analysis (Highlighting reasons for non-potability)
            st.subheader("🔍 Parameter Analysis (Risk Factors)")
            st.info("The following inputs deviate from standard safety benchmarks:")
            
            issues_found = False
            if not (6.5 <= ph <= 8.5):
                st.warning(f"⚠️ **pH Level ({ph}):** Outside WHO range (6.5 - 8.5).")
                issues_found = True
            if chloramines > 4.0:
                st.warning(f"⚠️ **Chloramines ({chloramines} ppm):** Exceeds safe drinking limit of 4.0 ppm.")
                issues_found = True
            if sulfate > 250.0:
                st.warning(f"⚠️ **Sulfate ({sulfate} mg/L):** High levels (>250 mg/L) can cause gastrointestinal issues.")
                issues_found = True
            if solids > 1000.0:
                st.warning(f"⚠️ **Total Dissolved Solids ({solids} ppm):** High TDS levels (>1000) indicate high mineralization.")
                issues_found = True
            
            if not issues_found:
                st.warning("⚠️ **Complex Interaction:** While individual parameters appear borderline, the model's multivariate analysis suggests chemical instability.")
    else:
        st.error("Assets not loaded.")

st.markdown("---")
st.caption("Developed by Aditya Atmaram | Mechatronics & AI Engineering Portfolio")
