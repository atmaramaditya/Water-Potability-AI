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

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Helper Functions
def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_water = load_lottieurl("https://lottie.host/677054f1-6718-47c0-8d54-1596541f92e8/4C0h0P8FPr.json")
lottie_warning = load_lottieurl("https://lottie.host/880a4b73-0f73-455b-8007-9f6874c7e627/7Z2LqO1L5L.json")

# 3. Load Model & Scaler
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        with open(os.path.join(base_path, 'water_model.pkl'), 'rb') as m_file:
            model = pickle.load(m_file)
        with open(os.path.join(base_path, 'scaler.pkl'), 'rb') as s_file:
            scaler = pickle.load(s_file)
        return model, scaler
    except:
        st.error("Model assets not found.")
        return None, None

model, scaler = load_assets()

# 4. Sidebar - Identification
with st.sidebar:
    st.title("🚀 Project Info")
    st.markdown("### **Developer**")
    st.info("Aditya Atmaram")
    st.write("Mechatronics Engineering, MPSTME")
    st.write("AI & Data Science, BIA")
    st.markdown("---")
    st.write("This tool uses a Random Forest Classifier to evaluate water safety based on 9 physicochemical metrics.")

# 5. Main Header
st.title("💧 Water Quality Analysis System")
st.markdown("---")

# 6. User Input Section
st.markdown("### 🛠️ Enter Water Parameters")
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

# 7. Analysis Logic
if st.button("Run Diagnostic Analysis"):
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                            conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    
    st.markdown("---")
    
    if prediction == 1:
        st.balloons()
        st.success("### ✅ Result: Potable (Safe for Consumption)")
        if lottie_water:
            st_lottie(lottie_water, height=150)
    else:
        st.error("### ❌ Result: Not Potable (Unsafe)")
        if lottie_warning:
            st_lottie(lottie_warning, height=150)
        
        # --- NEW: Parameter Analysis Section ---
        st.subheader("🔍 Analysis of Non-Potability")
        st.write("The following parameters are outside the typical recommended safety ranges:")
        
        # Checking against standard WHO/EPA guidelines
        issues = []
        if not (6.5 <= ph <= 8.5):
            issues.append(f"• **pH Level ({ph}):** Outside safe range (6.5 - 8.5). Extreme pH can be corrosive or impact taste.")
        if chloramines > 4.0:
            issues.append(f"• **Chloramines ({chloramines} ppm):** Higher than the recommended limit of 4.0 ppm.")
        if sulfate > 250.0:
            issues.append(f"• **Sulfate ({sulfate} mg/L):** High sulfate levels can cause a laxative effect and bitter taste.")
        if solids > 1000.0:
            issues.append(f"• **Total Dissolved Solids ({solids} ppm):** High TDS indicates heavy mineralization or contamination.")
        if turbidity > 5.0:
            issues.append(f"• **Turbidity ({turbidity} NTU):** High turbidity can shield bacteria from disinfection.")

        if issues:
            for issue in issues:
                st.warning(issue)
        else:
            st.info("The model determined this sample is unsafe based on a complex combination of all parameters (Multivariate Analysis), even though individual levels may seem near-normal.")

# 8. Footer
st.markdown("---")
st.caption("Developed by Aditya Atmaram | Mechatronics & AI Engineering Portfolio")
