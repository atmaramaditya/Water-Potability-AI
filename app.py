import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
from streamlit_lottie import st_lottie

# 1. Page Configuration
st.set_page_config(page_title="Water Potability AI", page_icon="💧", layout="centered")

# 2. Function to load smart animations
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Loading a professional water-related animation
lottie_water = load_lottieurl("https://lottie.host/677054f1-6718-47c0-8d54-1596541f92e8/4C0h0P8FPr.json")

# 3. Load the saved model and scaler (using the OS path fix)
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'water_model.pkl')
scaler_path = os.path.join(base_path, 'scaler.pkl')

@st.cache_resource # This makes the app faster
def load_assets():
    with open(model_path, 'rb') as m_file:
        model = pickle.load(m_file)
    with open(scaler_path, 'rb') as s_file:
        scaler = pickle.load(s_file)
    return model, scaler

model, scaler = load_assets()

# 4. App Header
st.title("💧 Water Quality Predictor")
st.subheader("AI and Data Science Project")
st.write("This application uses a **Random Forest** model to predict water safety based on physicochemical properties.")

# 5. User Input Section (Organized in Columns for better UI)
st.markdown("### 🛠️ Enter Water Parameters")
col1, col2 = st.columns(2)

with col1:
    ph = st.number_input("pH Level", 0.0, 14.0, 7.0, help="Measure of acidity/alkalinity")
    hardness = st.number_input("Hardness (mg/L)", value=196.36)
    solids = st.number_input("Solids (ppm)", value=22014.09)
    chloramines = st.number_input("Chloramines (ppm)", value=7.12)
    sulfate = st.number_input("Sulfate (mg/L)", value=333.60)

with col2:
    conductivity = st.number_input("Conductivity (μS/cm)", value=426.20)
    organic_carbon = st.number_input("Organic Carbon (ppm)", value=14.28)
    trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=66.40)
    turbidity = st.number_input("Turbidity (NTU)", value=3.96)

# 6. Prediction Logic
if st.button("Predict Potability", use_container_width=True):
    # Array in the EXACT order of the provided dataset columns
    input_features = np.array([[ph, hardness, solids, chloramines, sulfate, 
                                conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    # Scale inputs
    scaled_features = scaler.transform(input_features)
    
    # Generate prediction
    prediction = model.predict(scaled_features)
    
    st.markdown("---")
    
    if prediction[0] == 1:
        st.balloons() # SMART ANIMATION: Flying Balloons
        if lottie_water:
            st_lottie(lottie_water, height=200, key="water_anim")
        st.success("### ✅ Result: Potable (Safe to Drink)")
        st.confetti = True # Visual feedback
    else:
        st.error("### ❌ Result: Not Potable (Unsafe)")
        st.warning("High levels of certain parameters detected. Water treatment required.")

# 7. Professional Footer (Updated with your project details)
st.markdown("---")
st.subheader("📊 Project Information")
st.write("""
This tool was developed to demonstrate the application of Machine Learning in Environmental Engineering.
""")

info_col1, info_col2 = st.columns(2)
with info_col1:
    st.info("**Algorithm:** Random Forest")
    st.info("**Dataset:** Provided Project Dataset")

with info_col2:
    st.info("**Application:** Water Safety Analysis")
    st.info("**Tools:** Streamlit, Scikit-Learn")

st.caption("Developed for Mechatronics & AI Engineering Portfolio | MPSTME & BIA")
