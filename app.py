import requests
from streamlit_lottie import st_lottie
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# 1. Page Configuration
st.set_page_config(page_title="Water Potability AI", page_icon="💧")

# 2. Function to load animations
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_water = load_lottieurl("https://lottie.host/677054f1-6718-47c0-8d54-1596541f92e8/4C0h0P8FPr.json")

# 3. Load the saved model and scaler
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'water_model.pkl')
scaler_path = os.path.join(base_path, 'scaler.pkl')

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

# 4. UI Header
st.title("💧 Water Quality Predictor")
st.write("### AI and Data Science Project")
st.markdown("Enter the water parameters below to check if it is safe for consumption.")

# 5. Input fields (Layout adjusted for better look)
col_a, col_b = st.columns(2)

with col_a:
    ph = st.number_input("pH Level", 0.0, 14.0, 7.0)
    hardness = st.number_input("Hardness", value=196.36)
    solids = st.number_input("Solids (ppm)", value=22014.09)
    chloramines = st.number_input("Chloramines (ppm)", value=7.12)
    sulfate = st.number_input("Sulfate (mg/L)", value=333.60)

with col_b:
    conductivity = st.number_input("Conductivity (μS/cm)", value=426.20)
    organic_carbon = st.number_input("Organic Carbon (ppm)", value=14.28)
    trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=66.40)
    turbidity = st.number_input("Turbidity", value=3.96)

# 6. Prediction Logic
if st.button("Predict Potability"):
    # Prepare input array
    input_features = np.array([[ph, hardness, solids, chloramines, sulfate, 
                                conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    # Scale inputs
    scaled_features = scaler.transform(input_features)
    
    # Generate prediction
    prediction = model.predict(scaled_features)
    
    st.markdown("---")
    
    if prediction[0] == 1:
        st.balloons() # Flying balloons animation
        if lottie_water:
            st_lottie(lottie_water, height=200, key="success_water")
        st.success("### ✅ Result: Potable (Safe to Drink)")
    else:
        st.error("### ❌ Result: Not Potable (Unsafe)")

# 7. About Section (Always visible)
st.markdown("---")
st.subheader("📊 Project Details")
st.write("""
This project leverages a **Random Forest Classifier** to determine water safety. 
By analyzing physicochemical parameters, the model identifies patterns that distinguish potable water from contaminated samples.
""")

c1, c2 = st.columns(2)
with c1:
    st.info("**Algorithm:** Random Forest")
with c2:
    st.info("**Dataset:** Provided Project Dataset")

st.caption("Developed for AI and Data Science Diploma Portfolio")
