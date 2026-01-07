import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model and scaler
import os

# Get the directory where app.py is located
base_path = os.path.dirname(__file__)

# Create the full path to your files
model_path = os.path.join(base_path, 'water_model.pkl')
scaler_path = os.path.join(base_path, 'scaler.pkl')

# Load the files using the full path
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))
st.set_page_config(page_title="Water Potability AI", page_icon="💧")

st.title("💧 Water Quality Predictor")
st.write("AI and Data Science Project")

# Input fields matching your 'water_potability_cleaned.csv' columns
ph = st.number_input("pH Level", 0.0, 14.0, 7.0)
hardness = st.number_input("Hardness", value=196.36)
solids = st.number_input("Solids (ppm)", value=22014.09)
chloramines = st.number_input("Chloramines (ppm)", value=7.12)
sulfate = st.number_input("Sulfate (mg/L)", value=333.60)
conductivity = st.number_input("Conductivity (μS/cm)", value=426.20)
organic_carbon = st.number_input("Organic Carbon (ppm)", value=14.28)
trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=66.40)
turbidity = st.number_input("Turbidity", value=3.96)

if st.button("Predict Potability"):
    # Array must be in the EXACT order of your X_train columns
    input_features = np.array([[ph, hardness, solids, chloramines, sulfate, 
                                conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    # Scale inputs using your saved scaler
    scaled_features = scaler.transform(input_features)
    
    # Generate prediction
    prediction = model.predict(scaled_features)
    
    if prediction[0] == 1:
        st.success("✅ Result: Potable (Safe to Drink)")
    else:

        st.error("❌ Result: Not Potable (Unsafe)")
