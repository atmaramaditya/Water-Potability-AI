import requests
from streamlit_lottie import st_lottie
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
# Function to load animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_water = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_96bovdur.json")

# After your prediction button logic:
if prediction == 1:
    st.balloons() # This adds a smart flying balloon animation
    st_lottie(lottie_water, height=200, key="water")
    st.success("The water is Potable!")
else:
    st.error("The water is NOT Potable.")

# The Updated About Section
st.markdown("---")
st.subheader("📊 Project Details")
st.write("""
This **AI and Data Science Project** leverages a **Random Forest Classifier** to determine water safety. 
By analyzing physicochemical parameters, the model identifies patterns that distinguish potable water from contaminated samples.
""")

c1, c2 = st.columns(2)
with c1:
    st.info("**Algorithm:** Random Forest")
with c2:
    st.info("**Dataset:** Provided Project Dataset")

st.caption("Developed for Mechatronics & AI Engineering Portfolio")


