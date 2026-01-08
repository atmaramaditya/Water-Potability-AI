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

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .sidebar-text {
        font-size: 14px;
        color: #555;
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

# Loading Animations
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
    except FileNotFoundError:
        st.error("Model or Scaler files not found. Please ensure 'water_model.pkl' and 'scaler.pkl' are in the directory.")
        return None, None

model, scaler = load_assets()

# 4. Sidebar - Personal Branding & Info
with st.sidebar:
    st.title("👨‍💻 Developer Profile")
    st.markdown("### **Aditya Atmaram**")
    st.write("Mechatronics Engineering | MPSTME")
    st.write("AI & Data Science | BIA")
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info("**Algorithm:** Random Forest Classifier")
    st.info("**Accuracy:** [Insert your %]%")
    st.markdown("---")
    st.markdown("### 🔗 Connect")
    st.markdown("[LinkedIn](https://linkedin.com/in/yourprofile)")
    st.markdown("[GitHub](https://github.com/yourusername)")

# 5. Main Header Section
col_head1, col_head2 = st.columns([2, 1])
with col_head1:
    st.title("💧 Water Quality Analysis System")
    st.subheader("Predicting Potability with Machine Learning")
    st.write("""
        This intelligent system utilizes a **Random Forest** model trained on physicochemical water properties. 
        It evaluates 9 key parameters to determine if water is safe for human consumption according to WHO standards.
    """)

with col_head2:
    if lottie_water:
        st_lottie(lottie_water, height=180, key="header_anim")

st.markdown("---")

# 6. User Input Section (Organized into Logical Groups)
st.markdown("### 🛠️ Laboratory Parameters")
tab1, tab2 = st.tabs(["Chemical Levels", "Physical Properties"])

with tab1:
    c1, c2, c3 = st.columns(3)
    ph = c1.number_input("pH Level", 0.0, 14.0, 7.0, help="Measures Acid-Base balance (WHO range: 6.5 - 8.5)")
    hardness = c2.number_input("Hardness (mg/L)", 0.0, 500.0, 196.36)
    chloramines = c3.number_input("Chloramines (ppm)", 0.0, 20.0, 7.12)
    
    c4, c5, c6 = st.columns(3)
    sulfate = c4.number_input("Sulfate (mg/L)", 0.0, 500.0, 333.60)
    organic_carbon = c5.number_input("Organic Carbon (ppm)", 0.0, 40.0, 14.28)
    trihalomethanes = c6.number_input("Trihalomethanes (μg/L)", 0.0, 150.0, 66.40)

with tab2:
    p1, p2, p3 = st.columns(3)
    solids = p1.number_input("Total Dissolved Solids (ppm)", 0.0, 60000.0, 22014.0)
    conductivity = p2.number_input("Conductivity (μS/cm)", 0.0, 1000.0, 426.20)
    turbidity = p3.number_input("Turbidity (NTU)", 0.0, 10.0, 3.96)

# 7. Prediction Logic
if st.button("Run Diagnostic Analysis"):
    if model and scaler:
        input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                                conductivity, organic_carbon, trihalomethanes, turbidity]])
        
        # Scaling and Prediction
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][prediction] * 100

        st.markdown("---")
        
        if prediction == 1:
            st.balloons()
            st.success(f"### ✅ Result: Potable (Safe)")
            st.write(f"**Confidence Level:** {probability:.2f}%")
        else:
            st.error(f"### ❌ Result: Not Potable (Unsafe)")
            st.write(f"**Confidence Level:** {probability:.2f}%")
            
            with st.expander("📝 Detailed Scientific Analysis"):
                st.warning("One or more parameters deviate from safe drinking water standards.")
                st.write("""
                    - **pH Standards:** 6.5 to 8.5.
                    - **Chloramines:** Below 4 ppm is ideal.
                    - **Sulfate:** Below 250 mg/L is recommended.
                """)
    else:
        st.error("Error: Model assets not loaded properly.")

# 8. Professional Footer
st.markdown("---")
st.caption(f"© 2026 Aditya Atmaram | Mechatronics & AI Engineering Portfolio")
